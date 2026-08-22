import time
import re
import numpy

from dataclasses import dataclass, field
from collections import defaultdict
from operator import itemgetter

from .migoto_object.migoto_object_builder import MigotoObject, MigotoComponent
from ..migoto_model.migoto_mesh import MigotoMesh, WeightingType
from ..migoto_model.migoto_mesh import GeometryMatcherConfig, GeometryMatcher, VertexGroupsMatcher


class LODMatcherError(Exception):
    pass


class ObjectLowSimilarityError(LODMatcherError):
    pass


class ComponentLowSimilarityError(LODMatcherError):
    pass


@dataclass
class SimilarityGraph:

    data: dict[MigotoComponent, dict[MigotoComponent, float]]

    def calculate_object_similarity(self) -> float:
        total_similarity = 0
        for lod_component, similarities in self.data.items():
            if not similarities:
                continue
            similarity = next(iter(similarities.values()))
            total_similarity += similarity
        weighted_similarity = total_similarity / len(self.data)
        return weighted_similarity

    def find_optimal_matching(self, min_similarity: float = 0.0) -> "SimilarityGraph":
        """Match LoD/full components one-to-one by maximum total similarity."""
        rows = list(self.data)
        columns = []
        seen_columns = set()
        for similarities in self.data.values():
            for candidate in similarities:
                if candidate not in seen_columns:
                    seen_columns.add(candidate)
                    columns.append(candidate)
        if not rows or not columns:
            return SimilarityGraph({})

        column_indices = {component: index for index, component in enumerate(columns)}
        real_column_count = len(columns)
        # One dummy column per row lets legitimately unmatched components remain
        # unmatched while keeping the Hungarian problem complete.
        weights = numpy.full(
            (len(rows), real_column_count + len(rows)),
            float(min_similarity),
            dtype=numpy.float64,
        )
        weights[:, :real_column_count] = -numpy.inf
        for row_index, component in enumerate(rows):
            for candidate, similarity in self.data[component].items():
                if similarity >= min_similarity:
                    weights[row_index, column_indices[candidate]] = similarity

        finite = numpy.isfinite(weights)
        max_weight = float(weights[finite].max())
        costs = numpy.full_like(weights, numpy.inf)
        costs[finite] = max_weight - weights[finite]
        row_count, column_count = costs.shape
        potentials_row = numpy.zeros(row_count + 1)
        potentials_column = numpy.zeros(column_count + 1)
        column_to_row = numpy.zeros(column_count + 1, dtype=numpy.int32)
        predecessor = numpy.zeros(column_count + 1, dtype=numpy.int32)

        for row in range(1, row_count + 1):
            column_to_row[0] = row
            min_cost = numpy.full(column_count + 1, numpy.inf)
            used = numpy.zeros(column_count + 1, dtype=bool)
            current_column = 0
            while True:
                used[current_column] = True
                current_row = column_to_row[current_column]
                delta = numpy.inf
                next_column = 0
                for column in range(1, column_count + 1):
                    if used[column]:
                        continue
                    cost = costs[current_row - 1, column - 1]
                    reduced = (
                        cost
                        - potentials_row[current_row]
                        - potentials_column[column]
                    )
                    if reduced < min_cost[column]:
                        min_cost[column] = reduced
                        predecessor[column] = current_column
                    if min_cost[column] < delta:
                        delta = min_cost[column]
                        next_column = column
                if not numpy.isfinite(delta):
                    raise ValueError("LOD 组件不存在完整的可行匹配。")
                for column in range(column_count + 1):
                    if used[column]:
                        potentials_row[column_to_row[column]] += delta
                        potentials_column[column] -= delta
                    else:
                        min_cost[column] -= delta
                current_column = next_column
                if column_to_row[current_column] == 0:
                    break
            while True:
                previous_column = predecessor[current_column]
                column_to_row[current_column] = column_to_row[previous_column]
                current_column = previous_column
                if current_column == 0:
                    break

        result = {}
        for column in range(1, column_count + 1):
            row = int(column_to_row[column])
            if row == 0 or column > real_column_count:
                continue
            weight = float(weights[row - 1, column - 1])
            if not numpy.isfinite(weight):
                continue
            result[rows[row - 1]] = {columns[column - 1]: weight}
        return SimilarityGraph(result)

    def verify_endmin_similarity_graph(self):
        endmin_lod1_to_full_map = {
            "5c29f1fc": "3d9e52b8",
            "070d7b84": "5825df15",
            "2f3d2c97": "b1f947ec",
            "3fc2a3de": "bf3c08af",
            "9b189efd": "b3bf2e13",
            "7cdfa2a3": "b57bbb30",
        }

        for lod_component, similarities in self.data.items():
            lod_hash = lod_component.metadata.ib_hash
            full_hash = next(iter(similarities.keys())).metadata.ib_hash
            correct_full_hash = endmin_lod1_to_full_map.get(lod_hash, None)
            if correct_full_hash is None:
                continue
            if full_hash != correct_full_hash:
                raise ValueError(f"LOD {lod_hash} matched {full_hash}, while {correct_full_hash} was expected")
            else:
                print(f"LOD {lod_hash} matched {full_hash} as expected")


@dataclass
class LODMatcher:

    component_min_vertex_count: int
    component_hash_blacklist: str

    object_similarity_threshold: float
    component_similarity_threshold: float
    skip_components_below_similarity_threshold: bool

    geo_matcher_main_config: GeometryMatcherConfig

    geo_matcher_prefilter_config: GeometryMatcherConfig
    geo_matcher_prefilter_candidates_count: int

    vg_matcher_candidates_count: int

    geo_matcher: GeometryMatcher = field(init=False)
    vg_matcher: VertexGroupsMatcher = field(init=False)

    def __post_init__(self):
        self.geo_matcher = GeometryMatcher(self.geo_matcher_main_config)
        self.vg_matcher = VertexGroupsMatcher(candidates_count=self.vg_matcher_candidates_count)

    def find_matching_lods(
        self,
        full_object: MigotoObject,
        lod_candidate_objects: list[MigotoObject],
    ) -> tuple[MigotoObject, dict[MigotoComponent, tuple[MigotoComponent, dict[int, int] | None]]]:
        t = time.time()

        lod_object_candidates = self.prefilter_lod_object_candidates(full_object, lod_candidate_objects)

        lod_object, hash_matched_components = self.find_lod_object_by_hash(full_object, lod_object_candidates)

        if lod_object is None:
            lod_object, object_similarity, similarity_graph = self.find_lod_object_by_similarity(full_object, lod_object_candidates)
            if object_similarity < self.object_similarity_threshold:
                raise ObjectLowSimilarityError(f"Best matching LoD for object {full_object.id} has {object_similarity:.2f}% similarity!")
        else:
            similarity_graph = self.match_components_by_similarity(full_object, lod_object, hash_matched_components)

        # similarity_graph.verify_endmin_similarity_graph()

        geo_matched_components = self.get_best_matching_components(similarity_graph)

        matched_components: dict[MigotoComponent, MigotoComponent] = (
            hash_matched_components | geo_matched_components
        )
            
        for lod_component in lod_object.components:
            if lod_component.metadata.mesh_name.startswith("Skipped"):
                continue
            if lod_component not in matched_components:
                lod_component.metadata.mesh_name = f"Skipped Component ib={lod_component.metadata.ib_hash} (no matching full component found)"

        print(f'Meshes match time: {time.time()-t:.2f}s')

        vg_maps = self.remap_vertex_groups(matched_components)

        result: dict[MigotoComponent, tuple[MigotoComponent, dict[int, int] | None]] = {}

        for lod_component, full_component in matched_components.items():
            result[full_component] = (lod_component, vg_maps.get(lod_component))

        return lod_object, result

    def prefilter_lod_object_candidates(
        self,
        full_object: MigotoObject,
        lod_candidate_objects: list[MigotoObject],
    ) -> list[MigotoObject]:

        candidates = []

        component_hash_blacklist = set([x for x in re.split(r"[,; ]", self.component_hash_blacklist) if x])

        lod_hashes = {}
        for full_component in full_object.components:
            for lod in full_component.metadata.lods:
                if lod.ib_hash == full_component.metadata.ib_hash:
                    continue
                lod_hashes[lod.ib_hash] = lod.lod_object_name

        for lod_object in lod_candidate_objects:
            # Skip object with 2+ times fewer components.
            if len(lod_object.components) < len(full_object.components) / 2:
                continue

            for lod_component in lod_object.components:

                # Check if lod_component hash is already imported from other lod object.
                known_lod_object = lod_hashes.get(lod_component.metadata.ib_hash, None)
                if known_lod_object is not None and known_lod_object != lod_object.id:
                    lod_component.metadata.mesh_name = f"Skipped Component ib={lod_component.metadata.ib_hash} (already imported from {known_lod_object})"
                    continue

                if lod_component.metadata.ib_hash in component_hash_blacklist:
                    lod_component.metadata.mesh_name = f"Skipped Component ib={lod_component.metadata.ib_hash} (component hash blacklisted)"
                    continue

                if lod_component.metadata.vertex_count < self.component_min_vertex_count:
                    lod_component.metadata.mesh_name = f"Skipped Component ib={lod_component.metadata.ib_hash} (vertex count below minimum)"
                    continue

            candidates.append(lod_object)
            
        return candidates

    def remap_vertex_groups(
        self,
        matched_components: dict[MigotoComponent, MigotoComponent]
    ) -> dict[MigotoComponent, dict[int, int]]:

        print(f"Remapping Vertex Groups for {len(matched_components)} components...")

        t = time.time()

        vg_maps = {}

        for lod_component, full_component in matched_components.items():
            vg_map = self.vg_matcher.match_vertex_groups(
                full_component.mesh,
                lod_component.mesh,
            )

            remapped = sum(1 for k, v in vg_map.items() if k != v)

            component_desc = f"{full_component.metadata.mesh_name} LoD (full={full_component.metadata.ib_hash}, lod={lod_component.metadata.ib_hash})"

            if remapped > 0:
                vg_maps[lod_component] = vg_map
                print(f"{component_desc}: {remapped} out of used {len(vg_map) or 1} VGs are different (LoD mesh uses simplified skeleton)")
            else:
                print(f"{component_desc}: all {len(vg_map)} VGs are identical (LoD mesh uses full skeleton)")

        print(f"Vertex Groups match time: {time.time() - t:.03f}s")

        return vg_maps

    def find_lod_object_by_hash(
        self,
        full_object: MigotoObject,
        lod_object_candidates: list[MigotoObject],
    ) -> tuple[MigotoObject | None, dict[MigotoComponent, MigotoComponent]]:

        full_by_hash = {component.metadata.ib_hash: component for component in full_object.components}

        lods: dict[MigotoObject, dict[MigotoComponent, MigotoComponent]] = {}

        for lod_object in lod_object_candidates:
            matches = {}

            for lod_component in lod_object.components:
                if lod_component.metadata.mesh_name.startswith("Skipped"):
                    continue

                full_component = full_by_hash.get(lod_component.metadata.ib_hash)

                if full_component is None:
                    continue

                matches[lod_component] = full_component

                similarity = self.geo_matcher.calculate_similarity(full_component.mesh, lod_component.mesh)

                lod_component.metadata.mesh_name = self.make_matched_mesh_name(full_component, lod_component, "hash")

                print(f"Match by hash (mesh similarity: {similarity:.2f}%): {full_component.__repr__()} == {lod_component.__repr__()} ")

            if matches:
                lods[lod_object] = matches

        if not lods:
            return None, {}

        matched_lod_object = max(
            lods,
            key=lambda obj: len(lods[obj]),
        )

        return matched_lod_object, lods[matched_lod_object]

    def find_lod_object_by_similarity(
        self,
        full_object: MigotoObject,
        lod_object_candidates: list[MigotoObject],
    ) -> tuple[MigotoObject, float, SimilarityGraph]:

        lod_object_similarity_graphs = {}
        lod_object_similarities = {}

        for lod_object in lod_object_candidates:
            similarity_graph = self.calculate_similarity_graph(full_object.components, lod_object.components)
            lod_object_similarity_graphs[lod_object] = similarity_graph
            lod_object_similarities[lod_object] = similarity_graph.calculate_object_similarity()

        matched_lod_object = max(
            lod_object_similarity_graphs,
            key=lambda obj: lod_object_similarities[obj],
        )

        object_similarity = lod_object_similarities[matched_lod_object]
        similarity_graph = lod_object_similarity_graphs[matched_lod_object]

        return matched_lod_object, object_similarity, similarity_graph

    def calculate_component_similarities(
        self,
        component: MigotoComponent,
        candidates: list[MigotoComponent],
    ) -> dict[MigotoComponent, float]:
        mesh_similarities = {}

        for candidate_component in candidates:
            similarity = self.geo_matcher.calculate_similarity(candidate_component.mesh, component.mesh)
            mesh_similarities[candidate_component] = similarity

        mesh_similarities = dict(
            sorted(mesh_similarities.items(), key=itemgetter(1), reverse=True)
        )

        return mesh_similarities

    def calculate_similarity_graph(
        self,
        full_components: list[MigotoComponent],
        lod_components: list[MigotoComponent],
    ) -> SimilarityGraph:

        similarities = {}

        for lod_component in lod_components:
            if lod_component.metadata.mesh_name.startswith("Skipped"):
                continue

            self.geo_matcher.cfg = self.geo_matcher_prefilter_config

            valid_full_components = [
                full_component for full_component in full_components
                if full_component.metadata.vertex_count >= lod_component.metadata.vertex_count
            ]

            prefilter_similarities = self.calculate_component_similarities(lod_component, valid_full_components)

            self.geo_matcher.cfg = self.geo_matcher_main_config

            prefiltered_full_components = list(prefilter_similarities.keys())[:self.geo_matcher_prefilter_candidates_count]

            similarities[lod_component] = self.calculate_component_similarities(lod_component, prefiltered_full_components)

        return SimilarityGraph(data=similarities)

    def match_components_by_similarity(
        self,
        full_object: MigotoObject,
        lod_object: MigotoObject,
        matched_lod_to_full_components: dict[MigotoComponent, MigotoComponent],
    ) -> SimilarityGraph:

        # Exclude already matched full components from matching.
        full_components = [
            full_component for full_component in full_object.components
            if full_component not in matched_lod_to_full_components.values()
        ]

        # Exclude already matched lod components from matching.
        lod_components = [
            lod_component for lod_component in lod_object.components
            if lod_component not in matched_lod_to_full_components.keys()
        ]

        similarity_graph = self.calculate_similarity_graph(full_components, lod_components)

        return similarity_graph
    
    def make_matched_mesh_name(self, full_component: MigotoComponent, lod_component: MigotoComponent, similarity: float | str) -> str:
        match_type = f"{similarity:.2f}%" if isinstance(similarity, float) else similarity
        mesh_name = f"{full_component.metadata.mesh_name} full={full_component.metadata.ib_hash} lod={lod_component.metadata.ib_hash} match={match_type}"
        if lod_component.metadata.ib_hash == full_component.metadata.ib_hash:
            if lod_component.metadata.vg_map:
                mesh_name += f" (full mesh, full skeleton)"
            else:
                mesh_name += f" (full mesh, simplified skeleton)"
        else:
            mesh_name += f" (simplified mesh and skeleton)"
        return mesh_name

    def get_best_matching_components(self, similarity_graph: SimilarityGraph) -> dict[MigotoComponent, MigotoComponent]:
        result = {}
        # Remove invalid edges before solving the one-to-one assignment.  If
        # the threshold is applied only after Hungarian matching, two weak
        # pairs can steal one strong, semantically correct pair merely because
        # their combined score is larger.  Dummy columns let surplus LoD
        # components remain unmatched instead.
        matched_graph = similarity_graph.find_optimal_matching(
            min_similarity=self.component_similarity_threshold
        )
        for lod_component, similarities in matched_graph.data.items():
            if not similarities:
                continue
            full_component, similarity = next(iter(similarities.items()))

            if similarity < self.component_similarity_threshold:
                if self.skip_components_below_similarity_threshold:
                    print(f"Skipped match by geometry below {self.component_similarity_threshold:.2f}% threshold (mesh similarity: {similarity:.2f}%): {full_component.__repr__()} == {lod_component.__repr__()} ")
                    lod_component.metadata.mesh_name = f"Skipped Component ib={lod_component.metadata.ib_hash} (mesh similarity {similarity:.2f}% is below configured {self.component_similarity_threshold:.2f}% threshold)"
                    continue
                raise ComponentLowSimilarityError(f"Best matching LoD for {full_component.metadata.mesh_name} has {similarity:.2f}% similarity!")
            
            lod_component.metadata.mesh_name = self.make_matched_mesh_name(full_component, lod_component, similarity)
            
            result[lod_component] = full_component

            print(f"Match by geometry (mesh similarity: {similarity:.2f}%): {full_component.__repr__()} == {lod_component.__repr__()} ")

        return result
