"""Full Blender mesh -> EFMI buffer/INI export smoke test, no game required."""
from pathlib import Path
import importlib.util,json,sys,tempfile,types,unittest
try:
    import bpy
except ImportError:
    raise unittest.SkipTest('requires Blender 4.5')

ROOT=Path(__file__).resolve().parents[1]

def main():
    spec=importlib.util.spec_from_file_location('loyaltools_integration',ROOT/'__init__.py',submodule_search_locations=[str(ROOT)])
    addon=importlib.util.module_from_spec(spec);sys.modules[spec.name]=addon;spec.loader.exec_module(addon);addon.register()
    from loyaltools_integration.common.global_config import GlobalConfig
    from loyaltools_integration.common.efmi_merged_skeleton import write_profile
    from loyaltools_integration.common.draw_call_model import DrawCallModel
    from loyaltools_integration.ui.universal.efmi import ExportEFMI
    from loyaltools_integration.efmi_physics import rig
    from loyaltools_integration.blueprint.export_helper import BlueprintExportHelper
    import numpy as np
    with tempfile.TemporaryDirectory(prefix='loyaltools_full_export_') as folder:
        workspace=Path(folder);out=workspace/'export';(out/'Meshes').mkdir(parents=True)
        GlobalConfig.path_workspace_folder=classmethod(lambda cls:str(workspace))
        GlobalConfig.path_generate_mod_folder=classmethod(lambda cls:str(out))
        GlobalConfig.path_generatemod_buffer_folder=classmethod(lambda cls:str(out/'Meshes'))
        GlobalConfig.get_workspace_name=classmethod(lambda cls:'PhysicsIntegration')
        GlobalConfig.logic_name=rig.LogicName.EFMI
        components=[];objects=[]
        def element(name,fmt,width,index=0):
            return {'SemanticName':name,'SemanticIndex':index,'Format':fmt,'ByteWidth':width}
        categories=[
            {'Type':'Position','FileName':'position.buf','D3D11ElementList':[element('POSITION','R32G32B32_FLOAT',12),element('NORMAL','R32_UINT',4)]},
            {'Type':'Texcoord','FileName':'uv.buf','D3D11ElementList':[element('TEXCOORD','R32G32_FLOAT',8)]},
            {'Type':'Blend','FileName':'blend.buf','D3D11ElementList':[element('BLENDWEIGHTS','R16G16B16A16_UNORM',8),element('BLENDINDICES','R8G8B8A8_UINT',4)]},
        ]
        for cid,ib in enumerate(('11111111','22222222')):
            unique=ib+'-3-0';part=workspace/unique/'TYPE_GPU-EFMI';part.mkdir(parents=True)
            for item in categories:
                for e in item['D3D11ElementList']:
                    e.update(ExtractSlot={'Position':'vb0','Texcoord':'vb1','Blend':'vb2'}[item['Type']],ExtractTechnique='trianglelist',Category=item['Type'])
            data={'WorkGameType':'GPU-EFMI','GamePreset':'EFMI','GPU-PreSkinning':True,
                  'CategoryHash':{'Position':ib,'Texcoord':ib,'Blend':ib},
                  'CategoryDrawCategoryMap':{'Position':'Position','Texcoord':'Texcoord','Blend':'Blend'},
                  'IndexBufferList':[{'DXGI_FORMAT':'R16_UINT','FileName':'index.buf'}],
                  'CategoryBufferList':categories,'TextureMarkUpInfoList':[]}
            (part/(unique+'.json')).write_text(json.dumps(data),encoding='utf-8')
            components.append({'component_id':cid,'unique_str':unique,'ib_hash':ib,'index_count':3,'first_index':0,'first_vertex':0,'vertex_count':3,'cpu_posed':False,'vg_offset':cid*3,'vg_count':3,
                'vg_map':{str(i):cid*3+i for i in range(3)},
                'lods':[{'lod_object_name':'Fixture_LOD1','ib_hash':('aaaaaaaa','bbbbbbbb')[cid],'index_count':3,'first_index':0,'first_vertex':0,'vertex_count':3,'vg_map':{str(i):i for i in range(3)},
                    'vb_formats':{'VB1':{'stride':8,'semantics':[{'name':'TEXCOORD','index':0,'format':'R32G32_FLOAT','offset':0}]}}}]})
            mesh=bpy.data.meshes.new(unique);mesh.from_pydata([(0,0,0),(0,1,0),(1,1,0)],[],[(0,1,2)])
            obj=bpy.data.objects.new(unique+'.Cloth',mesh);bpy.context.scene.collection.objects.link(obj)
            obj[rig.OBJECT_MARKER_MERGED_SKELETON]=True;obj['LoyalTools:EFMIComponentId']=cid
            for i in range(6):obj.vertex_groups.new(name=str(i))
            obj.vertex_groups[str(cid*3)].add([0,1,2],1.0,'REPLACE')
            uv=mesh.uv_layers.new(name='TEXCOORD.xy')
            for loop,co in zip(uv.data,[(0,0),(0,1),(1,1)]):loop.uv=co
            objects.append(obj)
        (workspace/'Import.json').write_text(json.dumps({c['unique_str']:'GPU-EFMI' for c in components}),encoding='utf-8')
        profile={'mode':'EFMI_MERGED_SKELETON','format_version':1,'components':components,'max_instance_count':2,'object_guid':123456}
        write_profile(str(workspace),profile)
        from loyaltools_integration.common.efmi_merged_skeleton import load_profile
        profile=load_profile(str(workspace))
        def blueprint():
            return types.SimpleNamespace(ordered_draw_obj_data_model_list=[DrawCallModel(o.name) for o in objects],
                cross_ib_info_dict={},cross_ib_method_dict={},has_cross_ib=False,cross_ib_mapping_objects={},
                cross_ib_vb_condition_mapping={},cross_ib_source_to_target_dict={},cross_ib_object_vb_condition={},
                cross_ib_target_info={},cross_ib_match_mode='DRAW_IB',cross_ib_object_names=[],multi_file_export_nodes=[],keyname_mkey_dict={},
                _tree={'LoyalTools:EFMIMergedSkeleton':True})
        disabled=ExportEFMI(blueprint());disabled.export()
        ini=(out/'PhysicsIntegration.ini').read_text(encoding='utf-8')
        assert 'EFMIPhysics' not in ini
        settings=bpy.context.scene.loyaltools_efmi_physics
        settings.enabled=True
        for obj in objects:
            entry=rig.add_object(bpy.context,obj,profile)
            chain=rig.add_chain(bpy.context,entry,profile)
            chain.follow_group_id=objects.index(obj)*3
            rig.add_segment(bpy.context,entry,chain,profile)
            obj.vertex_groups[chain.segments[0].group_name].add([1],0.5,'REPLACE')
            obj.vertex_groups[chain.segments[1].group_name].add([2],0.5,'REPLACE')
        settings.enabled=True
        enabled=ExportEFMI(blueprint());enabled.export()
        ini=(out/'PhysicsIntegration.ini').read_text(encoding='utf-8')
        assert 'res/EFMIPhysics/Shaders/prepare.hlsl' in ini
        assert 'global $bones_count = 6' in ini
        assert 'ResourceEFMIPhysics_Extended' in ini
        assert '_LOD1' in ini and 'Fallback' in ini
        manifest=json.loads((out/'res/EFMIPhysics/authoring.json').read_text(encoding='utf-8'))
        assert len(enabled.efmi_physics_plan['bones'])==4
        assert len(enabled.efmi_physics_plan['chains'])==2
        used=[]
        for sm in enabled.submesh_model_list:
            stride=sm.d3d11_game_type.CategoryStrideDict['Blend']
            rows=sm.category_buffer_dict['Blend'].reshape(-1,stride)
            ids=np.ascontiguousarray(rows[:,8:16]).view(np.uint16)
            used.extend(ids.tolist())
        assert any(any(i>=6 for i in row) for row in used)
        print('EFMI_PHYSICS_FULL_EXPORT_OK',json.dumps({'bones':4,'chains':2,'meshes':2,'lods':[0,1],'manifest':bool(manifest)}))
    addon.unregister()

if __name__=='__main__':main()
