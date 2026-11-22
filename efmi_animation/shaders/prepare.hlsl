// EFMI rolling palette. Model-space poses are sampled from Blender and
// interpolated in parent space before applying native skin and inverse bind.
StructuredBuffer<float4> SourcePalette : register(t0);
StructuredBuffer<float4> Samples : register(t1);
Texture1D<float4> IniParams : register(t120);
RWStructuredBuffer<float4> ExpandedPalette : register(u0);

float4x4 Affine(float4 a, float4 b, float4 c)
{
    return float4x4(a, b, c, float4(0, 0, 0, 1));
}

float4x4 NativeSkin(uint row)
{
    return Affine(SourcePalette[row], SourcePalette[row + 1], SourcePalette[row + 2]);
}

bool ValidSkin(float4x4 skin)
{
    return all(isfinite(skin[0])) && all(isfinite(skin[1])) && all(isfinite(skin[2])) &&
           abs(dot(skin[0].xyz, cross(skin[1].xyz, skin[2].xyz))) > 1e-10;
}

float4 Slerp(float4 a, float4 b, float fraction)
{
    float d = dot(a, b);
    b = d < 0 ? -b : b;
    d = saturate(abs(d));
    if (d > 0.9995)
        return normalize(lerp(a, b, fraction));
    float angle = acos(d);
    return (sin((1 - fraction) * angle) * a + sin(fraction * angle) * b) / sin(angle);
}

float4x4 Pose(uint first, uint second, float fraction)
{
    float3 t = lerp(Samples[first].xyz, Samples[second].xyz, fraction);
    float4 q = Slerp(Samples[first + 1], Samples[second + 1], fraction);
    float3 s = lerp(Samples[first + 2].xyz, Samples[second + 2].xyz, fraction);
    float x = q.x, y = q.y, z = q.z, w = q.w;
    return Affine(
        float4(s * float3(1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)), t.x),
        float4(s * float3(2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)), t.y),
        float4(s * float3(2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)), t.z));
}

float4x4 AnimatedSkin(uint bone, uint boneStart, uint boneCount, uint clipBoneCount,
                      float4 clip, float elapsed)
{
    uint count = (uint)clip.y;
    float phase = count > 1 ? fmod(max(0, elapsed) * clip.z, clip.w) : 0;
    uint last = count - 1;
    uint first = min((uint)phase, last);
    uint second = first == last ? 0 : first + 1;
    float fraction = first < last ? frac(phase) : saturate((phase - last) / max(1, clip.w - last));
    uint firstRow = (uint)clip.x + first * clipBoneCount * 3;
    uint secondRow = (uint)clip.x + second * clipBoneCount * 3;
    uint infoRow = boneStart + 4 * bone;
    float4x4 skin = Affine(Samples[infoRow + 1], Samples[infoRow + 2], Samples[infoRow + 3]);
    [loop]
    for (uint depth = 0; depth < boneCount; ++depth)
    {
        float4 info = Samples[infoRow];
        uint sampleOffset = (uint)info.w * 3;
        skin = mul(Pose(firstRow + sampleOffset, secondRow + sampleOffset, fraction), skin);
        int parent = (int)info.z;
        if (parent < 0)
            break;
        infoRow = boneStart + 4 * (uint)parent;
    }
    return skin;
}

[numthreads(64, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    uint sampleRows, sampleStride, sourceRows, sourceStride, expandedRows, expandedStride;
    Samples.GetDimensions(sampleRows, sampleStride);
    SourcePalette.GetDimensions(sourceRows, sourceStride);
    ExpandedPalette.GetDimensions(expandedRows, expandedStride);
    if (sampleRows < 3 || sampleStride != 16 || sourceStride != 16 || expandedStride != 16)
        return;

    uint4 layout = (uint4)Samples[0];
    uint4 table = (uint4)Samples[1];
    uint physicsBones = (uint)Samples[2].x;
    float4 clock = IniParams.Load(int2(0, 0));
    uint implicitCount = (uint)IniParams.Load(int2(1, 0)).x;
    uint slot = (uint)clock.z;
    uint nativeRows = 3 * layout.x;
    uint drawRows = nativeRows + 3 * layout.y;
    uint implicitRows = 3 * implicitCount;
    uint sourceBase = implicitRows + 2 * nativeRows * slot;
    uint drawBase = implicitRows + 2 * drawRows * slot;
    if (!all(isfinite(clock)) || !nativeRows || implicitCount > layout.w || slot >= layout.z ||
        table.w + physicsBones + 4 * table.x > sampleRows || physicsBones + table.y != layout.y ||
        sourceBase + 2 * nativeRows > sourceRows || drawBase + 2 * drawRows > expandedRows)
        return;

    uint thread = threadId.x;
    if (thread < implicitRows)
    {
        ExpandedPalette[thread] = SourcePalette[thread];
        return;
    }
    thread -= implicitRows;
    if (thread < 2 * nativeRows)
    {
        ExpandedPalette[drawBase + (thread / nativeRows) * drawRows + thread % nativeRows] =
            SourcePalette[sourceBase + thread];
        return;
    }
    thread -= 2 * nativeRows;
    if (thread >= layout.y)
        return;

    bool isPhysics = thread < physicsBones;
    uint bone = isPhysics ? 0 : thread - physicsBones;
    float4 info = Samples[isPhysics ? table.w + thread : table.z + 4 * bone];
    uint palette = (uint)info.x;
    uint clipIndex = (uint)info.y;
    uint anchor = clipIndex;
    bool hasLink = isPhysics && info.z > 0;
    bool linked = hasLink && clock.w > 0;
    if (linked)
    {
        bone = (uint)info.z - 1;
        clipIndex = (uint)Samples[table.z + 4 * bone].y;
    }
    if (!isPhysics || linked)
        anchor = (uint)Samples[table.w + physicsBones + clipIndex].x;
    if (palette < layout.x || 3 * palette >= drawRows || 3 * anchor >= nativeRows)
        return;
    uint clipBoneCount = 0;
    bool animate = (!isPhysics || linked) && clock.w > 0;
    float4 clip = 0;
    float4x4 rootTransform = 0;
    if (animate)
    {
        clip = Samples[3 + clipIndex];
        uint rootRow = table.w + physicsBones + table.x + 3 * clipIndex;
        rootTransform = Affine(Samples[rootRow], Samples[rootRow + 1], Samples[rootRow + 2]);
        // The host packs each clip's samples frame-major. Its bone count
        // follows from the row offset of the next clip (or the resource end).
        uint clipEnd = clipIndex + 1 < table.x ? (uint)Samples[4 + clipIndex].x : sampleRows;
        clipBoneCount = (clipEnd - (uint)clip.x) / (3 * (uint)clip.y);
    }
    [unroll]
    for (uint bank = 0; bank < 2; ++bank)
    {
        float4x4 skin = NativeSkin(sourceBase + bank * nativeRows + 3 * anchor);
        // The dynamic native slot can still be uninitialized if its owner
        // has not drawn yet. Keep the mesh visible until the slot is imported.
        if (hasLink && !linked)
            skin = Affine(float4(1, 0, 0, 0), float4(0, 1, 0, 0), float4(0, 0, 1, 0));
        else if (!ValidSkin(skin))
        {
            skin = NativeSkin(sourceBase + 3 * anchor);
            if (!ValidSkin(skin))
                skin = Affine(float4(1, 0, 0, 0), float4(0, 1, 0, 0), float4(0, 0, 1, 0));
        }
        if (animate)
            skin = mul(mul(skin, rootTransform),
                       AnimatedSkin(bone, table.z, table.y, clipBoneCount, clip, clock[bank]));
        uint target = drawBase + bank * drawRows + 3 * palette;
        ExpandedPalette[target] = skin[0];
        ExpandedPalette[target + 1] = skin[1];
        ExpandedPalette[target + 2] = skin[2];
    }
}
