// EFMI rolling buffer instance CB adapter. cs_5_0 / main.
// Original bone count remains unchanged globally. Only the draw-local CB copy
// points at the extended current/history banks; static offsets stay intact.
cbuffer OriginalInstanceConfig : register(b0) { uint4 OriginalConfig[16]; };
StructuredBuffer<float4> Profile : register(t1);
Texture1D<float4> IniParams : register(t120);
RWBuffer<uint4> PatchedConfig : register(u0);
[numthreads(16,1,1)]
void main(uint3 id:SV_DispatchThreadID)
{
    uint row=id.x,n;PatchedConfig.GetDimensions(n);
    if(row>=16||n<16) return;
    uint4 value=OriginalConfig[row];
    uint count,stride;Profile.GetDimensions(count,stride);
    if(row==5&&count>=4&&stride==16)
    {
        float4 l=Profile[0],r=IniParams.Load(int2(0,0)),r2=IniParams.Load(int2(2,0));
        if(all(isfinite(l))&&all(isfinite(r))&&all(isfinite(r2))&&all(l>=0))
        {
            uint B=(uint)(l.x+0.5),K=(uint)(l.y+0.5),N=(uint)(l.z+0.5),I=(uint)max(0,floor(r2.y+0.5));
            uint slot=(uint)max(0,floor(r.y+0.5));
            if(B>0&&K>0&&K<=64&&B+K<=65536&&N>0&&N<=32&&slot<N&&I<=(uint)l.w)
            {
                uint base=(I+2*(B+K)*slot)*3;
                if(value.x>=I*3) value.x=base-3;
                if(value.y>=I*3) value.y=base+(B+K)*3-3;
            }
        }
    }
    PatchedConfig[row]=value;
}
