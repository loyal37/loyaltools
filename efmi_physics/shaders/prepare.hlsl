// EFMI rolling palette adapter v2. cs_5_0 / main. No engine injection.
// One dispatch selects ONE chain. Thread zero walks its connected segments;
// every chain is prepared before a draw. All current/history publication happens
// once per spatial slot and frame, independently of component/IB/pass order.
StructuredBuffer<float4> SourcePalette : register(t0);
StructuredBuffer<float4> Profile : register(t1);
Texture1D<float4> IniParams : register(t120);
RWStructuredBuffer<float4> ExpandedPalette : register(u0);
RWStructuredBuffer<float4> State : register(u1);
RWStructuredBuffer<float4> ChainState : register(u2);

// Metadata is stored as exact normal float numbers (all integers < 2^24).
// Bit-casting a small uint frame into a float creates a denormal; SM5 branch
// merges may flush that value to zero and silently break same-frame freezing.
static const uint MAGIC=4546502;
static const uint MASK=1048575;
static const uint SR=24;
static const uint CR=4;
static const uint PENDING=1, PUBLISHED=2, RESET_HISTORY=4;
// Only linked chains use the anchor snapshots in the first segment's
// previously unused state rows. The original per-segment ABI stays intact.
static const uint PENDING_ANCHOR=14, CURRENT_ANCHOR=17, HISTORY_ANCHOR=20;
uint Number(float v) { return (uint)max(0.0,floor(v+0.5)); }
bool Finite3(float3 v) { return all(isfinite(v)); }
bool Finite4(float4 v) { return all(isfinite(v)); }
float3 Unit(float3 v,float3 fallback)
{
    float q=dot(v,v);
    return (Finite3(v)&&q>1e-12&&q<1e16)?v*rsqrt(q):fallback;
}
float3 Perpendicular(float3 y)
{
    float3 a=abs(y.x)<0.8?float3(1,0,0):float3(0,0,1);
    return Unit(a-y*dot(a,y),float3(0,0,1));
}
float3 Point(float4 a,float4 b,float4 c,float3 p)
{ return float3(dot(a,float4(p,1)),dot(b,float4(p,1)),dot(c,float4(p,1))); }
float3 Direction(float4 a,float4 b,float4 c,float3 v)
{ return float3(dot(a.xyz,v),dot(b.xyz,v),dot(c.xyz,v)); }
void Identity(out float4 a,out float4 b,out float4 c)
{ a=float4(1,0,0,0);b=float4(0,1,0,0);c=float4(0,0,1,0); }
void Multiply(float4 a,float4 b,float4 c,float4 i0,float4 i1,float4 i2,
              out float4 s0,out float4 s1,out float4 s2)
{
    float4 last=float4(0,0,0,1);
    s0=a.x*i0+a.y*i1+a.z*i2+a.w*last;
    s1=b.x*i0+b.y*i1+b.z*i2+b.w*last;
    s2=c.x*i0+c.y*i1+c.z*i2+c.w*last;
}
void CopyAnchor(uint dest,uint source)
{
    State[dest]=State[source];State[dest+1]=State[source+1];State[dest+2]=State[source+2];
}
bool AnchorCorrection(uint desired,uint previous,out float4 d0,out float4 d1,out float4 d2)
{
    Identity(d0,d1,d2);
    float4 a=State[previous],b=State[previous+1],c=State[previous+2];
    float3 col0=cross(b.xyz,c.xyz),col1=cross(c.xyz,a.xyz),col2=cross(a.xyz,b.xyz);
    float det=dot(a.xyz,col0);
    if(!isfinite(det)||abs(det)<=1e-10) return false;
    float3 t=float3(a.w,b.w,c.w);
    float3 x=float3(col0.x,col1.x,col2.x)/det;
    float3 y=float3(col0.y,col1.y,col2.y)/det;
    float3 z=float3(col0.z,col1.z,col2.z)/det;
    Multiply(ExpandedPalette[desired],ExpandedPalette[desired+1],ExpandedPalette[desired+2],
             float4(x,-dot(x,t)),float4(y,-dot(y,t)),float4(z,-dot(z,t)),d0,d1,d2);
    return Finite4(d0)&&Finite4(d1)&&Finite4(d2);
}
void Skin(float3 head,float3 tail,float3 targetX,float3 targetY,float3 fallbackX,
          float4 i0,float4 i1,float4 i2,out float4 s0,out float4 s1,out float4 s2)
{
    float3 y=Unit(tail-head,targetY);
    float3 projected=targetX-y*dot(targetX,y);
    if(dot(projected,projected)<1e-8) projected=fallbackX-y*dot(fallbackX,y);
    float3 x=Unit(projected,Perpendicular(y));
    float3 z=Unit(cross(x,y),Perpendicular(y));
    x=Unit(cross(y,z),x);
    // G_phys * inverse_bind. The resulting SKIN matrix is consumed once by the
    // game's skinning path; never multiply the extra bone by gameSkin again.
    Multiply(float4(x.x,y.x,z.x,head.x),float4(x.y,y.y,z.y,head.y),
             float4(x.z,y.z,z.z,head.z),i0,i1,i2,s0,s1,s2);
}
float3 Limit(float3 p,float3 head,float3 target,float length,float cone)
{
    float3 d=Unit(p-head,target);
    float ct=clamp(dot(d,target),-1.0,1.0), bound=cos(cone);
    if(ct<bound) d=target*bound+Unit(d-target*ct,Perpendicular(target))*sin(cone);
    return head+Unit(d,target)*length;
}

[numthreads(64,1,1)]
void main(uint3 dispatchId:SV_DispatchThreadID)
{
    uint pc,ps;Profile.GetDimensions(pc,ps);
    if(pc<4||ps!=16) return;
    float4 l=Profile[0], tables=Profile[1], limits=Profile[2], abi=Profile[3];
    float4 r0=IniParams.Load(int2(0,0)),r1=IniParams.Load(int2(1,0)),r2=IniParams.Load(int2(2,0));
    if(!Finite4(l)||!Finite4(tables)||!Finite4(limits)||!Finite4(abi)||
       !Finite4(r0)||!Finite4(r1)||!Finite4(r2)) return;
    uint B=Number(l.x),K=Number(l.y),N=Number(l.z),I=Number(r2.y);
    uint C=Number(tables.x),chainStart=Number(tables.y),segmentStart=Number(tables.z),stride=Number(tables.w);
    uint slot=Number(r0.y),chain=Number(r2.z),frame=Number(r0.x)&MASK,generation=Number(r1.z);
    if(B==0||B+K>65536||K==0||K>64||N==0||N>32||I>Number(l.w)||
       C==0||C>K||chain>=C||slot>=N||stride!=12||Number(abi.y)!=2||
       Number(abi.z)!=SR||Number(abi.w)!=CR||chainStart!=4||segmentStart!=4+2*C||
       segmentStart+K*stride>pc) return;
    uint sc,ss,ec,es,stc,sts,cc,cs;
    SourcePalette.GetDimensions(sc,ss);ExpandedPalette.GetDimensions(ec,es);
    State.GetDimensions(stc,sts);ChainState.GetDimensions(cc,cs);
    // With animation, its dispatch already seeded both banks and any rigid
    // fallbacks. The signed width selects whether this chain may publish.
    uint E=B+(r2.w==0 ? K : Number(abs(r2.w))),srcBase=(I+2*B*slot)*3,extBase=(I+2*E*slot)*3;
    uint srcRows=3*B,extRows=3*E,implicitRows=3*I;
    uint ct=(slot*C+chain)*CR;
    if(E<B+K||E>65536||ss!=16||es!=16||sts!=16||cs!=16||srcBase+2*srcRows>sc||extBase+2*extRows>ec||
       implicitRows>sc||implicitRows>ec||N*K*SR>stc||ct+CR>cc) return;
    uint tid=dispatchId.x;
    // Disjoint source-copy and extra-bone writes. No cross-group dependency.
    if(r2.w==0 && tid<implicitRows) ExpandedPalette[tid]=SourcePalette[tid];
    else if(r2.w==0 && tid-implicitRows<2*srcRows)
    {
        uint n=tid-implicitRows;
        ExpandedPalette[extBase+(n/srcRows)*extRows+n%srcRows]=SourcePalette[srcBase+n];
    }
    if(tid!=0) return;
    float4 anchorInfo=Profile[chainStart+2*chain],range=Profile[chainStart+2*chain+1];
    uint first=Number(range.x),count=Number(range.y),anchorID=Number(anchorInfo.x);
    uint offset=Number(anchorInfo.z),componentCount=Number(anchorInfo.w);
    if(!Finite4(anchorInfo)||!Finite4(range)||count==0||count>64||first+count>K||
       anchorID>=B||offset>=B||componentCount==0||componentCount>B-offset||
       anchorID<offset||anchorID>=offset+componentCount) return;
    uint animationID=Number(range.z);
    bool linked=r2.x>=2.0&&animationID>B+K&&animationID<=E;
    uint animationAt=linked?extBase+3*(animationID-1):0;
    uint anchorState=(slot*K+first)*SR;

    float4 rawMeta=ChainState[ct];
    uint4 meta=(uint4)max(0.0,rawMeta);
    float4 times=ChainState[ct+1];
    uint flags=Number(times.z);
    bool freshState=!Finite4(rawMeta)||meta.x!=MAGIC||!Finite4(times)||
                    (flags&~(PENDING|PUBLISHED|RESET_HISTORY))!=0;
    bool reset=freshState||meta.w!=generation||times.y!=r1.w;
    if(reset)
    {
        // A late anchor LoD/epoch change must preserve an already consumed pose.
        bool preserve=!freshState&&meta.y==frame;
        uint keep=preserve?(flags&PUBLISHED):0;
        [loop] for(uint j=0;j<count;j++)
        {
            uint st=(slot*K+first+j)*SR;
            float4 a,b,c;Identity(a,b,c);
            if(!preserve)
            {
                State[st+5]=a;State[st+6]=b;State[st+7]=c;
                State[st+8]=a;State[st+9]=b;State[st+10]=c;
            }
            State[st]=0;State[st+1]=0;State[st+2]=0;
            State[st+3]=float4(0,1,0,0);State[st+4]=float4(1,0,0,0);
            State[st+11]=a;State[st+12]=b;State[st+13]=c;
        }
        meta=uint4(MAGIC,frame,MASK+1,generation);
        flags=keep|RESET_HISTORY;times=float4(r1.y,r1.w,0,0);
    }
    else if(meta.y!=frame)
    {
        if(linked)
        {
            // Pair the pending/published physics poses with their own anchor
            // snapshots so both banks can be rebased to this draw's pose.
            bool publish=(flags&PENDING)!=0;
            bool resetHistory=(flags&PUBLISHED)==0||(flags&RESET_HISTORY)!=0;
            uint history=publish&&resetHistory?PENDING_ANCHOR:CURRENT_ANCHOR;
            CopyAnchor(anchorState+HISTORY_ANCHOR,anchorState+history);
            if(publish) CopyAnchor(anchorState+CURRENT_ANCHOR,anchorState+PENDING_ANCHOR);
        }
        [loop] for(uint j=0;j<count;j++)
        {
            uint st=(slot*K+first+j)*SR;
            float4 a=State[st+5],b=State[st+6],c=State[st+7];
            float4 h0=a,h1=b,h2=c;
            if((flags&PENDING)!=0)
            {
                a=State[st+11];b=State[st+12];c=State[st+13];
                if((flags&PUBLISHED)==0||(flags&RESET_HISTORY)!=0) {h0=a;h1=b;h2=c;}
            }
            State[st+5]=a;State[st+6]=b;State[st+7]=c;
            State[st+8]=h0;State[st+9]=h1;State[st+10]=h2;
        }
        if((flags&PENDING)!=0) {flags|=PUBLISHED;flags&=~RESET_HISTORY;}
        meta.y=frame;
    }
    bool allowed=r0.z>0.5&&r0.w>0.5&&meta.z!=frame;
    uint reason=0;
    if(allowed)
    {
        uint at=srcBase+anchorID*3;
        float4 a0=SourcePalette[at],a1=SourcePalette[at+1],a2=SourcePalette[at+2];
        if(linked)
        {
            a0=ExpandedPalette[animationAt];
            a1=ExpandedPalette[animationAt+1];
            a2=ExpandedPalette[animationAt+2];
        }
        bool valid=Finite4(a0)&&Finite4(a1)&&Finite4(a2);
        float dt=r1.x,maxDt=clamp(limits.x,0.001,0.25);
        bool rigid=(r2.x-(linked?2.0:0.0))<0.5;
        bool restart=(flags&PENDING)==0||!isfinite(dt)||dt<0||dt>maxDt||
                     ((frame-meta.z)&MASK)>Number(limits.z);
        float4 root=Profile[segmentStart+first*stride+1];
        float4 root1=Profile[segmentStart+first*stride+2],root2=Profile[segmentStart+first*stride+3];
        float3 anchor=Point(a0,a1,a2,float3(root.w,root1.w,root2.w));
        if((flags&PENDING)!=0&&distance(anchor,ChainState[ct+2].xyz)>limits.y) restart=true;
        if(restart) flags|=RESET_HISTORY;
        valid=valid&&Finite3(anchor)&&all(abs(anchor)<abi.x);
        uint steps=restart||rigid?1:clamp(Number(limits.w),1u,16u);
        float h=dt/(float)steps;
        // PBD sweep: a segment head is its predecessor's tail. Parents are
        // solved first at each substep, so arbitrarily edited rest directions
        // and lengths propagate down a genuinely multi-segment chain.
        [loop] for(uint step=0;step<steps&&valid;step++)
        {
            float4 parent0=a0,parent1=a1,parent2=a2;
            float3 previousTail=anchor;
            [loop] for(uint j=0;j<count;j++)
            {
                uint seg=first+j,pr=segmentStart+seg*stride,st=(slot*K+seg)*SR;
                float4 info=Profile[pr],b0=Profile[pr+1],b1=Profile[pr+2],b2=Profile[pr+3];
                float4 i0=Profile[pr+4],i1=Profile[pr+5],i2=Profile[pr+6];
                float4 forces=Profile[pr+7],params=Profile[pr+8],extra=Profile[pr+9];
                float length=info.w;
                float3 head=j==0?anchor:previousTail;
                float3 rawY=Direction(parent0,parent1,parent2,float3(b0.y,b1.y,b2.y));
                float3 rawX=Direction(parent0,parent1,parent2,float3(b0.x,b1.x,b2.x));
                valid=valid&&Finite3(rawY)&&Finite3(rawX)&&dot(rawY,rawY)>1e-10&&dot(rawX,rawX)>1e-10;
                float3 dir=Unit(rawY,float3(0,1,0));
                float3 x=Unit(rawX,Perpendicular(dir));
                float3 tail=State[st].xyz,velocity=State[st+1].xyz,lastHead=State[st+2].xyz;
                bool bad=!Finite3(tail)||!Finite3(velocity)||!isfinite(length)||length<=1e-5;
                if(restart||rigid||bad) {tail=head+dir*length;velocity=0;}
                else if(h>0)
                {
                    if(step==0) tail+=(head-lastHead)*saturate(params.w);
                    float t=r1.y-dt+h*(step+1.0);
                    float3 wind=float3(params.z*(1.0+0.25*sin(t*3.14159265+seg)),0,0);
                    float3 accel=forces.xyz+wind+(120.0*params.x)*(head+dir*length-tail);
                    float3 before=tail;
                    float damping=exp(log(max(1.0-forces.w,1e-6))*h*60.0);
                    velocity=(velocity+accel*h)*damping;
                    velocity*=min(1.0,extra.x/max(sqrt(dot(velocity,velocity)),1e-6));
                    tail=Limit(tail+velocity*h,head,dir,length,params.y);
                    velocity=(tail-before)/h;
                    velocity*=min(1.0,extra.x/max(sqrt(dot(velocity,velocity)),1e-6));
                }
                if(!Finite3(tail)||!Finite3(velocity)||any(abs(tail)>abi.x))
                {tail=head+dir*length;velocity=0;reason=1;}
                float4 s0,s1,s2;
                Skin(head,tail,x,dir,State[st+4].xyz,i0,i1,i2,s0,s1,s2);
                // Exact rigid/reset comparison, including native affine scale:
                // S_anchor * B_segment * inverse(B_segment) == S_anchor.
                if(restart||rigid) {s0=a0;s1=a1;s2=a2;}
                valid=valid&&Finite4(s0)&&Finite4(s1)&&Finite4(s2)&&Number(info.x)>=B&&Number(info.x)<E;
                State[st]=float4(tail,0);State[st+1]=float4(velocity,0);
                State[st+2]=float4(head,0);State[st+3]=float4(dir,0);State[st+4]=float4(x,0);
                State[st+11]=s0;State[st+12]=s1;State[st+13]=s2;
                parent0=s0;parent1=s1;parent2=s2;previousTail=tail;
            }
        }
        if(valid)
        {
            flags|=PENDING;times.x=r1.y;times.w+=1;ChainState[ct+2]=float4(anchor,0);
            if(linked)
            {
                State[anchorState+PENDING_ANCHOR]=a0;
                State[anchorState+PENDING_ANCHOR+1]=a1;
                State[anchorState+PENDING_ANCHOR+2]=a2;
            }
        }
        else {flags&=~PENDING;flags|=RESET_HISTORY;reason=2;}
        meta.z=frame;
    }
    // Pending physics still advances once per frame. Rebase the published
    // bend to the current animation anchor so fast clips do not detach the
    // physics root for a frame. History uses its independent paired anchor.
    float4 d0,d1,d2,h0,h1,h2;
    Identity(d0,d1,d2);Identity(h0,h1,h2);
    bool publishLinked=true;
    if(linked)
    {
        publishLinked=(flags&PUBLISHED)!=0;
        if(publishLinked)
        {
            bool currentValid=AnchorCorrection(animationAt,anchorState+CURRENT_ANCHOR,d0,d1,d2);
            bool historyValid=AnchorCorrection(animationAt+extRows,anchorState+HISTORY_ANCHOR,h0,h1,h2);
            publishLinked=currentValid&&historyValid;
        }
    }
    [loop] for(uint j=0;r2.w>=0&&publishLinked&&j<count;j++)
    {
        uint seg=first+j,st=(slot*K+seg)*SR,palette=Number(Profile[segmentStart+seg*stride].x);
        if(palette<B||palette>=E) continue;
        uint outAt=extBase+3*palette;
        float4 a=State[st+5],b=State[st+6],c=State[st+7];
        float4 aPrev=State[st+8],bPrev=State[st+9],cPrev=State[st+10];
        if(linked)
        {
            Multiply(d0,d1,d2,a,b,c,a,b,c);
            Multiply(h0,h1,h2,aPrev,bPrev,cPrev,aPrev,bPrev,cPrev);
        }
        ExpandedPalette[outAt]=a;ExpandedPalette[outAt+1]=b;ExpandedPalette[outAt+2]=c;
        ExpandedPalette[outAt+extRows]=aPrev;ExpandedPalette[outAt+extRows+1]=bPrev;ExpandedPalette[outAt+extRows+2]=cPrev;
    }
    times.z=(float)flags;ChainState[ct]=(float4)meta;ChainState[ct+1]=times;
    ChainState[ct+3]=float4(frame,reason,allowed?1:0,(flags&PUBLISHED)!=0?1:0);
}
