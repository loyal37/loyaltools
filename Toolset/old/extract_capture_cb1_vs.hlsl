struct V2P {
    float4 pos : SV_Position;
    nointerpolation uint id : TEXCOORD0;
    nointerpolation uint4 raw_bits : TEXCOORD1;
};

cbuffer CaptureCB1 : register(b1) {
    float4 capture_cb1_data[4096];
};

V2P main(uint id : SV_VertexID) {
    V2P o;
    float x = -0.98 + (id % 64) * 0.03;
    float y = -0.98 + (id / 64) * 0.03;
    o.pos = float4(x, y, 0.5, 1.0);
    o.id = id;
    o.raw_bits = asuint(capture_cb1_data[id]);
    return o;
}
