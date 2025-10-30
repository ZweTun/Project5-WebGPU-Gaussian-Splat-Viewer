

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
   //TODO: information passed from vertex shader to fragment shader
  @location(0) uv: vec2<f32>,
  @location(1) frag_coord: vec2<f32>,
  @location(2) color: vec4<f32>,
  @location(3) conic: vec2<f32>,

};



struct Splat {
  //TODO: information defined in preprocess compute shader
  position: u32,         // packs ndc.x, ndc.y
  color: array<u32, 2>,    // packs color.rg, color.b + opacity
  conic: array<u32, 2>,    // packs conic.xy, conic.z + radius
  depth: u32,               // packs view-space z
};

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct RenderSettings {
    gaussian_scaling: f32,
    sh_deg: f32,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(1) @binding(0) var<storage, read> splats: array<Splat>;
@group(1) @binding(1) var<storage, read> sorted_splat_indices: array<u32>;



@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VertexOutput {
    let corner = vi % 6u;              // 6 vertices per splat (two triangles)
    let splatIndex = ii;               // one instance per splat

    // Early out if out of bounds
    if (splatIndex >= arrayLength(&sorted_splat_indices)) {
        return VertexOutput(vec4<f32>(0.0), vec2<f32>(0.0), vec2<f32>(0.0), vec4<f32>(0.0), vec2<f32>(0.0));
    }

    let s = splats[sorted_splat_indices[splatIndex]];


    var offset: vec2<f32>;
    if (corner == 0u) { offset = vec2<f32>(-1.0, -1.0); }
    else if (corner == 1u) { offset = vec2<f32>( 1.0, -1.0); }
    else if (corner == 2u) { offset = vec2<f32>(-1.0,  1.0); }
    else if (corner == 3u) { offset = vec2<f32>( 1.0, -1.0); }
    else if (corner == 4u) { offset = vec2<f32>( 1.0,  1.0); }
    else                  { offset = vec2<f32>(-1.0,  1.0); }



    let ndc_center = unpack2x16float(s.position);
    let radius_ndc = unpack2x16float(s.conic[1]);


    let ndc_xy = ndc_center + offset * radius_ndc;

 
    let z_view = unpack2x16float(s.depth).x; // negative is in front of camera
    let clip_z = camera.proj[2][2] * z_view + camera.proj[3][2];
    let clip_w = camera.proj[2][3] * z_view + camera.proj[3][3];

    var out: VertexOutput;
    let ndc_z = clip_z / clip_w;

    let clip_xy = ndc_xy * clip_w;
    out.position = vec4<f32>(clip_xy, clip_z, clip_w);


    out.uv = offset * 0.5 + vec2<f32>(0.5, 0.5); 
    out.frag_coord = ndc_center;
    out.color = vec4<f32>(
        unpack2x16float(s.color[0]).x,
        unpack2x16float(s.color[0]).y,
        unpack2x16float(s.color[1]).x,
        unpack2x16float(s.color[1]).y
    );

    out.conic = unpack2x16float(s.conic[1]);

    return out;
}






@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Local quad coordinates in [-1, 1]
    let p = in.uv * 2.0 - vec2<f32>(1.0, 1.0);

    let r2 = dot(p, p);
    
    if (r2 > 1.0) {
        discard;
    }

    let weight = exp(-0.5 * r2);
    let opacity = in.color.a;
    let alpha = clamp(weight * opacity, 0.0, 1.0);

    let rgb = in.color.rgb * alpha;

    return vec4<f32>(rgb, alpha);
}