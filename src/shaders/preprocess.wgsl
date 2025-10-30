const SH_C0: f32 = 0.28209479177387814;
const SH_C1 = 0.4886025119029199;
const SH_C2 = array<f32,5>(
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
);
const SH_C3 = array<f32,7>(
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
);

override workgroupSize: u32;
override sortKeyPerThread: u32;

struct DispatchIndirect {
    dispatch_x: atomic<u32>,
    dispatch_y: u32,
    dispatch_z: u32,
}

struct SortInfos {
    keys_size: atomic<u32>,  // instance_count in DrawIndirect
    //data below is for info inside radix sort 
    padded_size: u32, 
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
}

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

struct Gaussian {
    pos_opacity: array<u32,2>,
    rot: array<u32,2>,
    scale: array<u32,2>
};

struct Splat {
  //TODO: store information for 2D splat rendering
  position: u32,         // packs ndc.x, ndc.y
  color: array<u32, 2>,    // packs color.rg, color.b + opacity
  conic: array<u32, 2>,    // packs conic.xy, conic.z + radius
  depth: u32,               // packs view-space z
};

//TODO: bind your data here
@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> render_settings: RenderSettings;

@group(1) @binding(0) var<storage, read> gaussians: array<Gaussian>;
@group(1) @binding(1) var<storage, read_write> splats: array<Splat>;
// SH coefficients buffer packed as f16 stream; read as u32 pairs and unpack
@group(1) @binding(2) var<storage, read> sh_packed: array<u32>;


@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sort_depths : array<u32>;
@group(2) @binding(2)
var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;


fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    
    //TODO: access your binded sh_coeff, see load.ts for how it is stored
    let max_coefs: u32 = 16u;
    if (c_idx >= max_coefs) { 
        return vec3<f32>(0.0); 
    }
    let base_f16 = splat_idx * (max_coefs * 3u) + c_idx * 3u;

    let pair0_idx = base_f16 >> 1u;    
    let is_odd = (base_f16 & 1u) == 1u;

    let pair0  = unpack2x16float(sh_packed[pair0_idx]);
    let pair1  = unpack2x16float(sh_packed[pair0_idx + 1u]);

    if (!is_odd) {
        return vec3<f32>(pair0.x, pair0.y, pair1.x);
    } else {

        return vec3<f32>(pair0.y, pair1.x, pair1.y);
    }
}




fn float_flip(f: f32) -> u32 {
    var u: u32 = bitcast<u32>(f);
    // If sign bit is negative, flip all bits otherwise flip just sign bit.
    let mask: u32 = select(0x80000000u, 0xFFFFFFFFu, (u & 0x80000000u) != 0u);
    return u ^ mask;
}

// spherical harmonics evaluation with Condon–Shortley phase
fn computeColorFromSH(dir: vec3<f32>, v_idx: u32, sh_deg: u32) -> vec3<f32> {
    var result = SH_C0 * sh_coef(v_idx, 0u);

    if sh_deg > 0u {

        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result += - SH_C1 * y * sh_coef(v_idx, 1u) + SH_C1 * z * sh_coef(v_idx, 2u) - SH_C1 * x * sh_coef(v_idx, 3u);

        if sh_deg > 1u {

            let xx = dir.x * dir.x;
            let yy = dir.y * dir.y;
            let zz = dir.z * dir.z;
            let xy = dir.x * dir.y;
            let yz = dir.y * dir.z;
            let xz = dir.x * dir.z;

            result += SH_C2[0] * xy * sh_coef(v_idx, 4u) + SH_C2[1] * yz * sh_coef(v_idx, 5u) + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coef(v_idx, 6u) + SH_C2[3] * xz * sh_coef(v_idx, 7u) + SH_C2[4] * (xx - yy) * sh_coef(v_idx, 8u);

            if sh_deg > 2u {
                result += SH_C3[0] * y * (3.0 * xx - yy) * sh_coef(v_idx, 9u) + SH_C3[1] * xy * z * sh_coef(v_idx, 10u) + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_coef(v_idx, 11u) + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coef(v_idx, 12u) + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_coef(v_idx, 13u) + SH_C3[5] * z * (xx - yy) * sh_coef(v_idx, 14u) + SH_C3[6] * x * (xx - 3.0 * yy) * sh_coef(v_idx, 15u);
            }
        }
    }
    result += 0.5;

    return  max(vec3<f32>(0.), result);
}

@compute @workgroup_size(workgroupSize,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gid.x;
    //TODO: set up pipeline as described in instruction


    let keys_per_dispatch = workgroupSize * sortKeyPerThread;
    // increment DispatchIndirect.dispatchx each time you reach limit for one dispatch of keys

    if idx < arrayLength(&gaussians) {
 
        let g = gaussians[idx];


        let p01 = unpack2x16float(g.pos_opacity[0]); // x, y
        let p2o = unpack2x16float(g.pos_opacity[1]); // z, opacity
        let pos_world = vec4<f32>(p01.x, p01.y, p2o.x, 1.0);
        let opacity = 1.0 / (1.0 + exp(-p2o.y));


        let view_pos = camera.view * pos_world;
        let clip = camera.proj * view_pos;
        let w = clip.w;


        let safe_w = max(abs(w), 1e-4);
        let ndc = clip.xy / w;

        let rot0= unpack2x16float(g.rot[0]);
        let rot1 = unpack2x16float(g.rot[1]);
        let q = vec4<f32>(rot0.y, rot1.x, rot1.y, rot0.x); // XYZW quaternion
        let x2 = q.x * q.x; let y2 = q.y * q.y; let z2 = q.z * q.z; let xy = q.x * q.y; let xz = q.x * q.z; let yz = q.y * q.z; let wx = q.w * q.x; let wy = q.w * q.y; let wz = q.w * q.z;

     

        let R = mat3x3<f32>(
            vec3<f32>(1.0 - 2.0*(y2 + z2), 2.0*(xy - wz), 2.0*(xz + wy)),
            vec3<f32>(2.0*(xy + wz), 1.0 - 2.0*(x2 + z2), 2.0*(yz - wx)),
            vec3<f32>(2.0*(xz - wy), 2.0*(yz + wx), 1.0 - 2.0*(x2 + y2))
        );

        let s = vec3<f32>( exp(unpack2x16float(g.scale[0])).x,
                           exp(unpack2x16float(g.scale[0])).y,
                           exp(unpack2x16float(g.scale[1])).x );
        let S = mat3x3<f32>(
            vec3<f32>(s.x * s.x, 0.0, 0.0),
            vec3<f32>(0.0, s.y * s.y, 0.0),
            vec3<f32>(0.0, 0.0, s.z * s.z)
        );

        // 3D covariance 
        let cov3D = R * S * transpose(R);

        let proj = camera.proj;

        // extract projection rows
        let p0 = vec3<f32>(proj[0][0], proj[1][0], proj[2][0]);
        let p1 = vec3<f32>(proj[0][1], proj[1][1], proj[2][1]);
        let p3 = vec3<f32>(proj[0][3], proj[1][3], proj[2][3]);

        let clip_w = clip.w;

        let j00 = (p0.x - ndc.x * p3.x) / clip_w;
        let j01 = (p0.y - ndc.x * p3.y) / clip_w;
        let j02 = (p0.z - ndc.x * p3.z) / clip_w;

        let j10 = (p1.x - ndc.y * p3.x) / clip_w;
        let j11 = (p1.y - ndc.y * p3.y) / clip_w;
        let j12 = (p1.z - ndc.y * p3.z) / clip_w;

        // 2x2 projected covariance
        let cov2_00 = j00 * (cov3D[0][0]*j00 + cov3D[0][1]*j01 + cov3D[0][2]*j02)
                    + j01 * (cov3D[1][0]*j00 + cov3D[1][1]*j01 + cov3D[1][2]*j02)
                    + j02 * (cov3D[2][0]*j00 + cov3D[2][1]*j01 + cov3D[2][2]*j02);

        let cov2_01 = j00 * (cov3D[0][0]*j10 + cov3D[0][1]*j11 + cov3D[0][2]*j12)
                    + j01 * (cov3D[1][0]*j10 + cov3D[1][1]*j11 + cov3D[1][2]*j12)
                    + j02 * (cov3D[2][0]*j10 + cov3D[2][1]*j11 + cov3D[2][2]*j12);

        let cov2_11 = j10 * (cov3D[0][0]*j10 + cov3D[0][1]*j11 + cov3D[0][2]*j12)
                    + j11 * (cov3D[1][0]*j10 + cov3D[1][1]*j11 + cov3D[1][2]*j12)
                    + j12 * (cov3D[2][0]*j10 + cov3D[2][1]*j11 + cov3D[2][2]*j12);

        let a = cov2_00;
        let d = cov2_11;
        let b = cov2_01;

        let det = a * d - b * b;

        let mid = 0.5 * (a + d);
        let l1 = mid + sqrt(max(0.0, mid * mid - det));
        let l2 = mid - sqrt(max(0.0, mid * mid - det));
        
        let radiusPixel = ceil(2.5 * sqrt(max(l1, l2))) * render_settings.gaussian_scaling;
        var rx_ndc = (2.0 * radiusPixel) / camera.viewport.x;
        var ry_ndc = (2.0 * radiusPixel) / camera.viewport.y;

        let det_inv = 1.0 / det;
        let storage_conic = vec3<f32>(d * det_inv, -b * det_inv, a * det_inv);

        
        if (w <= 0.0) {
            return;
        }
        
        // Cull if outside view frustum 
        if (abs(ndc.x) > 1.0 + rx_ndc || abs(ndc.y) > 1.0 + ry_ndc) {
            return;
        }


        let dir = normalize(-view_pos.xyz);
        let color = computeColorFromSH(dir, idx, 3u);
        

        let draw_count = atomicAdd(&sort_infos.keys_size, 1u);
        if (draw_count < arrayLength(&sort_indices)) {
            sort_indices[draw_count] = draw_count;
            sort_depths[draw_count] = float_flip(-view_pos.z);
        } else {
            return;
        }


        splats[draw_count].position = pack2x16float(ndc);
        splats[draw_count].color[0] = pack2x16float(color.xy);
        splats[draw_count].color[1] = pack2x16float(vec2<f32>(color.z, opacity));
        splats[draw_count].conic[0] = pack2x16float(storage_conic.xy);
        splats[draw_count].conic[1] = pack2x16float(vec2<f32>(rx_ndc, ry_ndc));
        splats[draw_count].depth = pack2x16float(vec2<f32>(view_pos.z, 0.0));

     
        let written: u32 = draw_count + 1u;

        //Update dispatch_x
        if (written % keys_per_dispatch == 0u) {
            atomicAdd(&sort_dispatch.dispatch_x, 1u);
        }



    }
}