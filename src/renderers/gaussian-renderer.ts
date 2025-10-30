import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
import { Renderer } from './renderer';

export interface GaussianRenderer extends Renderer {
  set_gauss_mult: (v: number) => void,
}



// Utility to create GPU buffers
const createBuffer = (
  device: GPUDevice,
  label: string,
  size: number,
  usage: GPUBufferUsageFlags,
  data?: ArrayBuffer | ArrayBufferView
) => {
  const buffer = device.createBuffer({ label, size, usage });
  if (data) device.queue.writeBuffer(buffer, 0, (data as unknown) as BufferSource);
  return buffer;
};

export default function get_renderer(
  pc: PointCloud,
  device: GPUDevice,
  presentation_format: GPUTextureFormat,
  camera_buffer: GPUBuffer,
): GaussianRenderer {

  const sorter = get_sorter(pc.num_points, device);

  const bytes_per_splat = 32; 
  const splat_count = pc.num_points;
  const splat_data = new Uint32Array((bytes_per_splat / 4) * splat_count); // zero-initialized
  const indirect_data = new Uint32Array([6, 0, 0, 0]);




  // ===============================================
  //            Initialize GPU Buffers
  // ===============================================



  const nulling_data = new Uint32Array(4); // 16 bytes of zeros
  const nulling_buffer = createBuffer(
    device,
    'nulling buffer',
    nulling_data.byteLength,
    GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
    nulling_data
  );

  const indirect_buffer = createBuffer(
    device,
    'indirect buffer',
    indirect_data.byteLength, 
    GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    indirect_data
  );



  const splat_buffer = createBuffer(
    device,
    'splat buffer',
    splat_data.byteLength,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    splat_data
  );

  const render_settings_data = new Float32Array([1.0, pc.sh_deg, pc.num_points, 0.0]);
  const render_settings_buffer = device.createBuffer({
    label: 'render settings',
    size: render_settings_data.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(render_settings_buffer, 0, render_settings_data);

  // Write initial data to buffers
  device.queue.writeBuffer(splat_buffer, 0, splat_data);
  device.queue.writeBuffer(sorter.sort_info_buffer, 0, new Uint32Array([0]));
  device.queue.writeBuffer(indirect_buffer, 0, new Uint32Array([6, 0, 0, 0]));
  device.queue.writeBuffer(splat_buffer, 0, splat_data);
  device.queue.writeBuffer(nulling_buffer, 0, nulling_data);
  device.queue.writeBuffer(indirect_buffer, 0, indirect_data);
  
  // ===============================================
  //    Create Compute Pipeline and Bind Groups
  // ===============================================



  const preprocess_layout = device.createPipelineLayout({
    bindGroupLayouts: [
      device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ],
      }),
      device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        ],
      }),
      device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
          { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        ],
      }),
    ],
  });

  const preprocess_pipeline = device.createComputePipeline({
    label: 'preprocess',
    layout: preprocess_layout,
    compute: {
      module: device.createShaderModule({ code: preprocessWGSL }),
      entryPoint: 'preprocess',
      constants: {
        workgroupSize: C.histogram_wg_size,
        sortKeyPerThread: c_histogram_block_rows,
      },
    },
  });

  const preprocess_bind_group = device.createBindGroup({
    label: 'preprocess bind group',
    layout: preprocess_pipeline.getBindGroupLayout(0), 
    entries: [
      { binding: 0, resource: { buffer: camera_buffer } },  // group(0) binding 0
      { binding: 1, resource: { buffer: render_settings_buffer } },  // group(0) binding 1
    ],
  });

  // group(1) bind group (inputs/outputs to compute)
  const preprocess_io_bg = device.createBindGroup({
    label: 'preprocess io',
    layout: preprocess_pipeline.getBindGroupLayout(1),
    entries: [
      { binding: 0, resource: { buffer: pc.gaussian_3d_buffer } }, // gaussians
      { binding: 1, resource: { buffer: splat_buffer } },         // splats (read_write)
      { binding: 2, resource: { buffer: pc.sh_buffer } },         // SH coefficients

    ],
  });

  // group(2) control/sort buffers
  const preprocess_ctrl_bg = device.createBindGroup({
    label: 'preprocess control',
    layout: preprocess_pipeline.getBindGroupLayout(2),
    entries: [
      { binding: 0, resource: { buffer: sorter.sort_info_buffer } },      // sort_infos
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_depths_buffer } }, // sort_depths
      { binding: 2, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },// sort_indices
      { binding: 3, resource: { buffer: sorter.sort_dispatch_indirect_buffer } },   // indirect
    ],
  });






  // ===============================================
  //    Create Render Pipeline and Bind Groups
  // ===============================================

  const render_layout = device.createPipelineLayout({
  bindGroupLayouts: [
    device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }],
    }),
    device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],

    }),
  ],
  });

  const render_pipeline = device.createRenderPipeline({
    label: 'gaussian render',
    layout: render_layout,
    vertex: {
      module: device.createShaderModule({ code: renderWGSL }),
      entryPoint: 'vs_main',
    },
    fragment: {
      module: device.createShaderModule({ code: renderWGSL }),
      entryPoint: 'fs_main',
      targets: [{ 
        format: presentation_format,
        blend: {
          color: {
            srcFactor: 'one',
            dstFactor: 'one-minus-src-alpha',
            operation: 'add',
          },
          alpha: {
            srcFactor: 'one',
            dstFactor: 'one-minus-src-alpha',
            operation: 'add',
          }
        }
      }],
    },
    primitive: {
      topology: 'triangle-list', 
    },
  });

  

  const camera_bind_group = device.createBindGroup({
    label: 'gaussian camera',
    layout: render_pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: camera_buffer } }],
  });

  const splat_bind_group = device.createBindGroup({
    label: 'splat bind group',
    layout: render_pipeline.getBindGroupLayout(1),
    entries: [
      { binding: 0, resource: { buffer: splat_buffer } },
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
    ],
  });




  

  
  

  // ===============================================
  //    Command Encoder Functions
  // ====================================

    const render = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
    const pass = encoder.beginRenderPass({
      label: 'gaussian render pass',
      colorAttachments: [
        { 
          view: texture_view,
          loadOp: 'clear',
          storeOp: 'store',
        } 
      ],
    });
    pass.setPipeline(render_pipeline);
    pass.setBindGroup(0, camera_bind_group);
    pass.setBindGroup(1, splat_bind_group);
    pass.drawIndirect(indirect_buffer, 0);
    pass.end();
  };



  function set_gauss_mult(v: number) {
    device.queue.writeBuffer(render_settings_buffer, 0, new Float32Array([v, pc.sh_deg, pc.num_points, 0.0]));
  }






  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      //Clean buffers
      encoder.copyBufferToBuffer(nulling_buffer, 0, sorter.sort_info_buffer, 0, 4);
      encoder.copyBufferToBuffer(nulling_buffer, 0, sorter.sort_dispatch_indirect_buffer, 0, 4);


      // Preprocess compute pass
      const pass = encoder.beginComputePass({ label: 'preprocess compute pass' });
      pass.setPipeline(preprocess_pipeline);
      pass.setBindGroup(0, preprocess_bind_group);
      pass.setBindGroup(1, preprocess_io_bg);
      pass.setBindGroup(2, preprocess_ctrl_bg);
      pass.dispatchWorkgroups(Math.ceil(pc.num_points / C.histogram_wg_size));
      pass.end();

     
      //Sorting pass
      sorter.sort(encoder);

      // Copy number of sorted splats to indirect buffer
      encoder.copyBufferToBuffer(sorter.sort_info_buffer, 0, indirect_buffer, 4, 4);
      // Render pass
      render(encoder, texture_view);
    },
    camera_buffer,
    set_gauss_mult,

  };
}
