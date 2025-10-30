# University of Pennsylvania, CIS 5650: GPU Programming and Architecture
## Project 3 - WebGL Forward+ and Clustered Deferred Shading

* Zwe Tun
  * LinkedIn: https://www.linkedin.com/in/zwe-tun-6b7191256/
* Tested on: Intel(R) i7-14700HX, 2100 Mhz, RTX 5060 Laptop
![WebGPU](img/cover4.gif)


---

## Overview  
The WebGPU Gaussian Splat Viewer renders 3D scenes using Gaussian Splatting, where each point is represented by a smooth 3D Gaussian instead of a triangle. Each Gaussian defines its position, color, scale, and opacity, and when projected to screen space, becomes an elliptical “splat.” The viewer preprocesses and sorts these splats on the GPU using WebGPU, then blends them back-to-front to form continuous surfaces. 


### Implementation Summary  

- **Preprocessing (Compute Shader): **  
  Transforms each 3D Gaussian into camera space, performs view-frustum culling, and projects its covariance into a 2D ellipse. Spherical harmonics are evaluated for color, producing visible splats ready for sorting.

  **High-Level Steps:**  
- Transforms Gaussian means into camera space.

- Performs view-frustum culling to discard invisible splats.

- Projects 3D covariance matrices into 2D screen-space conics.

- Evaluates spherical harmonics based on view direction to compute color.

- Outputs visible Gaussians with updated screen-space properties.

- **Sorting (Compute Shader): **  
 Visible Gaussians are GPU-sorted by depth to ensure correct back-to-front transparency during rendering, preserving visual accuracy and avoiding blending artifacts.

  **High-Level Steps:**  
- Sorts visible splats by depth using radix sort to ensure correct front-to-back blending.

- Stores sorted indices in a GPU buffer for efficient draw calls.


- **Rasterization (Render Pipeline):**  
Each Gaussian becomes a screen-space quad. The vertex shader positions it, and the fragment shader computes opacity and color from the ellipse footprint, blending results  to form final image. 

  **High-Level Steps:**  
- The vertex shader expands each Gaussian into a 6-vertex quad (two triangles).

- The fragment shader evaluates the Gaussian density function to determine per-pixel opacity and color contribution.

- Blending accumulates splats to form the final image.

[![](img/thumb.png)](http://TODO.github.io/Project4-WebGPU-Forward-Plus-and-Clustered-Deferred)

### Demo Video/GIF

[![](img/video.mp4)](TODO)

### (TODO: Your README)

*DO NOT* leave the README to the last minute! It is a crucial part of the
project, and we will not be able to grade you without a good README.

This assignment has a considerable amount of performance analysis compared
to implementation work. Complete the implementation early to leave time!

### Credits

- [Vite](https://vitejs.dev/)
- [tweakpane](https://tweakpane.github.io/docs//v3/monitor-bindings/)
- [stats.js](https://github.com/mrdoob/stats.js)
- [wgpu-matrix](https://github.com/greggman/wgpu-matrix)
- Special Thanks to: Shrek Shao (Google WebGPU team) & [Differential Guassian Renderer](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
