import { tgpu } from 'typegpu'
import * as d from 'typegpu/data'
import { BrushUniforms } from './gpuSchema'

const VertexOutput = d.struct({
  position: d.builtin.position,
  localUV: d.location(0, d.vec2f),
  color: d.location(1, d.vec3f),
  opacity: d.location(2, d.f32),
  hardness: d.location(3, d.f32)
})

const brushVertexTemplate = `
@group(0) @binding(0) var<uniform> globals: BrushUniforms;

@vertex
fn vs(
  @location(0) quadPos: vec2<f32>,
  @location(1) pos: vec2<f32>,
  @location(2) size: f32,
  @location(3) pressure: f32
) -> VertexOutput {
  // Convert diameter to radius
  let radius = size * pressure;
  let pixelPos = pos + (quadPos * radius);
  
  // Convert pixel coordinates to Normalized Device Coordinates (NDC)
  let ndcX = (pixelPos.x / globals.screenSize.x) * 2.0 - 1.0;
  let ndcY = 1.0 - ((pixelPos.y / globals.screenSize.y) * 2.0);  // Flip Y axis for WebGPU coordinate system

  return VertexOutput(
    vec4<f32>(ndcX, ndcY, 0.0, 1.0),
    quadPos,
    globals.brushColor,
    pressure * globals.brushOpacity,
    globals.hardness
  );
}
`

export const brushVertex = tgpu.resolve({
  template: brushVertexTemplate,
  externals: {
    BrushUniforms,
    VertexOutput
  }
})

const brushFragmentTemplate = `
@group(0) @binding(0) var<uniform> globals: BrushUniforms;

@fragment
fn fs(v: VertexOutput) -> @location(0) vec4<f32> {
  var dist: f32;
  if (globals.brushShape == 1u) {
    // Calculate Chebyshev distance for square shape
    dist = max(abs(v.localUV.x), abs(v.localUV.y));
  } else {
    // Calculate Euclidean distance for circle shape
    dist = length(v.localUV);
  }

  if (dist > 1.0) { discard; }

  // Calculate alpha with hardness and anti-aliasing
  let edgeWidth = fwidth(dist);
  let startFade = min(v.hardness, 1.0 - edgeWidth * 2.0);
  let linearAlpha = 1.0 - smoothstep(startFade, 1.0, dist);
  // Apply quadratic falloff for smoother edges
  let alphaShape = pow(linearAlpha, 2.0);
  
  // Return premultiplied alpha color
  let alpha = alphaShape * v.opacity;
  return vec4<f32>(v.color * alpha, alpha);
}
`

export const brushFragment = tgpu.resolve({
  template: brushFragmentTemplate,
  externals: {
    VertexOutput,
    BrushUniforms
  }
})

const blitShaderTemplate = `
@vertex fn vs(@builtin(vertex_index) vIdx: u32) -> @builtin(position) vec4<f32> {
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0)
  );
  return vec4<f32>(pos[vIdx], 0.0, 1.0);
}

@group(0) @binding(0) var myTexture: texture_2d<f32>;

@fragment fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let c = textureLoad(myTexture, vec2<i32>(pos.xy), 0);
  // Treat texture as premultiplied to prevent double-darkening on overlaps
  return c;
}
`

export const blitShader = tgpu.resolve({
  template: blitShaderTemplate,
  externals: {}
})

const compositeShaderTemplate = `
@vertex fn vs(@builtin(vertex_index) vIdx: u32) -> @builtin(position) vec4<f32> {
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0)
  );
  return vec4<f32>(pos[vIdx], 0.0, 1.0);
}

@group(0) @binding(0) var myTexture: texture_2d<f32>;
@group(1) @binding(0) var<uniform> globals: BrushUniforms;

@fragment fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let sampled = textureLoad(myTexture, vec2<i32>(pos.xy), 0);
  // Apply global brush opacity to accumulated coverage
  return sampled * globals.brushOpacity;
}
`

export const compositeShader = tgpu.resolve({
  template: compositeShaderTemplate,
  externals: {
    BrushUniforms
  }
})

const readbackShaderTemplate = `
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> outputBuf: array<u32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let dims = textureDimensions(inputTex);
  if (id.x >= dims.x || id.y >= dims.y) { return; }

  let color = textureLoad(inputTex, vec2<i32>(id.xy), 0);
  
  var r = color.r;
  var g = color.g;
  var b = color.b;
  let a = color.a;

  if (a > 0.0) {
    r = r / a;
    g = g / a;
    b = b / a;
  }

  let ir = u32(clamp(r * 255.0, 0.0, 255.0));
  let ig = u32(clamp(g * 255.0, 0.0, 255.0));
  let ib = u32(clamp(b * 255.0, 0.0, 255.0));
  let ia = u32(clamp(a * 255.0, 0.0, 255.0));

  // Pack RGBA channels into a single u32 (Little Endian)
  let packed = ir | (ig << 8u) | (ib << 16u) | (ia << 24u);

  let index = id.y * dims.x + id.x;
  outputBuf[index] = packed;
}
`

export const readbackShader = tgpu.resolve({
  template: readbackShaderTemplate,
  externals: {}
})
