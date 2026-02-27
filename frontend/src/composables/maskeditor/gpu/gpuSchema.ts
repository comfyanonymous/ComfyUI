import * as d from 'typegpu/data'

// Global brush uniforms
export const BrushUniforms = d.struct({
  brushColor: d.vec3f,
  brushOpacity: d.f32,
  hardness: d.f32,
  screenSize: d.vec2f,
  brushShape: d.u32 // 0: Circle, 1: Square
})

// Per-point instance data
export const StrokePoint = d.struct({
  pos: d.location(0, d.vec2f), // Center position
  size: d.location(1, d.f32), // Brush radius
  pressure: d.location(2, d.f32) // Pressure value (0.0 - 1.0)
})
