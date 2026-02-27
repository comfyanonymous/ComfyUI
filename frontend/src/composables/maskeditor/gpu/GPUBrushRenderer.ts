import * as d from 'typegpu/data'
import { StrokePoint } from './gpuSchema'
import {
  brushFragment,
  brushVertex,
  blitShader,
  compositeShader,
  readbackShader
} from './brushShaders'

// ... (rest of the file)

const QUAD_VERTS = new Float32Array([-1, -1, 1, -1, 1, 1, -1, 1])
const QUAD_INDICES = new Uint16Array([0, 1, 2, 0, 2, 3])

const UNIFORM_SIZE = 48 // Uniform buffer size aligned to 16 bytes
const STROKE_STRIDE = d.sizeOf(StrokePoint) // 16
const MAX_STROKES = 10000

export class GPUBrushRenderer {
  private device: GPUDevice

  // Buffers
  private quadVertexBuffer: GPUBuffer
  private indexBuffer: GPUBuffer
  private instanceBuffer: GPUBuffer
  private uniformBuffer: GPUBuffer

  // Pipelines
  private renderPipeline: GPURenderPipeline // Standard alpha blending pipeline
  private accumulatePipeline: GPURenderPipeline // SourceOver blending pipeline for stroke accumulation
  private blitPipeline: GPURenderPipeline
  private compositePipeline: GPURenderPipeline // Composite pipeline that applies opacity
  private compositePipelinePreview: GPURenderPipeline // Pipeline for rendering to the preview canvas
  private erasePipeline: GPURenderPipeline // Pipeline for erasing (Destination Out)
  private erasePipelinePreview: GPURenderPipeline // Eraser pipeline for the preview canvas
  readbackPipeline: GPUComputePipeline // Compute pipeline for texture readback

  // Bind Group Layouts
  private uniformBindGroupLayout: GPUBindGroupLayout
  private textureBindGroupLayout: GPUBindGroupLayout

  // Shared Bind Groups
  private mainUniformBindGroup: GPUBindGroup

  // Textures
  private currentStrokeTexture: GPUTexture | null = null
  private currentStrokeView: GPUTextureView | null = null

  // Cached Bind Groups
  private compositeTextureBindGroup: GPUBindGroup | null = null
  private previewTextureBindGroup: GPUBindGroup | null = null

  // Removed separate uniform bind groups as we will use mainUniformBindGroup

  private lastReadbackTexture: GPUTexture | null = null
  private lastReadbackBuffer: GPUBuffer | null = null
  private readbackBindGroup: GPUBindGroup | null = null

  private lastBackgroundTexture: GPUTexture | null = null
  private backgroundBindGroup: GPUBindGroup | null = null

  constructor(
    device: GPUDevice,
    presentationFormat: GPUTextureFormat = 'rgba8unorm'
  ) {
    this.device = device

    // --- 1. Initialize Buffers ---
    this.quadVertexBuffer = device.createBuffer({
      size: QUAD_VERTS.byteLength,
      usage: GPUBufferUsage.VERTEX,
      mappedAtCreation: true
    })
    new Float32Array(this.quadVertexBuffer.getMappedRange()).set(QUAD_VERTS)
    this.quadVertexBuffer.unmap()

    this.indexBuffer = device.createBuffer({
      size: QUAD_INDICES.byteLength,
      usage: GPUBufferUsage.INDEX,
      mappedAtCreation: true
    })
    new Uint16Array(this.indexBuffer.getMappedRange()).set(QUAD_INDICES)
    this.indexBuffer.unmap()

    this.instanceBuffer = device.createBuffer({
      size: MAX_STROKES * STROKE_STRIDE,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
    })

    this.uniformBuffer = device.createBuffer({
      size: UNIFORM_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    })

    // --- 2. Brush Shader (Drawing) ---
    const brushModuleV = device.createShaderModule({ code: brushVertex })
    const brushModuleF = device.createShaderModule({ code: brushFragment })

    // Create explicit bind group layouts
    this.uniformBindGroupLayout = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform' }
        }
      ]
    })

    this.textureBindGroupLayout = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT,
          texture: {} // default is float, 2d
        }
      ]
    })

    this.mainUniformBindGroup = device.createBindGroup({
      layout: this.uniformBindGroupLayout,
      entries: [{ binding: 0, resource: { buffer: this.uniformBuffer } }]
    })

    const renderPipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.uniformBindGroupLayout]
    })

    // Standard Render Pipeline (Alpha Blend)
    this.renderPipeline = device.createRenderPipeline({
      layout: renderPipelineLayout,
      vertex: {
        module: brushModuleV,
        entryPoint: 'vs',
        buffers: [
          {
            arrayStride: 8,
            stepMode: 'vertex',
            attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }] // Quad vertex attributes
          },
          {
            arrayStride: 16,
            stepMode: 'instance',
            attributes: [
              { shaderLocation: 1, offset: 0, format: 'float32x2' }, // Instance attributes: position
              { shaderLocation: 2, offset: 8, format: 'float32' }, // size
              { shaderLocation: 3, offset: 12, format: 'float32' } // pressure
            ]
          }
        ]
      },
      fragment: {
        module: brushModuleF,
        entryPoint: 'fs',
        targets: [
          {
            format: 'rgba8unorm',
            blend: {
              color: {
                srcFactor: 'one',
                dstFactor: 'one-minus-src-alpha',
                operation: 'add'
              },
              alpha: {
                srcFactor: 'one',
                dstFactor: 'one-minus-src-alpha',
                operation: 'add'
              }
            }
          }
        ]
      },
      primitive: { topology: 'triangle-list' }
    })

    // Accumulate strokes using SourceOver blending to ensure smooth intersections.
    this.accumulatePipeline = device.createRenderPipeline({
      layout: renderPipelineLayout,
      vertex: {
        module: brushModuleV,
        entryPoint: 'vs',
        buffers: [
          {
            arrayStride: 8,
            stepMode: 'vertex',
            attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }]
          },
          {
            arrayStride: 16,
            stepMode: 'instance',
            attributes: [
              { shaderLocation: 1, offset: 0, format: 'float32x2' },
              { shaderLocation: 2, offset: 8, format: 'float32' },
              { shaderLocation: 3, offset: 12, format: 'float32' }
            ]
          }
        ]
      },
      fragment: {
        module: brushModuleF,
        entryPoint: 'fs',
        targets: [
          {
            format: 'rgba8unorm',
            blend: {
              // Use SourceOver blending for smooth stroke intersections.
              color: {
                srcFactor: 'one',
                dstFactor: 'one-minus-src-alpha',
                operation: 'add'
              },
              alpha: {
                srcFactor: 'one',
                dstFactor: 'one-minus-src-alpha',
                operation: 'add'
              }
            }
          }
        ]
      },
      primitive: { topology: 'triangle-list' }
    })

    // --- 3. Blit Pipeline (For Preview) ---
    const blitPipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.textureBindGroupLayout]
    })

    this.blitPipeline = device.createRenderPipeline({
      layout: blitPipelineLayout,
      vertex: {
        module: device.createShaderModule({ code: blitShader }),
        entryPoint: 'vs'
      },
      fragment: {
        module: device.createShaderModule({ code: blitShader }),
        entryPoint: 'fs',
        targets: [
          {
            format: presentationFormat, // Use the presentation format
            blend: {
              color: {
                srcFactor: 'one',
                dstFactor: 'one-minus-src-alpha',
                operation: 'add'
              },
              alpha: {
                srcFactor: 'one',
                dstFactor: 'one-minus-src-alpha',
                operation: 'add'
              }
            }
          }
        ]
      },
      primitive: { topology: 'triangle-list' }
    })

    // --- 4. Composite Pipeline ---

    const compositePipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [
        this.textureBindGroupLayout,
        this.uniformBindGroupLayout
      ]
    })

    // Standard composite pipeline for offscreen textures
    this.compositePipeline = device.createRenderPipeline({
      layout: compositePipelineLayout,
      vertex: {
        module: device.createShaderModule({ code: compositeShader }),
        entryPoint: 'vs'
      },
      fragment: {
        module: device.createShaderModule({ code: compositeShader }),
        entryPoint: 'fs',
        targets: [
          {
            format: 'rgba8unorm',
            blend: {
              color: {
                srcFactor: 'one',
                dstFactor: 'one-minus-src-alpha',
                operation: 'add'
              },
              alpha: {
                srcFactor: 'one',
                dstFactor: 'one-minus-src-alpha',
                operation: 'add'
              }
            }
          }
        ]
      },
      primitive: { topology: 'triangle-list' }
    })

    // Composite pipeline for the preview canvas
    this.compositePipelinePreview = device.createRenderPipeline({
      layout: compositePipelineLayout,
      vertex: {
        module: device.createShaderModule({ code: compositeShader }),
        entryPoint: 'vs'
      },
      fragment: {
        module: device.createShaderModule({ code: compositeShader }),
        entryPoint: 'fs',
        targets: [
          {
            format: presentationFormat,
            blend: {
              color: {
                srcFactor: 'one',
                dstFactor: 'one-minus-src-alpha',
                operation: 'add'
              },
              alpha: {
                srcFactor: 'one',
                dstFactor: 'one-minus-src-alpha',
                operation: 'add'
              }
            }
          }
        ]
      },
      primitive: { topology: 'triangle-list' }
    })

    // --- 5. Erase Pipeline (Destination Out) ---
    // Standard erase pipeline for offscreen textures
    this.erasePipeline = device.createRenderPipeline({
      layout: compositePipelineLayout,
      vertex: {
        module: device.createShaderModule({ code: compositeShader }),
        entryPoint: 'vs'
      },
      fragment: {
        module: device.createShaderModule({ code: compositeShader }),
        entryPoint: 'fs',
        targets: [
          {
            format: 'rgba8unorm',
            blend: {
              color: {
                srcFactor: 'zero',
                dstFactor: 'one-minus-src-alpha', // dst * (1 - src_alpha)
                operation: 'add'
              },
              alpha: {
                srcFactor: 'zero',
                dstFactor: 'one-minus-src-alpha', // dst_alpha * (1 - src_alpha)
                operation: 'add'
              }
            }
          }
        ]
      },
      primitive: { topology: 'triangle-list' }
    })

    // Erase pipeline for the preview canvas
    this.erasePipelinePreview = device.createRenderPipeline({
      layout: compositePipelineLayout,
      vertex: {
        module: device.createShaderModule({ code: compositeShader }),
        entryPoint: 'vs'
      },
      fragment: {
        module: device.createShaderModule({ code: compositeShader }),
        entryPoint: 'fs',
        targets: [
          {
            format: presentationFormat,
            blend: {
              color: {
                srcFactor: 'zero',
                dstFactor: 'one-minus-src-alpha',
                operation: 'add'
              },
              alpha: {
                srcFactor: 'zero',
                dstFactor: 'one-minus-src-alpha',
                operation: 'add'
              }
            }
          }
        ]
      },
      primitive: { topology: 'triangle-list' }
    })

    // --- 6. Readback Pipeline (Compute) ---
    this.readbackPipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({ code: readbackShader }),
        entryPoint: 'main'
      }
    })
  }

  public prepareStroke(width: number, height: number) {
    // Initialize or resize the accumulation texture
    if (
      !this.currentStrokeTexture ||
      this.currentStrokeTexture.width !== width ||
      this.currentStrokeTexture.height !== height
    ) {
      if (this.currentStrokeTexture) this.currentStrokeTexture.destroy()
      this.currentStrokeTexture = this.device.createTexture({
        size: [width, height],
        format: 'rgba8unorm',
        usage:
          GPUTextureUsage.RENDER_ATTACHMENT |
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_SRC
      })
      this.currentStrokeView = this.currentStrokeTexture.createView()

      // Invalidate texture-dependent bind groups
      this.compositeTextureBindGroup = null
      this.previewTextureBindGroup = null
      // Readback bind group might also be invalid if it was using the old texture
      if (this.lastReadbackTexture === this.currentStrokeTexture) {
        this.readbackBindGroup = null
        this.lastReadbackTexture = null
      }
    }

    // Clear the accumulation texture
    const encoder = this.device.createCommandEncoder()
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.currentStrokeView!,
          loadOp: 'clear',
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          storeOp: 'store'
        }
      ]
    })
    pass.end()
    this.device.queue.submit([encoder.finish()])
  }

  public renderStrokeToAccumulator(
    points: { x: number; y: number; pressure: number }[],
    settings: {
      size: number
      opacity: number
      hardness: number
      color: [number, number, number]
      width: number
      height: number
      brushShape: number
    }
  ) {
    if (!this.currentStrokeView) return
    // Render stroke using accumulation pipeline
    this.renderStrokeInternal(
      this.currentStrokeView,
      this.accumulatePipeline,
      points,
      settings
    )
  }

  public compositeStroke(
    targetView: GPUTextureView,
    settings: {
      opacity: number
      color: [number, number, number]
      hardness: number // Required for uniform buffer layout
      screenSize: [number, number]
      brushShape: number
      isErasing?: boolean
    }
  ) {
    if (!this.currentStrokeTexture) return

    // Update uniforms for the composite pass
    const buffer = new ArrayBuffer(UNIFORM_SIZE)
    const f32 = new Float32Array(buffer)
    const u32 = new Uint32Array(buffer)

    f32[0] = settings.color[0]
    f32[1] = settings.color[1]
    f32[2] = settings.color[2]
    f32[3] = settings.opacity
    f32[4] = settings.hardness
    f32[5] = 0 // Padding
    f32[6] = settings.screenSize[0]
    f32[7] = settings.screenSize[1]
    u32[8] = settings.brushShape // Brush shape: 0=Circle, 1=Square
    this.device.queue.writeBuffer(this.uniformBuffer, 0, buffer)

    const encoder = this.device.createCommandEncoder()

    // Choose pipeline based on operation
    const pipeline = settings.isErasing
      ? this.erasePipeline
      : this.compositePipeline

    // 1. Texture Bind Group (Group 0)
    if (!this.compositeTextureBindGroup) {
      this.compositeTextureBindGroup = this.device.createBindGroup({
        layout: this.textureBindGroupLayout,
        entries: [
          { binding: 0, resource: this.currentStrokeTexture.createView() }
        ]
      })
    }

    // 2. Uniform Bind Group (Group 1) - Use shared mainUniformBindGroup
    // It is compatible because we used the same layout

    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: targetView,
          loadOp: 'load',
          storeOp: 'store'
        }
      ]
    })

    pass.setPipeline(pipeline)
    pass.setBindGroup(0, this.compositeTextureBindGroup)
    pass.setBindGroup(1, this.mainUniformBindGroup)
    pass.draw(3)
    pass.end()

    this.device.queue.submit([encoder.finish()])
  }

  // Direct rendering method
  public renderStroke(
    targetView: GPUTextureView,
    points: { x: number; y: number; pressure: number }[],
    settings: {
      size: number
      opacity: number
      hardness: number
      color: [number, number, number]
      width: number
      height: number
      brushShape: number
    }
  ) {
    this.renderStrokeInternal(targetView, this.renderPipeline, points, settings)
  }

  private renderStrokeInternal(
    targetView: GPUTextureView,
    pipeline: GPURenderPipeline,
    points: { x: number; y: number; pressure: number }[],
    settings: {
      size: number
      opacity: number
      hardness: number
      color: [number, number, number]
      width: number
      height: number
      brushShape: number
    }
  ) {
    if (points.length === 0) return

    // 1. Update Uniforms
    const buffer = new ArrayBuffer(UNIFORM_SIZE)
    const f32 = new Float32Array(buffer)
    const u32 = new Uint32Array(buffer)

    f32[0] = settings.color[0]
    f32[1] = settings.color[1]
    f32[2] = settings.color[2]
    f32[3] = settings.opacity
    f32[4] = settings.hardness
    f32[5] = 0 // Padding
    f32[6] = settings.width
    f32[7] = settings.height
    u32[8] = settings.brushShape
    this.device.queue.writeBuffer(this.uniformBuffer, 0, buffer)

    // 2. Batch Rendering
    let processedPoints = 0
    while (processedPoints < points.length) {
      const batchSize = Math.min(points.length - processedPoints, MAX_STROKES)
      const iData = new Float32Array(batchSize * 4)

      for (let i = 0; i < batchSize; i++) {
        const p = points[processedPoints + i]
        iData[i * 4 + 0] = p.x
        iData[i * 4 + 1] = p.y
        iData[i * 4 + 2] = settings.size
        iData[i * 4 + 3] = p.pressure
      }

      this.device.queue.writeBuffer(this.instanceBuffer, 0, iData)

      // 3. Render Pass
      const encoder = this.device.createCommandEncoder()
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: targetView,
            loadOp: 'load',
            storeOp: 'store'
          }
        ]
      })

      pass.setPipeline(pipeline)
      pass.setBindGroup(0, this.mainUniformBindGroup)
      pass.setVertexBuffer(0, this.quadVertexBuffer)
      pass.setVertexBuffer(1, this.instanceBuffer)
      pass.setIndexBuffer(this.indexBuffer, 'uint16')
      pass.drawIndexed(6, batchSize)
      pass.end()

      this.device.queue.submit([encoder.finish()])

      processedPoints += batchSize
    }
  }

  // Blit the accumulated stroke to the preview canvas
  public blitToCanvas(
    destinationCtx: GPUCanvasContext,
    settings: {
      opacity: number
      color: [number, number, number]
      hardness: number
      screenSize: [number, number]
      brushShape: number
      isErasing?: boolean
    },
    backgroundTexture?: GPUTexture
  ) {
    const encoder = this.device.createCommandEncoder()
    const destView = destinationCtx.getCurrentTexture().createView()

    if (backgroundTexture) {
      // Draw background texture to allow erasing effect on existing content
      if (
        this.lastBackgroundTexture !== backgroundTexture ||
        !this.backgroundBindGroup
      ) {
        this.backgroundBindGroup = this.device.createBindGroup({
          layout: this.textureBindGroupLayout,
          entries: [{ binding: 0, resource: backgroundTexture.createView() }]
        })
        this.lastBackgroundTexture = backgroundTexture
      }

      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: destView,
            loadOp: 'clear', // Clear attachment before drawing
            clearValue: { r: 0, g: 0, b: 0, a: 0 },
            storeOp: 'store'
          }
        ]
      })
      pass.setPipeline(this.blitPipeline)
      pass.setBindGroup(0, this.backgroundBindGroup)
      pass.draw(3)
      pass.end()
    } else {
      // Clear the destination texture
      const clearPass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: destView,
            loadOp: 'clear',
            clearValue: { r: 0, g: 0, b: 0, a: 0 },
            storeOp: 'store'
          }
        ]
      })
      clearPass.end()
    }

    // Draw the accumulated stroke
    if (this.currentStrokeTexture) {
      // Update uniforms for the preview pass
      const buffer = new ArrayBuffer(UNIFORM_SIZE)
      const f32 = new Float32Array(buffer)
      const u32 = new Uint32Array(buffer)

      f32[0] = settings.color[0]
      f32[1] = settings.color[1]
      f32[2] = settings.color[2]
      f32[3] = settings.opacity
      f32[4] = settings.hardness
      f32[5] = 0 // Padding
      f32[6] = settings.screenSize[0]
      f32[7] = settings.screenSize[1]
      u32[8] = settings.brushShape
      this.device.queue.writeBuffer(this.uniformBuffer, 0, buffer)

      // Select preview pipeline based on operation
      const pipeline = settings.isErasing
        ? this.erasePipelinePreview
        : this.compositePipelinePreview

      // 1. Texture Bind Group (Group 0)
      if (!this.previewTextureBindGroup) {
        this.previewTextureBindGroup = this.device.createBindGroup({
          layout: this.textureBindGroupLayout,
          entries: [
            { binding: 0, resource: this.currentStrokeTexture.createView() }
          ]
        })
      }

      // 2. Uniform Bind Group (Group 1) - Use shared mainUniformBindGroup

      const passStroke = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: destView,
            loadOp: 'load', // Load the previous pass result
            storeOp: 'store'
          }
        ]
      })
      passStroke.setPipeline(pipeline)
      passStroke.setBindGroup(0, this.previewTextureBindGroup)
      passStroke.setBindGroup(1, this.mainUniformBindGroup)
      passStroke.draw(3)
      passStroke.end()
    }

    this.device.queue.submit([encoder.finish()])
  }

  // Clear the preview canvas
  public clearPreview(destinationCtx: GPUCanvasContext) {
    const encoder = this.device.createCommandEncoder()
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: destinationCtx.getCurrentTexture().createView(),
          loadOp: 'clear',
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          storeOp: 'store'
        }
      ]
    })
    pass.end()
    this.device.queue.submit([encoder.finish()])
  }

  public prepareReadback(texture: GPUTexture, outputBuffer: GPUBuffer) {
    if (
      this.lastReadbackTexture !== texture ||
      this.lastReadbackBuffer !== outputBuffer ||
      !this.readbackBindGroup
    ) {
      this.readbackBindGroup = this.device.createBindGroup({
        layout: this.readbackPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: texture.createView() },
          { binding: 1, resource: { buffer: outputBuffer } }
        ]
      })
      this.lastReadbackTexture = texture
      this.lastReadbackBuffer = outputBuffer
    }

    const encoder = this.device.createCommandEncoder()
    const pass = encoder.beginComputePass()
    pass.setPipeline(this.readbackPipeline)
    pass.setBindGroup(0, this.readbackBindGroup)

    const width = texture.width
    const height = texture.height
    // Dispatch workgroups based on texture dimensions (8x8 block size)
    pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8))
    pass.end()

    this.device.queue.submit([encoder.finish()])
  }

  public destroy() {
    this.quadVertexBuffer.destroy()
    this.indexBuffer.destroy()
    this.instanceBuffer.destroy()
    this.uniformBuffer.destroy()
    if (this.currentStrokeTexture) this.currentStrokeTexture.destroy()

    // Clear cached bind groups
    this.compositeTextureBindGroup = null
    this.previewTextureBindGroup = null
    this.readbackBindGroup = null
    this.backgroundBindGroup = null
    this.lastReadbackTexture = null
    this.lastReadbackBuffer = null
    this.lastBackgroundTexture = null
  }
}
