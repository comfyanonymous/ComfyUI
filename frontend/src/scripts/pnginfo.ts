import { LGraph, LGraphNode, LiteGraph } from '@/lib/litegraph/src/litegraph'
import type { IBaseWidget } from '@/lib/litegraph/src/types/widgets'

import { api } from './api'
import { getFromAvifFile } from './metadata/avif'
import { getFromFlacFile } from './metadata/flac'
import { getFromPngFile } from './metadata/png'

// Original functions left in for backwards compatibility
export function getPngMetadata(file: File): Promise<Record<string, string>> {
  return getFromPngFile(file)
}

export function getFlacMetadata(file: File): Promise<Record<string, string>> {
  return getFromFlacFile(file)
}

export function getAvifMetadata(file: File): Promise<Record<string, string>> {
  return getFromAvifFile(file)
}

function parseExifData(exifData: Uint8Array) {
  // Check for the correct TIFF header (0x4949 for little-endian or 0x4D4D for big-endian)
  const isLittleEndian = String.fromCharCode(...exifData.slice(0, 2)) === 'II'

  // Function to read 16-bit and 32-bit integers from binary data
  function readInt(
    offset: number,
    isLittleEndian: boolean,
    length: 2 | 4
  ): number {
    let arr = exifData.slice(offset, offset + length)
    if (length === 2) {
      return new DataView(arr.buffer, arr.byteOffset, arr.byteLength).getUint16(
        0,
        isLittleEndian
      )
    } else if (length === 4) {
      return new DataView(arr.buffer, arr.byteOffset, arr.byteLength).getUint32(
        0,
        isLittleEndian
      )
    }
    return 0
  }

  // Read the offset to the first IFD (Image File Directory)
  const ifdOffset = readInt(4, isLittleEndian, 4)

  function parseIFD(offset: number): Record<number, string | undefined> {
    const numEntries = readInt(offset, isLittleEndian, 2)
    const result: Record<number, string | undefined> = {}

    for (let i = 0; i < numEntries; i++) {
      const entryOffset = offset + 2 + i * 12
      const tag = readInt(entryOffset, isLittleEndian, 2)
      const type = readInt(entryOffset + 2, isLittleEndian, 2)
      const numValues = readInt(entryOffset + 4, isLittleEndian, 4)
      const valueOffset = readInt(entryOffset + 8, isLittleEndian, 4)

      // Read the value(s) based on the data type
      let value
      if (type === 2) {
        // ASCII string
        value = new TextDecoder('utf-8').decode(
          exifData.subarray(valueOffset, valueOffset + numValues - 1)
        )
      }

      result[tag] = value
    }

    return result
  }

  // Parse the first IFD
  const ifdData = parseIFD(ifdOffset)
  return ifdData
}

export function getWebpMetadata(file: File) {
  return new Promise<Record<string, string>>((r) => {
    const reader = new FileReader()
    reader.onload = (event) => {
      const webp = new Uint8Array(event.target?.result as ArrayBuffer)
      const dataView = new DataView(webp.buffer)

      // Check that the WEBP signature is present
      if (
        dataView.getUint32(0) !== 0x52494646 ||
        dataView.getUint32(8) !== 0x57454250
      ) {
        console.error('Not a valid WEBP file')
        r({})
        return
      }

      // Start searching for chunks after the WEBP signature
      let offset = 12
      const txt_chunks: Record<string, string> = {}
      // Loop through the chunks in the WEBP file
      while (offset < webp.length) {
        const chunk_length = dataView.getUint32(offset + 4, true)
        const chunk_type = String.fromCharCode(
          ...webp.slice(offset, offset + 4)
        )
        if (chunk_type === 'EXIF') {
          if (
            String.fromCharCode(...webp.slice(offset + 8, offset + 8 + 6)) ==
            'Exif\0\0'
          ) {
            offset += 6
          }
          let data = parseExifData(
            webp.slice(offset + 8, offset + 8 + chunk_length)
          )
          for (const key in data) {
            const value = data[Number(key)]
            if (typeof value === 'string') {
              const index = value.indexOf(':')
              txt_chunks[value.slice(0, index)] = value.slice(index + 1)
            }
          }
          break
        }

        // RIFF spec requires odd-sized chunks to be padded with a single byte
        // https://developers.google.com/speed/webp/docs/riff_container#riff_file_format
        offset += 8 + chunk_length + (chunk_length % 2)
      }

      r(txt_chunks)
    }

    reader.readAsArrayBuffer(file)
  })
}

export function getLatentMetadata(file: File): Promise<Record<string, string>> {
  return new Promise((r) => {
    const reader = new FileReader()
    reader.onload = (event) => {
      const safetensorsData = new Uint8Array(
        event.target?.result as ArrayBuffer
      )
      const dataView = new DataView(safetensorsData.buffer)
      let header_size = dataView.getUint32(0, true)
      let offset = 8
      let header = JSON.parse(
        new TextDecoder().decode(
          safetensorsData.slice(offset, offset + header_size)
        )
      )
      r(header.__metadata__)
    }

    var slice = file.slice(0, 1024 * 1024 * 4)
    reader.readAsArrayBuffer(slice)
  })
}

interface NodeConnection {
  node: LGraphNode
  index: number
}

interface LoraEntry {
  name: string
  weight: number
}

export async function importA1111(
  graph: LGraph,
  parameters: string
): Promise<void> {
  const p = parameters.lastIndexOf('\nSteps:')
  if (p > -1) {
    const embeddings = await api.getEmbeddings()
    const matchResult = parameters
      .substr(p)
      .split('\n')[1]
      .match(
        new RegExp('\\s*([^:]+:\\s*([^"\\{].*?|".*?"|\\{.*?\\}))\\s*(,|$)', 'g')
      )
    if (!matchResult) return

    const opts: Record<string, string> = matchResult.reduce(
      (acc: Record<string, string>, n: string) => {
        const s = n.split(':')
        if (s[1].endsWith(',')) {
          s[1] = s[1].substr(0, s[1].length - 1)
        }
        acc[s[0].trim().toLowerCase()] = s[1].trim()
        return acc
      },
      {}
    )
    const p2 = parameters.lastIndexOf('\nNegative prompt:', p)
    if (p2 > -1) {
      let positive = parameters.substr(0, p2).trim()
      let negative = parameters.substring(p2 + 18, p).trim()

      const ckptNode = LiteGraph.createNode('CheckpointLoaderSimple')
      const clipSkipNode = LiteGraph.createNode('CLIPSetLastLayer')
      const positiveNode = LiteGraph.createNode('CLIPTextEncode')
      const negativeNode = LiteGraph.createNode('CLIPTextEncode')
      const samplerNode = LiteGraph.createNode('KSampler')
      const imageNode = LiteGraph.createNode('EmptyLatentImage')
      const vaeNode = LiteGraph.createNode('VAEDecode')
      const saveNode = LiteGraph.createNode('SaveImage')

      if (
        !ckptNode ||
        !clipSkipNode ||
        !positiveNode ||
        !negativeNode ||
        !samplerNode ||
        !imageNode ||
        !vaeNode ||
        !saveNode
      ) {
        console.error('Failed to create required nodes for A1111 import')
        return
      }

      let hrSamplerNode: LGraphNode | null = null
      let hrSteps: string | null = null

      const ceil64 = (v: number) => Math.ceil(v / 64) * 64

      function getWidget(
        node: LGraphNode | null,
        name: string
      ): IBaseWidget | undefined {
        return node?.widgets?.find((w) => w.name === name)
      }

      function setWidgetValue(
        node: LGraphNode | null,
        name: string,
        value: string | number,
        isOptionPrefix?: boolean
      ): void {
        const w = getWidget(node, name)
        if (!w) return

        if (isOptionPrefix) {
          const values = w.options.values as string[] | undefined
          const o = values?.find((v) => v.startsWith(String(value)))
          if (o) {
            w.value = o
          } else {
            console.warn(`Unknown value '${value}' for widget '${name}'`, node)
            w.value = value
          }
        } else {
          w.value = value
        }
      }

      function createLoraNodes(
        clipNode: LGraphNode,
        text: string,
        prevClip: NodeConnection,
        prevModel: NodeConnection,
        targetSamplerNode: LGraphNode
      ): { text: string; prevModel: NodeConnection; prevClip: NodeConnection } {
        const loras: LoraEntry[] = []
        text = text.replace(/<lora:([^:]+:[^>]+)>/g, (_m, c: string) => {
          const s = c.split(':')
          const weight = parseFloat(s[1])
          if (isNaN(weight)) {
            console.warn('Invalid LORA', _m)
          } else {
            loras.push({ name: s[0], weight })
          }
          return ''
        })

        for (const l of loras) {
          const loraNode = LiteGraph.createNode('LoraLoader')
          if (!loraNode) continue
          graph.add(loraNode)
          setWidgetValue(loraNode, 'lora_name', l.name, true)
          setWidgetValue(loraNode, 'strength_model', l.weight)
          setWidgetValue(loraNode, 'strength_clip', l.weight)
          prevModel.node.connect(prevModel.index, loraNode, 0)
          prevClip.node.connect(prevClip.index, loraNode, 1)
          prevModel = { node: loraNode, index: 0 }
          prevClip = { node: loraNode, index: 1 }
        }

        prevClip.node.connect(1, clipNode, 0)
        prevModel.node.connect(0, targetSamplerNode, 0)
        if (hrSamplerNode) {
          prevModel.node.connect(0, hrSamplerNode, 0)
        }

        return { text, prevModel, prevClip }
      }

      function replaceEmbeddings(text: string): string {
        if (!embeddings.length) return text
        return text.replaceAll(
          new RegExp(
            '\\b(' +
              embeddings
                .map((e) => e.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
                .join('\\b|\\b') +
              ')\\b',
            'ig'
          ),
          'embedding:$1'
        )
      }

      function popOpt(name: string): string | undefined {
        const v = opts[name]
        delete opts[name]
        return v
      }

      graph.clear()
      graph.add(ckptNode)
      graph.add(clipSkipNode)
      graph.add(positiveNode)
      graph.add(negativeNode)
      graph.add(samplerNode)
      graph.add(imageNode)
      graph.add(vaeNode)
      graph.add(saveNode)

      ckptNode.connect(1, clipSkipNode, 0)
      clipSkipNode.connect(0, positiveNode, 0)
      clipSkipNode.connect(0, negativeNode, 0)
      ckptNode.connect(0, samplerNode, 0)
      positiveNode.connect(0, samplerNode, 1)
      negativeNode.connect(0, samplerNode, 2)
      imageNode.connect(0, samplerNode, 3)
      vaeNode.connect(0, saveNode, 0)
      samplerNode.connect(0, vaeNode, 0)
      ckptNode.connect(2, vaeNode, 1)

      const handlers: Record<string, (v: string) => void> = {
        model(v: string) {
          setWidgetValue(ckptNode, 'ckpt_name', v, true)
        },
        vae() {},
        'cfg scale'(v: string) {
          setWidgetValue(samplerNode, 'cfg', +v)
        },
        'clip skip'(v: string) {
          setWidgetValue(clipSkipNode, 'stop_at_clip_layer', -Number(v))
        },
        sampler(v: string) {
          let name = v.toLowerCase().replace('++', 'pp').replaceAll(' ', '_')
          if (name.includes('karras')) {
            name = name.replace('karras', '').replace(/_+$/, '')
            setWidgetValue(samplerNode, 'scheduler', 'karras')
          } else {
            setWidgetValue(samplerNode, 'scheduler', 'normal')
          }
          const w = getWidget(samplerNode, 'sampler_name')
          const values = w?.options.values as string[] | undefined
          const o = values?.find((v) => v === name || v === 'sample_' + name)
          if (o) {
            setWidgetValue(samplerNode, 'sampler_name', o)
          }
        },
        size(v: string) {
          const wxh = v.split('x')
          const w = ceil64(+wxh[0])
          const h = ceil64(+wxh[1])
          const hrUp = popOpt('hires upscale')
          const hrSz = popOpt('hires resize')
          hrSteps = popOpt('hires steps') ?? null
          let hrMethod = popOpt('hires upscaler')

          setWidgetValue(imageNode, 'width', w)
          setWidgetValue(imageNode, 'height', h)

          if (hrUp || hrSz) {
            let uw: number, uh: number
            if (hrUp) {
              uw = w * Number(hrUp)
              uh = h * Number(hrUp)
            } else if (hrSz) {
              const s = hrSz.split('x')
              uw = +s[0]
              uh = +s[1]
            } else {
              return
            }

            let upscaleNode: LGraphNode | null
            let latentNode: LGraphNode | null

            if (hrMethod?.startsWith('Latent')) {
              latentNode = upscaleNode = LiteGraph.createNode('LatentUpscale')
              if (!upscaleNode) return
              graph.add(upscaleNode)
              samplerNode.connect(0, upscaleNode, 0)

              switch (hrMethod) {
                case 'Latent (nearest-exact)':
                  hrMethod = 'nearest-exact'
                  break
              }
              setWidgetValue(upscaleNode, 'upscale_method', hrMethod, true)
            } else {
              const decode = LiteGraph.createNode('VAEDecodeTiled')
              if (!decode) return
              graph.add(decode)
              samplerNode.connect(0, decode, 0)
              ckptNode.connect(2, decode, 1)

              const upscaleLoaderNode =
                LiteGraph.createNode('UpscaleModelLoader')
              if (!upscaleLoaderNode) return
              graph.add(upscaleLoaderNode)
              setWidgetValue(
                upscaleLoaderNode,
                'model_name',
                hrMethod ?? '',
                true
              )

              const modelUpscaleNode = LiteGraph.createNode(
                'ImageUpscaleWithModel'
              )
              if (!modelUpscaleNode) return
              graph.add(modelUpscaleNode)
              decode.connect(0, modelUpscaleNode, 1)
              upscaleLoaderNode.connect(0, modelUpscaleNode, 0)

              upscaleNode = LiteGraph.createNode('ImageScale')
              if (!upscaleNode) return
              graph.add(upscaleNode)
              modelUpscaleNode.connect(0, upscaleNode, 0)

              const vaeEncodeNode = LiteGraph.createNode('VAEEncodeTiled')
              if (!vaeEncodeNode) return
              latentNode = vaeEncodeNode
              graph.add(vaeEncodeNode)
              upscaleNode.connect(0, vaeEncodeNode, 0)
              ckptNode.connect(2, vaeEncodeNode, 1)
            }

            setWidgetValue(upscaleNode, 'width', ceil64(uw))
            setWidgetValue(upscaleNode, 'height', ceil64(uh))

            hrSamplerNode = LiteGraph.createNode('KSampler')
            if (!hrSamplerNode || !latentNode) return
            graph.add(hrSamplerNode)
            ckptNode.connect(0, hrSamplerNode, 0)
            positiveNode.connect(0, hrSamplerNode, 1)
            negativeNode.connect(0, hrSamplerNode, 2)
            latentNode.connect(0, hrSamplerNode, 3)
            hrSamplerNode.connect(0, vaeNode, 0)
          }
        },
        steps(v: string) {
          setWidgetValue(samplerNode, 'steps', +v)
        },
        seed(v: string) {
          setWidgetValue(samplerNode, 'seed', +v)
        }
      }

      for (const opt in opts) {
        const handler = handlers[opt]
        if (handler) {
          const value = popOpt(opt)
          if (value !== undefined) handler(value)
        }
      }

      if (hrSamplerNode) {
        setWidgetValue(
          hrSamplerNode,
          'steps',
          hrSteps
            ? +hrSteps
            : (getWidget(samplerNode, 'steps')?.value as number)
        )
        setWidgetValue(
          hrSamplerNode,
          'cfg',
          getWidget(samplerNode, 'cfg')?.value as number
        )
        setWidgetValue(
          hrSamplerNode,
          'scheduler',
          getWidget(samplerNode, 'scheduler')?.value as string
        )
        setWidgetValue(
          hrSamplerNode,
          'sampler_name',
          getWidget(samplerNode, 'sampler_name')?.value as string
        )
        setWidgetValue(
          hrSamplerNode,
          'denoise',
          +(popOpt('denoising strength') ?? '1')
        )
      }

      let n = createLoraNodes(
        positiveNode,
        positive,
        { node: clipSkipNode, index: 0 },
        { node: ckptNode, index: 0 },
        samplerNode
      )
      positive = n.text
      n = createLoraNodes(
        negativeNode,
        negative,
        n.prevClip,
        n.prevModel,
        samplerNode
      )
      negative = n.text

      setWidgetValue(positiveNode, 'text', replaceEmbeddings(positive))
      setWidgetValue(negativeNode, 'text', replaceEmbeddings(negative))

      graph.arrange()

      for (const opt of [
        'model hash',
        'ensd',
        'version',
        'vae hash',
        'ti hashes',
        'lora hashes',
        'hashes'
      ]) {
        delete opts[opt]
      }

      console.warn('Unhandled parameters:', opts)
    }
  }
}
