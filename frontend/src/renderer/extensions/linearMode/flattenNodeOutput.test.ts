import { describe, expect, it } from 'vitest'

import { flattenNodeOutput } from '@/renderer/extensions/linearMode/flattenNodeOutput'
import type { NodeExecutionOutput } from '@/schemas/apiSchema'

function makeOutput(
  overrides: Partial<NodeExecutionOutput> = {}
): NodeExecutionOutput {
  return { ...overrides }
}

describe(flattenNodeOutput, () => {
  it('returns empty array for output with no known media types', () => {
    const result = flattenNodeOutput(['1', makeOutput({ text: 'hello' })])
    expect(result).toEqual([])
  })

  it('flattens images into ResultItemImpl instances', () => {
    const output = makeOutput({
      images: [
        { filename: 'a.png', subfolder: '', type: 'output' },
        { filename: 'b.png', subfolder: 'sub', type: 'output' }
      ]
    })

    const result = flattenNodeOutput(['42', output])

    expect(result).toHaveLength(2)
    expect(result[0].filename).toBe('a.png')
    expect(result[0].nodeId).toBe('42')
    expect(result[0].mediaType).toBe('images')
    expect(result[1].filename).toBe('b.png')
    expect(result[1].subfolder).toBe('sub')
  })

  it('flattens audio outputs', () => {
    const output = makeOutput({
      audio: [{ filename: 'sound.wav', subfolder: '', type: 'output' }]
    })

    const result = flattenNodeOutput([7, output])

    expect(result).toHaveLength(1)
    expect(result[0].mediaType).toBe('audio')
    expect(result[0].nodeId).toBe(7)
  })

  it('flattens multiple media types in a single output', () => {
    const output = makeOutput({
      images: [{ filename: 'img.png', subfolder: '', type: 'output' }],
      video: [{ filename: 'vid.mp4', subfolder: '', type: 'output' }]
    })

    const result = flattenNodeOutput(['1', output])

    expect(result).toHaveLength(2)
    const types = result.map((r) => r.mediaType)
    expect(types).toContain('images')
    expect(types).toContain('video')
  })

  it('handles gifs and 3d output types', () => {
    const output = makeOutput({
      gifs: [
        { filename: 'anim.gif', subfolder: '', type: 'output' }
      ] as NodeExecutionOutput['gifs'],
      '3d': [
        { filename: 'model.glb', subfolder: '', type: 'output' }
      ] as NodeExecutionOutput['3d']
    })

    const result = flattenNodeOutput(['5', output])

    expect(result).toHaveLength(2)
    const types = result.map((r) => r.mediaType)
    expect(types).toContain('gifs')
    expect(types).toContain('3d')
  })

  it('ignores empty arrays', () => {
    const output = makeOutput({ images: [], audio: [] })
    const result = flattenNodeOutput(['1', output])
    expect(result).toEqual([])
  })
})
