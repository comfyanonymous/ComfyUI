import { describe, expect, it, vi } from 'vitest'

vi.mock('@/i18n', () => ({
  t: vi.fn((key: string) => key)
}))

import { resolveEssentialsDisplayName } from '@/constants/essentialsDisplayNames'

describe('resolveEssentialsDisplayName', () => {
  describe('exact name matches', () => {
    it.each([
      ['LoadImage', 'essentials.loadImage'],
      ['SaveImage', 'essentials.saveImage'],
      ['PrimitiveStringMultiline', 'essentials.text'],
      ['ImageScale', 'essentials.resizeImage'],
      ['LoraLoader', 'essentials.loadStyleLora'],
      ['OpenAIChatNode', 'essentials.textGenerationLLM'],
      ['RecraftRemoveBackgroundNode', 'essentials.removeBackground'],
      ['ImageCompare', 'essentials.imageCompare'],
      ['StabilityTextToAudio', 'essentials.musicGeneration'],
      ['BatchImagesNode', 'essentials.batchImage'],
      ['Video Slice', 'essentials.extractFrame'],
      ['KlingLipSyncAudioToVideoNode', 'essentials.lipsync'],
      ['KlingLipSyncTextToVideoNode', 'essentials.lipsync']
    ])('%s -> %s', (name, expected) => {
      expect(resolveEssentialsDisplayName({ name })).toBe(expected)
    })
  })

  describe('3D API node alternatives', () => {
    it.each([
      ['TencentTextToModelNode', 'essentials.textTo3DModel'],
      ['MeshyTextToModelNode', 'essentials.textTo3DModel'],
      ['TripoTextToModelNode', 'essentials.textTo3DModel'],
      ['TencentImageToModelNode', 'essentials.imageTo3DModel'],
      ['MeshyImageToModelNode', 'essentials.imageTo3DModel'],
      ['TripoImageToModelNode', 'essentials.imageTo3DModel']
    ])('%s -> %s', (name, expected) => {
      expect(resolveEssentialsDisplayName({ name })).toBe(expected)
    })
  })

  describe('blueprint prefix matches', () => {
    it.each([
      [
        'SubgraphBlueprint.text_to_image_flux_schnell.json',
        'essentials.textToImage'
      ],
      ['SubgraphBlueprint.text_to_image_sd15.json', 'essentials.textToImage'],
      [
        'SubgraphBlueprint.image_edit_something.json',
        'essentials.imageToImage'
      ],
      ['SubgraphBlueprint.pose_to_image_v2.json', 'essentials.poseToImage'],
      [
        'SubgraphBlueprint.canny_to_image_z_image_turbo.json',
        'essentials.cannyToImage'
      ],
      [
        'SubgraphBlueprint.depth_to_image_z_image_turbo.json',
        'essentials.depthToImage'
      ],
      ['SubgraphBlueprint.text_to_video_ltx.json', 'essentials.textToVideo'],
      ['SubgraphBlueprint.image_to_video_wan.json', 'essentials.imageToVideo'],
      [
        'SubgraphBlueprint.pose_to_video_ltx_2_0.json',
        'essentials.poseToVideo'
      ],
      [
        'SubgraphBlueprint.canny_to_video_ltx_2_0.json',
        'essentials.cannyToVideo'
      ],
      [
        'SubgraphBlueprint.depth_to_video_ltx_2_0.json',
        'essentials.depthToVideo'
      ],
      [
        'SubgraphBlueprint.image_inpainting_qwen_image_instantx.json',
        'essentials.inpaintImage'
      ],
      [
        'SubgraphBlueprint.image_outpainting_qwen_image_instantx.json',
        'essentials.outpaintImage'
      ]
    ])('%s -> %s', (name, expected) => {
      expect(resolveEssentialsDisplayName({ name })).toBe(expected)
    })
  })

  describe('unmapped nodes', () => {
    it('returns undefined for unknown node names', () => {
      expect(resolveEssentialsDisplayName({ name: 'SomeRandomNode' })).toBe(
        undefined
      )
    })

    it('returns undefined for unknown blueprint prefixes', () => {
      expect(
        resolveEssentialsDisplayName({
          name: 'SubgraphBlueprint.unknown_workflow.json'
        })
      ).toBe(undefined)
    })
  })
})
