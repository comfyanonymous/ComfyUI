import { t } from '@/i18n'
import type { ComfyNodeDefImpl } from '@/stores/nodeDefStore'

const BLUEPRINT_PREFIX = 'SubgraphBlueprint.'

/**
 * Static mapping of node names to their Essentials tab display name i18n keys.
 */
const EXACT_NAME_MAP: Record<string, string> = {
  // Basics
  LoadImage: 'essentials.loadImage',
  SaveImage: 'essentials.saveImage',
  LoadVideo: 'essentials.loadVideo',
  SaveVideo: 'essentials.saveVideo',
  Load3D: 'essentials.load3DModel',
  SaveGLB: 'essentials.save3DModel',
  PrimitiveStringMultiline: 'essentials.text',

  // Image Tools
  BatchImagesNode: 'essentials.batchImage',
  ImageCrop: 'essentials.cropImage',
  ImageScale: 'essentials.resizeImage',
  ImageRotate: 'essentials.rotate',
  ImageInvert: 'essentials.invert',
  Canny: 'essentials.canny',
  RecraftRemoveBackgroundNode: 'essentials.removeBackground',
  ImageCompare: 'essentials.imageCompare',

  // Video Tools
  'Video Slice': 'essentials.extractFrame',

  // Image Generation
  LoraLoader: 'essentials.loadStyleLora',

  // Video Generation
  KlingLipSyncAudioToVideoNode: 'essentials.lipsync',
  KlingLipSyncTextToVideoNode: 'essentials.lipsync',

  // Text Generation
  OpenAIChatNode: 'essentials.textGenerationLLM',

  // 3D
  TencentTextToModelNode: 'essentials.textTo3DModel',
  TencentImageToModelNode: 'essentials.imageTo3DModel',
  MeshyTextToModelNode: 'essentials.textTo3DModel',
  MeshyImageToModelNode: 'essentials.imageTo3DModel',
  TripoTextToModelNode: 'essentials.textTo3DModel',
  TripoImageToModelNode: 'essentials.imageTo3DModel',

  // Audio
  StabilityTextToAudio: 'essentials.musicGeneration',
  LoadAudio: 'essentials.loadAudio',
  SaveAudio: 'essentials.saveAudio'
}

/**
 * Blueprint prefix patterns mapped to display name i18n keys.
 * Entries are matched by checking if the blueprint filename
 * (after removing the SubgraphBlueprint. prefix) starts with the key.
 * Ordered longest-first so more specific prefixes match before shorter ones.
 */
const BLUEPRINT_PREFIX_MAP: [prefix: string, displayNameKey: string][] = [
  // Image Generation
  ['image_inpainting_', 'essentials.inpaintImage'],
  ['image_outpainting_', 'essentials.outpaintImage'],
  ['image_edit', 'essentials.imageToImage'],
  ['text_to_image', 'essentials.textToImage'],
  ['pose_to_image', 'essentials.poseToImage'],
  ['canny_to_image', 'essentials.cannyToImage'],
  ['depth_to_image', 'essentials.depthToImage'],

  // Video Generation
  ['text_to_video', 'essentials.textToVideo'],
  ['image_to_video', 'essentials.imageToVideo'],
  ['pose_to_video', 'essentials.poseToVideo'],
  ['canny_to_video', 'essentials.cannyToVideo'],
  ['depth_to_video', 'essentials.depthToVideo']
]

function resolveBlueprintDisplayName(
  blueprintName: string
): string | undefined {
  for (const [prefix, displayNameKey] of BLUEPRINT_PREFIX_MAP) {
    if (blueprintName.startsWith(prefix)) {
      return t(displayNameKey)
    }
  }
  return undefined
}

/**
 * Resolves the Essentials tab display name for a given node definition.
 * Returns `undefined` if the node has no Essentials display name mapping.
 */
export function resolveEssentialsDisplayName(
  nodeDef: Pick<ComfyNodeDefImpl, 'name'> | undefined
): string | undefined {
  if (!nodeDef) return undefined
  const { name } = nodeDef

  if (name.startsWith(BLUEPRINT_PREFIX)) {
    const blueprintName = name.slice(BLUEPRINT_PREFIX.length)
    return resolveBlueprintDisplayName(blueprintName)
  }

  const key = EXACT_NAME_MAP[name]
  return key ? t(key) : undefined
}
