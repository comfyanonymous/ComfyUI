/**
 * Single source of truth for Essentials tab node categorization and ordering.
 *
 * Adding a new node to the Essentials tab? Add it here and nowhere else.
 *
 * Source: https://www.notion.so/comfy-org/2fe6d73d365080d0a951d14cdf540778
 */

export const ESSENTIALS_CATEGORIES = [
  'basics',
  'text generation',
  'image generation',
  'video generation',
  'image tools',
  'video tools',
  'audio',
  '3D'
] as const

export type EssentialsCategory = (typeof ESSENTIALS_CATEGORIES)[number]

/**
 * Ordered list of nodes per category.
 * Array order = display order in the Essentials tab.
 * Presence in a category = the node's essentials_category mock fallback.
 */
export const ESSENTIALS_NODES: Record<EssentialsCategory, readonly string[]> = {
  basics: [
    'LoadImage',
    'LoadVideo',
    'Load3D',
    'SaveImage',
    'SaveVideo',
    'SaveGLB',
    'PrimitiveStringMultiline',
    'PreviewImage'
  ],
  'text generation': ['OpenAIChatNode'],
  'image generation': [
    'LoraLoader',
    'LoraLoaderModelOnly',
    'ConditioningCombine'
  ],
  'video generation': [
    'SubgraphBlueprint.pose_to_video_ltx_2_0',
    'SubgraphBlueprint.canny_to_video_ltx_2_0',
    'KlingLipSyncAudioToVideoNode',
    'KlingOmniProEditVideoNode'
  ],
  'image tools': [
    'ImageBatch',
    'ImageCrop',
    'ImageCropV2',
    'ImageScale',
    'ImageScaleBy',
    'ImageRotate',
    'ImageBlur',
    'ImageBlend',
    'ImageInvert',
    'ImageCompare',
    'Canny',
    'RecraftRemoveBackgroundNode',
    'RecraftVectorizeImageNode',
    'LoadImageMask',
    'GLSLShader'
  ],
  'video tools': ['GetVideoComponents', 'CreateVideo', 'Video Slice'],
  audio: [
    'LoadAudio',
    'SaveAudio',
    'SaveAudioMP3',
    'StabilityTextToAudio',
    'EmptyLatentAudio'
  ],
  '3D': ['TencentTextToModelNode', 'TencentImageToModelNode']
}

/**
 * Flat map: node name → category (derived from ESSENTIALS_NODES).
 * Used as mock/fallback when backend doesn't provide essentials_category.
 */
export const ESSENTIALS_CATEGORY_MAP: Record<string, EssentialsCategory> =
  Object.fromEntries(
    Object.entries(ESSENTIALS_NODES).flatMap(([category, nodes]) =>
      nodes.map((node) => [node, category])
    )
  ) as Record<string, EssentialsCategory>

/**
 * Case-insensitive lookup: lowercase category → canonical category.
 * Used to normalize backend categories (which may be title-cased) to the
 * canonical form used in ESSENTIALS_CATEGORIES.
 */
export const ESSENTIALS_CATEGORY_CANONICAL: ReadonlyMap<
  string,
  EssentialsCategory
> = new Map(ESSENTIALS_CATEGORIES.map((c) => [c.toLowerCase(), c]))

/**
 * "Novel" toolkit nodes for telemetry — basics excluded.
 * Derived from ESSENTIALS_NODES minus the 'basics' category.
 */
export const TOOLKIT_NOVEL_NODE_NAMES: ReadonlySet<string> = new Set(
  Object.entries(ESSENTIALS_NODES)
    .filter(([cat]) => cat !== 'basics')
    .flatMap(([, nodes]) => nodes)
    .filter((n) => !n.startsWith('SubgraphBlueprint.'))
)

/**
 * python_module values that identify toolkit blueprint nodes.
 */
export const TOOLKIT_BLUEPRINT_MODULES: ReadonlySet<string> = new Set([
  'comfy_essentials'
])
