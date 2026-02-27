/**
 * Maps category IDs to their corresponding Lucide icon classes
 */
export const getCategoryIcon = (categoryId: string): string => {
  const iconMap: Record<string, string> = {
    // Main categories
    all: 'icon-[lucide--list]',
    'getting-started': 'icon-[lucide--graduation-cap]',

    // Generation types
    'generation-image': 'icon-[lucide--image]',
    image: 'icon-[lucide--image]',
    'generation-video': 'icon-[lucide--film]',
    video: 'icon-[lucide--film]',
    'generation-3d': 'icon-[lucide--box]',
    '3d': 'icon-[lucide--box]',
    'generation-audio': 'icon-[lucide--volume-2]',
    audio: 'icon-[lucide--volume-2]',
    'generation-llm': 'icon-[lucide--message-square-text]',

    // API and models
    'api-nodes': 'icon-[lucide--hand-coins]',
    'closed-models': 'icon-[lucide--hand-coins]',

    // LLMs and AI
    llm: 'icon-[lucide--message-square-text]',
    llms: 'icon-[lucide--message-square-text]',
    'llm-api': 'icon-[lucide--message-square-text]',

    // Performance and hardware
    'small-models': 'icon-[lucide--zap]',
    performance: 'icon-[lucide--zap]',
    'mac-compatible': 'icon-[lucide--command]',
    'runs-on-mac': 'icon-[lucide--command]',

    // Training
    'lora-training': 'icon-[lucide--dumbbell]',
    training: 'icon-[lucide--dumbbell]',

    // Extensions and tools
    extensions: 'icon-[lucide--puzzle]',
    tools: 'icon-[lucide--wrench]',

    // Fallbacks for common patterns
    upscaling: 'icon-[lucide--maximize-2]',
    controlnet: 'icon-[lucide--sliders-horizontal]',
    'area-composition': 'icon-[lucide--layout-grid]'
  }

  // Return mapped icon or fallback to folder
  return iconMap[categoryId.toLowerCase()] || 'icon-[lucide--folder]'
}

/**
 * Provider brand colors extracted from SVG icons.
 * Each entry can be a single color or [color1, color2] for gradient.
 */
const PROVIDER_COLORS: Record<string, string | [string, string]> = {
  bfl: '#ffffff',
  bria: '#B6B6B6',
  bytedance: ['#00C8D2', '#325AB4'],
  gemini: ['#3186FF', '#FABC12'],
  grok: '#B6B6B6',
  hitpaw: '#B6B6B6',
  ideogram: '#B6B6B6',
  kling: ['#0BF2F9', '#FFF959'],
  ltxv: '#B6B6B6',
  luma: ['#004EFF', '#00FFFF'],
  magnific: ['#EA5A3D', '#F1A64A'],
  meshy: ['#67B700', '#FA418C'],
  minimax: ['#E2167E', '#FE603C'],
  'moonvalley-marey': '#DAD9C5',
  openai: '#B6B6B6',
  pixverse: ['#B465E6', '#E8632A'],
  recraft: '#B6B6B6',
  rodin: '#F7F7F7',
  runway: '#B6B6B6',
  sora: ['#6BB6FE', '#ffffff'],
  'stability-ai': ['#9D39FF', '#E80000'],
  tencent: ['#004BE5', '#00B3FE'],
  topaz: '#B6B6B6',
  tripo: ['#F6D85A', '#B6B6B6'],
  veo: ['#4285F4', '#EB4335'],
  vidu: ['#047FFE', '#40EDD8'],
  wan: ['#6156EC', '#F4F3FD'],
  wavespeed: '#B6B6B6'
}

/**
 * Extracts the provider name from a node category path.
 * e.g. "api/image/BFL" -> "BFL"
 */
export function getProviderName(category: string): string {
  return category.split('/').at(-1) ?? ''
}

/**
 * Returns the icon class for an API node provider (e.g., BFL, OpenAI, Stability AI)
 * @param providerName - The provider name from the node category
 * @returns The icon class string (e.g., 'icon-[comfy--bfl]')
 */
export function getProviderIcon(providerName: string): string {
  const iconKey = providerName.toLowerCase().replaceAll(/\s+/g, '-')
  return `icon-[comfy--${iconKey}]`
}

/**
 * Returns the border color(s) for an API node provider badge.
 * @param providerName - The provider name from the node category
 * @returns CSS color string or gradient definition
 */
export function getProviderBorderStyle(providerName: string): string {
  const iconKey = providerName.toLowerCase().replaceAll(/\s+/g, '-')
  const colors = PROVIDER_COLORS[iconKey]

  if (!colors) {
    return '#525252' // neutral-600 fallback
  }

  if (Array.isArray(colors)) {
    return `linear-gradient(90deg, ${colors[0]}, ${colors[1]})`
  }

  return colors
}

/**
 * Generates a unique category ID from a category group and title
 */
export function generateCategoryId(
  categoryGroup: string,
  categoryTitle: string
) {
  return `${categoryGroup.toLowerCase().replace(/\s+/g, '-')}-${categoryTitle.toLowerCase().replace(/\s+/g, '-')}`
}
