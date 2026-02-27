import {
  ESSENTIALS_CATEGORY_CANONICAL,
  ESSENTIALS_CATEGORY_MAP
} from '@/constants/essentialsNodes'

export enum NodeSourceType {
  Core = 'core',
  CustomNodes = 'custom_nodes',
  Blueprint = 'blueprint',
  Essentials = 'essentials',
  Unknown = 'unknown'
}

export type NodeSource = {
  type: NodeSourceType
  className: string
  displayText: string
  badgeText: string
}

const UNKNOWN_NODE_SOURCE: NodeSource = {
  type: NodeSourceType.Unknown,
  className: 'comfy-unknown',
  displayText: 'Unknown',
  badgeText: '?'
}

const shortenNodeName = (name: string) => {
  return name
    .replace(/^(ComfyUI-|ComfyUI_|Comfy-|Comfy_)/, '')
    .replace(/(-ComfyUI|_ComfyUI|-Comfy|_Comfy)$/, '')
}

/**
 * Get the essentials category for a node, falling back to mock data if not provided.
 */
export function getEssentialsCategory(
  name?: string,
  essentials_category?: string
): string | undefined {
  const normalizedCategory = essentials_category?.trim().toLowerCase()
  const canonical = normalizedCategory
    ? (ESSENTIALS_CATEGORY_CANONICAL.get(normalizedCategory) ??
      normalizedCategory)
    : undefined
  return canonical ?? (name ? ESSENTIALS_CATEGORY_MAP[name] : undefined)
}

export const getNodeSource = (
  python_module?: string,
  essentials_category?: string,
  name?: string
): NodeSource => {
  const resolvedEssentialsCategory = getEssentialsCategory(
    name,
    essentials_category
  )
  if (!python_module) {
    return UNKNOWN_NODE_SOURCE
  }
  const modules = python_module.split('.')
  if (resolvedEssentialsCategory) {
    const moduleName = modules[1] ?? modules[0] ?? 'essentials'
    const displayName = shortenNodeName(moduleName.split('@')[0])
    return {
      type: NodeSourceType.Essentials,
      className: 'comfy-essentials',
      displayText: displayName,
      badgeText: displayName
    }
  } else if (
    ['nodes', 'comfy_extras', 'comfy_api_nodes'].includes(modules[0])
  ) {
    return {
      type: NodeSourceType.Core,
      className: 'comfy-core',
      displayText: 'Comfy Core',
      badgeText: '🦊'
    }
  } else if (modules[0] === 'blueprint') {
    return {
      type: NodeSourceType.Blueprint,
      className: 'blueprint',
      displayText: 'Blueprint',
      badgeText: 'bp'
    }
  } else if (modules[0] === 'custom_nodes') {
    const moduleName = modules[1]
    const customNodeName = moduleName.split('@')[0]
    const displayName = shortenNodeName(customNodeName)
    return {
      type: NodeSourceType.CustomNodes,
      className: 'comfy-custom-nodes',
      displayText: displayName,
      badgeText: displayName
    }
  } else {
    return UNKNOWN_NODE_SOURCE
  }
}

export enum NodeBadgeMode {
  None = 'None',
  ShowAll = 'Show all',
  HideBuiltIn = 'Hide built-in'
}
