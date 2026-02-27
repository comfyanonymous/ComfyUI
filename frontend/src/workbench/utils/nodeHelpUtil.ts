import type { ComfyNodeDefImpl } from '@/stores/nodeDefStore'
import { NodeSourceType, getNodeSource } from '@/types/nodeSource'
import { normalizePackId } from '@/utils/packUtils'

export function extractCustomNodeName(
  pythonModule: string | undefined
): string | null {
  const modules = pythonModule?.split('.') || []
  if (modules.length >= 2 && modules[0] === 'custom_nodes') {
    // Use normalizePackId to remove version suffix
    return normalizePackId(modules[1])
  }
  return null
}

export function getNodeHelpBaseUrl(node: ComfyNodeDefImpl): string {
  const nodeSource = getNodeSource(node.python_module)
  if (nodeSource.type === NodeSourceType.Blueprint) {
    return ''
  }
  if (nodeSource.type === NodeSourceType.CustomNodes) {
    const customNodeName = extractCustomNodeName(node.python_module)
    if (customNodeName) {
      return `/extensions/${customNodeName}/docs/`
    }
  }
  return `/docs/${node.name}/`
}
