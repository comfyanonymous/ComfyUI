import type { MaybeRefOrGetter } from 'vue'
import { computed, toValue } from 'vue'

import type { LGraphNode } from '@/lib/litegraph/src/LGraphNode'
import { SubgraphNode } from '@/lib/litegraph/src/subgraph/SubgraphNode'
import { useNodeOutputStore } from '@/stores/imagePreviewStore'
import { usePromotionStore } from '@/stores/promotionStore'
import { createNodeLocatorId } from '@/types/nodeIdentification'

interface PromotedPreview {
  interiorNodeId: string
  widgetName: string
  type: 'image' | 'video' | 'audio'
  urls: string[]
}

/**
 * Returns reactive preview media from promoted `$$` pseudo-widgets
 * on a SubgraphNode. Each promoted preview interior node produces
 * a separate entry so they render independently.
 */
export function usePromotedPreviews(
  lgraphNode: MaybeRefOrGetter<LGraphNode | null | undefined>
) {
  const promotionStore = usePromotionStore()
  const nodeOutputStore = useNodeOutputStore()

  const promotedPreviews = computed((): PromotedPreview[] => {
    const node = toValue(lgraphNode)
    if (!(node instanceof SubgraphNode)) return []

    const entries = promotionStore.getPromotions(node.rootGraph.id, node.id)
    const pseudoEntries = entries.filter((e) => e.widgetName.startsWith('$$'))
    if (!pseudoEntries.length) return []

    const previews: PromotedPreview[] = []

    for (const entry of pseudoEntries) {
      const interiorNode = node.subgraph.getNodeById(entry.interiorNodeId)
      if (!interiorNode) continue

      // Read from the reactive nodeOutputs ref to establish Vue
      // dependency tracking. getNodeImageUrls reads from the
      // non-reactive app.nodeOutputs, so without this access the
      // computed would never re-evaluate when outputs change.
      const locatorId = createNodeLocatorId(
        node.subgraph.id,
        entry.interiorNodeId
      )
      const _reactiveOutputs = nodeOutputStore.nodeOutputs[locatorId]
      if (!_reactiveOutputs?.images?.length) continue

      const urls = nodeOutputStore.getNodeImageUrls(interiorNode)
      if (!urls?.length) continue

      const type =
        interiorNode.previewMediaType === 'video'
          ? 'video'
          : interiorNode.previewMediaType === 'audio'
            ? 'audio'
            : 'image'

      previews.push({
        interiorNodeId: entry.interiorNodeId,
        widgetName: entry.widgetName,
        type,
        urls
      })
    }

    return previews
  })

  return { promotedPreviews }
}
