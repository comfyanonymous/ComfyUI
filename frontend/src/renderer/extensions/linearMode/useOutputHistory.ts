import { useAsyncState } from '@vueuse/core'
import type { MaybeRef } from 'vue'

import { useMediaAssets } from '@/platform/assets/composables/media/useMediaAssets'
import type { IAssetsProvider } from '@/platform/assets/composables/media/IAssetsProvider'
import { getOutputAssetMetadata } from '@/platform/assets/schemas/assetMetadataSchema'
import type { AssetItem } from '@/platform/assets/schemas/assetSchema'
import { flattenNodeOutput } from '@/renderer/extensions/linearMode/flattenNodeOutput'
import { useLinearOutputStore } from '@/renderer/extensions/linearMode/linearOutputStore'
import { getJobDetail } from '@/services/jobOutputCache'
import type { ResultItemImpl } from '@/stores/queueStore'

export function useOutputHistory(): {
  outputs: IAssetsProvider
  allOutputs: (item?: AssetItem) => MaybeRef<ResultItemImpl[]>
} {
  const outputs = useMediaAssets('output')
  void outputs.fetchMediaList()
  const linearStore = useLinearOutputStore()

  const outputsCache: Record<string, MaybeRef<ResultItemImpl[]>> = {}

  function allOutputs(item?: AssetItem): MaybeRef<ResultItemImpl[]> {
    if (item?.id && outputsCache[item.id]) return outputsCache[item.id]

    const user_metadata = getOutputAssetMetadata(item?.user_metadata)
    if (!user_metadata) return []

    // For recently completed jobs still pending resolve, derive order from
    // the in-progress items which are in correct execution order.
    if (linearStore.pendingResolve.has(user_metadata.jobId)) {
      const ordered = linearStore.inProgressItems
        .filter((i) => i.jobId === user_metadata.jobId && i.output)
        .map((i) => i.output!)
      if (ordered.length > 0) {
        outputsCache[item!.id] = ordered
        return ordered
      }
    }

    if (
      user_metadata.allOutputs &&
      user_metadata.outputCount &&
      user_metadata.outputCount <= user_metadata.allOutputs.length
    ) {
      const reversed = user_metadata.allOutputs.toReversed()
      outputsCache[item!.id] = reversed
      return reversed
    }

    const outputRef = useAsyncState(
      getJobDetail(user_metadata.jobId).then((jobDetail) => {
        if (!jobDetail?.outputs) return []
        return Object.entries(jobDetail.outputs)
          .flatMap(flattenNodeOutput)
          .toReversed()
      }),
      []
    ).state
    outputsCache[item!.id] = outputRef
    return outputRef
  }

  return { outputs, allOutputs }
}
