<script setup lang="ts">
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'

import { downloadFile } from '@/base/common/downloadUtil'
import Popover from '@/components/ui/Popover.vue'
import Button from '@/components/ui/button/Button.vue'
import { useMediaAssetActions } from '@/platform/assets/composables/useMediaAssetActions'
import { getOutputAssetMetadata } from '@/platform/assets/schemas/assetMetadataSchema'
import type { AssetItem } from '@/platform/assets/schemas/assetSchema'
import { useWorkflowStore } from '@/platform/workflow/management/stores/workflowStore'
import { extractWorkflowFromAsset } from '@/platform/workflow/utils/workflowExtractionUtil'
import ImagePreview from '@/renderer/extensions/linearMode/ImagePreview.vue'
import LatentPreview from '@/renderer/extensions/linearMode/LatentPreview.vue'
import LinearWelcome from '@/renderer/extensions/linearMode/LinearWelcome.vue'
import LinearArrange from '@/renderer/extensions/linearMode/LinearArrange.vue'
import OutputHistory from '@/renderer/extensions/linearMode/OutputHistory.vue'
import type { OutputSelection } from '@/renderer/extensions/linearMode/linearModeTypes'
// Lazy-loaded to avoid pulling THREE.js into the main bundle
const Preview3d = () => import('@/renderer/extensions/linearMode/Preview3d.vue')
import VideoPreview from '@/renderer/extensions/linearMode/VideoPreview.vue'
import { getMediaType } from '@/renderer/extensions/linearMode/mediaTypes'
import { app } from '@/scripts/app'
import { useCommandStore } from '@/stores/commandStore'
import { useExecutionStore } from '@/stores/executionStore'
import { useQueueStore } from '@/stores/queueStore'
import type { ResultItemImpl } from '@/stores/queueStore'
import { collectAllNodes } from '@/utils/graphTraversalUtil'
import { executeWidgetsCallback } from '@/utils/litegraphUtil'
import { useAppMode } from '@/composables/useAppMode'
const { t } = useI18n()
const commandStore = useCommandStore()
const executionStore = useExecutionStore()
const mediaActions = useMediaAssetActions()
const queueStore = useQueueStore()
const { mode: appModeValue } = useAppMode()
const { runButtonClick } = defineProps<{
  runButtonClick?: (e: Event) => void
  mobile?: boolean
}>()

const selectedItem = ref<AssetItem>()
const selectedOutput = ref<ResultItemImpl>()
const canShowPreview = ref(true)
const latentPreview = ref<string>()

function handleSelection(sel: OutputSelection) {
  selectedItem.value = sel.asset
  selectedOutput.value = sel.output
  canShowPreview.value = sel.canShowPreview
  latentPreview.value = sel.latentPreviewUrl
}

function downloadAsset(item?: AssetItem) {
  const user_metadata = getOutputAssetMetadata(item?.user_metadata)
  for (const output of user_metadata?.allOutputs ?? [])
    downloadFile(output.url, output.filename)
}

async function loadWorkflow(item: AssetItem | undefined) {
  if (!item) return
  const { workflow } = await extractWorkflowFromAsset(item)
  if (!workflow) return

  if (workflow.id !== app.rootGraph.id) return app.loadGraphData(workflow)
  //update graph to new version, set old to top of undo queue
  const changeTracker = useWorkflowStore().activeWorkflow?.changeTracker
  if (!changeTracker) return app.loadGraphData(workflow)
  changeTracker.redoQueue = []
  changeTracker.updateState([workflow], changeTracker.undoQueue)
}

async function rerun(e: Event) {
  if (!runButtonClick) return
  await loadWorkflow(selectedItem.value)
  //FIXME don't use timeouts here
  //Currently seeds fail to properly update even with timeouts?
  await new Promise((r) => setTimeout(r, 500))
  executeWidgetsCallback(collectAllNodes(app.rootGraph), 'afterQueued')

  runButtonClick(e)
}
</script>
<template>
  <section
    v-if="selectedItem || selectedOutput || !executionStore.isIdle"
    data-testid="linear-output-info"
    class="flex flex-wrap gap-2 p-4 w-full md:z-10 tabular-nums justify-center text-sm"
  >
    <template v-if="selectedItem">
      <Button size="md" @click="rerun">
        {{ t('linearMode.rerun') }}
        <i class="icon-[lucide--refresh-cw]" />
      </Button>
      <Button size="md" @click="() => loadWorkflow(selectedItem)">
        {{ t('linearMode.reuseParameters') }}
        <i class="icon-[lucide--list-restart]" />
      </Button>
      <div class="border-r border-border-subtle mx-1" />
    </template>
    <Button
      v-if="selectedOutput"
      size="icon"
      :aria-label="t('g.download')"
      @click="
        () => {
          if (selectedOutput?.url) downloadFile(selectedOutput.url)
        }
      "
    >
      <i class="icon-[lucide--download]" />
    </Button>
    <Button
      v-if="!executionStore.isIdle && !selectedItem"
      variant="destructive"
      size="icon"
      :aria-label="t('menu.interrupt')"
      @click="commandStore.execute('Comfy.Interrupt')"
    >
      <i class="icon-[lucide--x]" />
    </Button>
    <Popover
      v-if="selectedItem"
      :entries="[
        {
          icon: 'icon-[lucide--download]',
          label: t('linearMode.downloadAll'),
          command: () => downloadAsset(selectedItem!)
        },
        { separator: true },
        {
          icon: 'icon-[lucide--trash-2]',
          label: t('queue.jobMenu.deleteAsset'),
          command: () => mediaActions.deleteAssets(selectedItem!)
        }
      ]"
    />
  </section>
  <ImagePreview
    v-if="
      (canShowPreview && latentPreview) ||
      getMediaType(selectedOutput) === 'images'
    "
    :mobile
    :src="(canShowPreview && latentPreview) || selectedOutput!.url"
  />
  <VideoPreview
    v-else-if="getMediaType(selectedOutput) === 'video'"
    :src="selectedOutput!.url"
    class="object-contain flex-1 md:contain-size md:p-3"
  />
  <audio
    v-else-if="getMediaType(selectedOutput) === 'audio'"
    class="w-full m-auto"
    controls
    :src="selectedOutput!.url"
  />
  <article
    v-else-if="getMediaType(selectedOutput) === 'text'"
    class="w-full max-w-128 m-auto my-12 overflow-y-auto"
    v-text="selectedOutput!.url"
  />
  <Preview3d
    v-else-if="getMediaType(selectedOutput) === '3d'"
    :model-url="selectedOutput!.url"
  />
  <LatentPreview v-else-if="queueStore.runningTasks.length > 0" />
  <LinearArrange v-else-if="appModeValue === 'builder:arrange'" />
  <LinearWelcome v-else />
  <OutputHistory class="not-md:mx-40" @update-selection="handleSelection" />
</template>
