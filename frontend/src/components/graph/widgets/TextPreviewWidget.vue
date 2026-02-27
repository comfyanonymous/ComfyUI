<template>
  <div
    class="relative max-h-[200px] min-h-[28px] w-full overflow-y-auto rounded-lg px-4 py-2 text-xs"
  >
    <div class="flex items-center gap-2">
      <div class="flex flex-1 items-center gap-2 break-all">
        <span v-html="formattedText"></span>
        <Skeleton v-if="isParentNodeExecuting" class="h-4! flex-1!" />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import Skeleton from 'primevue/skeleton'
import { computed, onMounted, ref, watch } from 'vue'

import type { NodeId } from '@/lib/litegraph/src/litegraph'
import { useExecutionStore } from '@/stores/executionStore'
import { linkifyHtml, nl2br } from '@/utils/formatUtil'

const modelValue = defineModel<string>({ required: true })
const props = defineProps<{
  nodeId: NodeId
}>()

const executionStore = useExecutionStore()
const isParentNodeExecuting = ref(true)
const formattedText = computed(() => {
  const src = modelValue.value
  // Turn [[label|url]] into placeholders to avoid interfering with linkifyHtml
  const tokens: { label: string; url: string }[] = []
  const holed = src.replace(
    /\[\[([^|\]]+)\|([^\]]+)\]\]/g,
    (_m, label, url) => {
      tokens.push({ label: String(label), url: String(url) })
      return `__LNK${tokens.length - 1}__`
    }
  )

  // Keep current behavior (auto-link bare URLs + \n -> <br>)
  let html = nl2br(linkifyHtml(holed))

  // Restore placeholders as <a>...</a> (minimal escaping + http default)
  html = html.replace(/__LNK(\d+)__/g, (_m, i) => {
    const { label, url } = tokens[+i]
    const safeHref = url.replace(/"/g, '&quot;')
    const safeLabel = label.replace(/</g, '&lt;').replace(/>/g, '&gt;')
    return /^https?:\/\//i.test(url)
      ? `<a href="${safeHref}" target="_blank" rel="noopener noreferrer">${safeLabel}</a>`
      : safeLabel
  })

  return html
})

let parentNodeId: NodeId | null = null
onMounted(() => {
  // Get the parent node ID from props if provided
  // For backward compatibility, fall back to the first executing node
  parentNodeId = props.nodeId
})

// Watch for either a new node has starting execution or overall execution ending
const stopWatching = watch(
  [() => executionStore.executingNodeIds, () => executionStore.isIdle],
  () => {
    if (executionStore.isIdle) {
      isParentNodeExecuting.value = false
      stopWatching()
      return
    }

    // Check if parent node is no longer in the executing nodes list
    if (
      parentNodeId &&
      !executionStore.executingNodeIds.includes(parentNodeId)
    ) {
      isParentNodeExecuting.value = false
      stopWatching()
    }

    // Set parent node ID if not set yet
    if (!parentNodeId && executionStore.executingNodeIds.length > 0) {
      parentNodeId = executionStore.executingNodeIds[0]
    }
  }
)
</script>
