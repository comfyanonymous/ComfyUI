import { useTitle } from '@vueuse/core'
import { computed } from 'vue'

import { t } from '@/i18n'
import { useSettingStore } from '@/platform/settings/settingStore'
import { useWorkflowStore } from '@/platform/workflow/management/stores/workflowStore'
import { useExecutionStore } from '@/stores/executionStore'
import { useWorkspaceStore } from '@/stores/workspaceStore'

const DEFAULT_TITLE = 'ComfyUI'
const TITLE_SUFFIX = ' - ComfyUI'

export const useBrowserTabTitle = () => {
  const executionStore = useExecutionStore()
  const settingStore = useSettingStore()
  const workflowStore = useWorkflowStore()
  const workspaceStore = useWorkspaceStore()

  const executionText = computed(() =>
    executionStore.isIdle
      ? ''
      : `[${Math.round(executionStore.executionProgress * 100)}%]`
  )

  const newMenuEnabled = computed(
    () => settingStore.get('Comfy.UseNewMenu') !== 'Disabled'
  )

  const isAutoSaveEnabled = computed(
    () => settingStore.get('Comfy.Workflow.AutoSave') === 'after delay'
  )

  const isActiveWorkflowModified = computed(
    () => !!workflowStore.activeWorkflow?.isModified
  )
  const isActiveWorkflowPersisted = computed(
    () => !!workflowStore.activeWorkflow?.isPersisted
  )

  const shouldShowUnsavedIndicator = computed(() => {
    if (workspaceStore.shiftDown) return false
    if (isAutoSaveEnabled.value) return false
    if (!isActiveWorkflowPersisted.value) return true
    if (isActiveWorkflowModified.value) return true
    return false
  })

  const isUnsavedText = computed(() =>
    shouldShowUnsavedIndicator.value ? ' *' : ''
  )
  const workflowNameText = computed(() => {
    const workflowName = workflowStore.activeWorkflow?.filename
    return workflowName
      ? isUnsavedText.value + workflowName + TITLE_SUFFIX
      : DEFAULT_TITLE
  })

  const nodeExecutionTitle = computed(() => {
    // Check if any nodes are in progress
    const nodeProgressEntries = Object.entries(
      executionStore.nodeProgressStates
    )
    const runningNodes = nodeProgressEntries.filter(
      ([_, state]) => state.state === 'running'
    )

    if (runningNodes.length === 0) {
      return ''
    }

    // If multiple nodes are running
    if (runningNodes.length > 1) {
      return `${executionText.value}[${runningNodes.length} ${t('g.nodesRunning', 'nodes running')}]`
    }

    // If only one node is running
    const [nodeId, state] = runningNodes[0]
    const progress = Math.round((state.value / state.max) * 100)
    const nodeType =
      executionStore.activeJob?.workflow?.changeTracker?.activeState.nodes.find(
        (n) => String(n.id) === nodeId
      )?.type || 'Node'

    return `${executionText.value}[${progress}%] ${nodeType}`
  })

  const workflowTitle = computed(
    () =>
      executionText.value +
      (newMenuEnabled.value ? workflowNameText.value : DEFAULT_TITLE)
  )

  const title = computed(() => nodeExecutionTitle.value || workflowTitle.value)
  useTitle(title)
}
