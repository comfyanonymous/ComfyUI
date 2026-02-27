import { computed, ref } from 'vue'

import { useWorkflowStore } from '@/platform/workflow/management/stores/workflowStore'

export type AppMode = 'graph' | 'app' | 'builder:select' | 'builder:arrange'

const enableAppBuilder = ref(true)

export function useAppMode() {
  const workflowStore = useWorkflowStore()
  const mode = computed(
    () =>
      workflowStore.activeWorkflow?.activeMode ??
      workflowStore.activeWorkflow?.initialMode ??
      'graph'
  )

  const isBuilderMode = computed(
    () => mode.value === 'builder:select' || mode.value === 'builder:arrange'
  )
  const isAppMode = computed(
    () => mode.value === 'app' || mode.value === 'builder:arrange'
  )
  const isGraphMode = computed(
    () => mode.value === 'graph' || mode.value === 'builder:select'
  )

  function setMode(newMode: AppMode) {
    if (newMode === mode.value) return

    const workflow = workflowStore.activeWorkflow
    if (workflow) workflow.activeMode = newMode
  }

  return {
    mode,
    enableAppBuilder,
    isBuilderMode,
    isAppMode,
    isGraphMode,
    setMode
  }
}
