import { computed, onUnmounted, watch } from 'vue'

import { useSettingStore } from '@/platform/settings/settingStore'
import { useWorkflowService } from '@/platform/workflow/core/services/workflowService'
import { useWorkflowStore } from '@/platform/workflow/management/stores/workflowStore'
import { api } from '@/scripts/api'

export function useWorkflowAutoSave() {
  const workflowStore = useWorkflowStore()
  const settingStore = useSettingStore()
  const workflowService = useWorkflowService()

  // Use computed refs to cache autosave settings
  const autoSaveSetting = computed(() =>
    settingStore.get('Comfy.Workflow.AutoSave')
  )
  const autoSaveDelay = computed(() =>
    settingStore.get('Comfy.Workflow.AutoSaveDelay')
  )

  let autoSaveTimeout: NodeJS.Timeout | null = null
  let isSaving = false
  let needsAutoSave = false

  const scheduleAutoSave = () => {
    // Clear any existing timeout
    if (autoSaveTimeout) {
      clearTimeout(autoSaveTimeout)
      autoSaveTimeout = null
    }

    // If autosave is enabled
    if (autoSaveSetting.value === 'after delay') {
      // If a save is in progress, mark that we need an autosave after saving
      if (isSaving) {
        needsAutoSave = true
        return
      }
      const delay = autoSaveDelay.value
      autoSaveTimeout = setTimeout(async () => {
        const activeWorkflow = workflowStore.activeWorkflow
        if (activeWorkflow?.isModified && activeWorkflow.isPersisted) {
          try {
            isSaving = true
            await workflowService.saveWorkflow(activeWorkflow)
          } catch (err) {
            console.error('Auto save failed:', err)
          } finally {
            isSaving = false
            if (needsAutoSave) {
              needsAutoSave = false
              scheduleAutoSave()
            }
          }
        }
      }, delay)
    }
  }

  // Watch for autosave setting changes
  watch(
    autoSaveSetting,
    (newSetting) => {
      // Clear any existing timeout when settings change
      if (autoSaveTimeout) {
        clearTimeout(autoSaveTimeout)
        autoSaveTimeout = null
      }

      // If there's an active modified workflow and autosave is enabled, schedule a save
      if (
        newSetting === 'after delay' &&
        workflowStore.activeWorkflow?.isModified
      ) {
        scheduleAutoSave()
      }
    },
    { immediate: true }
  )

  // Listen for graph changes and schedule autosave when they occur
  const onGraphChanged = () => {
    scheduleAutoSave()
  }

  api.addEventListener('graphChanged', onGraphChanged)

  onUnmounted(() => {
    if (autoSaveTimeout) {
      clearTimeout(autoSaveTimeout)
      autoSaveTimeout = null
    }
    api.removeEventListener('graphChanged', onGraphChanged)
  })
}
