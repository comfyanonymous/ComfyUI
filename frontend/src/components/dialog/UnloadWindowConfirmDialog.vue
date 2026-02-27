<template>
  <div>
    <!--
    UnloadWindowConfirmDialog: This component does not render
    anything visible. It is used to confirm the user wants to
    close the window, and if they do, it will call the
    beforeunload event.
    -->
  </div>
</template>

<script setup lang="ts">
import { onBeforeUnmount, onMounted } from 'vue'

import { useSettingStore } from '@/platform/settings/settingStore'
import { useWorkflowStore } from '@/platform/workflow/management/stores/workflowStore'

const settingStore = useSettingStore()
const workflowStore = useWorkflowStore()

const handleBeforeUnload = (event: BeforeUnloadEvent) => {
  if (
    settingStore.get('Comfy.Window.UnloadConfirmation') &&
    workflowStore.modifiedWorkflows.length > 0
  ) {
    event.preventDefault()
    return true
  }
  return undefined
}

onMounted(() => {
  window.addEventListener('beforeunload', handleBeforeUnload)
})

onBeforeUnmount(() => {
  window.removeEventListener('beforeunload', handleBeforeUnload)
})
</script>
