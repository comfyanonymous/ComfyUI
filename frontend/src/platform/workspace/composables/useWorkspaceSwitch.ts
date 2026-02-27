import { storeToRefs } from 'pinia'
import { useI18n } from 'vue-i18n'

import { useWorkflowStore } from '@/platform/workflow/management/stores/workflowStore'
import { useTeamWorkspaceStore } from '@/platform/workspace/stores/teamWorkspaceStore'
import { useDialogService } from '@/services/dialogService'

export function useWorkspaceSwitch() {
  const { t } = useI18n()
  const workspaceStore = useTeamWorkspaceStore()
  const { activeWorkspace } = storeToRefs(workspaceStore)
  const workflowStore = useWorkflowStore()
  const dialogService = useDialogService()

  function hasUnsavedChanges(): boolean {
    return workflowStore.modifiedWorkflows.length > 0
  }

  async function switchWithConfirmation(workspaceId: string): Promise<boolean> {
    if (activeWorkspace.value?.id === workspaceId) {
      return true
    }

    if (hasUnsavedChanges()) {
      const confirmed = await dialogService.confirm({
        title: t('workspace.unsavedChanges.title'),
        message: t('workspace.unsavedChanges.message'),
        type: 'dirtyClose'
      })

      if (!confirmed) {
        return false
      }
    }

    try {
      await workspaceStore.switchWorkspace(workspaceId)
      // Note: switchWorkspace triggers page reload internally
      return true
    } catch {
      return false
    }
  }

  return {
    hasUnsavedChanges,
    switchWithConfirmation
  }
}
