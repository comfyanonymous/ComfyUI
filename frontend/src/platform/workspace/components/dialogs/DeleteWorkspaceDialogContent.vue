<template>
  <div
    class="flex w-full max-w-[360px] flex-col rounded-2xl border border-border-default bg-base-background"
  >
    <!-- Header -->
    <div
      class="flex h-12 items-center justify-between border-b border-border-default px-4"
    >
      <h2 class="m-0 text-sm font-normal text-base-foreground">
        {{ $t('workspacePanel.deleteDialog.title') }}
      </h2>
      <button
        class="cursor-pointer rounded border-none bg-transparent p-0 text-muted-foreground transition-colors hover:text-base-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-secondary-foreground"
        :aria-label="$t('g.close')"
        @click="onCancel"
      >
        <i class="pi pi-times size-4" />
      </button>
    </div>

    <!-- Body -->
    <div class="px-4 py-4">
      <p class="m-0 text-sm text-muted-foreground">
        {{
          workspaceName
            ? $t('workspacePanel.deleteDialog.messageWithName', {
                name: workspaceName
              })
            : $t('workspacePanel.deleteDialog.message')
        }}
      </p>
    </div>

    <!-- Footer -->
    <div class="flex items-center justify-end gap-4 px-4 py-4">
      <Button variant="muted-textonly" @click="onCancel">
        {{ $t('g.cancel') }}
      </Button>
      <Button variant="destructive" size="lg" :loading @click="onDelete">
        {{ $t('g.delete') }}
      </Button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useToast } from 'primevue/usetoast'
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'

import Button from '@/components/ui/button/Button.vue'
import { useTeamWorkspaceStore } from '@/platform/workspace/stores/teamWorkspaceStore'
import { useDialogStore } from '@/stores/dialogStore'

const { workspaceId, workspaceName } = defineProps<{
  workspaceId?: string
  workspaceName?: string
}>()

const { t } = useI18n()
const toast = useToast()
const dialogStore = useDialogStore()
const workspaceStore = useTeamWorkspaceStore()
const loading = ref(false)

function onCancel() {
  dialogStore.closeDialog({ key: 'delete-workspace' })
}

async function onDelete() {
  loading.value = true
  try {
    // Delete workspace (uses workspaceId if provided, otherwise current workspace)
    await workspaceStore.deleteWorkspace(workspaceId)
    dialogStore.closeDialog({ key: 'delete-workspace' })
    window.location.reload()
  } catch (error) {
    console.error('[DeleteWorkspaceDialog] Failed to delete workspace:', error)
    toast.add({
      severity: 'error',
      summary: t('workspacePanel.toast.failedToDeleteWorkspace'),
      detail: error instanceof Error ? error.message : t('g.unknownError'),
      life: 5000
    })
  } finally {
    loading.value = false
  }
}
</script>
