<template>
  <div
    class="flex w-full min-w-[400px] flex-col rounded-2xl border border-border-default bg-base-background"
  >
    <!-- Header -->
    <div
      class="flex h-12 items-center justify-between border-b border-border-default px-4"
    >
      <h2 class="m-0 text-sm font-normal text-base-foreground">
        {{ $t('workspacePanel.editWorkspaceDialog.title') }}
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
    <div class="flex flex-col gap-4 px-4 py-4">
      <div class="flex flex-col gap-2">
        <label class="text-sm text-base-foreground">
          {{ $t('workspacePanel.editWorkspaceDialog.nameLabel') }}
        </label>
        <input
          v-model="newWorkspaceName"
          type="text"
          class="w-full rounded-lg border border-border-default bg-transparent px-3 py-2 text-sm text-base-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-secondary-foreground"
          @keydown.enter="isValidName && onSave()"
        />
      </div>
    </div>

    <!-- Footer -->
    <div class="flex items-center justify-end gap-4 px-4 py-4">
      <Button variant="muted-textonly" @click="onCancel">
        {{ $t('g.cancel') }}
      </Button>
      <Button
        variant="primary"
        size="lg"
        :loading
        :disabled="!isValidName"
        @click="onSave"
      >
        {{ $t('workspacePanel.editWorkspaceDialog.save') }}
      </Button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useToast } from 'primevue/usetoast'
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'

import Button from '@/components/ui/button/Button.vue'
import { useTeamWorkspaceStore } from '@/platform/workspace/stores/teamWorkspaceStore'
import { useDialogStore } from '@/stores/dialogStore'

const { t } = useI18n()
const toast = useToast()
const dialogStore = useDialogStore()
const workspaceStore = useTeamWorkspaceStore()
const loading = ref(false)
const newWorkspaceName = ref(workspaceStore.workspaceName)

const isValidName = computed(() => {
  const name = newWorkspaceName.value.trim()
  const safeNameRegex = /^[a-zA-Z0-9][a-zA-Z0-9\s\-_'.,()&+]*$/
  return name.length >= 1 && name.length <= 50 && safeNameRegex.test(name)
})

function onCancel() {
  dialogStore.closeDialog({ key: 'edit-workspace' })
}

async function onSave() {
  if (!isValidName.value) return
  loading.value = true
  try {
    await workspaceStore.updateWorkspaceName(newWorkspaceName.value.trim())
    dialogStore.closeDialog({ key: 'edit-workspace' })
    toast.add({
      severity: 'success',
      summary: t('workspacePanel.toast.workspaceUpdated.title'),
      detail: t('workspacePanel.toast.workspaceUpdated.message'),
      life: 5000
    })
  } catch (error) {
    console.error('[EditWorkspaceDialog] Failed to update workspace:', error)
    toast.add({
      severity: 'error',
      summary: t('workspacePanel.toast.failedToUpdateWorkspace'),
      detail: error instanceof Error ? error.message : t('g.unknownError'),
      life: 5000
    })
  } finally {
    loading.value = false
  }
}
</script>
