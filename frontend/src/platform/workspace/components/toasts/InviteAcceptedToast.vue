<template>
  <Toast group="invite-accepted" position="top-right">
    <template #message="slotProps">
      <div class="flex items-center gap-2 justify-between w-full">
        <div class="flex flex-col justify-start">
          <div class="text-base">
            {{ slotProps.message.summary }}
          </div>
          <div class="mt-1 text-sm text-foreground">
            {{ slotProps.message.detail.text }} <br />
            {{ slotProps.message.detail.workspaceName }}
          </div>
        </div>
        <Button
          size="md"
          variant="inverted"
          @click="viewWorkspace(slotProps.message.detail.workspaceId)"
        >
          {{ t('workspace.viewWorkspace') }}
        </Button>
      </div>
    </template>
  </Toast>
</template>

<script setup lang="ts">
import { useToast } from 'primevue'
import Toast from 'primevue/toast'
import { useI18n } from 'vue-i18n'

import Button from '@/components/ui/button/Button.vue'
import { useWorkspaceSwitch } from '@/platform/workspace/composables/useWorkspaceSwitch'

const { t } = useI18n()
const toast = useToast()
const { switchWithConfirmation } = useWorkspaceSwitch()

function viewWorkspace(workspaceId: string) {
  void switchWithConfirmation(workspaceId)
  toast.removeGroup('invite-accepted')
}
</script>
