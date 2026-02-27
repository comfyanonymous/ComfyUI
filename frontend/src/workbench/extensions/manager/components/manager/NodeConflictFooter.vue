<template>
  <div class="flex w-full items-center justify-between px-3 py-4">
    <div class="flex w-full items-center justify-between gap-2 pr-1">
      <Button
        variant="muted-textonly"
        size="sm"
        class="text-sm"
        @click="handleConflictInfoClick"
      >
        <i class="pi pi-info-circle" />
        {{ $t('manager.conflicts.conflictInfoTitle') }}
      </Button>
      <Button
        v-if="buttonText"
        variant="secondary"
        size="sm"
        @click="handleButtonClick"
      >
        {{ buttonText }}
      </Button>
    </div>
  </div>
</template>

<script setup lang="ts">
import Button from '@/components/ui/button/Button.vue'
import { useExternalLink } from '@/composables/useExternalLink'
import { useDialogStore } from '@/stores/dialogStore'

interface Props {
  buttonText?: string
  onButtonClick?: () => void
}
const { buttonText, onButtonClick } = defineProps<Props>()
const { buildDocsUrl } = useExternalLink()
const dialogStore = useDialogStore()
const handleConflictInfoClick = () => {
  window.open(
    buildDocsUrl('/troubleshooting/custom-node-issues', {
      includeLocale: true
    }),
    '_blank'
  )
}
const handleButtonClick = () => {
  // Close the conflict dialog
  dialogStore.closeDialog({ key: 'global-node-conflict' })
  // Execute the custom button action if provided
  if (onButtonClick) {
    onButtonClick()
  }
}
</script>
