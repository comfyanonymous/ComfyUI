<template>
  <BuilderDialog @close="onClose">
    <template #header-icon>
      <i class="icon-[lucide--circle-check-big] size-4 text-green-500" />
    </template>
    <template #title>
      {{ $t('builderToolbar.saveSuccess') }}
    </template>

    <p v-if="savedAsApp" class="m-0 text-sm text-muted-foreground">
      {{ $t('builderToolbar.saveSuccessAppMessage', { name: workflowName }) }}
    </p>
    <p v-if="savedAsApp" class="m-0 text-sm text-muted-foreground">
      {{ $t('builderToolbar.saveSuccessAppPrompt') }}
    </p>
    <p v-else class="m-0 text-sm text-muted-foreground">
      {{ $t('builderToolbar.saveSuccessGraphMessage', { name: workflowName }) }}
    </p>

    <template #footer>
      <Button
        :variant="savedAsApp ? 'muted-textonly' : 'secondary'"
        size="lg"
        @click="onClose"
      >
        {{ $t('g.close') }}
      </Button>
      <Button
        v-if="savedAsApp && onViewApp"
        variant="primary"
        size="lg"
        @click="onViewApp"
      >
        {{ $t('builderToolbar.viewApp') }}
      </Button>
    </template>
  </BuilderDialog>
</template>

<script setup lang="ts">
import Button from '@/components/ui/button/Button.vue'

import BuilderDialog from './BuilderDialog.vue'

defineProps<{
  workflowName: string
  savedAsApp: boolean
  onViewApp?: () => void
  onClose: () => void
}>()
</script>
