<template>
  <div
    class="flex items-center justify-between rounded-lg border border-border-default bg-base-raised-surface p-4"
  >
    <div class="flex flex-col gap-1">
      <div class="flex items-center gap-2">
        <span class="font-medium text-base-foreground">{{ secret.name }}</span>
        <img
          v-if="providerLogo"
          :src="providerLogo"
          :alt="providerLabel"
          class="h-5 w-5"
        />
        <span
          v-else-if="secret.provider"
          class="rounded bg-base-surface px-2 py-0.5 text-xs text-muted"
        >
          {{ providerLabel }}
        </span>
      </div>
      <div class="flex gap-3 text-xs text-muted">
        <span>{{ $t('secrets.createdAt', { date: createdDate }) }}</span>
        <span v-if="secret.last_used_at">
          {{ $t('secrets.lastUsed', { date: lastUsedDate }) }}
        </span>
      </div>
    </div>
    <div class="flex items-center gap-2">
      <i v-if="loading" class="pi pi-spinner pi-spin text-muted" />
      <template v-else>
        <Button
          v-tooltip="{ value: $t('g.edit'), showDelay: 300 }"
          variant="muted-textonly"
          size="icon-sm"
          :aria-label="$t('g.edit')"
          :disabled="disabled"
          @click="emit('edit')"
        >
          <i class="pi pi-pen-to-square" />
        </Button>
        <Button
          v-tooltip="{ value: $t('g.delete'), showDelay: 300 }"
          variant="muted-textonly"
          size="icon-sm"
          :aria-label="$t('g.delete')"
          :disabled="disabled"
          @click="emit('delete')"
        >
          <i class="pi pi-trash" />
        </Button>
      </template>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

import Button from '@/components/ui/button/Button.vue'

import { getProviderLabel, getProviderLogo } from '../providers'
import type { SecretMetadata } from '../types'

const {
  secret,
  loading = false,
  disabled = false
} = defineProps<{
  secret: SecretMetadata
  loading?: boolean
  disabled?: boolean
}>()

const emit = defineEmits<{
  edit: []
  delete: []
}>()

const providerLabel = computed(() => getProviderLabel(secret.provider))
const providerLogo = computed(() => getProviderLogo(secret.provider))

function formatDateString(dateString: string): string {
  const date = new Date(dateString)
  return date.toLocaleDateString()
}

const createdDate = computed(() => formatDateString(secret.created_at))
const lastUsedDate = computed(() =>
  secret.last_used_at ? formatDateString(secret.last_used_at) : ''
)
</script>
