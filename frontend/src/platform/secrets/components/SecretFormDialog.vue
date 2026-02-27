<template>
  <Dialog
    v-model:visible="visible"
    :header="
      mode === 'create' ? $t('secrets.addSecret') : $t('secrets.editSecret')
    "
    modal
    class="w-full max-w-md"
  >
    <form class="flex flex-col gap-4" @submit.prevent="handleSubmit">
      <div class="flex flex-col gap-1">
        <label for="secret-provider" class="text-sm font-medium">
          {{ $t('secrets.provider') }}
        </label>
        <Select v-model="form.provider" :disabled="mode === 'edit'">
          <SelectTrigger id="secret-provider" class="w-full" autofocus>
            <SelectValue :placeholder="$t('g.none')" />
          </SelectTrigger>
          <SelectContent disable-portal>
            <SelectItem
              v-for="option in providerOptions"
              :key="option.value || 'none'"
              :value="option.value"
              :disabled="option.disabled"
            >
              {{ option.label }}
            </SelectItem>
          </SelectContent>
        </Select>
        <small v-if="errors.provider" class="text-red-500">
          {{ errors.provider }}
        </small>
      </div>

      <div class="flex flex-col gap-1">
        <label for="secret-name" class="text-sm font-medium">
          {{ $t('secrets.name') }}
        </label>
        <InputText
          id="secret-name"
          v-model="form.name"
          :placeholder="$t('secrets.namePlaceholder')"
          :class="{ 'p-invalid': errors.name }"
        />
        <small v-if="errors.name" class="text-red-500">{{ errors.name }}</small>
      </div>

      <div class="flex flex-col gap-1">
        <label for="secret-value" class="text-sm font-medium">
          {{ $t('secrets.secretValue') }}
        </label>
        <Password
          id="secret-value"
          v-model="form.secretValue"
          :placeholder="
            mode === 'edit'
              ? $t('secrets.secretValuePlaceholderEdit')
              : $t('secrets.secretValuePlaceholder')
          "
          :feedback="false"
          toggle-mask
          fluid
          :class="{ 'p-invalid': errors.secretValue }"
        />
        <small v-if="errors.secretValue" class="text-red-500">
          {{ errors.secretValue }}
        </small>
        <small v-else class="text-muted">
          {{
            mode === 'edit'
              ? $t('secrets.secretValueHintEdit')
              : $t('secrets.secretValueHint')
          }}
        </small>
      </div>

      <span v-if="apiError" class="text-sm text-destructive">
        {{ apiError }}
      </span>

      <div class="flex justify-end gap-2 pt-2">
        <Button
          variant="secondary"
          type="button"
          tabindex="0"
          @click="visible = false"
        >
          {{ $t('g.cancel') }}
        </Button>
        <Button type="submit" tabindex="0" :loading="loading">
          {{ $t('g.save') }}
        </Button>
      </div>
    </form>
  </Dialog>
</template>

<script setup lang="ts">
import Dialog from 'primevue/dialog'
import InputText from 'primevue/inputtext'
import Password from 'primevue/password'

import Button from '@/components/ui/button/Button.vue'
import Select from '@/components/ui/select/Select.vue'
import SelectContent from '@/components/ui/select/SelectContent.vue'
import SelectItem from '@/components/ui/select/SelectItem.vue'
import SelectTrigger from '@/components/ui/select/SelectTrigger.vue'
import SelectValue from '@/components/ui/select/SelectValue.vue'

import { useSecretForm } from '../composables/useSecretForm'
import type { SecretMetadata, SecretProvider } from '../types'

const {
  secret,
  existingProviders = [],
  mode = 'create'
} = defineProps<{
  secret?: SecretMetadata
  existingProviders?: SecretProvider[]
  mode?: 'create' | 'edit'
}>()

const visible = defineModel<boolean>('visible', { default: false })

const emit = defineEmits<{
  saved: []
}>()

const { form, errors, loading, apiError, providerOptions, handleSubmit } =
  useSecretForm({
    mode,
    secret: () => secret,
    existingProviders: () => existingProviders,
    visible,
    onSaved: () => emit('saved')
  })
</script>
