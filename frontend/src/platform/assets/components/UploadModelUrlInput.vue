<template>
  <div class="flex flex-col justify-between h-full gap-6 text-sm">
    <div class="flex flex-col gap-6">
      <div class="flex flex-col gap-2">
        <p class="m-0 text-foreground">
          {{ $t('assetBrowser.uploadModelDescription1Generic') }}
        </p>
        <div class="m-0">
          <p class="m-0 text-muted-foreground">
            {{ $t('assetBrowser.uploadModelDescription2Generic') }}
          </p>
          <span class="inline-flex items-center gap-1 flex-wrap mt-2">
            <span class="inline-flex items-center gap-1">
              <img
                :src="civitaiIcon"
                :alt="$t('assetBrowser.providerCivitai')"
                class="w-4 h-4"
              />
              <a
                :href="civitaiUrl"
                target="_blank"
                rel="noopener noreferrer"
                class="text-muted-foreground underline"
              >
                {{ $t('assetBrowser.providerCivitai') }}</a
              ><span>,</span>
            </span>
            <span class="inline-flex items-center gap-1">
              <img
                :src="huggingFaceIcon"
                :alt="$t('assetBrowser.providerHuggingFace')"
                class="w-4 h-4"
              />
              <a
                :href="huggingFaceUrl"
                target="_blank"
                rel="noopener noreferrer"
                class="text-muted-foreground underline"
              >
                {{ $t('assetBrowser.providerHuggingFace') }}
              </a>
            </span>
          </span>
        </div>
      </div>

      <div class="flex flex-col gap-2">
        <InputText
          v-model="url"
          autofocus
          :placeholder="$t('assetBrowser.genericLinkPlaceholder')"
          class="w-full border-0 bg-secondary-background p-4"
          data-attr="upload-model-step1-url-input"
        />
        <p v-if="error" class="text-sm text-error">
          {{ error }}
        </p>
      </div>
    </div>

    <div class="flex flex-col gap-6 text-sm text-muted-foreground">
      <div v-if="showSecretsHint">
        <i18n-t keypath="assetBrowser.apiKeyHint" tag="span">
          <template #link>
            <Button
              variant="textonly"
              size="unset"
              class="text-muted-foreground underline p-0"
              @click="openSecretsSettings"
            >
              {{ $t('assetBrowser.apiKeyHintLink') }}
            </Button>
          </template>
        </i18n-t>
      </div>
      <div>
        {{ $t('assetBrowser.uploadModelHelpFooterText') }}
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import InputText from 'primevue/inputtext'
import { computed } from 'vue'

import Button from '@/components/ui/button/Button.vue'
import { useFeatureFlags } from '@/composables/useFeatureFlags'
import { useSettingsDialog } from '@/platform/settings/composables/useSettingsDialog'

const { flags } = useFeatureFlags()
const settingsDialog = useSettingsDialog()

const showSecretsHint = computed(() => flags.userSecretsEnabled)

function openSecretsSettings() {
  settingsDialog.show('secrets')
}

const props = defineProps<{
  modelValue: string
  error?: string
}>()

const emit = defineEmits<{
  'update:modelValue': [value: string]
}>()

const url = computed({
  get: () => props.modelValue,
  set: (value: string) => emit('update:modelValue', value)
})

const civitaiIcon = '/assets/images/civitai.svg'
const civitaiUrl = 'https://civitai.com/models'
const huggingFaceIcon = '/assets/images/hf-logo.svg'
const huggingFaceUrl = 'https://huggingface.co'
</script>
