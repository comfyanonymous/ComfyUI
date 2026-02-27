<template>
  <div class="flex justify-end gap-2 w-full">
    <div v-if="currentStep === 1" class="mr-auto flex items-center gap-2">
      <i class="icon-[lucide--circle-question-mark] text-muted-foreground" />
      <Button
        variant="muted-textonly"
        size="sm"
        data-attr="upload-model-step1-help-civitai"
        @click="showCivitaiHelp = true"
      >
        {{ $t('assetBrowser.providerCivitai') }}
      </Button>
      <Button
        variant="muted-textonly"
        size="sm"
        data-attr="upload-model-step1-help-huggingface"
        @click="showHuggingFaceHelp = true"
      >
        {{ $t('assetBrowser.providerHuggingFace') }}
      </Button>
    </div>
    <Button
      v-if="currentStep === 1"
      variant="muted-textonly"
      size="lg"
      data-attr="upload-model-step1-cancel-button"
      :disabled="isFetchingMetadata || isUploading"
      @click="emit('close')"
    >
      {{ $t('g.cancel') }}
    </Button>
    <Button
      v-if="currentStep !== 1 && currentStep !== 3"
      variant="muted-textonly"
      size="lg"
      :data-attr="`upload-model-step${currentStep}-back-button`"
      :disabled="isFetchingMetadata || isUploading"
      @click="emit('back')"
    >
      {{ $t('g.back') }}
    </Button>
    <span v-else />

    <Button
      v-if="currentStep === 1"
      variant="secondary"
      size="lg"
      data-attr="upload-model-step1-continue-button"
      :disabled="!canFetchMetadata || isFetchingMetadata"
      @click="emit('fetchMetadata')"
    >
      <i
        v-if="isFetchingMetadata"
        class="icon-[lucide--loader-circle] animate-spin"
      />
      <span>{{ $t('g.continue') }}</span>
    </Button>
    <Button
      v-else-if="currentStep === 2"
      variant="secondary"
      size="lg"
      data-attr="upload-model-step2-confirm-button"
      :disabled="!canUploadModel || isUploading"
      @click="emit('upload')"
    >
      <i v-if="isUploading" class="icon-[lucide--loader-circle] animate-spin" />
      <span>{{ $t('assetBrowser.upload') }}</span>
    </Button>
    <template
      v-else-if="
        currentStep === 3 &&
        (uploadStatus === 'success' || uploadStatus === 'processing')
      "
    >
      <Button
        variant="muted-textonly"
        size="lg"
        data-attr="upload-model-step3-import-another-button"
        @click="emit('importAnother')"
      >
        {{ $t('assetBrowser.importAnother') }}
      </Button>
      <Button
        variant="secondary"
        size="lg"
        data-attr="upload-model-step3-finish-button"
        @click="emit('close')"
      >
        {{
          uploadStatus === 'processing'
            ? $t('g.close')
            : $t('assetBrowser.finish')
        }}
      </Button>
    </template>
    <VideoHelpDialog
      v-model="showCivitaiHelp"
      video-url="https://media.comfy.org/compressed_768/civitai_howto.webm"
      :aria-label="$t('assetBrowser.uploadModelHelpVideo')"
    />
    <VideoHelpDialog
      v-model="showHuggingFaceHelp"
      video-url="https://media.comfy.org/byom/huggingfacehowto.mp4"
      :aria-label="$t('assetBrowser.uploadModelHelpVideo')"
    />
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

import Button from '@/components/ui/button/Button.vue'
import VideoHelpDialog from '@/platform/assets/components/VideoHelpDialog.vue'

const showCivitaiHelp = ref(false)
const showHuggingFaceHelp = ref(false)

defineProps<{
  currentStep: number
  isFetchingMetadata: boolean
  isUploading: boolean
  canFetchMetadata: boolean
  canUploadModel: boolean
  uploadStatus?: 'processing' | 'success' | 'error'
}>()

const emit = defineEmits<{
  (e: 'back'): void
  (e: 'fetchMetadata'): void
  (e: 'upload'): void
  (e: 'close'): void
  (e: 'importAnother'): void
}>()
</script>
