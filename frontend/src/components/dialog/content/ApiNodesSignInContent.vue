<template>
  <div class="flex h-110 max-w-96 flex-col gap-4 p-2">
    <div class="mb-2 text-2xl font-medium">
      {{ t('apiNodesSignInDialog.title') }}
    </div>

    <div class="mb-4 text-base">
      {{ t('apiNodesSignInDialog.message') }}
    </div>

    <ApiNodesList :node-names="apiNodeNames" />

    <div class="flex items-center justify-between">
      <Button variant="textonly" @click="handleLearnMoreClick">
        {{ t('g.learnMore') }}
      </Button>
      <div class="flex gap-2">
        <Button variant="secondary" @click="onCancel?.()">
          {{ t('g.cancel') }}
        </Button>
        <Button @click="onLogin?.()">
          {{ t('g.login') }}
        </Button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useI18n } from 'vue-i18n'

import Button from '@/components/ui/button/Button.vue'
import { useExternalLink } from '@/composables/useExternalLink'

const { t } = useI18n()
const { buildDocsUrl } = useExternalLink()

const { apiNodeNames, onLogin, onCancel } = defineProps<{
  apiNodeNames: string[]
  onLogin?: () => void
  onCancel?: () => void
}>()

const handleLearnMoreClick = () => {
  window.open(
    buildDocsUrl('/tutorials/api-nodes/faq', { includeLocale: true }),
    '_blank'
  )
}
</script>
