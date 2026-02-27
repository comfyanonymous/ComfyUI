<template>
  <BaseViewTemplate dark hide-language-selector>
    <div class="h-full p-8 2xl:p-16 flex flex-col items-center justify-center">
      <div
        class="bg-neutral-800 rounded-lg shadow-lg p-6 w-full max-w-[600px] flex flex-col gap-6"
      >
        <h2 class="text-3xl font-semibold text-neutral-100">
          {{ $t('install.helpImprove') }}
        </h2>
        <p class="text-neutral-400">
          {{ $t('install.updateConsent') }}
        </p>
        <p class="text-neutral-400">
          {{ $t('install.moreInfo') }}
          <a
            href="https://comfy.org/privacy"
            target="_blank"
            class="text-blue-400 hover:text-blue-300 underline"
          >
            {{ $t('install.privacyPolicy') }} </a
          >.
        </p>
        <div class="flex items-center gap-4">
          <ToggleSwitch
            v-model="allowMetrics"
            aria-describedby="metricsDescription"
          />
          <span id="metricsDescription" class="text-neutral-100">
            {{
              allowMetrics
                ? $t('install.metricsEnabled')
                : $t('install.metricsDisabled')
            }}
          </span>
        </div>
        <div class="flex pt-6 justify-end">
          <Button
            :label="$t('g.ok')"
            icon="pi pi-check"
            :loading="isUpdating"
            icon-pos="right"
            @click="updateConsent"
          />
        </div>
      </div>
    </div>
  </BaseViewTemplate>
</template>

<script setup lang="ts">
import Button from 'primevue/button'
import ToggleSwitch from 'primevue/toggleswitch'
import { useToast } from 'primevue/usetoast'
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRouter } from 'vue-router'

import { electronAPI } from '@/utils/envUtil'

const toast = useToast()
const { t } = useI18n()

const allowMetrics = ref(true)
const router = useRouter()
const isUpdating = ref(false)

const updateConsent = async () => {
  isUpdating.value = true
  try {
    await electronAPI().setMetricsConsent(allowMetrics.value)
  } catch (error) {
    toast.add({
      severity: 'error',
      summary: t('install.settings.errorUpdatingConsent'),
      detail: t('install.settings.errorUpdatingConsentDetail'),
      life: 3000
    })
  } finally {
    isUpdating.value = false
  }
  await router.push('/')
}
</script>
