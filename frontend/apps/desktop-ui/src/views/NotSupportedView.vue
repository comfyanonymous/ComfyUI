<template>
  <BaseViewTemplate>
    <div class="sad-container grid items-center justify-evenly">
      <!-- Right side image -->
      <img
        class="sad-girl"
        src="/assets/images/sad_girl.png"
        :alt="$t('notSupported.illustrationAlt')"
      />

      <div class="no-drag sad-text flex items-center">
        <div class="flex flex-col gap-8 p-8 min-w-110">
          <!-- Header -->
          <h1 class="text-4xl font-bold text-red-500">
            {{ $t('notSupported.title') }}
          </h1>

          <!-- Message -->
          <div class="space-y-4">
            <p class="text-xl">
              {{ $t('notSupported.message') }}
            </p>
            <ul class="list-disc list-inside space-y-1 text-neutral-800">
              <li>{{ $t('notSupported.supportedDevices.macos') }}</li>
              <li>{{ $t('notSupported.supportedDevices.windows') }}</li>
            </ul>
          </div>

          <!-- Actions -->
          <div class="flex gap-4">
            <Button
              :label="$t('notSupported.learnMore')"
              icon="pi pi-github"
              severity="secondary"
              @click="openDocs"
            />
            <Button
              :label="$t('notSupported.reportIssue')"
              icon="pi pi-flag"
              severity="secondary"
              @click="reportIssue"
            />
            <Button
              v-tooltip="$t('notSupported.continueTooltip')"
              :label="$t('notSupported.continue')"
              icon="pi pi-arrow-right"
              icon-pos="right"
              severity="danger"
              @click="continueToInstall"
            />
          </div>
        </div>
      </div>
    </div>
  </BaseViewTemplate>
</template>

<script setup lang="ts">
import Button from 'primevue/button'
import { useRouter } from 'vue-router'

import BaseViewTemplate from '@/views/templates/BaseViewTemplate.vue'

const openDocs = () => {
  window.open(
    'https://github.com/Comfy-Org/desktop#currently-supported-platforms',
    '_blank'
  )
}

const reportIssue = () => {
  window.open('https://forum.comfy.org/c/v1-feedback/', '_blank')
}

const router = useRouter()
const continueToInstall = async () => {
  await router.push('/install')
}
</script>

<style scoped>
.sad-container {
  grid-template-columns: 25rem 1fr;

  & > * {
    grid-row: 1;
  }
}

.sad-text {
  grid-column: 1/3;
}

.sad-girl {
  grid-column: 2/3;
  width: min(75vw, 100vh);
}
</style>
