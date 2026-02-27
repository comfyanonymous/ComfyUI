<template>
  <div class="flex">
    <BaseViewTemplate dark class="flex-1">
      <template #header>
        <CloudLogo />
      </template>
      <slot />
      <template #footer>
        <CloudTemplateFooter />
      </template>
    </BaseViewTemplate>
    <div class="relative hidden flex-1 overflow-hidden bg-black lg:block">
      <!-- Video Background -->
      <video
        class="absolute inset-0 h-full w-full object-cover"
        autoplay
        muted
        loop
        playsinline
        :poster="videoPoster"
      >
        <source :src="videoSrc" type="video/mp4" />
      </video>

      <div class="absolute inset-0 h-full w-full bg-black/30"></div>

      <!-- Optional Overlay for better visual -->
      <div
        class="absolute inset-0 flex items-center justify-center text-center text-white"
      >
        <div>
          <h1 class="font-abcrom hero-title font-black uppercase italic">
            {{ t('cloudStart_title') }}
          </h1>
          <p class="m-2 text-center text-xl text-white">
            {{ t('cloudStart_desc') }}
          </p>
          <p class="m-0 text-center text-xl text-white">
            {{ t('cloudStart_explain') }}
          </p>
        </div>
      </div>
      <div class="absolute inset-0 flex flex-col justify-end px-14 pb-[64px]">
        <div class="flex items-center justify-end">
          <div class="flex items-center gap-3">
            <p class="text-md text-white">
              {{ t('cloudStart_wantToRun') }}
            </p>
            <Button
              type="button"
              class="h-10 bg-black font-bold text-white"
              variant="secondary"
              @click="handleDownloadClick"
            >
              {{ t('cloudStart_download') }}
            </Button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useI18n } from 'vue-i18n'

import Button from '@/components/ui/button/Button.vue'
import videoPoster from '@/platform/cloud/onboarding/assets/videos/thumbnail.png'
import videoSrc from '@/platform/cloud/onboarding/assets/videos/video.mp4'
import CloudLogo from '@/platform/cloud/onboarding/components/CloudLogo.vue'
import CloudTemplateFooter from '@/platform/cloud/onboarding/components/CloudTemplateFooter.vue'
import BaseViewTemplate from '@/views/templates/BaseViewTemplate.vue'

const { t } = useI18n()

const handleDownloadClick = () => {
  window.open('https://www.comfy.org/download', '_blank')
}
</script>
<style>
@import '../assets/css/fonts.css';
</style>
