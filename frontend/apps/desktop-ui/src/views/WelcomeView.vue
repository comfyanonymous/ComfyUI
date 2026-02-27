<template>
  <BaseViewTemplate dark>
    <div class="flex items-center justify-center min-h-screen">
      <div class="grid gap-8">
        <!-- Top container: Logo -->
        <div class="flex items-end justify-center">
          <img
            src="/assets/images/comfy-brand-mark.svg"
            :alt="$t('g.logoAlt')"
            class="w-60"
          />
        </div>
        <!-- Bottom container: Title and button -->
        <div class="flex flex-col items-center justify-center gap-4">
          <Button
            :label="$t('welcome.getStarted')"
            class="px-8 mt-4 bg-brand-yellow hover:bg-brand-yellow/90 border-0 rounded-lg transition-colors"
            :pt="{
              label: { class: 'font-inter text-neutral-900 font-black' }
            }"
            @click="navigateTo('/install')"
          />
        </div>
      </div>
    </div>
  </BaseViewTemplate>
</template>

<script setup lang="ts">
import Button from 'primevue/button'
import { useRouter } from 'vue-router'

import BaseViewTemplate from '@/views/templates/BaseViewTemplate.vue'

const router = useRouter()
const navigateTo = async (path: string) => {
  await router.push(path)
}
</script>
