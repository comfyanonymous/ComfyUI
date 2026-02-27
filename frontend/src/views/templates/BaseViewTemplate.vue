<template>
  <div
    class="flex h-svh w-screen flex-col font-sans"
    :class="[
      dark
        ? 'dark-theme bg-neutral-900 text-neutral-300'
        : 'bg-neutral-300 text-neutral-900'
    ]"
  >
    <!-- Virtual top menu for native window (drag handle) -->
    <div
      v-show="isNativeWindow()"
      ref="topMenuRef"
      class="app-drag h-(--comfy-topbar-height) w-full"
    />
    <div class="flex w-full grow items-center justify-center overflow-auto">
      <slot />
    </div>
  </div>
</template>

<script setup lang="ts">
import { nextTick, onMounted, ref } from 'vue'

import { electronAPI, isNativeWindow } from '@/utils/envUtil'
import { isDesktop } from '@/platform/distribution/types'

const { dark = false } = defineProps<{
  dark?: boolean
}>()

const darkTheme = {
  color: 'rgba(0, 0, 0, 0)',
  symbolColor: '#d4d4d4'
}

const lightTheme = {
  color: 'rgba(0, 0, 0, 0)',
  symbolColor: '#171717'
}

const topMenuRef = ref<HTMLDivElement | null>(null)
onMounted(async () => {
  if (isDesktop) {
    await nextTick()

    electronAPI().changeTheme({
      ...(dark ? darkTheme : lightTheme),
      height: topMenuRef.value?.getBoundingClientRect().height ?? 0
    })
  }
})
</script>
