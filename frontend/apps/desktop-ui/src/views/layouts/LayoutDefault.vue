<template>
  <main class="w-full h-full overflow-hidden relative">
    <router-view />
  </main>
</template>

<script setup lang="ts">
import { useFavicon } from '@vueuse/core'

useFavicon('/assets/favicon.ico')
</script>
