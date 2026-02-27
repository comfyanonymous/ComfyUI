<template>
  <BaseViewTemplate dark>
    <div
      class="h-screen w-screen grid items-center justify-around overflow-y-auto"
    >
      <div class="relative m-8 text-center">
        <!-- Header -->
        <h1 class="download-bg pi-download text-4xl font-bold">
          {{ $t('desktopUpdate.title') }}
        </h1>

        <div class="m-8">
          <span>{{ $t('desktopUpdate.description') }}</span>
        </div>

        <ProgressSpinner class="m-8 w-48 h-48" />

        <!-- Console button -->
        <Button
          style="transform: translateX(-50%)"
          class="fixed bottom-0 left-1/2 my-8"
          :label="$t('maintenance.consoleLogs')"
          icon="pi pi-desktop"
          icon-pos="left"
          severity="secondary"
          @click="toggleConsoleDrawer"
        />

        <TerminalOutputDrawer
          v-model="terminalVisible"
          :header="$t('g.terminal')"
          :default-message="$t('desktopUpdate.terminalDefaultMessage')"
        />
      </div>
    </div>
    <Toast />
  </BaseViewTemplate>
</template>

<script setup lang="ts">
import Button from 'primevue/button'
import ProgressSpinner from 'primevue/progressspinner'
import Toast from 'primevue/toast'
import { onUnmounted, ref } from 'vue'

import TerminalOutputDrawer from '@/components/maintenance/TerminalOutputDrawer.vue'
import { electronAPI } from '@/utils/envUtil'

import BaseViewTemplate from './templates/BaseViewTemplate.vue'

const electron = electronAPI()

const terminalVisible = ref(false)

const toggleConsoleDrawer = () => {
  terminalVisible.value = !terminalVisible.value
}

onUnmounted(() => electron.Validation.dispose())
</script>

<style scoped>
.download-bg::before {
  position: absolute;
  margin: 0;
  color: var(--muted-foreground);
  font-family: 'primeicons', sans-serif;
  top: -2rem;
  right: 2rem;
  speak: none;
  font-style: normal;
  font-weight: 400;
  font-variant: normal;
  text-transform: none;
  line-height: 1;
  display: inline-block;
  -webkit-font-smoothing: antialiased;
  opacity: 0.02;
  font-size: min(14rem, 90vw);
  z-index: 0;
}
</style>
