<template>
  <div v-if="inProgressDownloads.length > 0" class="mx-6 mb-4">
    <div class="my-4 text-lg">
      {{ $t('electronFileDownload.inProgress') }}
    </div>

    <template v-for="download in inProgressDownloads" :key="download.url">
      <DownloadItem :download="download" />
    </template>
  </div>
</template>

<script setup lang="ts">
import { storeToRefs } from 'pinia'

import { useElectronDownloadStore } from '@/stores/electronDownloadStore'

import DownloadItem from './DownloadItem.vue'

const electronDownloadStore = useElectronDownloadStore()
const { inProgressDownloads } = storeToRefs(electronDownloadStore)
</script>
