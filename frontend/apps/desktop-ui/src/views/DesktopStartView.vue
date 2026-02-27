<template>
  <BaseViewTemplate dark>
    <StartupDisplay :title="$t('desktopStart.initialising')" />
  </BaseViewTemplate>
</template>

<script setup lang="ts">
import StartupDisplay from '@/components/common/StartupDisplay.vue'

import BaseViewTemplate from './templates/BaseViewTemplate.vue'
</script>
