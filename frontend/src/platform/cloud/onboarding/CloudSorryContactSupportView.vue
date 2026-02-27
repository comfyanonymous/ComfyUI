<template>
  <div class="cloud-sorry-contact-support">
    <h1>{{ t('cloudSorryContactSupport_title') }}</h1>
  </div>
</template>

<script setup lang="ts">
import { useI18n } from 'vue-i18n'

const { t } = useI18n()
</script>

<style scoped>
.cloud-sorry-contact-support {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
  font-family: monospace;
  font-size: 1.5rem;
}
</style>
