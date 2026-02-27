<template>
  <CloudTemplate>
    <!-- This will render the nested route components -->
    <RouterView />
  </CloudTemplate>
  <!-- Global Toast for displaying notifications -->
  <GlobalToast />
</template>

<script setup lang="ts">
import { RouterView } from 'vue-router'

import GlobalToast from '@/components/toast/GlobalToast.vue'

import CloudTemplate from './CloudTemplate.vue'
</script>
