<template>
  <p
    class="m-0 line-clamp-2 text-sm text-base-foreground leading-tight break-all"
    :title="fileName"
  >
    {{ fileName }}
  </p>
</template>

<script setup lang="ts">
defineProps<{
  fileName: string
}>()
</script>
