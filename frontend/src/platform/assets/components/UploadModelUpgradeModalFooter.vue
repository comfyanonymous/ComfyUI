<template>
  <div class="flex flex-wrap justify-end gap-2 w-full">
    <a
      href="https://blog.comfy.org/p/comfy-cloud-new-features-and-pricing"
      target="_blank"
      rel="noopener noreferrer"
      class="text-muted-foreground mr-auto underline flex items-center gap-2"
    >
      <i class="icon-[lucide--external-link]" />
      <span>{{ $t('g.learnMore') }}</span>
    </a>
    <Button variant="textonly" @click="emit('close')">{{
      $t('g.close')
    }}</Button>
    <Button variant="secondary" @click="emit('subscribe')">
      {{ $t('subscription.required.subscribe') }}
    </Button>
  </div>
</template>

<script setup lang="ts">
import Button from '@/components/ui/button/Button.vue'

const emit = defineEmits<{
  close: []
  subscribe: []
}>()
</script>
