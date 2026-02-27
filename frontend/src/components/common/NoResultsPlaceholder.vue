<template>
  <div class="no-results-placeholder h-full p-8" :class="props.class">
    <Card>
      <template #content>
        <div class="flex flex-col items-center">
          <i :class="icon" style="font-size: 3rem; margin-bottom: 1rem" />
          <h3>{{ title }}</h3>
          <p :class="textClass" class="text-center whitespace-pre-line">
            {{ message }}
          </p>
          <Button
            v-if="buttonLabel"
            variant="textonly"
            @click="$emit('action')"
          >
            {{ buttonLabel }}
          </Button>
        </div>
      </template>
    </Card>
  </div>
</template>

<script setup lang="ts">
import Card from 'primevue/card'

import Button from '@/components/ui/button/Button.vue'

const props = defineProps<{
  class?: string
  icon?: string
  title: string
  message: string
  textClass?: string
  buttonLabel?: string
}>()

defineEmits(['action'])
</script>

<style scoped>
.no-results-placeholder :deep(.p-card) {
  background-color: var(--surface-ground);
  text-align: center;
  box-shadow: unset;
}

.no-results-placeholder h3 {
  color: var(--text-color);
  margin-bottom: 0.5rem;
}

.no-results-placeholder p {
  color: var(--text-color-secondary);
  margin-bottom: 1rem;
}
</style>
