<template>
  <div class="image-upload-wrapper">
    <div class="flex items-center gap-2">
      <div
        class="preview-box flex h-16 w-16 items-center justify-center rounded border p-2"
        :class="{ 'bg-base-background': !modelValue }"
      >
        <img
          v-if="modelValue"
          :src="modelValue"
          class="max-h-full max-w-full object-contain"
        />
        <i v-else class="pi pi-image text-xl text-smoke-400" />
      </div>

      <div class="flex flex-col gap-2">
        <Button size="sm" @click="triggerFileInput">
          <i class="pi pi-upload" />
          {{ $t('g.upload') }}
        </Button>
        <Button
          v-if="modelValue"
          class="w-full"
          variant="destructive"
          size="sm"
          :aria-label="$t('g.delete')"
          @click="clearImage"
        >
          <i class="pi pi-trash" />
        </Button>
      </div>
    </div>
    <input
      ref="fileInput"
      type="file"
      class="hidden"
      accept="image/*"
      @change="handleFileUpload"
    />
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

import Button from '@/components/ui/button/Button.vue'

defineProps<{
  modelValue: string
}>()

const emit = defineEmits<{
  (e: 'update:modelValue', value: string): void
}>()

const fileInput = ref<HTMLInputElement | null>(null)

const triggerFileInput = () => {
  fileInput.value?.click()
}

const handleFileUpload = (event: Event) => {
  const target = event.target as HTMLInputElement
  if (target.files && target.files[0]) {
    const file = target.files[0]
    const reader = new FileReader()
    reader.onload = (e) => {
      emit('update:modelValue', e.target?.result as string)
    }
    reader.readAsDataURL(file)
  }
}

const clearImage = () => {
  emit('update:modelValue', '')
  if (fileInput.value) {
    fileInput.value.value = ''
  }
}
</script>
