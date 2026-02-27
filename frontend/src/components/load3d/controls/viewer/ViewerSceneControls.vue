<template>
  <div class="space-y-4">
    <div v-if="!hasBackgroundImage">
      <label>
        {{ $t('load3d.backgroundColor') }}
      </label>
      <input v-model="backgroundColor" type="color" class="w-full" />
    </div>

    <div>
      <Checkbox v-model="showGrid" input-id="showGrid" binary name="showGrid" />
      <label for="showGrid" class="pl-2">
        {{ $t('load3d.showGrid') }}
      </label>
    </div>

    <div v-if="!hasBackgroundImage && !disableBackgroundUpload">
      <Button variant="secondary" class="w-full" @click="openImagePicker">
        <i class="pi pi-image" />
        {{ $t('load3d.uploadBackgroundImage') }}
      </Button>
      <input
        ref="imagePickerRef"
        type="file"
        accept="image/*"
        class="hidden"
        @change="handleImageUpload"
      />
    </div>

    <div v-if="hasBackgroundImage" class="space-y-2">
      <div class="flex gap-2">
        <Button
          :variant="backgroundRenderMode === 'tiled' ? 'primary' : 'secondary'"
          class="flex-1"
          @click="setBackgroundRenderMode('tiled')"
        >
          <i class="pi pi-th-large" />
          {{ $t('load3d.tiledMode') }}
        </Button>
        <Button
          :variant="
            backgroundRenderMode === 'panorama' ? 'primary' : 'secondary'
          "
          class="flex-1"
          @click="setBackgroundRenderMode('panorama')"
        >
          <i class="pi pi-globe" />
          {{ $t('load3d.panoramaMode') }}
        </Button>
      </div>
      <Button variant="secondary" class="w-full" @click="removeBackgroundImage">
        <i class="pi pi-times" />
        {{ $t('load3d.removeBackgroundImage') }}
      </Button>
    </div>
  </div>
</template>

<script setup lang="ts">
import Checkbox from 'primevue/checkbox'
import { ref } from 'vue'

import Button from '@/components/ui/button/Button.vue'

const backgroundColor = defineModel<string>('backgroundColor')
const showGrid = defineModel<boolean>('showGrid')
const backgroundRenderMode = defineModel<'tiled' | 'panorama'>(
  'backgroundRenderMode'
)

defineProps<{
  hasBackgroundImage?: boolean
  disableBackgroundUpload?: boolean
}>()

const emit = defineEmits<{
  (e: 'updateBackgroundImage', file: File | null): void
}>()

const imagePickerRef = ref<HTMLInputElement | null>(null)

const openImagePicker = () => {
  imagePickerRef.value?.click()
}

const handleImageUpload = (event: Event) => {
  const input = event.target as HTMLInputElement
  if (input.files && input.files[0]) {
    emit('updateBackgroundImage', input.files[0])
  }

  input.value = ''
}

const removeBackgroundImage = () => {
  emit('updateBackgroundImage', null)
}

const setBackgroundRenderMode = (mode: 'tiled' | 'panorama') => {
  backgroundRenderMode.value = mode
}
</script>
