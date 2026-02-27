<template>
  <div class="flex gap-2">
    <InputText
      v-model="modelValue"
      class="flex-1"
      :placeholder="$t('g.imageUrl')"
    />
    <Button
      v-tooltip="$t('g.upload')"
      variant="secondary"
      size="sm"
      :aria-label="$t('g.upload')"
      :disabled="isUploading"
      @click="triggerFileInput"
    >
      <i :class="isUploading ? 'pi pi-spin pi-spinner' : 'pi pi-upload'" />
    </Button>
    <Button
      v-tooltip="$t('g.clear')"
      variant="destructive"
      size="sm"
      :aria-label="$t('g.clear')"
      :disabled="!modelValue"
      @click="clearImage"
    >
      <i class="pi pi-trash" />
    </Button>
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
import InputText from 'primevue/inputtext'
import { ref } from 'vue'

import Button from '@/components/ui/button/Button.vue'
import { useToastStore } from '@/platform/updates/common/toastStore'
import { api } from '@/scripts/api'

const modelValue = defineModel<string>()

const fileInput = ref<HTMLInputElement | null>(null)
const isUploading = ref(false)

const triggerFileInput = () => {
  fileInput.value?.click()
}

const uploadFile = async (file: File): Promise<string | null> => {
  const body = new FormData()
  body.append('image', file)
  body.append('subfolder', 'backgrounds')

  const resp = await api.fetchApi('/upload/image', {
    method: 'POST',
    body
  })

  if (resp.status !== 200) {
    useToastStore().addAlert(
      `Upload failed: ${resp.status} - ${resp.statusText}`
    )
    return null
  }

  const data = await resp.json()
  return data.subfolder ? `${data.subfolder}/${data.name}` : data.name
}

const handleFileUpload = async (event: Event) => {
  const target = event.target as HTMLInputElement
  if (target.files && target.files[0]) {
    const file = target.files[0]

    isUploading.value = true
    try {
      const uploadedPath = await uploadFile(file)
      if (uploadedPath) {
        // Set the value to the API view URL with subfolder parameter
        const params = new URLSearchParams({
          filename: uploadedPath,
          type: 'input',
          subfolder: 'backgrounds'
        })
        modelValue.value = `/api/view?${params.toString()}`
      }
    } catch (error) {
      useToastStore().addAlert(`Upload error: ${String(error)}`)
    } finally {
      isUploading.value = false
    }
  }
}

const clearImage = () => {
  modelValue.value = ''
  if (fileInput.value) {
    fileInput.value.value = ''
  }
}
</script>
