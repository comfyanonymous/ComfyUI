<template>
  <div class="flex flex-col gap-6 w-[600px]">
    <!-- Source Location Section -->
    <div class="flex flex-col gap-4">
      <p class="text-neutral-400 my-0">
        {{ $t('install.migrationSourcePathDescription') }}
      </p>

      <div class="flex gap-2">
        <InputText
          v-model="sourcePath"
          :placeholder="$t('install.locationPicker.migrationPathPlaceholder')"
          class="flex-1"
          :class="{ 'p-invalid': pathError }"
          @update:model-value="validateSource"
        />
        <Button icon="pi pi-folder" class="w-12" @click="browsePath" />
      </div>

      <Message v-if="pathError" severity="error">
        {{ pathError }}
      </Message>
    </div>

    <!-- Migration Options -->
    <div v-if="isValidSource" class="flex flex-col gap-4 p-4 rounded-lg">
      <h3 class="text-lg mt-0 font-medium text-neutral-100">
        {{ $t('install.selectItemsToMigrate') }}
      </h3>

      <div class="flex flex-col gap-3">
        <div
          v-for="item in migrationItems"
          :key="item.id"
          class="flex items-center gap-3 p-2 hover:bg-neutral-700 rounded"
          @click="item.selected = !item.selected"
        >
          <Checkbox
            v-model="item.selected"
            :input-id="item.id"
            :binary="true"
            @click.stop
          />
          <div>
            <label :for="item.id" class="text-neutral-200 font-medium">
              {{ item.label }}
            </label>
            <p class="text-sm text-neutral-400 my-1">
              {{ item.description }}
            </p>
          </div>
        </div>
      </div>
    </div>

    <!-- Skip Migration -->
    <div v-else class="text-neutral-400 italic">
      {{ $t('install.migrationOptional') }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { MigrationItems } from '@comfyorg/comfyui-electron-types'
import Button from 'primevue/button'
import Checkbox from 'primevue/checkbox'
import InputText from 'primevue/inputtext'
import Message from 'primevue/message'
import { computed, ref, watchEffect } from 'vue'
import { useI18n } from 'vue-i18n'

import { electronAPI } from '@/utils/envUtil'

const { t } = useI18n()

const electron = electronAPI()

const sourcePath = defineModel<string>('sourcePath', { required: false })
const migrationItemIds = defineModel<string[]>('migrationItemIds', {
  required: false
})

const migrationItems = ref(
  MigrationItems.map((item) => ({
    ...item,
    selected: true
  }))
)

const pathError = ref('')
const isValidSource = computed(
  () => sourcePath.value !== '' && pathError.value === ''
)

const validateSource = async (sourcePath: string | undefined) => {
  if (!sourcePath) {
    pathError.value = ''
    return
  }

  try {
    pathError.value = ''
    const validation = await electron.validateComfyUISource(sourcePath)

    if (!validation.isValid) pathError.value = validation.error ?? 'ERROR'
  } catch (error) {
    console.error(error)
    pathError.value = t('install.pathValidationFailed')
  }
}

const browsePath = async () => {
  try {
    const result = await electron.showDirectoryPicker()
    if (result) {
      sourcePath.value = result
      await validateSource(result)
    }
  } catch (error) {
    console.error(error)
    pathError.value = t('install.failedToSelectDirectory')
  }
}

watchEffect(() => {
  migrationItemIds.value = migrationItems.value
    .filter((item) => item.selected)
    .map((item) => item.id)
})
</script>
