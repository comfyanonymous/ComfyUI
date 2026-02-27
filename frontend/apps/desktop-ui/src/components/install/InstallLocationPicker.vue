<template>
  <div class="flex flex-col gap-8 w-full max-w-3xl mx-auto select-none">
    <!-- Installation Path Section -->
    <div class="grow flex flex-col gap-6 text-neutral-300">
      <h2 class="font-inter font-bold text-3xl text-neutral-100 text-center">
        {{ $t('install.locationPicker.title') }}
      </h2>

      <p class="text-center text-neutral-400 px-12">
        {{ $t('install.locationPicker.subtitle') }}
      </p>

      <!-- Path Input -->
      <div class="flex gap-2 px-12">
        <InputText
          v-model="installPath"
          :placeholder="$t('install.locationPicker.pathPlaceholder')"
          class="flex-1 bg-neutral-800/50 border-neutral-700 text-neutral-200 placeholder:text-neutral-500"
          :class="{ 'p-invalid': pathError }"
          @update:model-value="validatePath"
          @focus="onFocus"
        />
        <Button
          icon="pi pi-folder-open"
          severity="secondary"
          class="bg-neutral-700 hover:bg-neutral-600 border-0"
          @click="browsePath"
        />
      </div>

      <!-- Error Messages -->
      <div v-if="pathError || pathExists || nonDefaultDrive" class="px-12">
        <Message
          v-if="pathError"
          severity="error"
          class="whitespace-pre-line w-full"
        >
          {{ pathError }}
        </Message>
        <Message v-if="pathExists" severity="warn" class="w-full">
          {{ $t('install.pathExists') }}
        </Message>
        <Message v-if="nonDefaultDrive" severity="warn" class="w-full">
          {{ $t('install.nonDefaultDrive') }}
        </Message>
      </div>

      <!-- Collapsible Sections using PrimeVue Accordion -->
      <Accordion
        v-model:value="activeAccordionIndex"
        :multiple="true"
        class="location-picker-accordion"
        :pt="{
          root: 'bg-transparent border-0',
          panel: {
            root: 'border-0 mb-0'
          },
          header: {
            root: 'border-0',
            content:
              'text-neutral-400 hover:text-neutral-300 px-4 py-2 flex items-center gap-3',
            toggleicon: 'text-xs order-first mr-0'
          },
          content: {
            root: 'bg-transparent border-0',
            content: 'text-neutral-500 text-sm pl-11 pb-3 pt-0'
          }
        }"
      >
        <AccordionPanel value="0">
          <AccordionHeader>
            {{ $t('install.locationPicker.migrateFromExisting') }}
          </AccordionHeader>
          <AccordionContent>
            <MigrationPicker
              v-model:source-path="migrationSourcePath"
              v-model:migration-item-ids="migrationItemIds"
            />
          </AccordionContent>
        </AccordionPanel>

        <AccordionPanel value="1">
          <AccordionHeader>
            {{ $t('install.locationPicker.chooseDownloadServers') }}
          </AccordionHeader>
          <AccordionContent>
            <template
              v-for="([item, modelValue], index) in mirrors"
              :key="item.settingId + item.mirror"
            >
              <Divider v-if="index > 0" class="my-8" />

              <MirrorItem
                v-model="modelValue.value"
                :item="item"
                @state-change="validationStates[index] = $event"
              />
            </template>
          </AccordionContent>
        </AccordionPanel>
      </Accordion>
    </div>
  </div>
</template>

<script setup lang="ts">
import { TorchMirrorUrl } from '@comfyorg/comfyui-electron-types'
import type { TorchDeviceType } from '@comfyorg/comfyui-electron-types'
import { isInChina } from '@comfyorg/shared-frontend-utils/networkUtil'
import Accordion from 'primevue/accordion'
import AccordionContent from 'primevue/accordioncontent'
import AccordionHeader from 'primevue/accordionheader'
import AccordionPanel from 'primevue/accordionpanel'
import Button from 'primevue/button'
import Divider from 'primevue/divider'
import InputText from 'primevue/inputtext'
import Message from 'primevue/message'
import { computed, onMounted, ref } from 'vue'
import type { ModelRef } from 'vue'
import { useI18n } from 'vue-i18n'

import { PYPI_MIRROR, PYTHON_MIRROR } from '@/constants/uvMirrors'
import type { UVMirror } from '@/constants/uvMirrors'
import { electronAPI } from '@/utils/envUtil'
import { ValidationState } from '@/utils/validationUtil'

import MigrationPicker from './MigrationPicker.vue'
import MirrorItem from './mirror/MirrorItem.vue'

const { t } = useI18n()

const installPath = defineModel<string>('installPath', { required: true })
const pathError = defineModel<string>('pathError', { required: true })
const migrationSourcePath = defineModel<string>('migrationSourcePath')
const migrationItemIds = defineModel<string[]>('migrationItemIds')
const pythonMirror = defineModel<string>('pythonMirror', {
  default: ''
})
const pypiMirror = defineModel<string>('pypiMirror', {
  default: ''
})
const torchMirror = defineModel<string>('torchMirror', {
  default: ''
})

const { device } = defineProps<{ device: TorchDeviceType | null }>()

const pathExists = ref(false)
const nonDefaultDrive = ref(false)
const inputTouched = ref(false)

// Accordion state - array of active panel values
const activeAccordionIndex = ref<string[] | undefined>(undefined)

const electron = electronAPI()

// Mirror configuration logic
function getTorchMirrorItem(device: TorchDeviceType): UVMirror {
  const settingId = 'Comfy-Desktop.UV.TorchInstallMirror'
  switch (device) {
    case 'mps':
      return {
        settingId,
        mirror: TorchMirrorUrl.NightlyCpu,
        fallbackMirror: TorchMirrorUrl.NightlyCpu
      }
    case 'nvidia':
      return {
        settingId,
        mirror: TorchMirrorUrl.Cuda,
        fallbackMirror: TorchMirrorUrl.Cuda
      }
    case 'amd':
    case 'cpu':
    default:
      return {
        settingId,
        mirror: PYPI_MIRROR.mirror,
        fallbackMirror: PYPI_MIRROR.fallbackMirror
      }
  }
}

const userIsInChina = ref(false)
const useFallbackMirror = (mirror: UVMirror) => ({
  ...mirror,
  mirror: mirror.fallbackMirror
})

const mirrors = computed<[UVMirror, ModelRef<string>][]>(() =>
  (
    [
      [PYTHON_MIRROR, pythonMirror],
      [PYPI_MIRROR, pypiMirror],
      [getTorchMirrorItem(device ?? 'cpu'), torchMirror]
    ] as [UVMirror, ModelRef<string>][]
  ).map(([item, modelValue]) => [
    userIsInChina.value ? useFallbackMirror(item) : item,
    modelValue
  ])
)

const validationStates = ref<ValidationState[]>(
  mirrors.value.map(() => ValidationState.IDLE)
)

// Get default install path on component mount
onMounted(async () => {
  const paths = await electron.getSystemPaths()
  installPath.value = paths.defaultInstallPath
  await validatePath(paths.defaultInstallPath)
  userIsInChina.value = await isInChina()
})

const validatePath = async (path: string | undefined) => {
  try {
    pathError.value = ''
    pathExists.value = false
    nonDefaultDrive.value = false
    const validation = await electron.validateInstallPath(path ?? '')

    // Create a pre-formatted list of errors
    if (!validation.isValid) {
      const errors: string[] = []
      if (validation.cannotWrite) errors.push(t('install.cannotWrite'))
      if (validation.freeSpace < validation.requiredSpace) {
        const requiredGB = validation.requiredSpace / 1024 / 1024 / 1024
        errors.push(`${t('install.insufficientFreeSpace')}: ${requiredGB} GB`)
      }
      if (validation.parentMissing) errors.push(t('install.parentMissing'))
      if (validation.isOneDrive) errors.push(t('install.isOneDrive'))
      if (validation.isInsideAppInstallDir)
        errors.push(t('install.insideAppInstallDir'))
      if (validation.isInsideUpdaterCache)
        errors.push(t('install.insideUpdaterCache'))

      if (validation.error)
        errors.push(`${t('install.unhandledError')}: ${validation.error}`)
      pathError.value = errors.join('\n')
    }

    if (validation.isNonDefaultDrive) nonDefaultDrive.value = true
    if (validation.exists) pathExists.value = true
  } catch (error) {
    pathError.value = t('install.pathValidationFailed')
  }
}

const browsePath = async () => {
  try {
    const result = await electron.showDirectoryPicker()
    if (result) {
      installPath.value = result
      await validatePath(result)
    }
  } catch (error) {
    pathError.value = t('install.failedToSelectDirectory')
  }
}

const onFocus = async () => {
  if (!inputTouched.value) {
    inputTouched.value = true
    return
  }
  // Refresh validation on re-focus
  await validatePath(installPath.value)
}
</script>

<style scoped>
:deep(.location-picker-accordion) {
  padding-inline: calc(var(--spacing) * 12);

  .p-accordionpanel {
    border: 0;
    background-color: transparent;
  }

  .p-accordionheader {
    margin-top: calc(var(--spacing) * 2);
    border: 0;
    border-radius: var(--radius-xl);
    background-color: color-mix(
      in srgb,
      var(--color-neutral-800) 50%,
      transparent
    );
    transition:
      background-color 0.2s ease,
      border-radius 0.5s ease;

    &:hover {
      background-color: color-mix(
        in srgb,
        var(--color-neutral-700) 50%,
        transparent
      );
    }
  }

  /* When panel is expanded, adjust header border radius */
  .p-accordionpanel-active {
    .p-accordionheader {
      border-top-left-radius: var(--radius-xl);
      border-top-right-radius: var(--radius-xl);
      border-bottom-right-radius: 0;
      border-bottom-left-radius: 0;
    }

    .p-accordionheader-toggle-icon {
      &::before {
        content: '\e902';
      }
    }
  }

  .p-accordioncontent {
    border: 0;
    border-top-left-radius: 0;
    border-top-right-radius: 0;
    border-bottom-right-radius: var(--radius-xl);
    border-bottom-left-radius: var(--radius-xl);
    background-color: color-mix(
      in srgb,
      var(--color-neutral-800) 50%,
      transparent
    );
  }

  .p-accordioncontent-content {
    background-color: transparent;
    padding-top: calc(var(--spacing) * 3);
    padding-right: calc(var(--spacing) * 5);
    padding-bottom: calc(var(--spacing) * 5);
    padding-left: calc(var(--spacing) * 5);
  }

  /* Override default chevron icons to use up/down */
  .p-accordionheader-toggle-icon {
    &::before {
      content: '\e933';
    }
  }
}
</style>
