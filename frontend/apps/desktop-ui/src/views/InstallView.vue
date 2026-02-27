<template>
  <BaseViewTemplate dark>
    <!-- Fixed height container with flexbox layout for proper content management -->
    <div class="w-full h-full flex flex-col">
      <Stepper
        v-model:value="currentStep"
        class="flex flex-col h-full"
        @update:value="handleStepChange"
      >
        <!-- Main content area that grows to fill available space -->
        <StepPanels
          class="flex-1 overflow-auto"
          :style="{ scrollbarGutter: 'stable' }"
        >
          <StepPanel value="1" class="flex">
            <GpuPicker v-model:device="device" />
          </StepPanel>
          <StepPanel value="2">
            <InstallLocationPicker
              v-model:install-path="installPath"
              v-model:path-error="pathError"
              v-model:migration-source-path="migrationSourcePath"
              v-model:migration-item-ids="migrationItemIds"
              v-model:python-mirror="pythonMirror"
              v-model:pypi-mirror="pypiMirror"
              v-model:torch-mirror="torchMirror"
              :device="device"
            />
          </StepPanel>
          <StepPanel value="3">
            <DesktopSettingsConfiguration
              v-model:auto-update="autoUpdate"
              v-model:allow-metrics="allowMetrics"
            />
          </StepPanel>
        </StepPanels>

        <!-- Install footer with navigation -->
        <InstallFooter
          class="w-full max-w-2xl my-6 mx-auto"
          :current-step
          :can-proceed
          :disable-location-step="noGpu"
          :disable-migration-step="noGpu || hasError || highestStep < 2"
          :disable-settings-step="noGpu || hasError || highestStep < 3"
          @previous="goToPreviousStep"
          @next="goToNextStep"
          @install="install"
        />
      </Stepper>
    </div>
  </BaseViewTemplate>
</template>

<script setup lang="ts">
import type {
  InstallOptions,
  TorchDeviceType
} from '@comfyorg/comfyui-electron-types'
import StepPanel from 'primevue/steppanel'
import StepPanels from 'primevue/steppanels'
import Stepper from 'primevue/stepper'
import { computed, onMounted, ref, toRaw } from 'vue'
import { useRouter } from 'vue-router'

import DesktopSettingsConfiguration from '@/components/install/DesktopSettingsConfiguration.vue'
import GpuPicker from '@/components/install/GpuPicker.vue'
import InstallFooter from '@/components/install/InstallFooter.vue'
import InstallLocationPicker from '@/components/install/InstallLocationPicker.vue'
import { electronAPI } from '@/utils/envUtil'
import BaseViewTemplate from '@/views/templates/BaseViewTemplate.vue'

const device = ref<TorchDeviceType | null>(null)

const installPath = ref('')
const pathError = ref('')

const migrationSourcePath = ref('')
const migrationItemIds = ref<string[]>([])

const autoUpdate = ref(true)
const allowMetrics = ref(true)
const pythonMirror = ref('')
const pypiMirror = ref('')
const torchMirror = ref('')

/** Current step in the stepper */
const currentStep = ref('1')

/** Forces each install step to be visited at least once. */
const highestStep = ref(0)

const handleStepChange = (value: string | number) => {
  setHighestStep(value)

  electronAPI().Events.trackEvent('install_stepper_change', {
    step: value
  })
}

const setHighestStep = (value: string | number) => {
  const int = typeof value === 'number' ? value : parseInt(value, 10)
  if (!isNaN(int) && int > highestStep.value) highestStep.value = int
}

const hasError = computed(() => pathError.value !== '')
const noGpu = computed(() => typeof device.value !== 'string')

// Computed property to determine if user can proceed to next step
const regex = /^Insufficient space - minimum free space: \d+ GB$/

const canProceed = computed(() => {
  switch (currentStep.value) {
    case '1':
      return typeof device.value === 'string'
    case '2':
      return pathError.value === '' || regex.test(pathError.value)
    case '3':
      return !hasError.value
    default:
      return false
  }
})

// Navigation methods
const goToNextStep = () => {
  const nextStep = (parseInt(currentStep.value) + 1).toString()
  currentStep.value = nextStep
  setHighestStep(nextStep)
  electronAPI().Events.trackEvent('install_stepper_change', {
    step: nextStep
  })
}

const goToPreviousStep = () => {
  const prevStep = (parseInt(currentStep.value) - 1).toString()
  currentStep.value = prevStep
  electronAPI().Events.trackEvent('install_stepper_change', {
    step: prevStep
  })
}

const electron = electronAPI()
const router = useRouter()
const install = async () => {
  if (!device.value) return

  const options: InstallOptions = {
    installPath: installPath.value,
    autoUpdate: autoUpdate.value,
    allowMetrics: allowMetrics.value,
    migrationSourcePath: migrationSourcePath.value,
    migrationItemIds: toRaw(migrationItemIds.value),
    pythonMirror: pythonMirror.value,
    pypiMirror: pypiMirror.value,
    torchMirror: torchMirror.value,
    device: device.value
  }
  electron.installComfyUI(options)

  const nextPage =
    options.device === 'unsupported' ? '/manual-configuration' : '/server-start'
  await router.push(nextPage)
}

onMounted(async () => {
  if (!electron) return

  const detectedGpu = await electron.Config.getDetectedGpu()
  if (
    detectedGpu === 'mps' ||
    detectedGpu === 'nvidia' ||
    detectedGpu === 'amd'
  ) {
    device.value = detectedGpu
  }

  electronAPI().Events.trackEvent('install_stepper_change', {
    step: currentStep.value,
    gpu: detectedGpu
  })
})
</script>

<style scoped>
:deep(.p-steppanel) {
  margin-top: calc(var(--spacing) * 8);
  display: flex;
  justify-content: center;
  background-color: transparent;
}

/* Remove default padding/margin from StepPanels to make scrollbar flush */
:deep(.p-steppanels) {
  margin: 0;
  padding: 0;
}

/* Ensure StepPanel content container has no top/bottom padding */
:deep(.p-steppanel-content) {
  padding: 0;
}

/* Custom overlay scrollbar for WebKit browsers (Electron, Chrome) */
:deep(.p-steppanels::-webkit-scrollbar) {
  width: calc(var(--spacing) * 4);
}

:deep(.p-steppanels::-webkit-scrollbar-track) {
  background-color: transparent;
}

:deep(.p-steppanels::-webkit-scrollbar-thumb) {
  border: 4px solid transparent;
  border-radius: var(--radius-lg);
  background-color: color-mix(in srgb, var(--color-white) 20%, transparent);
  background-clip: content-box;
}
</style>
