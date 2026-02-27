<template>
  <div>
    <Stepper
      value="1"
      class="flex h-[638px] max-h-[80vh] w-[320px] max-w-[90vw] flex-col"
    >
      <ProgressBar
        :value="progressPercent"
        :show-value="false"
        class="mb-8 h-2"
      />

      <StepPanels class="flex flex-1 flex-col p-0">
        <StepPanel
          v-slot="{ activateCallback }"
          value="1"
          class="flex min-h-full flex-1 flex-col justify-between bg-transparent"
        >
          <div>
            <label class="mb-8 block text-lg font-medium">{{
              t('cloudSurvey_steps_familiarity')
            }}</label>
            <div class="flex flex-col gap-6">
              <div
                v-for="opt in familiarityOptions"
                :key="opt.value"
                class="flex items-center gap-3"
              >
                <RadioButton
                  v-model="surveyData.familiarity"
                  :input-id="`fam-${opt.value}`"
                  name="familiarity"
                  :value="opt.value"
                />
                <label
                  :for="`fam-${opt.value}`"
                  class="cursor-pointer text-sm"
                  >{{ opt.label }}</label
                >
              </div>
            </div>
          </div>

          <div class="flex justify-between pt-4">
            <span />
            <Button
              :disabled="!validStep1"
              class="h-10 w-full border-none text-white"
              @click="goTo(2, activateCallback)"
            >
              {{ $t('g.next') }}
            </Button>
          </div>
        </StepPanel>

        <StepPanel
          v-slot="{ activateCallback }"
          value="2"
          class="flex min-h-full flex-1 flex-col justify-between bg-transparent"
        >
          <div>
            <label class="mb-8 block text-lg font-medium">{{
              t('cloudSurvey_steps_purpose')
            }}</label>
            <div class="flex flex-col gap-6">
              <div
                v-for="opt in purposeOptions"
                :key="opt.value"
                class="flex items-center gap-3"
              >
                <RadioButton
                  v-model="surveyData.useCase"
                  :input-id="`purpose-${opt.value}`"
                  name="purpose"
                  :value="opt.value"
                />
                <label
                  :for="`purpose-${opt.value}`"
                  class="cursor-pointer text-sm"
                  >{{ opt.label }}</label
                >
              </div>
            </div>
            <div v-if="surveyData.useCase === 'other'" class="mt-4 ml-8">
              <InputText
                v-model="surveyData.useCaseOther"
                class="w-full"
                :placeholder="
                  $t('cloudOnboarding.survey.options.industry.otherPlaceholder')
                "
              />
            </div>
          </div>

          <div class="flex gap-6 pt-4">
            <Button
              variant="secondary"
              class="flex-1 text-white"
              @click="goTo(1, activateCallback)"
            >
              {{ $t('g.back') }}
            </Button>
            <Button
              :disabled="!validStep2"
              class="h-10 flex-1 text-white"
              @click="goTo(3, activateCallback)"
            >
              {{ $t('g.next') }}
            </Button>
          </div>
        </StepPanel>

        <StepPanel
          v-slot="{ activateCallback }"
          value="3"
          class="flex min-h-full flex-1 flex-col justify-between bg-transparent"
        >
          <div>
            <label class="mb-8 block text-lg font-medium">{{
              t('cloudSurvey_steps_industry')
            }}</label>
            <div class="flex flex-col gap-6">
              <div
                v-for="opt in industryOptions"
                :key="opt.value"
                class="flex items-center gap-3"
              >
                <RadioButton
                  v-model="surveyData.industry"
                  :input-id="`industry-${opt.value}`"
                  name="industry"
                  :value="opt.value"
                />
                <label
                  :for="`industry-${opt.value}`"
                  class="cursor-pointer text-sm"
                  >{{ opt.label }}</label
                >
              </div>
            </div>
            <div v-if="surveyData.industry === 'other'" class="mt-4 ml-8">
              <InputText
                v-model="surveyData.industryOther"
                class="w-full"
                :placeholder="
                  $t('cloudOnboarding.survey.options.industry.otherPlaceholder')
                "
              />
            </div>
          </div>

          <div class="flex gap-6 pt-4">
            <Button
              variant="secondary"
              class="flex-1 text-white"
              @click="goTo(2, activateCallback)"
            >
              {{ $t('g.back') }}
            </Button>
            <Button
              :disabled="!validStep3"
              class="h-10 flex-1 border-none text-white"
              @click="goTo(4, activateCallback)"
            >
              {{ $t('g.next') }}
            </Button>
          </div>
        </StepPanel>

        <StepPanel
          v-slot="{ activateCallback }"
          value="4"
          class="flex min-h-full flex-1 flex-col justify-between bg-transparent"
        >
          <div>
            <label class="mb-8 block text-lg font-medium">{{
              t('cloudSurvey_steps_making')
            }}</label>
            <div class="flex flex-col gap-6">
              <div
                v-for="opt in makingOptions"
                :key="opt.value"
                class="flex items-center gap-3"
              >
                <Checkbox
                  v-model="surveyData.making"
                  :input-id="`making-${opt.value}`"
                  :value="opt.value"
                />
                <label
                  :for="`making-${opt.value}`"
                  class="cursor-pointer text-sm"
                  >{{ opt.label }}</label
                >
              </div>
            </div>
          </div>

          <div class="flex gap-6 pt-4">
            <Button
              variant="secondary"
              class="flex-1 text-white"
              @click="goTo(3, activateCallback)"
            >
              {{ $t('g.back') }}
            </Button>
            <Button
              :disabled="!validStep4 || isSubmitting"
              :loading="isSubmitting"
              class="h-10 flex-1 border-none text-white"
              @click="onSubmitSurvey"
            >
              {{ $t('g.submit') }}
            </Button>
          </div>
        </StepPanel>
      </StepPanels>
    </Stepper>
  </div>
</template>

<script setup lang="ts">
import Checkbox from 'primevue/checkbox'
import InputText from 'primevue/inputtext'
import ProgressBar from 'primevue/progressbar'
import RadioButton from 'primevue/radiobutton'
import StepPanel from 'primevue/steppanel'
import StepPanels from 'primevue/steppanels'
import Stepper from 'primevue/stepper'
import { computed, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRouter } from 'vue-router'

import Button from '@/components/ui/button/Button.vue'
import { useFeatureFlags } from '@/composables/useFeatureFlags'
import {
  getSurveyCompletedStatus,
  submitSurvey
} from '@/platform/cloud/onboarding/auth'
import { isCloud } from '@/platform/distribution/types'
import { useTelemetry } from '@/platform/telemetry'

const { t } = useI18n()
const router = useRouter()
const { flags } = useFeatureFlags()
const onboardingSurveyEnabled = computed(() => flags.onboardingSurveyEnabled)

// Check if survey is already completed on mount
onMounted(async () => {
  if (!onboardingSurveyEnabled.value) {
    await router.replace({ name: 'cloud-user-check' })
    return
  }
  try {
    const surveyCompleted = await getSurveyCompletedStatus()
    if (surveyCompleted) {
      // User already completed survey, return to onboarding flow
      await router.replace({ name: 'cloud-user-check' })
    } else {
      // Track survey opened event
      if (isCloud) {
        useTelemetry()?.trackSurvey('opened')
      }
    }
  } catch (error) {
    console.error('Failed to check survey status:', error)
  }
})

const activeStep = ref(1)
const totalSteps = 4
const progressPercent = computed(() =>
  Math.max(20, Math.min(100, ((activeStep.value - 1) / (totalSteps - 1)) * 100))
)

const isSubmitting = ref(false)

const surveyData = ref({
  familiarity: '',
  useCase: '',
  useCaseOther: '',
  industry: '',
  industryOther: '',
  making: [] as string[]
})

// Options
const familiarityOptions = [
  { label: 'New to ComfyUI (never used it before)', value: 'new' },
  { label: 'Just getting started (following tutorials)', value: 'starting' },
  { label: 'Comfortable with basics', value: 'basics' },
  { label: 'Advanced user (custom workflows)', value: 'advanced' },
  { label: 'Expert (help others)', value: 'expert' }
]

const purposeOptions = [
  { label: 'Personal projects/hobby', value: 'personal' },
  {
    label: 'Community contributions (nodes, workflows, etc.)',
    value: 'community'
  },
  { label: 'Client work (freelance)', value: 'client' },
  { label: 'My own workplace (in-house)', value: 'inhouse' },
  { label: 'Academic research', value: 'research' },
  { label: 'Other', value: 'other' }
]

const industryOptions = [
  { label: 'Film, TV, & animation', value: 'film_tv_animation' },
  { label: 'Gaming', value: 'gaming' },
  { label: 'Marketing & advertising', value: 'marketing' },
  { label: 'Architecture', value: 'architecture' },
  { label: 'Product & graphic design', value: 'product_design' },
  { label: 'Fine art & illustration', value: 'fine_art' },
  { label: 'Software & technology', value: 'software' },
  { label: 'Education', value: 'education' },
  { label: 'Other', value: 'other' }
]

const makingOptions = [
  { label: 'Images', value: 'images' },
  { label: 'Video & animation', value: 'video' },
  { label: '3D assets', value: '3d' },
  { label: 'Audio/music', value: 'audio' },
  { label: 'Custom nodes & workflows', value: 'custom_nodes' }
]

// Validation per step
const validStep1 = computed(() => !!surveyData.value.familiarity)
const validStep2 = computed(() => {
  if (!surveyData.value.useCase) return false
  if (surveyData.value.useCase === 'other') {
    return !!surveyData.value.useCaseOther?.trim()
  }
  return true
})
const validStep3 = computed(() => {
  if (!surveyData.value.industry) return false
  if (surveyData.value.industry === 'other') {
    return !!surveyData.value.industryOther?.trim()
  }
  return true
})
const validStep4 = computed(() => surveyData.value.making.length > 0)

const changeActiveStep = (step: number) => {
  activeStep.value = step
}

const goTo = (step: number, activate: (val: string | number) => void) => {
  // keep Stepper panel and progress bar in sync; Stepper values are strings
  changeActiveStep(step)
  activate(String(step))
}

// Submit
const onSubmitSurvey = async () => {
  try {
    if (!onboardingSurveyEnabled.value) {
      await router.replace({ name: 'cloud-user-check' })
      return
    }
    isSubmitting.value = true
    // prepare payload with consistent structure
    const payload = {
      familiarity: surveyData.value.familiarity,
      useCase:
        surveyData.value.useCase === 'other'
          ? surveyData.value.useCaseOther?.trim() || 'other'
          : surveyData.value.useCase,
      industry:
        surveyData.value.industry === 'other'
          ? surveyData.value.industryOther?.trim() || 'other'
          : surveyData.value.industry,
      making: surveyData.value.making
    }

    await submitSurvey(payload)

    // Track survey submitted event with responses
    if (isCloud) {
      useTelemetry()?.trackSurvey('submitted', {
        industry: payload.industry,
        useCase: payload.useCase,
        familiarity: payload.familiarity,
        making: payload.making
      })
    }

    await router.push({ name: 'cloud-user-check' })
  } finally {
    isSubmitting.value = false
  }
}
</script>

<style scoped>
:deep(.p-progressbar .p-progressbar-value) {
  background-color: #f0ff41 !important;
}
:deep(.p-radiobutton-checked .p-radiobutton-box) {
  background-color: #f0ff41 !important;
  border-color: #f0ff41 !important;
}
:deep(.p-checkbox-checked .p-checkbox-box) {
  background-color: #f0ff41 !important;
  border-color: #f0ff41 !important;
}
</style>
