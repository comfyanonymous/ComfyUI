<template>
  <div class="grid grid-cols-[1fr_auto_1fr] items-center gap-4">
    <!-- Back button -->
    <Button
      v-if="currentStep !== '1'"
      :label="$t('g.back')"
      severity="secondary"
      icon="pi pi-arrow-left"
      class="font-inter rounded-lg border-0 px-6 py-2 justify-self-start"
      @click="$emit('previous')"
    />
    <div v-else></div>

    <!-- Step indicators in center -->
    <StepList class="flex justify-center items-center gap-3 select-none">
      <Step value="1" :pt="stepPassthrough">
        {{ $t('install.gpu') }}
      </Step>
      <Step value="2" :disabled="disableLocationStep" :pt="stepPassthrough">
        {{ $t('install.installLocation') }}
      </Step>
      <Step value="3" :disabled="disableSettingsStep" :pt="stepPassthrough">
        {{ $t('install.desktopSettings') }}
      </Step>
    </StepList>

    <!-- Next/Install button -->
    <Button
      :label="currentStep !== '3' ? $t('g.next') : $t('g.install')"
      class="px-8 py-2 bg-brand-yellow hover:bg-brand-yellow/90 font-inter rounded-lg border-0 transition-colors justify-self-end"
      :pt="{
        label: { class: 'text-neutral-900 font-inter font-black' }
      }"
      :disabled="!canProceed"
      @click="currentStep !== '3' ? $emit('next') : $emit('install')"
    />
  </div>
</template>

<script setup lang="ts">
import type { PassThrough } from '@primevue/core'
import Button from 'primevue/button'
import Step from 'primevue/step'
import type { StepPassThroughOptions } from 'primevue/step'
import StepList from 'primevue/steplist'

defineProps<{
  /** Current step index as string ('1', '2', '3', '4') */
  currentStep: string
  /** Whether the user can proceed to the next step */
  canProceed: boolean
  /** Whether the location step should be disabled */
  disableLocationStep: boolean
  /** Whether the settings step should be disabled */
  disableSettingsStep: boolean
}>()

defineEmits<{
  previous: []
  next: []
  install: []
}>()

const stepPassthrough: PassThrough<StepPassThroughOptions> = {
  root: { class: 'flex-none p-0 m-0' },
  header: ({ context }) => ({
    class: [
      'h-2.5 p-0 m-0 border-0 rounded-full transition-all duration-300',
      context.active
        ? 'bg-brand-yellow w-8 rounded-sm'
        : 'bg-neutral-700 w-2.5',
      context.disabled ? 'opacity-60 cursor-not-allowed' : ''
    ].join(' ')
  }),
  number: { class: 'hidden' },
  title: { class: 'hidden' }
}
</script>
