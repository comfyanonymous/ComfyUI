<template>
  <nav
    class="fixed top-[calc(var(--workflow-tabs-height)+var(--spacing)*1.5)] left-1/2 z-[1000] -translate-x-1/2"
    :aria-label="t('builderToolbar.label')"
  >
    <div
      class="inline-flex items-center gap-1 rounded-2xl border border-border-default bg-base-background p-2 shadow-interface"
    >
      <template
        v-for="(step, index) in [selectStep, arrangeStep]"
        :key="step.id"
      >
        <button
          :class="
            cn(
              stepClasses,
              activeStep === step.id && 'bg-interface-builder-mode-background',
              activeStep !== step.id &&
                'hover:bg-secondary-background bg-transparent'
            )
          "
          :aria-current="activeStep === step.id ? 'step' : undefined"
          @click="setMode(step.id)"
        >
          <StepBadge :step :index :model-value="activeStep" />
          <StepLabel :step />
        </button>

        <div class="mx-1 h-px w-4 bg-border-default" role="separator" />
      </template>

      <!-- Save -->
      <ConnectOutputPopover
        v-if="!hasOutputs"
        :is-select-active="activeStep === 'builder:select'"
        @switch="setMode('builder:select')"
      >
        <button :class="cn(stepClasses, 'opacity-30 bg-transparent')">
          <StepBadge :step="saveStep" :index="2" :model-value="activeStep" />
          <StepLabel :step="saveStep" />
        </button>
      </ConnectOutputPopover>
      <button
        v-else
        :class="
          cn(
            stepClasses,
            activeStep === 'save'
              ? 'bg-interface-builder-mode-background'
              : 'hover:bg-secondary-background bg-transparent'
          )
        "
        @click="setSaving(true)"
      >
        <StepBadge :step="saveStep" :index="2" :model-value="activeStep" />
        <StepLabel :step="saveStep" />
      </button>
    </div>
  </nav>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

import { useAppMode } from '@/composables/useAppMode'
import type { AppMode } from '@/composables/useAppMode'
import { useAppModeStore } from '@/stores/appModeStore'
import { cn } from '@/utils/tailwindUtil'

import { useBuilderSave } from './useBuilderSave'
import ConnectOutputPopover from './ConnectOutputPopover.vue'
import StepBadge from './StepBadge.vue'
import StepLabel from './StepLabel.vue'
import type { BuilderToolbarStep } from './types'
import { storeToRefs } from 'pinia'

const { t } = useI18n()
const { mode, setMode } = useAppMode()
const { hasOutputs } = storeToRefs(useAppModeStore())
const { saving, setSaving } = useBuilderSave()

const activeStep = computed(() => (saving.value ? 'save' : mode.value))

const stepClasses =
  'inline-flex h-14 min-h-8 cursor-pointer items-center gap-3 rounded-lg py-2 pr-4 pl-2 transition-colors border-none'

const selectStep: BuilderToolbarStep<AppMode> = {
  id: 'builder:select',
  title: t('builderToolbar.select'),
  subtitle: t('builderToolbar.selectDescription'),
  icon: 'icon-[lucide--mouse-pointer-click]'
}

const arrangeStep: BuilderToolbarStep<AppMode> = {
  id: 'builder:arrange',
  title: t('builderToolbar.arrange'),
  subtitle: t('builderToolbar.arrangeDescription'),
  icon: 'icon-[lucide--layout-panel-left]'
}

const saveStep: BuilderToolbarStep<'save'> = {
  id: 'save',
  title: t('builderToolbar.save'),
  subtitle: t('builderToolbar.saveDescription'),
  icon: 'icon-[lucide--cloud-upload]'
}
</script>
