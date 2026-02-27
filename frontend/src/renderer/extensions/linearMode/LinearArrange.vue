<script setup lang="ts">
import { useI18n } from 'vue-i18n'
import { useAppMode } from '@/composables/useAppMode'
import { useAppModeStore } from '@/stores/appModeStore'
import Button from '@/components/ui/button/Button.vue'
import { storeToRefs } from 'pinia'

const { t } = useI18n()
const { setMode } = useAppMode()
const { hasOutputs } = storeToRefs(useAppModeStore())
</script>

<template>
  <div
    v-if="hasOutputs"
    role="article"
    data-testid="arrange-preview"
    class="flex flex-col items-center justify-center h-full w-3/4 gap-6 p-8 mx-auto"
  >
    <div
      class="border-warning-background border-2 border-dashed rounded-2xl w-full h-4/5 flex items-center justify-center flex-col p-12"
    >
      <p class="text-base-foreground font-bold mb-0">
        {{ t('linearMode.arrange.outputs') }}
      </p>
      <p>{{ t('linearMode.arrange.resultsLabel') }}</p>
    </div>
  </div>
  <div
    v-else
    role="article"
    data-testid="arrange-no-outputs"
    class="flex flex-col items-center justify-center h-full gap-6 p-8 w-lg mx-auto text-center"
  >
    <p class="m-0 text-base-foreground">
      {{ t('linearMode.arrange.noOutputs') }}
    </p>

    <div class="flex flex-col gap-1 text-muted-foreground w-lg text-[14px]">
      <p class="mt-0 p-0">{{ t('linearMode.arrange.switchToSelect') }}</p>

      <i18n-t keypath="linearMode.arrange.connectAtLeastOne" tag="div">
        <template #atLeastOne>
          <span class="italic font-bold">
            {{ t('linearMode.arrange.atLeastOne') }}
          </span>
        </template>
      </i18n-t>

      <p class="mt-0 p-0">{{ t('linearMode.arrange.outputExamples') }}</p>
    </div>
    <div class="flex flex-row gap-2">
      <Button variant="primary" size="lg" @click="setMode('builder:select')">
        {{ t('linearMode.arrange.switchToSelectButton') }}
      </Button>
    </div>
  </div>
</template>
