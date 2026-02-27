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
    role="article"
    data-testid="linear-welcome"
    class="flex flex-col items-center justify-center h-full gap-6 p-8 max-w-lg mx-auto text-center"
  >
    <div class="flex flex-col gap-2">
      <h2 class="text-3xl font-semibold text-muted-foreground">
        {{ t('linearMode.welcome.title') }}
      </h2>
    </div>

    <div class="flex flex-col gap-3 text-muted-foreground max-w-md text-[14px]">
      <p class="mt-0">{{ t('linearMode.welcome.message') }}</p>
      <p class="mt-0">{{ t('linearMode.welcome.controls') }}</p>
      <p class="mt-0">{{ t('linearMode.welcome.sharing') }}</p>
    </div>
    <div v-if="hasOutputs" class="flex flex-row gap-2 text-[14px]">
      <p class="mt-0 text-base-foreground">
        <i18n-t keypath="linearMode.welcome.getStarted" tag="span">
          <template #runButton>
            <span
              class="inline-flex items-center px-3.5 py-0.5 mx-0.5 transform -translate-y-0.5 rounded bg-primary-background text-base-foreground text-xxs font-medium cursor-default"
            >
              {{ t('menu.run') }}
            </span>
          </template>
        </i18n-t>
      </p>
    </div>
    <div v-else class="flex flex-row gap-2">
      <Button variant="textonly" size="lg" @click="setMode('graph')">
        {{ t('linearMode.welcome.backToWorkflow') }}
      </Button>
      <Button variant="primary" size="lg" @click="setMode('builder:select')">
        <i class="icon-[lucide--hammer]" />
        {{ t('linearMode.welcome.buildApp') }}
        <div
          class="bg-base-foreground text-base-background text-xxs rounded-full absolute -top-2 -right-2 px-1"
        >
          {{ t('g.experimental') }}
        </div>
      </Button>
    </div>
  </div>
</template>
