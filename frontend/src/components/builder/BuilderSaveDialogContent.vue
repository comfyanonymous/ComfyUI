<template>
  <BuilderDialog @close="onClose">
    <template #title>
      {{ $t('builderToolbar.saveAs') }}
    </template>

    <!-- Filename -->
    <div class="flex flex-col gap-2">
      <label :for="inputId" class="text-sm text-muted-foreground">
        {{ $t('builderToolbar.filename') }}
      </label>
      <input
        :id="inputId"
        v-model="filename"
        autofocus
        type="text"
        class="flex h-10 min-h-8 items-center self-stretch rounded-lg border-none bg-secondary-background pl-4 text-sm text-base-foreground focus:outline-none"
        @keydown.enter="filename.trim() && onSave(filename.trim(), openAsApp)"
      />
    </div>

    <!-- Save as type -->
    <div class="flex flex-col gap-2">
      <label class="text-sm text-muted-foreground">
        {{ $t('builderToolbar.saveAsLabel') }}
      </label>
      <div role="radiogroup" class="flex flex-col gap-2">
        <Button
          v-for="option in saveTypeOptions"
          :key="option.value.toString()"
          role="radio"
          :aria-checked="openAsApp === option.value"
          :class="
            cn(
              itemClasses,
              openAsApp === option.value && 'bg-secondary-background'
            )
          "
          variant="textonly"
          @click="openAsApp = option.value"
        >
          <div
            class="flex size-8 min-h-8 items-center justify-center rounded-lg bg-secondary-background-hover"
          >
            <i :class="cn(option.icon, 'size-4')" />
          </div>
          <div class="mx-2 flex flex-1 flex-col items-start">
            <span class="text-sm font-medium text-base-foreground">
              {{ option.title }}
            </span>
            <span class="text-xs text-muted-foreground">
              {{ option.subtitle }}
            </span>
          </div>
          <i
            v-if="openAsApp === option.value"
            class="icon-[lucide--check] size-4 text-base-foreground"
          />
        </Button>
      </div>
    </div>

    <template #footer>
      <Button variant="muted-textonly" size="lg" @click="onClose">
        {{ $t('g.cancel') }}
      </Button>
      <Button
        variant="secondary"
        size="lg"
        :disabled="!filename.trim()"
        @click="onSave(filename.trim(), openAsApp)"
      >
        {{ $t('g.save') }}
      </Button>
    </template>
  </BuilderDialog>
</template>

<script setup lang="ts">
import { ref, useId } from 'vue'
import { useI18n } from 'vue-i18n'

import Button from '@/components/ui/button/Button.vue'
import { cn } from '@/utils/tailwindUtil'

import BuilderDialog from './BuilderDialog.vue'

const { t } = useI18n()

const { defaultFilename, onSave, onClose } = defineProps<{
  defaultFilename: string
  onSave: (filename: string, openAsApp: boolean) => void
  onClose: () => void
}>()

const inputId = useId()
const filename = ref(defaultFilename)
const openAsApp = ref(true)

const saveTypeOptions = [
  {
    value: true,
    icon: 'icon-[lucide--app-window]',
    title: t('builderToolbar.app'),
    subtitle: t('builderToolbar.appDescription')
  },
  {
    value: false,
    icon: 'icon-[comfy--workflow]',
    title: t('builderToolbar.nodeGraph'),
    subtitle: t('builderToolbar.nodeGraphDescription')
  }
]

const itemClasses =
  'flex h-14 cursor-pointer items-center gap-2 self-stretch rounded-lg border-none bg-transparent py-2 pr-4 pl-2 text-base-foreground transition-colors hover:bg-secondary-background'
</script>
