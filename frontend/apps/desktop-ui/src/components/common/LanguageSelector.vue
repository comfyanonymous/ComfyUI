<template>
  <Select
    :id="dropdownId"
    v-model="selectedLocale"
    :options="localeOptions"
    option-label="label"
    option-value="value"
    :disabled="isSwitching"
    :pt="dropdownPt"
    :size="size"
    class="language-selector"
    @change="onLocaleChange"
  >
    <template #value="{ value }">
      <span :class="valueClass">
        <i class="pi pi-language" :class="iconClass" />
        <span>{{ displayLabel(value as SupportedLocale) }}</span>
      </span>
    </template>
    <template #option="{ option }">
      <span :class="optionClass">
        <i class="pi pi-language" :class="iconClass" />
        <span class="leading-none">{{ option.label }}</span>
      </span>
    </template>
  </Select>
</template>

<script setup lang="ts">
import Select from 'primevue/select'
import type { SelectChangeEvent } from 'primevue/select'
import { computed, ref, watch } from 'vue'

import { i18n, loadLocale, st } from '@/i18n'

type VariantKey = 'dark' | 'light'
type SizeKey = 'small' | 'large'

const { variant = 'dark', size = 'small' } = defineProps<{
  variant?: VariantKey
  size?: SizeKey
}>()

const dropdownId = `language-select-${Math.random().toString(36).slice(2)}`

const LOCALES = [
  ['en', 'English'],
  ['zh', '中文'],
  ['zh-TW', '繁體中文'],
  ['ru', 'Русский'],
  ['ja', '日本語'],
  ['ko', '한국어'],
  ['fr', 'Français'],
  ['es', 'Español'],
  ['ar', 'عربي'],
  ['tr', 'Türkçe'],
  ['pt-BR', 'Português (BR)']
] as const satisfies ReadonlyArray<[string, string]>

type SupportedLocale = (typeof LOCALES)[number][0]

const SIZE_PRESETS = {
  large: {
    wrapper: 'px-3 py-1 min-w-[7rem]',
    gap: 'gap-2',
    valueText: 'text-xs',
    optionText: 'text-sm',
    icon: 'text-sm'
  },
  small: {
    wrapper: 'px-2 py-0.5 min-w-[5rem]',
    gap: 'gap-1',
    valueText: 'text-[0.65rem]',
    optionText: 'text-xs',
    icon: 'text-xs'
  }
} as const satisfies Record<SizeKey, Record<string, string>>

const VARIANT_PRESETS = {
  light: {
    root: 'bg-white/80 border border-neutral-200 text-neutral-700 rounded-full shadow-sm backdrop-blur hover:border-neutral-400 transition-colors focus-visible:ring-offset-2 focus-visible:ring-offset-white',
    trigger: 'text-neutral-500 hover:text-neutral-700',
    item: 'text-neutral-700 bg-transparent hover:bg-neutral-100 focus-visible:outline-none',
    valueText: 'text-neutral-600',
    optionText: 'text-neutral-600',
    icon: 'text-neutral-500'
  },
  dark: {
    root: 'bg-neutral-900/70 border border-neutral-700 text-neutral-200 rounded-full shadow-sm backdrop-blur hover:border-neutral-500 transition-colors focus-visible:ring-offset-2 focus-visible:ring-offset-neutral-900',
    trigger: 'text-neutral-400 hover:text-neutral-200',
    item: 'text-neutral-200 bg-transparent hover:bg-neutral-800/80 focus-visible:outline-none',
    valueText: 'text-neutral-100',
    optionText: 'text-neutral-100',
    icon: 'text-neutral-300'
  }
} as const satisfies Record<VariantKey, Record<string, string>>

const selectedLocale = ref<string>(i18n.global.locale.value)
const isSwitching = ref(false)

const sizePreset = computed(() => SIZE_PRESETS[size])
const variantPreset = computed(() => VARIANT_PRESETS[variant])

const dropdownPt = computed(() => ({
  root: {
    class: `${variantPreset.value.root} ${sizePreset.value.wrapper}`
  },
  trigger: {
    class: variantPreset.value.trigger
  },
  item: {
    class: `${variantPreset.value.item} ${sizePreset.value.optionText}`
  }
}))

const valueClass = computed(() =>
  [
    'flex items-center font-medium uppercase tracking-wide leading-tight',
    sizePreset.value.gap,
    sizePreset.value.valueText,
    variantPreset.value.valueText
  ].join(' ')
)

const optionClass = computed(() =>
  [
    'flex items-center leading-tight',
    sizePreset.value.gap,
    variantPreset.value.optionText,
    sizePreset.value.optionText
  ].join(' ')
)

const iconClass = computed(() =>
  [sizePreset.value.icon, variantPreset.value.icon].join(' ')
)

const localeOptions = computed(() =>
  LOCALES.map(([value, fallback]) => ({
    value,
    label: st(`settings.Comfy_Locale.options.${value}`, fallback)
  }))
)

const labelLookup = computed(() =>
  localeOptions.value.reduce<Record<string, string>>((acc, option) => {
    acc[option.value] = option.label
    return acc
  }, {})
)

function displayLabel(locale?: SupportedLocale) {
  if (!locale) {
    return st('settings.Comfy_Locale.name', 'Language')
  }

  return labelLookup.value[locale] ?? locale
}

watch(
  () => i18n.global.locale.value,
  (newLocale) => {
    if (newLocale !== selectedLocale.value) {
      selectedLocale.value = newLocale
    }
  }
)

async function onLocaleChange(event: SelectChangeEvent) {
  const nextLocale = event.value as SupportedLocale | undefined

  if (!nextLocale || nextLocale === i18n.global.locale.value) {
    return
  }

  isSwitching.value = true
  try {
    await loadLocale(nextLocale)
    i18n.global.locale.value = nextLocale
  } catch (error) {
    console.error(`Failed to change locale to "${nextLocale}"`, error)
    selectedLocale.value = i18n.global.locale.value
  } finally {
    isSwitching.value = false
  }
}
</script>

<style scoped>
:deep(.p-dropdown-panel .p-dropdown-item) {
  transition-property: color, background-color, border-color;
  transition-duration: var(--default-transition-duration);
}

:deep(.p-dropdown) {
  &:focus-visible {
    outline: none;
    box-shadow:
      0 0 0 2px var(--color-neutral-900),
      0 0 0 4px color-mix(in srgb, var(--color-brand-yellow) 60%, transparent);
  }
}
</style>
