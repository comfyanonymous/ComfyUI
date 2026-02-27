<template>
  <div
    v-for="logo in validLogos"
    :key="logo.key"
    :class="
      cn('pointer-events-none absolute z-10', logo.position ?? defaultPosition)
    "
  >
    <div
      v-show="!hasAllFailed(logo.providers)"
      data-testid="logo-pill"
      class="flex items-center gap-1.5 rounded-full bg-black/20 py-1 pr-2"
      :style="{ opacity: logo.opacity ?? 0.85 }"
    >
      <div class="ml-0.5 flex items-center">
        <img
          v-for="(provider, providerIndex) in logo.providers"
          :key="provider"
          data-testid="logo-img"
          :src="logo.urls[providerIndex]"
          :alt="provider"
          class="h-6 w-6 rounded-full border-2 border-white object-cover"
          :class="{ relative: providerIndex > 0 }"
          :style="
            providerIndex > 0 ? { marginLeft: `${logo.gap ?? -6}px` } : {}
          "
          draggable="false"
          @error="onImageError(provider)"
        />
      </div>
      <span class="text-sm font-medium text-white">
        {{ logo.label }}
      </span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'

import type { LogoInfo } from '@/platform/workflow/templates/types/template'
import { cn } from '@/utils/tailwindUtil'

const { t, locale } = useI18n()

function formatProviderList(providers: string[]): string {
  const localeValue = String(locale.value)
  try {
    return new Intl.ListFormat(localeValue, {
      style: 'long',
      type: 'conjunction'
    }).format(providers)
  } catch {
    return providers.join(t('templates.logoProviderSeparator'))
  }
}

const {
  logos,
  getLogoUrl,
  defaultPosition = 'top-2 left-2'
} = defineProps<{
  logos: LogoInfo[]
  getLogoUrl: (provider: string) => string
  defaultPosition?: string
}>()

const failedLogos = ref(new Set<string>())

function onImageError(provider: string) {
  failedLogos.value = new Set([...failedLogos.value, provider])
}

function hasAllFailed(providers: string[]): boolean {
  return providers.every((p) => failedLogos.value.has(p))
}

interface ValidatedLogo {
  key: string
  providers: string[]
  urls: string[]
  label: string
  position: string | undefined
  opacity: number | undefined
  gap: number | undefined
}

const validLogos = computed<ValidatedLogo[]>(() => {
  const result: ValidatedLogo[] = []

  logos.forEach((logo, index) => {
    const providers = Array.isArray(logo.provider)
      ? logo.provider
      : [logo.provider]
    const urls = providers.map((p) => getLogoUrl(p))
    const validProviders = providers.filter((_, i) => urls[i])
    const validUrls = urls.filter((url) => url)

    if (validProviders.length === 0) return

    const providerKey = validProviders.join('-')
    const layoutKey = `${logo.position ?? ''}-${logo.opacity ?? ''}-${logo.gap ?? ''}`
    result.push({
      key: providerKey ? `${providerKey}-${layoutKey}` : `logo-${index}`,
      providers: validProviders,
      urls: validUrls,
      label: logo.label ?? formatProviderList(validProviders),
      position: logo.position,
      opacity: logo.opacity,
      gap: logo.gap
    })
  })

  return result
})
</script>
