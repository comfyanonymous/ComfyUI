<template>
  <div
    :class="cn('rounded-2xl overflow-hidden relative', sizeClasses)"
    @keydown.esc.capture="handleEscape"
  >
    <div
      class="grid h-full w-full transition-[grid-template-columns] duration-300 ease-out"
      :style="gridStyle"
    >
      <nav
        class="h-full overflow-hidden bg-modal-panel-background flex flex-col"
        :inert="!showLeftPanel"
        :aria-hidden="!showLeftPanel"
      >
        <header
          data-component-id="LeftPanelHeader"
          class="flex w-full h-18 shrink-0 gap-2 pl-6 pr-3 items-center-safe"
        >
          <slot name="leftPanelHeaderTitle" />
          <Button
            v-if="!notMobile && showLeftPanel"
            size="lg"
            class="w-10 p-0 ml-auto"
            :aria-label="t('g.hideLeftPanel')"
            @click="toggleLeftPanel"
          >
            <i class="icon-[lucide--panel-left-close]" />
          </Button>
        </header>
        <slot name="leftPanel" />
      </nav>

      <div class="flex flex-col bg-base-background overflow-hidden">
        <header
          v-if="$slots.header"
          class="w-full h-18 px-6 flex items-center justify-between gap-2"
        >
          <div class="flex flex-1 shrink-0 gap-2">
            <Button
              v-if="!notMobile && !showLeftPanel"
              size="lg"
              class="w-10 p-0"
              :aria-label="t('g.showLeftPanel')"
              @click="toggleLeftPanel"
            >
              <i class="icon-[lucide--panel-left]" />
            </Button>
            <slot name="header" />
          </div>
          <slot name="header-right-area" />
          <template v-if="!isRightPanelOpen">
            <Button
              v-if="hasRightPanel"
              size="lg"
              class="w-10 p-0"
              :aria-label="t('g.showRightPanel')"
              @click="toggleRightPanel"
            >
              <i class="icon-[lucide--panel-right] size-4" />
            </Button>
            <Button
              size="lg"
              class="w-10"
              :aria-label="t('g.closeDialog')"
              @click="closeDialog"
            >
              <i class="pi pi-times" />
            </Button>
          </template>
        </header>

        <main class="flex min-h-0 flex-1 flex-col">
          <slot name="contentFilter" />
          <h2
            v-if="!hasLeftPanel"
            class="text-xxl m-0 select-none px-6 pt-2 pb-6 capitalize"
          >
            {{ contentTitle }}
          </h2>
          <div
            class="min-h-0 flex-1 px-6 pt-0 pb-10 overflow-y-auto scrollbar-custom"
          >
            <slot name="content" />
          </div>
        </main>
      </div>

      <aside
        v-if="hasRightPanel"
        class="overflow-hidden"
        :inert="!isRightPanelOpen"
        :aria-hidden="!isRightPanelOpen"
      >
        <div
          class="min-w-72 w-72 flex flex-col bg-modal-panel-background h-full"
        >
          <header
            data-component-id="RightPanelHeader"
            class="flex h-18 shrink-0 items-center gap-2 px-6"
          >
            <h2
              v-if="rightPanelTitle"
              class="flex-1 select-none text-base font-semibold"
            >
              {{ rightPanelTitle }}
            </h2>
            <div v-else class="flex-1">
              <slot name="rightPanelHeaderTitle" />
            </div>
            <slot name="rightPanelHeaderActions" />
            <Button
              size="lg"
              class="w-10 p-0"
              :aria-label="t('g.hideRightPanel')"
              @click="toggleRightPanel"
            >
              <i class="icon-[lucide--panel-right-close] size-4" />
            </Button>
            <Button
              size="lg"
              class="w-10 p-0"
              :aria-label="t('g.closeDialog')"
              @click="closeDialog"
            >
              <i class="pi pi-times" />
            </Button>
          </header>
          <div class="min-h-0 flex-1 overflow-y-auto">
            <slot name="rightPanel" />
          </div>
        </div>
      </aside>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useBreakpoints } from '@vueuse/core'
import { computed, inject, ref, useSlots, watch } from 'vue'
import { useI18n } from 'vue-i18n'

import Button from '@/components/ui/button/Button.vue'
import { OnCloseKey } from '@/types/widgetTypes'
import { cn } from '@/utils/tailwindUtil'

const { t } = useI18n()

const SIZE_CLASSES = {
  sm: 'h-[80vh] w-[90vw] max-w-[960px]',
  md: 'h-[80vh] w-[90vw] max-w-[1400px]',
  lg: 'h-[80vh] w-[90vw] max-w-[1280px] aspect-[20/13] min-[1450px]:max-w-[1724px]',
  full: 'h-full w-full max-w-[1400px] 2xl:max-w-[1600px]'
} as const

type ModalSize = keyof typeof SIZE_CLASSES

const {
  contentTitle,
  rightPanelTitle,
  size = 'lg'
} = defineProps<{
  contentTitle: string
  rightPanelTitle?: string
  size?: ModalSize
}>()

const sizeClasses = computed(() => SIZE_CLASSES[size])

const isRightPanelOpen = defineModel<boolean>('rightPanelOpen', {
  default: false
})

const slots = useSlots()
const hasLeftPanel = computed(() => !!slots.leftPanel)
const hasRightPanel = computed(() => !!slots.rightPanel)

const BREAKPOINTS = { md: 880 }

const closeDialog = inject(OnCloseKey, () => {})

const breakpoints = useBreakpoints(BREAKPOINTS)
const notMobile = breakpoints.greater('md')

const isLeftPanelOpen = ref<boolean>(true)
const mobileMenuOpen = ref<boolean>(false)

watch(notMobile, (isDesktop) => {
  if (!isDesktop) {
    mobileMenuOpen.value = false
  }
})

const showLeftPanel = computed(() => {
  const shouldShow = notMobile.value
    ? isLeftPanelOpen.value
    : mobileMenuOpen.value
  return shouldShow
})

const gridStyle = computed(() => ({
  gridTemplateColumns: hasRightPanel.value
    ? `${hasLeftPanel.value && showLeftPanel.value ? '14rem' : '0rem'} 1fr ${isRightPanelOpen.value ? '18rem' : '0rem'}`
    : `${hasLeftPanel.value && showLeftPanel.value ? '14rem' : '0rem'} 1fr`
}))

const toggleLeftPanel = () => {
  if (notMobile.value) {
    isLeftPanelOpen.value = !isLeftPanelOpen.value
  } else {
    mobileMenuOpen.value = !mobileMenuOpen.value
  }
}

const toggleRightPanel = () => {
  isRightPanelOpen.value = !isRightPanelOpen.value
}

function handleEscape(event: KeyboardEvent) {
  const target = event.target
  if (!(target instanceof HTMLElement)) return
  if (
    target instanceof HTMLInputElement ||
    target instanceof HTMLTextAreaElement ||
    target instanceof HTMLSelectElement ||
    target.isContentEditable
  ) {
    return
  }
  if (isRightPanelOpen.value) {
    event.stopPropagation()
    isRightPanelOpen.value = false
  }
}
</script>
