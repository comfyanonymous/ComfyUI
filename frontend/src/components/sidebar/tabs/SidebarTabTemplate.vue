<template>
  <div
    ref="containerRef"
    :class="
      cn(
        'comfy-vue-side-bar-container group/sidebar-tab flex h-full flex-col w-full',
        props.class
      )
    "
  >
    <div class="comfy-vue-side-bar-header flex flex-col gap-2">
      <Toolbar
        class="min-h-16 bg-transparent rounded-none border-x-0 border-t-0 px-2 2xl:px-4"
        :pt="sidebarPt"
      >
        <template #start>
          <span class="truncate font-bold" :title="props.title">
            {{ props.title }}
          </span>
          <slot name="alt-title" />
        </template>
        <template #end>
          <div
            class="touch:w-auto touch:opacity-100 [&_.p-button]:py-1 2xl:[&_.p-button]:py-2 flex flex-row overflow-hidden transition-all duration-200 motion-safe:w-0 motion-safe:opacity-0 motion-safe:group-focus-within/sidebar-tab:w-auto motion-safe:group-focus-within/sidebar-tab:opacity-100 motion-safe:group-hover/sidebar-tab:w-auto motion-safe:group-hover/sidebar-tab:opacity-100"
          >
            <slot name="tool-buttons" />
          </div>
        </template>
      </Toolbar>
      <slot name="header" />
    </div>
    <!-- h-0 to force scrollpanel to grow -->
    <ScrollPanel class="comfy-vue-side-bar-body h-0 grow">
      <slot name="body" />
    </ScrollPanel>
    <slot name="footer" />
  </div>
</template>

<script lang="ts">
import type { InjectionKey, Ref } from 'vue'

export const SidebarContainerKey: InjectionKey<Ref<HTMLElement | null>> =
  Symbol('SidebarContainer')
</script>

<script setup lang="ts">
import ScrollPanel from 'primevue/scrollpanel'
import Toolbar from 'primevue/toolbar'
import { provide, ref } from 'vue'

import { cn } from '@/utils/tailwindUtil'

const props = defineProps<{
  title: string
  class?: string
}>()
const sidebarPt = {
  start: 'min-w-0 flex-1 overflow-hidden'
}

const containerRef = ref<HTMLElement | null>(null)
provide(SidebarContainerKey, containerRef)
</script>
