<script setup lang="ts">
import { cn } from '@/utils/tailwindUtil'

const { visible } = defineProps<{
  visible: boolean
}>()

const isExpanded = defineModel<boolean>('expanded', { default: false })

function toggle() {
  isExpanded.value = !isExpanded.value
}
</script>

<template>
  <Teleport to="body">
    <Transition
      enter-active-class="transition-all duration-300 ease-out"
      enter-from-class="translate-y-full opacity-0"
      enter-to-class="translate-y-0 opacity-100"
      leave-active-class="transition-all duration-200 ease-in"
      leave-from-class="translate-y-0 opacity-100"
      leave-to-class="translate-y-full opacity-0"
    >
      <div
        v-if="visible"
        role="status"
        aria-live="polite"
        class="fixed inset-x-0 bottom-6 z-9999 mx-auto max-w-3xl overflow-hidden rounded-lg border border-border-default bg-base-background shadow-lg min-w-0 w-min transition-all duration-300"
      >
        <div
          :class="
            cn(
              'overflow-hidden transition-all duration-300 min-w-0 max-w-full',
              isExpanded ? 'w-[max(400px,40vw)] max-h-100' : 'w-0 max-h-0'
            )
          "
        >
          <slot :is-expanded />
        </div>

        <slot name="footer" :is-expanded :toggle />
      </div>
    </Transition>
  </Teleport>
</template>
