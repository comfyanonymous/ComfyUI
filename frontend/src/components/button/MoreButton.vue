<template>
  <div class="relative inline-flex items-center">
    <Button
      size="icon"
      variant="secondary"
      v-bind="$attrs"
      @click="popover?.toggle"
    >
      <i
        :class="
          cn(
            !isVertical
              ? 'icon-[lucide--ellipsis]'
              : 'icon-[lucide--more-vertical]',
            'text-sm'
          )
        "
      />
    </Button>

    <Popover
      ref="popover"
      append-to="body"
      auto-z-index
      dismissable
      close-on-escape
      unstyled
      :base-z-index="1000"
      :pt="{
        root: {
          class: cn('absolute z-50')
        },
        content: {
          class: cn(
            'mt-1 rounded-lg',
            'bg-secondary-background text-base-foreground',
            'shadow-lg'
          )
        }
      }"
      @show="
        () => {
          isOpen = true
          $emit('menuOpened')
        }
      "
      @hide="
        () => {
          isOpen = false
          $emit('menuClosed')
        }
      "
    >
      <div class="flex min-w-40 flex-col gap-2 p-2">
        <slot :close="hide" />
      </div>
    </Popover>
  </div>
</template>

<script setup lang="ts">
import Popover from 'primevue/popover'
import { ref } from 'vue'

import Button from '@/components/ui/button/Button.vue'
import { cn } from '@/utils/tailwindUtil'

defineOptions({
  inheritAttrs: false
})

interface MoreButtonProps {
  isVertical?: boolean
}

const { isVertical = false } = defineProps<MoreButtonProps>()

defineEmits<{
  menuOpened: []
  menuClosed: []
}>()

const isOpen = ref(false)
const popover = ref<InstanceType<typeof Popover>>()

function hide() {
  popover.value?.hide()
}

defineExpose({
  hide,
  isOpen
})
</script>
