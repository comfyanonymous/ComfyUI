<template>
  <Popover
    ref="popoverRef"
    :auto-z-index="true"
    :base-z-index="1100"
    :dismissable="true"
    :close-on-escape="true"
    unstyled
    :pt="{
      root: {
        class: 'absolute z-60'
      },
      content: {
        class: [
          'text-base-foreground rounded-lg',
          'shadow-lg border border-base-background',
          'bg-interface-panel-surface'
        ]
      }
    }"
  >
    <div
      :class="
        isColorSubmenu
          ? 'flex flex-col gap-1 p-2'
          : 'flex flex-col p-2 min-w-40'
      "
    >
      <div
        v-for="subOption in option.submenu"
        :key="subOption.label"
        :class="
          cn(
            'hover:bg-secondary-background-hover rounded cursor-pointer',
            isColorSubmenu
              ? 'w-7 h-7 flex items-center justify-center'
              : 'flex items-center gap-2 px-3 py-1.5 text-sm',
            subOption.disabled
              ? 'cursor-not-allowed pointer-events-none text-node-icon-disabled'
              : 'hover:bg-secondary-background-hover'
          )
        "
        :title="subOption.label"
        @click="handleSubmenuClick(subOption)"
      >
        <div
          v-if="subOption.color"
          class="size-5 rounded-full border border-border-default"
          :style="{ backgroundColor: subOption.color }"
        />
        <template v-else-if="!subOption.color">
          <i
            v-if="isShapeSelected(subOption)"
            class="icon-[lucide--check] size-4 flex-shrink-0"
          />
          <div v-else class="w-4 flex-shrink-0" />
          <span>{{ subOption.label }}</span>
        </template>
      </div>
    </div>
  </Popover>
</template>

<script setup lang="ts">
import { cn } from '@comfyorg/tailwind-utils'
import Popover from 'primevue/popover'
import { computed, ref } from 'vue'

import type {
  MenuOption,
  SubMenuOption
} from '@/composables/graph/useMoreOptionsMenu'
import { useNodeCustomization } from '@/composables/graph/useNodeCustomization'

interface Props {
  option: MenuOption
}

interface Emits {
  (e: 'submenu-click', subOption: SubMenuOption): void
}

const props = defineProps<Props>()
const emit = defineEmits<Emits>()

const { getCurrentShape } = useNodeCustomization()

const popoverRef = ref<InstanceType<typeof Popover>>()

const toggle = (event: Event, target?: HTMLElement) => {
  popoverRef.value?.toggle(event, target)
}
defineExpose({
  toggle
})

const handleSubmenuClick = (subOption: SubMenuOption) => {
  if (subOption.disabled) {
    return
  }
  emit('submenu-click', subOption)
  popoverRef.value?.hide()
}

const isShapeSelected = (subOption: SubMenuOption): boolean => {
  if (subOption.color) return false

  const currentShape = getCurrentShape()
  if (!currentShape) return false

  return currentShape.localizedName === subOption.label
}

const isColorSubmenu = computed(() => {
  return (
    props.option.submenu &&
    props.option.submenu.length > 0 &&
    props.option.submenu.every((item) => item.color && !item.icon)
  )
})
</script>
