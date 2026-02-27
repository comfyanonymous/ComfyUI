<script setup lang="ts" generic="T extends WidgetValue">
import { computed, defineAsyncComponent, ref, watch } from 'vue'
import type { Component } from 'vue'

import Popover from '@/components/ui/Popover.vue'
import Button from '@/components/ui/button/Button.vue'
import type {
  SimplifiedControlWidget,
  WidgetValue
} from '@/types/simplifiedWidget'

const ValueControlPopover = defineAsyncComponent(
  () => import('./ValueControlPopover.vue')
)

const props = defineProps<{
  widget: SimplifiedControlWidget<T>
  component: Component
}>()

const modelValue = defineModel<T>()

const controlModel = ref(props.widget.controlWidget.value)

const controlButtonIcon = computed(() => {
  switch (controlModel.value) {
    case 'increment':
      return 'pi pi-plus'
    case 'decrement':
      return 'pi pi-minus'
    case 'fixed':
      return 'icon-[lucide--pencil-off]'
    default:
      return 'icon-[lucide--shuffle]'
  }
})

watch(controlModel, props.widget.controlWidget.update)
</script>
<template>
  <div class="relative grid grid-cols-subgrid">
    <component :is="component" v-bind="$attrs" v-model="modelValue" :widget>
      <Popover>
        <template #button>
          <Button
            variant="textonly"
            size="sm"
            class="h-4 w-7 p-0 self-center rounded-xl bg-primary-background/30 hover:bg-primary-background-hover/30"
          >
            <i
              :class="`${controlButtonIcon} text-primary-background text-xs w-full`"
            />
          </Button>
        </template>
        <ValueControlPopover v-model="controlModel" />
      </Popover>
    </component>
  </div>
</template>
