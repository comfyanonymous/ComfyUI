<script setup lang="ts" generic="T">
import { onBeforeUnmount, ref, useTemplateRef, watchPostEffect } from 'vue'

import { DraggableList } from '@/scripts/ui/draggableList'

const modelValue = defineModel<T[]>({ required: true })
const draggableList = ref<DraggableList>()
const draggableItems = useTemplateRef('draggableItems')

watchPostEffect(() => {
  void modelValue.value.length
  draggableList.value?.dispose()
  if (!draggableItems.value?.children?.length) return
  draggableList.value = new DraggableList(
    draggableItems.value,
    '.draggable-item'
  )
  draggableList.value.applyNewItemsOrder = function () {
    const reorderedItems = []

    let oldPosition = -1
    this.getAllItems().forEach((item, index) => {
      if (item === this.draggableItem) {
        oldPosition = index
        return
      }
      if (!this.isItemToggled(item)) {
        reorderedItems[index] = item
        return
      }
      const newIndex = this.isItemAbove(item) ? index + 1 : index - 1
      reorderedItems[newIndex] = item
    })

    for (let index = 0; index < this.getAllItems().length; index++) {
      const item = reorderedItems[index]
      if (typeof item === 'undefined') {
        reorderedItems[index] = this.draggableItem
      }
    }
    const newPosition = reorderedItems.indexOf(this.draggableItem)
    const itemList = modelValue.value
    const [item] = itemList.splice(oldPosition, 1)
    itemList.splice(newPosition, 0, item)
    modelValue.value = [...itemList]
  }
})

onBeforeUnmount(() => {
  draggableList.value?.dispose()
})
</script>
<template>
  <div ref="draggableItems" class="pb-2 px-2 space-y-0.5 mt-0.5">
    <slot
      drag-class="draggable-item drag-handle cursor-grab [&.is-draggable]:cursor-grabbing"
    />
  </div>
</template>
