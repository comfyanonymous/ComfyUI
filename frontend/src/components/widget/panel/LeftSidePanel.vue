<template>
  <div
    class="flex w-full flex-auto overflow-y-auto gap-1 min-h-0 flex-col bg-modal-panel-background scrollbar-hide px-3"
  >
    <template
      v-for="item in navItems"
      :key="'title' in item ? item.title : item.id"
    >
      <div v-if="'items' in item" class="flex flex-col gap-2">
        <NavTitle
          v-model="collapsedGroups[item.title]"
          :title="item.title"
          :collapsible="item.collapsible"
        />
        <template v-if="!item.collapsible || !collapsedGroups[item.title]">
          <NavItem
            v-for="subItem in item.items"
            :key="subItem.id"
            :icon="subItem.icon"
            :badge="subItem.badge"
            :active="activeItem === subItem.id"
            @click="activeItem = subItem.id"
          >
            {{ subItem.label }}
          </NavItem>
        </template>
      </div>
      <div v-else class="flex flex-col gap-2">
        <NavItem
          :icon="item.icon"
          :badge="item.badge"
          :active="activeItem === item.id"
          @click="activeItem = item.id"
        >
          {{ item.label }}
        </NavItem>
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'

import NavItem from '@/components/widget/nav/NavItem.vue'
import NavTitle from '@/components/widget/nav/NavTitle.vue'
import type { NavGroupData, NavItemData } from '@/types/navTypes'

const { navItems = [], modelValue } = defineProps<{
  navItems?: (NavItemData | NavGroupData)[]
  modelValue?: string | null
}>()

const emit = defineEmits<{
  'update:modelValue': [value: string | null]
}>()

// Track collapsed state for each group
const collapsedGroups = ref<Record<string, boolean>>({})

const getFirstItemId = () => {
  if (!navItems || navItems.length === 0) {
    return null
  }

  const firstEntry = navItems[0]

  if ('items' in firstEntry && firstEntry.items.length > 0) {
    return firstEntry.items[0].id
  }
  if ('id' in firstEntry) {
    return firstEntry.id
  }

  return null
}

const activeItem = computed({
  get: () => modelValue ?? getFirstItemId(),
  set: (value: string | null) => emit('update:modelValue', value)
})
</script>
