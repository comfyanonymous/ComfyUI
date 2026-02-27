<!--
TODO: Extract checkbox pattern into reusable Checkbox component
- Create src/components/input/Checkbox.vue with:
  - Hidden native <input type="checkbox"> for accessibility
  - Custom visual styling matching this implementation
  - Semantic tokens (--primary-background, --input-surface, etc.)
- Use this Checkbox component in:
  - MediaAssetFilterMenu.vue (this file)
  - MultiSelect.vue option template
  - SingleSelect.vue if needed
- Benefits: Consistent checkbox UI, better maintainability, reusable design system component
-->
<template>
  <div class="flex flex-col gap-0 p-0 m-0">
    <div
      v-for="filter in filters"
      :key="filter.type"
      class="flex h-10 min-w-32 cursor-pointer items-center gap-2 rounded-lg px-2 hover:bg-secondary-background-hover"
      tabindex="0"
      role="checkbox"
      :aria-checked="mediaTypeFilters.includes(filter.type)"
      @click="toggleMediaType(filter.type)"
      @keydown.enter.prevent="toggleMediaType(filter.type)"
      @keydown.space.prevent="toggleMediaType(filter.type)"
    >
      <div
        class="flex h-4 w-4 shrink-0 items-center justify-center rounded p-0.5 transition-all duration-200"
        :class="
          mediaTypeFilters.includes(filter.type)
            ? 'bg-primary-background border-primary-background'
            : 'bg-secondary-background'
        "
      >
        <i
          v-if="mediaTypeFilters.includes(filter.type)"
          class="icon-[lucide--check] text-xs font-bold text-white"
        />
      </div>
      <span class="text-sm">{{ $t(filter.label) }}</span>
    </div>
  </div>
</template>

<script setup lang="ts">
const { mediaTypeFilters } = defineProps<{
  mediaTypeFilters: string[]
}>()

const emit = defineEmits<{
  'update:mediaTypeFilters': [value: string[]]
}>()

const filters = [
  { type: 'image', label: 'sideToolbar.mediaAssets.filterImage' },
  { type: 'video', label: 'sideToolbar.mediaAssets.filterVideo' },
  { type: 'audio', label: 'sideToolbar.mediaAssets.filterAudio' },
  { type: '3d', label: 'sideToolbar.mediaAssets.filter3D' }
]

const toggleMediaType = (type: string) => {
  const isCurrentlySelected = mediaTypeFilters.includes(type)
  if (isCurrentlySelected) {
    emit(
      'update:mediaTypeFilters',
      mediaTypeFilters.filter((t) => t !== type)
    )
  } else {
    emit('update:mediaTypeFilters', [...mediaTypeFilters, type])
  }
}
</script>
