<template>
  <div class="flex flex-col">
    <Button
      variant="textonly"
      class="w-full"
      @click="handleViewModeChange('list')"
    >
      <span class="flex items-center gap-2">
        <i class="icon-[lucide--table-of-contents] size-4" />
        <span>{{ $t('sideToolbar.queueProgressOverlay.viewList') }}</span>
      </span>
      <i
        class="icon-[lucide--check] ml-auto size-4"
        :class="viewMode !== 'list' && 'opacity-0'"
      />
    </Button>

    <Button
      variant="textonly"
      class="w-full"
      @click="handleViewModeChange('grid')"
    >
      <span class="flex items-center gap-2">
        <i class="icon-[lucide--layout-grid] size-4" />
        <span>{{ $t('sideToolbar.queueProgressOverlay.viewGrid') }}</span>
      </span>
      <i
        class="icon-[lucide--check] ml-auto size-4"
        :class="viewMode !== 'grid' && 'opacity-0'"
      />
    </Button>

    <template v-if="showSortOptions">
      <div class="border-b w-full border-border-subtle my-1" />

      <Button
        variant="textonly"
        class="w-full"
        @click="handleSortChange('newest')"
      >
        <span>{{ $t('sideToolbar.mediaAssets.sortNewestFirst') }}</span>
        <i
          class="icon-[lucide--check] ml-auto size-4"
          :class="sortBy !== 'newest' && 'opacity-0'"
        />
      </Button>

      <Button
        variant="textonly"
        class="w-full"
        @click="handleSortChange('oldest')"
      >
        <span>{{ $t('sideToolbar.mediaAssets.sortOldestFirst') }}</span>
        <i
          class="icon-[lucide--check] ml-auto size-4"
          :class="sortBy !== 'oldest' && 'opacity-0'"
        />
      </Button>

      <template v-if="showGenerationTimeSort">
        <Button
          variant="textonly"
          class="w-full"
          @click="handleSortChange('longest')"
        >
          <span>{{ $t('sideToolbar.mediaAssets.sortLongestFirst') }}</span>
          <i
            class="icon-[lucide--check] ml-auto size-4"
            :class="sortBy !== 'longest' && 'opacity-0'"
          />
        </Button>

        <Button
          variant="textonly"
          class="w-full"
          @click="handleSortChange('fastest')"
        >
          <span>{{ $t('sideToolbar.mediaAssets.sortFastestFirst') }}</span>
          <i
            class="icon-[lucide--check] ml-auto size-4"
            :class="sortBy !== 'fastest' && 'opacity-0'"
          />
        </Button>
      </template>
    </template>
  </div>
</template>

<script setup lang="ts">
import Button from '@/components/ui/button/Button.vue'

export type SortBy = 'newest' | 'oldest' | 'longest' | 'fastest'

const { showSortOptions = false, showGenerationTimeSort = false } =
  defineProps<{
    showSortOptions?: boolean
    showGenerationTimeSort?: boolean
  }>()

const viewMode = defineModel<'list' | 'grid'>('viewMode', { required: true })
const sortBy = defineModel<SortBy>('sortBy', { required: true })

function handleViewModeChange(value: 'list' | 'grid') {
  viewMode.value = value
}

function handleSortChange(value: SortBy) {
  sortBy.value = value
}
</script>
