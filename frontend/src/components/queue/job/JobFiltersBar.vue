<template>
  <div class="flex items-center justify-between gap-2 px-3">
    <JobFilterTabs
      :selected-job-tab="selectedJobTab"
      :has-failed-jobs="hasFailedJobs"
      @update:selected-job-tab="$emit('update:selectedJobTab', $event)"
    />
    <JobFilterActions
      :selected-workflow-filter="selectedWorkflowFilter"
      :selected-sort-mode="selectedSortMode"
      :hide-show-assets-action="hideShowAssetsAction"
      @show-assets="$emit('showAssets')"
      @update:selected-workflow-filter="
        $emit('update:selectedWorkflowFilter', $event)
      "
      @update:selected-sort-mode="$emit('update:selectedSortMode', $event)"
    />
  </div>
</template>

<script setup lang="ts">
import JobFilterActions from '@/components/queue/job/JobFilterActions.vue'
import JobFilterTabs from '@/components/queue/job/JobFilterTabs.vue'
import type { JobSortMode, JobTab } from '@/composables/queue/useJobList'

const {
  selectedJobTab,
  selectedWorkflowFilter,
  selectedSortMode,
  hasFailedJobs,
  hideShowAssetsAction
} = defineProps<{
  selectedJobTab: JobTab
  selectedWorkflowFilter: 'all' | 'current'
  selectedSortMode: JobSortMode
  hasFailedJobs: boolean
  hideShowAssetsAction?: boolean
}>()

defineEmits<{
  (e: 'showAssets'): void
  (e: 'update:selectedJobTab', value: JobTab): void
  (e: 'update:selectedWorkflowFilter', value: 'all' | 'current'): void
  (e: 'update:selectedSortMode', value: JobSortMode): void
}>()
</script>
