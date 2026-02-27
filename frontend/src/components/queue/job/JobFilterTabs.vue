<template>
  <div class="min-w-0 flex-1 overflow-x-auto">
    <div class="inline-flex items-center gap-1 whitespace-nowrap">
      <Button
        v-for="tab in visibleJobTabs"
        :key="tab"
        :variant="selectedJobTab === tab ? 'secondary' : 'muted-textonly'"
        size="md"
        @click="$emit('update:selectedJobTab', tab)"
      >
        {{ tabLabel(tab) }}
      </Button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

import Button from '@/components/ui/button/Button.vue'
import { jobTabs } from '@/composables/queue/useJobList'
import type { JobTab } from '@/composables/queue/useJobList'

const { selectedJobTab, hasFailedJobs } = defineProps<{
  selectedJobTab: JobTab
  hasFailedJobs: boolean
}>()

defineEmits<{
  (e: 'update:selectedJobTab', value: JobTab): void
}>()

const { t } = useI18n()

const visibleJobTabs = computed(() =>
  hasFailedJobs ? jobTabs : jobTabs.filter((tab) => tab !== 'Failed')
)

const tabLabel = (tab: JobTab) => {
  if (tab === 'All') return t('g.all')
  if (tab === 'Completed') return t('g.completed')
  return t('g.failed')
}
</script>
