<template>
  <div v-if="props.settingGroups.length > 0">
    <SettingGroup
      v-for="(group, i) in props.settingGroups"
      :key="group.label"
      :divider="i !== 0"
      :group="group"
    />
  </div>
  <NoResultsPlaceholder
    v-else
    icon="pi pi-search"
    :title="$t('g.noResultsFound')"
    :message="$t('g.searchFailedMessage')"
  />
</template>

<script setup lang="ts">
import NoResultsPlaceholder from '@/components/common/NoResultsPlaceholder.vue'
import SettingGroup from '@/platform/settings/components/SettingGroup.vue'
import type { ISettingGroup } from '@/platform/settings/types'

const props = defineProps<{
  settingGroups: ISettingGroup[]
}>()
</script>
