<template>
  <div class="setting-group">
    <Divider v-if="divider" />
    <h3>
      <span v-if="group.category" class="text-muted">
        {{
          $t(
            `settingsCategories.${normalizeI18nKey(group.category)}`,
            group.category
          )
        }}
        &#8250;
      </span>
      {{
        $t(`settingsCategories.${normalizeI18nKey(group.label)}`, group.label)
      }}
    </h3>
    <div
      v-for="setting in group.settings.filter((s) => !s.deprecated)"
      :key="setting.id"
      :data-setting-id="setting.id"
      class="setting-item mb-4"
    >
      <SettingItem :setting="setting" />
    </div>
  </div>
</template>

<script setup lang="ts">
import Divider from 'primevue/divider'

import SettingItem from '@/platform/settings/components/SettingItem.vue'
import type { SettingParams } from '@/platform/settings/types'
import { normalizeI18nKey } from '@/utils/formatUtil'

defineProps<{
  group: {
    label: string
    category?: string
    settings: SettingParams[]
  }
  divider?: boolean
}>()
</script>
