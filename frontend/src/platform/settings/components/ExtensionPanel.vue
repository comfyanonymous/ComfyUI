<template>
  <div class="extension-panel flex flex-col gap-2">
    <SearchBox
      v-model="filters['global'].value"
      :placeholder="$t('g.searchPlaceholder', { subject: $t('g.extensions') })"
    />
    <Message
      v-if="hasChanges"
      severity="info"
      pt:text="w-full"
      class="max-h-96 overflow-y-auto"
    >
      <ul>
        <li v-for="ext in changedExtensions" :key="ext.name">
          <span>
            {{ extensionStore.isExtensionEnabled(ext.name) ? '[-]' : '[+]' }}
          </span>
          {{ ext.name }}
        </li>
      </ul>
      <div class="flex justify-end">
        <Button variant="destructive" @click="applyChanges">
          {{ $t('g.reloadToApplyChanges') }}
        </Button>
      </div>
    </Message>
    <div class="mb-3 flex gap-2">
      <SelectButton
        v-model="filterType"
        :options="filterTypes"
        option-label="label"
        option-value="value"
      />
    </div>
    <DataTable
      v-model:selection="selectedExtensions"
      :value="filteredExtensions"
      striped-rows
      size="small"
      :filters="filters"
      selection-mode="multiple"
      data-key="name"
    >
      <Column selection-mode="multiple" :frozen="true" style="width: 3rem" />
      <Column :header="$t('g.extensionName')" sortable field="name">
        <template #body="slotProps">
          {{ slotProps.data.name }}
          <Tag
            v-if="extensionStore.isCoreExtension(slotProps.data.name)"
            :value="$t('g.core')"
          />
          <Tag v-else :value="$t('g.custom')" severity="info" />
        </template>
      </Column>
      <Column
        :pt="{
          headerCell: 'flex items-center justify-end',
          bodyCell: 'flex items-center justify-end'
        }"
      >
        <template #header>
          <Button
            size="icon"
            variant="muted-textonly"
            @click="menu?.show($event)"
          >
            <i class="pi pi-ellipsis-h" />
          </Button>
          <ContextMenu ref="menu" :model="contextMenuItems" />
        </template>
        <template #body="slotProps">
          <ToggleSwitch
            v-model="editingEnabledExtensions[slotProps.data.name]"
            :disabled="extensionStore.isExtensionReadOnly(slotProps.data.name)"
            @change="updateExtensionStatus"
          />
        </template>
      </Column>
    </DataTable>
  </div>
</template>

<script setup lang="ts">
import { FilterMatchMode } from '@primevue/core/api'
import Column from 'primevue/column'
import ContextMenu from 'primevue/contextmenu'
import DataTable from 'primevue/datatable'
import Message from 'primevue/message'
import SelectButton from 'primevue/selectbutton'
import Tag from 'primevue/tag'
import ToggleSwitch from 'primevue/toggleswitch'
import { computed, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'

import SearchBox from '@/components/common/SearchBox.vue'
import Button from '@/components/ui/button/Button.vue'
import { useSettingStore } from '@/platform/settings/settingStore'
import { useExtensionStore } from '@/stores/extensionStore'
import type { ComfyExtension } from '@/types/comfy'

const { t } = useI18n()

const filterTypeKeys = ['all', 'core', 'custom'] as const
type FilterTypeKey = (typeof filterTypeKeys)[number]
const filterTypes = computed(() =>
  filterTypeKeys.map((key) => ({
    label: t(`g.${key}`),
    value: key
  }))
)
const filterType = ref<FilterTypeKey>('all')
const selectedExtensions = ref<ComfyExtension[]>([])

const filters = ref({
  global: { value: '', matchMode: FilterMatchMode.CONTAINS }
})

const extensionStore = useExtensionStore()
const settingStore = useSettingStore()

const editingEnabledExtensions = ref<Record<string, boolean>>({})

const filteredExtensions = computed(() => {
  const extensions = extensionStore.extensions
  switch (filterType.value) {
    case 'core':
      return extensions.filter((ext) =>
        extensionStore.isCoreExtension(ext.name)
      )
    case 'custom':
      return extensions.filter(
        (ext) => !extensionStore.isCoreExtension(ext.name)
      )
    default:
      return extensions
  }
})

onMounted(() => {
  extensionStore.extensions.forEach((ext) => {
    editingEnabledExtensions.value[ext.name] =
      extensionStore.isExtensionEnabled(ext.name)
  })
})

const changedExtensions = computed(() => {
  return extensionStore.extensions.filter(
    (ext) =>
      editingEnabledExtensions.value[ext.name] !==
      extensionStore.isExtensionEnabled(ext.name)
  )
})

const hasChanges = computed(() => {
  return changedExtensions.value.length > 0
})

const updateExtensionStatus = async () => {
  const editingDisabledExtensionNames = Object.entries(
    editingEnabledExtensions.value
  )
    .filter(([_, enabled]) => !enabled)
    .map(([name]) => name)

  await settingStore.set('Comfy.Extension.Disabled', [
    ...extensionStore.inactiveDisabledExtensionNames,
    ...editingDisabledExtensionNames
  ])
}

const enableAllExtensions = async () => {
  extensionStore.extensions.forEach((ext) => {
    if (extensionStore.isExtensionReadOnly(ext.name)) return

    editingEnabledExtensions.value[ext.name] = true
  })
  await updateExtensionStatus()
}

const disableAllExtensions = async () => {
  extensionStore.extensions.forEach((ext) => {
    if (extensionStore.isExtensionReadOnly(ext.name)) return

    editingEnabledExtensions.value[ext.name] = false
  })
  await updateExtensionStatus()
}

const disableThirdPartyExtensions = async () => {
  extensionStore.extensions.forEach((ext) => {
    if (extensionStore.isCoreExtension(ext.name)) return

    editingEnabledExtensions.value[ext.name] = false
  })
  await updateExtensionStatus()
}

const applyChanges = () => {
  // Refresh the page to apply changes
  window.location.reload()
}

const menu = ref<InstanceType<typeof ContextMenu>>()
const contextMenuItems = computed(() => [
  {
    label: t('g.enableSelected'),
    icon: 'pi pi-check',
    command: async () => {
      selectedExtensions.value.forEach((ext) => {
        if (!extensionStore.isExtensionReadOnly(ext.name)) {
          editingEnabledExtensions.value[ext.name] = true
        }
      })
      await updateExtensionStatus()
    }
  },
  {
    label: t('g.disableSelected'),
    icon: 'pi pi-times',
    command: async () => {
      selectedExtensions.value.forEach((ext) => {
        if (!extensionStore.isExtensionReadOnly(ext.name)) {
          editingEnabledExtensions.value[ext.name] = false
        }
      })
      await updateExtensionStatus()
    }
  },
  {
    separator: true
  },
  {
    label: t('g.enableAll'),
    icon: 'pi pi-check',
    command: enableAllExtensions
  },
  {
    label: t('g.disableAll'),
    icon: 'pi pi-times',
    command: disableAllExtensions
  },
  {
    label: t('g.disableThirdParty'),
    icon: 'pi pi-times',
    command: disableThirdPartyExtensions,
    disabled: !extensionStore.hasThirdPartyExtensions
  }
])
</script>
