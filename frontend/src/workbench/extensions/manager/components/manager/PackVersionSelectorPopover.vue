<template>
  <div class="w-64 pt-1">
    <div class="py-2">
      <span class="text-md pl-3 font-semibold text-neutral-500">
        {{ $t('manager.selectVersion') }}
      </span>
    </div>
    <div
      v-if="isLoadingVersions || isQueueing"
      class="flex flex-col items-center py-4 text-center text-muted"
    >
      <ProgressSpinner class="mb-2 h-8 w-8" />
      {{ $t('manager.loadingVersions') }}
    </div>
    <div v-else-if="versionOptions.length === 0" class="py-2">
      <NoResultsPlaceholder
        :title="$t('g.noResultsFound')"
        :message="$t('manager.tryAgainLater')"
        icon="pi pi-exclamation-circle"
        class="p-0"
      />
    </div>
    <Listbox
      v-else
      v-model="selectedVersion"
      option-label="label"
      option-value="value"
      option-disabled="isInstalled"
      :options="processedVersionOptions"
      :highlight-on-select="false"
      class="max-h-[50vh] w-full rounded-md border-none shadow-none"
      :pt="{
        listContainer: { class: 'scrollbar-hide' }
      }"
    >
      <template #option="slotProps">
        <div class="flex w-full items-center justify-between p-1">
          <div class="flex items-center gap-2">
            <template v-if="slotProps.option.value === 'nightly'">
              <div class="w-4"></div>
            </template>
            <template v-else>
              <i
                v-if="slotProps.option.hasConflict"
                v-tooltip="{
                  value: slotProps.option.conflictMessage,
                  showDelay: 300
                }"
                class="icon-[lucide--triangle-alert] text-warning-background"
              />
              <VerifiedIcon v-else :size="20" class="relative right-0.5" />
            </template>
            <span>{{ slotProps.option.label }}</span>
          </div>
          <i
            v-if="slotProps.option.isSelected"
            class="pi pi-check text-highlight"
          />
        </div>
      </template>
    </Listbox>
    <ContentDivider class="my-2" />
    <div class="flex justify-end gap-2 px-3 py-1">
      <Button
        variant="muted-textonly"
        class="text-sm"
        :disabled="isQueueing"
        @click="emit('cancel')"
      >
        {{ $t('g.cancel') }}
      </Button>
      <Button
        variant="secondary"
        class="rounded-lg bg-secondary-background px-4 py-2.5 text-sm text-base-foreground"
        :disabled="isInstallDisabled"
        @click="handleSubmit"
      >
        {{ $t('g.install') }}
      </Button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { whenever } from '@vueuse/core'
import Listbox from 'primevue/listbox'
import ProgressSpinner from 'primevue/progressspinner'
import { valid as validSemver } from 'semver'
import { computed, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'

import ContentDivider from '@/components/common/ContentDivider.vue'
import NoResultsPlaceholder from '@/components/common/NoResultsPlaceholder.vue'
import VerifiedIcon from '@/components/icons/VerifiedIcon.vue'
import Button from '@/components/ui/button/Button.vue'
import { useComfyRegistryService } from '@/services/comfyRegistryService'
import type { components } from '@/types/comfyRegistryTypes'
import { useConflictDetection } from '@/workbench/extensions/manager/composables/useConflictDetection'
import { useComfyManagerStore } from '@/workbench/extensions/manager/stores/comfyManagerStore'
import type { components as ManagerComponents } from '@/workbench/extensions/manager/types/generatedManagerTypes'
import { getJoinedConflictMessages } from '@/workbench/extensions/manager/utils/conflictMessageUtil'

type ManagerChannel = ManagerComponents['schemas']['ManagerChannel']
type ManagerDatabaseSource =
  ManagerComponents['schemas']['ManagerDatabaseSource']
type SelectedVersion = ManagerComponents['schemas']['SelectedVersion']

// Enum values for runtime use
const SelectedVersionValues = {
  LATEST: 'latest' as SelectedVersion,
  NIGHTLY: 'nightly' as SelectedVersion
}

const ManagerChannelValues: Record<string, ManagerChannel> = {
  DEFAULT: 'default',
  DEV: 'dev'
}

const ManagerDatabaseSourceValues: Record<string, ManagerDatabaseSource> = {
  CACHE: 'cache',
  REMOTE: 'remote',
  LOCAL: 'local'
}

const { nodePack } = defineProps<{
  nodePack: components['schemas']['Node']
}>()

const emit = defineEmits<{
  cancel: []
  submit: []
}>()

const { t } = useI18n()
const registryService = useComfyRegistryService()
const managerStore = useComfyManagerStore()
const { checkNodeCompatibility } = useConflictDetection()

const isQueueing = ref(false)
const selectedVersion = ref<string>(SelectedVersionValues.LATEST)
const isInstallDisabled = computed(
  () => isQueueing.value || isVersionInstalled(selectedVersion.value)
)
onMounted(() => {
  const initialVersion =
    getInitialSelectedVersion() ?? SelectedVersionValues.LATEST
  selectedVersion.value =
    // Use NIGHTLY when version is a Git hash
    validSemver(initialVersion) ? initialVersion : SelectedVersionValues.NIGHTLY
})

const getInitialSelectedVersion = () => {
  if (!nodePack.id) return

  // If unclaimed, set selected version to nightly
  if (nodePack.publisher?.name === 'Unclaimed')
    return SelectedVersionValues.NIGHTLY

  // If node pack is installed, set selected version to the installed version
  if (managerStore.isPackInstalled(nodePack.id))
    return managerStore.getInstalledPackVersion(nodePack.id)

  // If node pack is not installed, set selected version to latest
  return nodePack.latest_version?.version
}

const fetchVersions = async () => {
  if (!nodePack?.id) return []
  return (await registryService.getPackVersions(nodePack.id)) || []
}

const versionOptions = ref<
  {
    value: string
    label: string
  }[]
>([])

const fetchedVersions = ref<components['schemas']['NodeVersion'][]>([])

const isLoadingVersions = ref(false)

const onNodePackChange = async () => {
  isLoadingVersions.value = true

  // Fetch versions from the registry
  const versions = await fetchVersions()
  fetchedVersions.value = versions

  const latestVersionNumber = nodePack.latest_version?.version

  const availableVersionOptions = versions
    .map((version) => ({
      value: version.version ?? '',
      label: version.version ?? ''
    }))
    .filter((option) => option.value && option.value !== latestVersionNumber) // Exclude latest version from the list

  // Add Latest option with actual version number
  const latestLabel = latestVersionNumber
    ? `${t('manager.latestVersion')} (${latestVersionNumber})`
    : t('manager.latestVersion')

  // Add Latest option
  const defaultVersions = [
    {
      value: SelectedVersionValues.LATEST,
      label: latestLabel
    }
  ]

  // Add Nightly option if there is a non-empty `repository` field
  if (nodePack.repository?.length) {
    defaultVersions.push({
      value: SelectedVersionValues.NIGHTLY,
      label: t('manager.nightlyVersion')
    })
  }

  versionOptions.value = [...defaultVersions, ...availableVersionOptions]
  isLoadingVersions.value = false
}

whenever(
  () => nodePack.id,
  (nodePackId, oldNodePackId) => {
    if (nodePackId !== oldNodePackId) {
      void onNodePackChange()
    }
  },
  { deep: true, immediate: true }
)

const handleSubmit = async () => {
  isQueueing.value = true

  if (!nodePack.id) {
    throw new Error('Node ID is required for installation')
  }
  // Convert 'latest' to actual version number for installation
  const actualVersion =
    selectedVersion.value === 'latest'
      ? (nodePack.latest_version?.version ?? 'latest')
      : selectedVersion.value

  await managerStore.installPack.call({
    id: nodePack.id,
    repository: nodePack.repository ?? '',
    channel: ManagerChannelValues.DEFAULT,
    mode: ManagerDatabaseSourceValues.CACHE,
    version: actualVersion,
    selected_version: selectedVersion.value
  })

  isQueueing.value = false
  emit('submit')
}

const getVersionData = (version: string) => {
  const latestVersionNumber = nodePack.latest_version?.version
  const useLatestVersionData =
    version === 'latest' || version === latestVersionNumber
  if (useLatestVersionData) {
    const latestVersionData = nodePack.latest_version
    return {
      ...latestVersionData
    }
  }
  const versionData = fetchedVersions.value.find((v) => v.version === version)
  if (versionData) {
    return {
      ...versionData
    }
  }
  // Fallback to nodePack data
  return {
    ...nodePack
  }
}
// Main function to get version compatibility info
const getVersionCompatibility = (version: string) => {
  const versionData = getVersionData(version)
  const compatibility = checkNodeCompatibility(versionData)
  const conflictMessage = compatibility.hasConflict
    ? getJoinedConflictMessages(compatibility.conflicts, t)
    : ''
  return {
    hasConflict: compatibility.hasConflict,
    conflictMessage
  }
}
// Helper to determine if an option is selected.
const isOptionSelected = (optionValue: string) => {
  if (selectedVersion.value === optionValue) {
    return true
  }
  if (
    optionValue === 'latest' &&
    selectedVersion.value === nodePack.latest_version?.version
  ) {
    return true
  }
  return false
}
const isVersionInstalled = (version: string) => {
  const installed = nodePack.id
    ? managerStore.getInstalledPackVersion(nodePack.id)
    : undefined
  if (!installed) return false
  if (version === 'latest')
    return installed === nodePack.latest_version?.version
  return version === installed
}

const processedVersionOptions = computed(() => {
  return versionOptions.value.map((option) => {
    const compatibility = getVersionCompatibility(option.value)
    return {
      ...option,
      hasConflict: compatibility.hasConflict,
      conflictMessage: compatibility.conflictMessage,
      isSelected: isOptionSelected(option.value),
      isInstalled: isVersionInstalled(option.value)
    }
  })
})
</script>
