<script setup lang="ts">
import { capitalize } from 'es-toolkit'
import { computed, provide, ref, toRef, watch } from 'vue'
import { useI18n } from 'vue-i18n'

import { useTransformCompatOverlayProps } from '@/composables/useTransformCompatOverlayProps'
import { SUPPORTED_EXTENSIONS_ACCEPT } from '@/extensions/core/load3d/constants'
import { useAssetFilterOptions } from '@/platform/assets/composables/useAssetFilterOptions'
import {
  filterItemByBaseModels,
  filterItemByOwnership
} from '@/platform/assets/utils/assetFilterUtils'
import {
  getAssetBaseModels,
  getAssetDisplayName,
  getAssetFilename
} from '@/platform/assets/utils/assetMetadataUtils'
import { useToastStore } from '@/platform/updates/common/toastStore'
import FormDropdown from '@/renderer/extensions/vueNodes/widgets/components/form/dropdown/FormDropdown.vue'
import type {
  FilterOption,
  OwnershipOption
} from '@/platform/assets/types/filterTypes'
import { AssetKindKey } from '@/renderer/extensions/vueNodes/widgets/components/form/dropdown/types'
import type {
  FormDropdownItem,
  LayoutMode
} from '@/renderer/extensions/vueNodes/widgets/components/form/dropdown/types'
import WidgetLayoutField from '@/renderer/extensions/vueNodes/widgets/components/layout/WidgetLayoutField.vue'
import { useAssetWidgetData } from '@/renderer/extensions/vueNodes/widgets/composables/useAssetWidgetData'
import type { ResultItemType } from '@/schemas/apiSchema'
import { api } from '@/scripts/api'
import { useAssetsStore } from '@/stores/assetsStore'
import { useQueueStore } from '@/stores/queueStore'
import type { SimplifiedWidget } from '@/types/simplifiedWidget'
import type { AssetKind } from '@/types/widgetTypes'
import {
  PANEL_EXCLUDED_PROPS,
  filterWidgetProps
} from '@/utils/widgetPropFilter'

interface Props {
  widget: SimplifiedWidget<string | undefined>
  nodeType?: string
  assetKind?: AssetKind
  allowUpload?: boolean
  uploadFolder?: ResultItemType
  uploadSubfolder?: string
  isAssetMode?: boolean
  defaultLayoutMode?: LayoutMode
}

const props = defineProps<Props>()

provide(
  AssetKindKey,
  computed(() => props.assetKind)
)

const modelValue = defineModel<string | undefined>({
  default(props: Props) {
    const values = props.widget.options?.values
    return (Array.isArray(values) ? values[0] : undefined) ?? ''
  }
})

const { t } = useI18n()
const toastStore = useToastStore()
const queueStore = useQueueStore()

const transformCompatProps = useTransformCompatOverlayProps()

const combinedProps = computed(() => ({
  ...filterWidgetProps(props.widget.options, PANEL_EXCLUDED_PROPS),
  ...transformCompatProps.value
}))

const getAssetData = () => {
  const nodeType: string | undefined =
    props.widget.options?.nodeType ?? props.nodeType
  if (props.isAssetMode && nodeType) {
    return useAssetWidgetData(toRef(nodeType))
  }
  return null
}
const assetData = getAssetData()

const filterSelected = ref('all')
const filterOptions = computed<FilterOption[]>(() => {
  if (props.isAssetMode) {
    const categoryName = assetData?.category.value ?? 'All'
    return [{ name: capitalize(categoryName), value: 'all' }]
  }
  return [
    { name: 'All', value: 'all' },
    { name: 'Inputs', value: 'inputs' },
    { name: 'Outputs', value: 'outputs' }
  ]
})

const ownershipSelected = ref<OwnershipOption>('all')
const showOwnershipFilter = computed(() => props.isAssetMode)

const { ownershipOptions, availableBaseModels } = useAssetFilterOptions(
  () => assetData?.assets.value ?? []
)

const baseModelSelected = ref<Set<string>>(new Set())
const showBaseModelFilter = computed(() => props.isAssetMode)
const baseModelOptions = computed<FilterOption[]>(() => {
  if (!props.isAssetMode || !assetData) return []
  return availableBaseModels.value
})

const selectedSet = ref<Set<string>>(new Set())

/**
 * Transforms a value using getOptionLabel if available.
 * Falls back to the original value if getOptionLabel is not provided,
 * returns undefined/null, or throws an error.
 */
function getDisplayLabel(value: string): string {
  const getOptionLabel = props.widget.options?.getOptionLabel
  if (!getOptionLabel) return value

  try {
    return getOptionLabel(value) || value
  } catch (e) {
    console.error('Failed to map value:', e)
    return value
  }
}

const inputItems = computed<FormDropdownItem[]>(() => {
  const values = props.widget.options?.values || []

  if (!Array.isArray(values)) {
    return []
  }

  return values.map((value, index) => ({
    id: `input-${index}`,
    preview_url: getMediaUrl(String(value), 'input'),
    name: String(value),
    label: getDisplayLabel(String(value))
  }))
})
const outputItems = computed<FormDropdownItem[]>(() => {
  if (!['image', 'video', 'mesh'].includes(props.assetKind ?? '')) return []

  const outputs = new Set<string>()

  // Extract output images/videos from queue history
  queueStore.historyTasks.forEach((task) => {
    task.flatOutputs.forEach((output) => {
      const isTargetType =
        (props.assetKind === 'image' && output.mediaType === 'images') ||
        (props.assetKind === 'video' && output.mediaType === 'video') ||
        (props.assetKind === 'mesh' && output.is3D)

      if (output.type === 'output' && isTargetType) {
        const path = output.subfolder
          ? `${output.subfolder}/${output.filename}`
          : output.filename
        // Add [output] annotation so the preview component knows the type
        const annotatedPath = `${path} [output]`
        outputs.add(annotatedPath)
      }
    })
  })

  return Array.from(outputs).map((output) => ({
    id: `output-${output}`,
    preview_url: getMediaUrl(output.replace(' [output]', ''), 'output'),
    name: output,
    label: getDisplayLabel(output)
  }))
})

/**
 * Creates a fallback item for the current modelValue when it doesn't exist
 * in the available items list. This handles cases like template-loaded nodes
 * where the saved value may not exist in the current server environment.
 * Works for both local mode (inputItems/outputItems) and cloud mode (assetData).
 */
const missingValueItem = computed<FormDropdownItem | undefined>(() => {
  const currentValue = modelValue.value
  if (!currentValue) return undefined

  // Check in cloud mode assets
  if (props.isAssetMode && assetData) {
    const existsInAssets = assetData.assets.value.some(
      (asset) => getAssetFilename(asset) === currentValue
    )
    if (existsInAssets) return undefined

    return {
      id: `missing-${currentValue}`,
      preview_url: '',
      name: currentValue,
      label: getDisplayLabel(currentValue)
    }
  }

  // Check in local mode inputs/outputs
  const existsInInputs = inputItems.value.some(
    (item) => item.name === currentValue
  )
  const existsInOutputs = outputItems.value.some(
    (item) => item.name === currentValue
  )

  if (existsInInputs || existsInOutputs) return undefined

  const isOutput = currentValue.endsWith(' [output]')
  const strippedValue = isOutput
    ? currentValue.replace(' [output]', '')
    : currentValue

  return {
    id: `missing-${currentValue}`,
    preview_url: getMediaUrl(strippedValue, isOutput ? 'output' : 'input'),
    name: currentValue,
    label: getDisplayLabel(currentValue)
  }
})

/**
 * Transforms AssetItem[] to FormDropdownItem[] for cloud mode.
 * Uses getAssetFilename for display name, asset.name for label.
 */
const assetItems = computed<FormDropdownItem[]>(() => {
  if (!props.isAssetMode || !assetData) return []
  return assetData.assets.value.map((asset) => ({
    id: asset.id,
    name: getAssetFilename(asset),
    label: getAssetDisplayName(asset),
    preview_url: asset.preview_url,
    is_immutable: asset.is_immutable,
    base_models: getAssetBaseModels(asset)
  }))
})

const ownershipFilteredAssetItems = computed<FormDropdownItem[]>(() =>
  filterItemByOwnership(assetItems.value, ownershipSelected.value)
)

const baseModelFilteredAssetItems = computed<FormDropdownItem[]>(() =>
  filterItemByBaseModels(
    ownershipFilteredAssetItems.value,
    baseModelSelected.value
  )
)

const allItems = computed<FormDropdownItem[]>(() => {
  if (props.isAssetMode && assetData) {
    // Cloud assets not in user's library shouldn't appear as search results (COM-14333).
    // Unlike local mode, cloud users can't access files they don't own.
    return baseModelFilteredAssetItems.value
  }
  return [
    ...(missingValueItem.value ? [missingValueItem.value] : []),
    ...inputItems.value,
    ...outputItems.value
  ]
})

const dropdownItems = computed<FormDropdownItem[]>(() => {
  if (props.isAssetMode) {
    return allItems.value
  }

  switch (filterSelected.value) {
    case 'inputs':
      return inputItems.value
    case 'outputs':
      return outputItems.value
    case 'all':
    default:
      return allItems.value
  }
})

/**
 * Items used for display in the input field. In cloud mode, includes
 * missing items so users can see their selected value even if not in library.
 */
const displayItems = computed<FormDropdownItem[]>(() => {
  if (props.isAssetMode && assetData && missingValueItem.value) {
    return [missingValueItem.value, ...baseModelFilteredAssetItems.value]
  }
  return dropdownItems.value
})

const mediaPlaceholder = computed(() => {
  const options = props.widget.options

  if (options?.placeholder) {
    return options.placeholder
  }

  switch (props.assetKind) {
    case 'image':
      return t('widgets.uploadSelect.placeholderImage')
    case 'video':
      return t('widgets.uploadSelect.placeholderVideo')
    case 'audio':
      return t('widgets.uploadSelect.placeholderAudio')
    case 'mesh':
      return t('widgets.uploadSelect.placeholderMesh')
    case 'model':
      return t('widgets.uploadSelect.placeholderModel')
    case 'unknown':
      return t('widgets.uploadSelect.placeholderUnknown')
  }

  return t('widgets.uploadSelect.placeholder')
})

const uploadable = computed(() => {
  if (props.isAssetMode) return false
  return props.allowUpload === true
})

const acceptTypes = computed(() => {
  // Be permissive with accept types because backend uses libraries
  // that can handle a wide range of formats
  switch (props.assetKind) {
    case 'image':
      return 'image/*'
    case 'video':
      return 'video/*'
    case 'audio':
      return 'audio/*'
    case 'mesh':
      return SUPPORTED_EXTENSIONS_ACCEPT
    default:
      return undefined // model or unknown
  }
})

const layoutMode = ref<LayoutMode>(props.defaultLayoutMode ?? 'grid')

watch(
  [modelValue, displayItems],
  ([currentValue]) => {
    if (currentValue === undefined) {
      selectedSet.value.clear()
      return
    }

    const item = displayItems.value.find((item) => item.name === currentValue)
    if (!item) {
      selectedSet.value.clear()
      return
    }
    selectedSet.value.clear()
    selectedSet.value.add(item.id)
  },
  { immediate: true }
)

function updateSelectedItems(selectedItems: Set<string>) {
  let id: string | undefined = undefined
  if (selectedItems.size > 0) {
    id = selectedItems.values().next().value!
  }
  if (id == null) {
    modelValue.value = undefined
    return
  }
  const name = dropdownItems.value.find((item) => item.id === id)?.name
  if (!name) {
    modelValue.value = undefined
    return
  }
  modelValue.value = name
}

const uploadFile = async (
  file: File,
  isPasted: boolean = false,
  formFields: Partial<{ type: ResultItemType }> = {}
) => {
  const body = new FormData()
  body.append('image', file)
  if (isPasted) body.append('subfolder', 'pasted')
  else if (props.uploadSubfolder)
    body.append('subfolder', props.uploadSubfolder)
  if (formFields.type) body.append('type', formFields.type)

  const resp = await api.fetchApi('/upload/image', {
    method: 'POST',
    body
  })

  if (resp.status !== 200) {
    toastStore.addAlert(resp.status + ' - ' + resp.statusText)
    return null
  }

  const data = await resp.json()

  // Update AssetsStore when uploading to input folder
  if (formFields.type === 'input' || (!formFields.type && !isPasted)) {
    const assetsStore = useAssetsStore()
    await assetsStore.updateInputs()
  }

  return data.subfolder ? `${data.subfolder}/${data.name}` : data.name
}

const uploadFiles = async (files: File[]): Promise<string[]> => {
  const folder = props.uploadFolder ?? 'input'
  const uploadPromises = files.map((file) =>
    uploadFile(file, false, { type: folder })
  )
  const results = await Promise.all(uploadPromises)
  return results.filter((path): path is string => path !== null)
}

async function handleFilesUpdate(files: File[]) {
  if (!files || files.length === 0) return

  try {
    // 1. Upload files to server
    const uploadedPaths = await uploadFiles(files)

    if (uploadedPaths.length === 0) {
      toastStore.addAlert('File upload failed')
      return
    }

    // 2. Update widget options to include new files
    // This simulates what addToComboValues does but for SimplifiedWidget
    const values = props.widget.options?.values
    if (Array.isArray(values)) {
      uploadedPaths.forEach((path) => {
        if (!values.includes(path)) {
          values.push(path)
        }
      })
    }

    // 3. Update widget value to the first uploaded file
    modelValue.value = uploadedPaths[0]

    // 4. Trigger callback to notify underlying LiteGraph widget
    if (props.widget.callback) {
      props.widget.callback(uploadedPaths[0])
    }
  } catch (error) {
    console.error('Upload error:', error)
    toastStore.addAlert(`Upload failed: ${error}`)
  }
}

function getMediaUrl(
  filename: string,
  type: 'input' | 'output' = 'input'
): string {
  if (!['image', 'video'].includes(props.assetKind ?? '')) return ''
  return `/api/view?filename=${encodeURIComponent(filename)}&type=${type}`
}
</script>

<template>
  <WidgetLayoutField :widget>
    <FormDropdown
      v-model:selected="selectedSet"
      v-model:filter-selected="filterSelected"
      v-model:layout-mode="layoutMode"
      v-model:ownership-selected="ownershipSelected"
      v-model:base-model-selected="baseModelSelected"
      :items="dropdownItems"
      :display-items="displayItems"
      :placeholder="mediaPlaceholder"
      :multiple="false"
      :uploadable
      :accept="acceptTypes"
      :filter-options
      :show-ownership-filter
      :ownership-options
      :show-base-model-filter
      :base-model-options
      v-bind="combinedProps"
      class="w-full"
      @update:selected="updateSelectedItems"
      @update:files="handleFilesUpdate"
    />
  </WidgetLayoutField>
</template>
