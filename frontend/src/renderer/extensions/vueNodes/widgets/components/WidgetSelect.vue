<template>
  <WidgetSelectDropdown
    v-if="isDropdownUIWidget"
    v-model="modelValue"
    :widget
    :node-type="widget.nodeType ?? nodeType"
    :asset-kind="assetKind"
    :allow-upload="allowUpload"
    :upload-folder="uploadFolder"
    :upload-subfolder="uploadSubfolder"
    :is-asset-mode="isAssetMode"
    :default-layout-mode="defaultLayoutMode"
  />
  <WidgetWithControl
    v-else-if="widget.controlWidget"
    v-model="modelValue"
    :component="WidgetSelectDefault"
    :widget="widget as StringControlWidget"
  />
  <WidgetSelectDefault v-else v-model="modelValue" :widget />
</template>

<script setup lang="ts">
import { computed } from 'vue'

import { assetService } from '@/platform/assets/services/assetService'
import WidgetSelectDefault from '@/renderer/extensions/vueNodes/widgets/components/WidgetSelectDefault.vue'
import WidgetSelectDropdown from '@/renderer/extensions/vueNodes/widgets/components/WidgetSelectDropdown.vue'
import WidgetWithControl from '@/renderer/extensions/vueNodes/widgets/components/WidgetWithControl.vue'
import type { LayoutMode } from '@/renderer/extensions/vueNodes/widgets/components/form/dropdown/types'
import type { ResultItemType } from '@/schemas/apiSchema'
import { isComboInputSpec } from '@/schemas/nodeDef/nodeDefSchemaV2'
import type { ComboInputSpec } from '@/schemas/nodeDef/nodeDefSchemaV2'
import type {
  SimplifiedControlWidget,
  SimplifiedWidget
} from '@/types/simplifiedWidget'
import type { AssetKind } from '@/types/widgetTypes'

type StringControlWidget = SimplifiedControlWidget<string | undefined>

const props = defineProps<{
  widget: SimplifiedWidget<string | undefined>
  nodeType?: string
}>()

const modelValue = defineModel<string | undefined>()

const comboSpec = computed<ComboInputSpec | undefined>(() => {
  if (props.widget.spec && isComboInputSpec(props.widget.spec)) {
    return props.widget.spec
  }
  return undefined
})

const specDescriptor = computed<{
  kind: AssetKind
  allowUpload: boolean
  folder: ResultItemType | undefined
  subfolder: string | undefined
}>(() => {
  const spec = comboSpec.value
  if (!spec) {
    return {
      kind: 'unknown',
      allowUpload: false,
      folder: undefined,
      subfolder: undefined
    }
  }

  const {
    image_upload,
    animated_image_upload,
    video_upload,
    image_folder,
    audio_upload,
    mesh_upload,
    upload_subfolder
  } = spec

  let kind: AssetKind = 'unknown'
  if (video_upload) {
    kind = 'video'
  } else if (image_upload || animated_image_upload) {
    kind = 'image'
  } else if (audio_upload) {
    kind = 'audio'
  } else if (mesh_upload) {
    kind = 'mesh'
  }

  // TODO: add support for models (checkpoints, VAE, LoRAs, etc.) -- get widgetType from spec

  const allowUpload =
    image_upload === true ||
    animated_image_upload === true ||
    video_upload === true ||
    audio_upload === true ||
    mesh_upload === true

  const folder = mesh_upload ? 'input' : image_folder

  return {
    kind,
    allowUpload,
    folder,
    subfolder: upload_subfolder
  }
})

const isAssetMode = computed(
  () =>
    assetService.shouldUseAssetBrowser(props.nodeType, props.widget.name) ||
    (assetService.isAssetAPIEnabled() && props.widget.type === 'asset')
)

const assetKind = computed(() => specDescriptor.value.kind)
const isDropdownUIWidget = computed(
  () => isAssetMode.value || assetKind.value !== 'unknown'
)
const allowUpload = computed(() => specDescriptor.value.allowUpload)
const uploadFolder = computed<ResultItemType>(() => {
  return specDescriptor.value.folder ?? 'input'
})
const uploadSubfolder = computed(() => specDescriptor.value.subfolder)
const defaultLayoutMode = computed<LayoutMode>(() => {
  return isAssetMode.value ? 'list' : 'grid'
})
</script>
