import { ref } from 'vue'

import MultiSelectWidget from '@/components/graph/widgets/MultiSelectWidget.vue'
import { t } from '@/i18n'
import type { LGraphNode } from '@/lib/litegraph/src/litegraph'
import { isComboWidget } from '@/lib/litegraph/src/litegraph'
import type { IBaseWidget } from '@/lib/litegraph/src/types/widgets'
import { assetService } from '@/platform/assets/services/assetService'
import { createAssetWidget } from '@/platform/assets/utils/createAssetWidget'
import { isCloud } from '@/platform/distribution/types'
import type {
  ComboInputSpec,
  InputSpec
} from '@/schemas/nodeDef/nodeDefSchemaV2'
import { isComboInputSpec } from '@/schemas/nodeDef/nodeDefSchemaV2'
import { transformInputSpecV2ToV1 } from '@/schemas/nodeDef/migration'
import { ComponentWidgetImpl, addWidget } from '@/scripts/domWidget'
import type { BaseDOMWidget } from '@/scripts/domWidget'
import type { ComfyWidgetConstructorV2 } from '@/scripts/widgets'
import { addValueControlWidgets } from '@/scripts/widgets'
import { useAssetsStore } from '@/stores/assetsStore'
import { getMediaTypeFromFilename } from '@/utils/formatUtil'

import { useRemoteWidget } from './useRemoteWidget'

const getDefaultValue = (inputSpec: ComboInputSpec) => {
  if (inputSpec.default) return inputSpec.default
  if (inputSpec.options?.length) return inputSpec.options[0]
  if (inputSpec.remote) return 'Loading...'
  return undefined
}

// Map node types to expected media types
const NODE_MEDIA_TYPE_MAP: Record<string, 'image' | 'video' | 'audio'> = {
  LoadImage: 'image',
  LoadVideo: 'video',
  LoadAudio: 'audio'
}

// Map node types to placeholder i18n keys
const NODE_PLACEHOLDER_MAP: Record<string, string> = {
  LoadImage: 'widgets.uploadSelect.placeholderImage',
  LoadVideo: 'widgets.uploadSelect.placeholderVideo',
  LoadAudio: 'widgets.uploadSelect.placeholderAudio'
}

const addMultiSelectWidget = (
  node: LGraphNode,
  inputSpec: ComboInputSpec
): IBaseWidget => {
  const widgetValue = ref<string[]>([])
  const widget = new ComponentWidgetImpl({
    node,
    name: inputSpec.name,
    component: MultiSelectWidget,
    inputSpec,
    options: {
      getValue: () => widgetValue.value,
      setValue: (value: string[]) => {
        widgetValue.value = value
      }
    }
  })
  addWidget(node, widget as BaseDOMWidget<object | string>)
  // TODO: Add remote support to multi-select widget
  // https://github.com/Comfy-Org/ComfyUI_frontend/issues/3003
  if (inputSpec.control_after_generate) {
    const defaultType =
      typeof inputSpec.control_after_generate === 'string'
        ? inputSpec.control_after_generate
        : 'fixed'
    widget.linkedWidgets = addValueControlWidgets(
      node,
      widget,
      defaultType,
      undefined,
      transformInputSpecV2ToV1(inputSpec)
    )
  }

  return widget
}

function createAssetBrowserWidget(
  node: LGraphNode,
  inputSpec: ComboInputSpec,
  defaultValue: string | undefined
): IBaseWidget {
  return createAssetWidget({
    node,
    widgetName: inputSpec.name,
    nodeTypeForBrowser: node.comfyClass ?? '',
    inputNameForBrowser: inputSpec.name,
    defaultValue,
    onValueChange: (widget, newValue, oldValue) => {
      node.onWidgetChanged?.(widget.name, newValue, oldValue, widget)
    }
  })
}

const createInputMappingWidget = (
  node: LGraphNode,
  inputSpec: ComboInputSpec,
  defaultValue: string | undefined
): IBaseWidget => {
  const assetsStore = useAssetsStore()

  const widget = node.addWidget(
    'combo',
    inputSpec.name,
    defaultValue ?? '',
    () => {},
    {
      values: [],
      getOptionLabel: (value?: string | null) => {
        if (!value) {
          const placeholderKey =
            NODE_PLACEHOLDER_MAP[node.comfyClass ?? ''] ??
            'widgets.uploadSelect.placeholder'
          return t(placeholderKey)
        }
        return assetsStore.getInputName(value)
      }
    }
  )

  if (assetsStore.inputAssets.length === 0 && !assetsStore.inputLoading) {
    void assetsStore.updateInputs().then(() => {
      // edge for users using nodes with 0 prior inputs
      // force canvas refresh the first time they add an asset
      // so they see filenames instead of hashes.
      node.setDirtyCanvas(true, false)
    })
  }

  const origOptions = widget.options
  widget.options = new Proxy(origOptions, {
    get(target, prop) {
      if (prop !== 'values') {
        return target[prop as keyof typeof target]
      }
      return assetsStore.inputAssets
        .filter(
          (asset) =>
            getMediaTypeFromFilename(asset.name) ===
            NODE_MEDIA_TYPE_MAP[node.comfyClass ?? '']
        )
        .map((asset) => asset.asset_hash)
        .filter((hash): hash is string => !!hash)
    }
  })

  if (inputSpec.control_after_generate) {
    if (!isComboWidget(widget)) {
      throw new Error(`Expected combo widget but received ${widget.type}`)
    }
    const defaultType =
      typeof inputSpec.control_after_generate === 'string'
        ? inputSpec.control_after_generate
        : 'randomize'
    widget.linkedWidgets = addValueControlWidgets(
      node,
      widget,
      defaultType,
      undefined,
      transformInputSpecV2ToV1(inputSpec)
    )
  }

  return widget
}

const addComboWidget = (
  node: LGraphNode,
  inputSpec: ComboInputSpec
): IBaseWidget => {
  const defaultValue = getDefaultValue(inputSpec)

  if (isCloud) {
    if (assetService.shouldUseAssetBrowser(node.comfyClass, inputSpec.name)) {
      return createAssetBrowserWidget(node, inputSpec, defaultValue)
    }

    if (NODE_MEDIA_TYPE_MAP[node.comfyClass ?? '']) {
      return createInputMappingWidget(node, inputSpec, defaultValue)
    }
  }

  // Standard combo widget
  const widget = node.addWidget(
    'combo',
    inputSpec.name,
    defaultValue,
    () => {},
    {
      values: inputSpec.options ?? []
    }
  )

  if (inputSpec.remote) {
    if (!isComboWidget(widget)) {
      throw new Error(`Expected combo widget but received ${widget.type}`)
    }

    const remoteWidget = useRemoteWidget({
      remoteConfig: inputSpec.remote,
      defaultValue,
      node,
      widget
    })
    if (inputSpec.remote.refresh_button) remoteWidget.addRefreshButton()

    const origOptions = widget.options
    widget.options = new Proxy(origOptions, {
      get(target, prop) {
        // Assertion: Proxy handler passthrough
        return prop !== 'values'
          ? target[prop as keyof typeof target]
          : remoteWidget.getValue()
      }
    })
  }

  if (inputSpec.control_after_generate) {
    if (!isComboWidget(widget)) {
      throw new Error(`Expected combo widget but received ${widget.type}`)
    }

    const defaultType =
      typeof inputSpec.control_after_generate === 'string'
        ? inputSpec.control_after_generate
        : 'randomize'
    widget.linkedWidgets = addValueControlWidgets(
      node,
      widget,
      defaultType,
      undefined,
      transformInputSpecV2ToV1(inputSpec)
    )
  }

  return widget
}

export const useComboWidget = () => {
  const widgetConstructor: ComfyWidgetConstructorV2 = (
    node: LGraphNode,
    inputSpec: InputSpec
  ) => {
    if (!isComboInputSpec(inputSpec)) {
      throw new Error(`Invalid input data: ${inputSpec}`)
    }
    return inputSpec.multi_select
      ? addMultiSelectWidget(node, inputSpec)
      : addComboWidget(node, inputSpec)
  }

  return widgetConstructor
}
