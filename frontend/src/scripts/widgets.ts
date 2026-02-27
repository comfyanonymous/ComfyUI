import { t } from '@/i18n'
import { type LGraphNode, isComboWidget } from '@/lib/litegraph/src/litegraph'
import type {
  IBaseWidget,
  IComboWidget,
  IStringWidget
} from '@/lib/litegraph/src/types/widgets'
import { useSettingStore } from '@/platform/settings/settingStore'
import { dynamicWidgets } from '@/core/graph/widgets/dynamicWidgets'
import { useBooleanWidget } from '@/renderer/extensions/vueNodes/widgets/composables/useBooleanWidget'
import { useBoundingBoxWidget } from '@/renderer/extensions/vueNodes/widgets/composables/useBoundingBoxWidget'
import { useCurveWidget } from '@/renderer/extensions/vueNodes/widgets/composables/useCurveWidget'
import { useChartWidget } from '@/renderer/extensions/vueNodes/widgets/composables/useChartWidget'
import { useColorWidget } from '@/renderer/extensions/vueNodes/widgets/composables/useColorWidget'
import { useComboWidget } from '@/renderer/extensions/vueNodes/widgets/composables/useComboWidget'
import { useFloatWidget } from '@/renderer/extensions/vueNodes/widgets/composables/useFloatWidget'
import { useGalleriaWidget } from '@/renderer/extensions/vueNodes/widgets/composables/useGalleriaWidget'
import { useImageCompareWidget } from '@/renderer/extensions/vueNodes/widgets/composables/useImageCompareWidget'
import { useImageUploadWidget } from '@/renderer/extensions/vueNodes/widgets/composables/useImageUploadWidget'
import { useIntWidget } from '@/renderer/extensions/vueNodes/widgets/composables/useIntWidget'
import { useMarkdownWidget } from '@/renderer/extensions/vueNodes/widgets/composables/useMarkdownWidget'
import { usePainterWidget } from '@/renderer/extensions/vueNodes/widgets/composables/usePainterWidget'
import { useStringWidget } from '@/renderer/extensions/vueNodes/widgets/composables/useStringWidget'
import { useTextareaWidget } from '@/renderer/extensions/vueNodes/widgets/composables/useTextareaWidget'
import { transformInputSpecV1ToV2 } from '@/schemas/nodeDef/migration'
import type { InputSpec as InputSpecV2 } from '@/schemas/nodeDef/nodeDefSchemaV2'
import type { InputSpec } from '@/schemas/nodeDefSchema'

import type { ComfyApp } from './app'
import './domWidget'
import './errorNodeWidgets'

export type ComfyWidgetConstructorV2 = (
  node: LGraphNode,
  inputSpec: InputSpecV2
) => IBaseWidget

export type ComfyWidgetConstructor = (
  node: LGraphNode,
  inputName: string,
  inputData: InputSpec,
  app: ComfyApp,
  widgetName?: string
) => { widget: IBaseWidget; minWidth?: number; minHeight?: number }

/**
 * Transforms a V2 widget constructor to a V1 widget constructor.
 * @param widgetConstructorV2 The V2 widget constructor to transform.
 * @returns The transformed V1 widget constructor.
 */
const transformWidgetConstructorV2ToV1 = (
  widgetConstructorV2: ComfyWidgetConstructorV2
): ComfyWidgetConstructor => {
  return (node, inputName, inputData) => {
    const inputSpec = transformInputSpecV1ToV2(inputData, {
      name: inputName
    })
    const widget = widgetConstructorV2(node, inputSpec)
    return {
      widget,
      minWidth: widget.options.minNodeSize?.[0],
      minHeight: widget.options.minNodeSize?.[1]
    }
  }
}

function controlValueRunBefore() {
  return useSettingStore().get('Comfy.WidgetControlMode') === 'before'
}

export function updateControlWidgetLabel(widget: IBaseWidget) {
  if (controlValueRunBefore()) {
    widget.label = t('g.control_before_generate')
  } else {
    widget.label = t('g.control_after_generate')
  }
}

export const IS_CONTROL_WIDGET = Symbol()
const HAS_EXECUTED = Symbol()

export function addValueControlWidget(
  node: LGraphNode,
  targetWidget: IBaseWidget,
  defaultValue?: string,
  _values?: unknown,
  widgetName?: string,
  inputData?: InputSpec
): IComboWidget {
  let name = inputData?.[1]?.control_after_generate
  if (typeof name !== 'string') {
    name = widgetName
  }
  const widgets = addValueControlWidgets(
    node,
    targetWidget,
    defaultValue ?? 'randomize',
    {
      addFilterList: false,
      controlAfterGenerateName: name
    },
    inputData
  )
  return widgets[0]
}

export function addValueControlWidgets(
  node: LGraphNode,
  targetWidget: IBaseWidget,
  defaultValue?: string,
  options?: Record<string, any>,
  inputData?: InputSpec
): [IComboWidget, ...IStringWidget[]] {
  if (!defaultValue) defaultValue = 'randomize'
  if (!options) options = {}

  const getName = (defaultName: string, optionName: string) => {
    let name = defaultName
    if (options[optionName]) {
      name = options[optionName]
    } else if (typeof inputData?.[1]?.[defaultName] === 'string') {
      name = inputData?.[1]?.[defaultName]
    } else if (inputData?.[1]?.control_prefix) {
      name = inputData?.[1]?.control_prefix + ' ' + name
    }
    return name
  }

  const valueControl = node.addWidget(
    'combo',
    getName('control_after_generate', 'controlAfterGenerateName'),
    defaultValue,
    function () {},
    {
      values: ['fixed', 'increment', 'decrement', 'randomize'],
      serialize: false, // Don't include this in prompt.
      canvasOnly: true
    }
  ) as IComboWidget

  valueControl.tooltip =
    'Allows the linked widget to be changed automatically, for example randomizing the noise seed.'
  valueControl[IS_CONTROL_WIDGET] = true
  updateControlWidgetLabel(valueControl)
  Object.defineProperty(valueControl, 'disabled', {
    get: () => targetWidget.computedDisabled
  })
  const widgets: [IComboWidget, ...IStringWidget[]] = [valueControl]

  const isCombo = isComboWidget(targetWidget)
  let comboFilter: IStringWidget
  if (isCombo && valueControl.options.values) {
    // @ts-expect-error Combo widget values may be a dictionary or legacy function type
    valueControl.options.values.push('increment-wrap')
  }
  if (isCombo && options.addFilterList !== false) {
    comboFilter = node.addWidget(
      'string',
      getName('control_filter_list', 'controlFilterListName'),
      '',
      function () {},
      {
        serialize: false // Don't include this in prompt.
      }
    ) as IStringWidget
    updateControlWidgetLabel(comboFilter)
    comboFilter.tooltip =
      "Allows for filtering the list of values when changing the value via the control generate mode. Allows for RegEx matches in the format /abc/ to only filter to values containing 'abc'."
    Object.defineProperty(comboFilter, 'disabled', {
      get: () => targetWidget.computedDisabled
    })

    widgets.push(comboFilter)
  }

  const applyWidgetControl = () => {
    var v = valueControl.value

    if (isCombo && v !== 'fixed') {
      let values = targetWidget.options.values ?? []
      const filter = comboFilter?.value
      if (filter) {
        let check
        if (filter.startsWith('/') && filter.endsWith('/')) {
          try {
            const regex = new RegExp(filter.substring(1, filter.length - 1))
            check = (item: string) => regex.test(item)
          } catch (error) {
            console.error(
              'Error constructing RegExp filter for node ' + node.id,
              filter,
              error
            )
          }
        }
        if (!check) {
          const lower = filter.toLocaleLowerCase()
          check = (item: string) => item.toLocaleLowerCase().includes(lower)
        }
        // @ts-expect-error Combo widget values may be a dictionary or legacy function type
        values = values.filter((item: string) => check(item))
        if (!values.length && targetWidget.options.values?.length) {
          console.warn(
            'Filter for node ' + node.id + ' has filtered out all items',
            filter
          )
        }
      }
      // @ts-expect-error targetWidget.value can be number or string
      let current_index = values.indexOf(targetWidget.value)
      let current_length = values.length

      switch (v) {
        case 'increment':
          current_index += 1
          break
        case 'increment-wrap':
          current_index += 1
          if (current_index >= current_length) {
            current_index = 0
          }
          break
        case 'decrement':
          current_index -= 1
          break
        case 'randomize':
          // @ts-expect-error Combo widget values may be a dictionary or legacy function type
          current_index = Math.floor(Math.random() * current_length)
          break
        default:
          break
      }
      current_index = Math.max(0, current_index)
      // @ts-expect-error Combo widget values may be a dictionary or legacy function type
      current_index = Math.min(current_length - 1, current_index)
      if (current_index >= 0) {
        // @ts-expect-error Combo widget values may be a dictionary or legacy function type
        let value = values[current_index]
        targetWidget.value = value
        targetWidget.callback?.(value)
      }
    } else {
      //number
      let { min = 0, max = 1, step2 = 1 } = targetWidget.options
      // limit to something that javascript can handle
      max = Math.min(1125899906842624, max)
      min = Math.max(-1125899906842624, min)
      let range = (max - min) / step2

      //adjust values based on valueControl Behaviour
      switch (v) {
        case 'fixed':
          break
        case 'increment':
          // @ts-expect-error targetWidget.value can be number or string
          targetWidget.value += step2
          break
        case 'decrement':
          // @ts-expect-error targetWidget.value can be number or string
          targetWidget.value -= step2
          break
        case 'randomize':
          targetWidget.value = Math.floor(Math.random() * range) * step2 + min
          break
        default:
          break
      }
      /*check if values are over or under their respective
       * ranges and set them to min or max.*/
      // @ts-expect-error targetWidget.value can be number or string
      if (targetWidget.value < min) targetWidget.value = min
      // @ts-expect-error targetWidget.value can be number or string
      if (targetWidget.value > max) targetWidget.value = max
      targetWidget.callback?.(targetWidget.value)
    }
  }

  valueControl.beforeQueued = ({ isPartialExecution } = {}) => {
    if (!isPartialExecution && controlValueRunBefore()) {
      // Don't run on first execution
      if (valueControl[HAS_EXECUTED]) {
        applyWidgetControl()
      }
    }
    valueControl[HAS_EXECUTED] = true
  }

  valueControl.afterQueued = ({ isPartialExecution } = {}) => {
    if (!isPartialExecution && !controlValueRunBefore()) {
      applyWidgetControl()
    }
  }

  return widgets
}

export const ComfyWidgets = {
  INT: transformWidgetConstructorV2ToV1(useIntWidget()),
  FLOAT: transformWidgetConstructorV2ToV1(useFloatWidget()),
  BOOLEAN: transformWidgetConstructorV2ToV1(useBooleanWidget()),
  STRING: transformWidgetConstructorV2ToV1(useStringWidget()),
  MARKDOWN: transformWidgetConstructorV2ToV1(useMarkdownWidget()),
  COMBO: transformWidgetConstructorV2ToV1(useComboWidget()),
  IMAGEUPLOAD: useImageUploadWidget(),
  COLOR: transformWidgetConstructorV2ToV1(useColorWidget()),
  IMAGECOMPARE: transformWidgetConstructorV2ToV1(useImageCompareWidget()),
  BOUNDING_BOX: transformWidgetConstructorV2ToV1(useBoundingBoxWidget()),
  CHART: transformWidgetConstructorV2ToV1(useChartWidget()),
  GALLERIA: transformWidgetConstructorV2ToV1(useGalleriaWidget()),
  PAINTER: transformWidgetConstructorV2ToV1(usePainterWidget()),
  TEXTAREA: transformWidgetConstructorV2ToV1(useTextareaWidget()),
  CURVE: transformWidgetConstructorV2ToV1(useCurveWidget()),
  ...dynamicWidgets
} as const

export function isValidWidgetType(
  key: unknown
): key is keyof typeof ComfyWidgets {
  return ComfyWidgets[key as keyof typeof ComfyWidgets] !== undefined
}
