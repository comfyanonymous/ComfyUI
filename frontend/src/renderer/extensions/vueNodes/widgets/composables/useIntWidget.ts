import type { LGraphNode } from '@/lib/litegraph/src/litegraph'
import type { INumericWidget } from '@/lib/litegraph/src/types/widgets'
import { useSettingStore } from '@/platform/settings/settingStore'
import type { InputSpec } from '@/schemas/nodeDef/nodeDefSchemaV2'
import { isIntInputSpec } from '@/schemas/nodeDef/nodeDefSchemaV2'
import type { ComfyWidgetConstructorV2 } from '@/scripts/widgets'
import { addValueControlWidget } from '@/scripts/widgets'
import { transformInputSpecV2ToV1 } from '@/schemas/nodeDef/migration'

function onValueChange(this: INumericWidget, v: number) {
  // For integers, always round to the nearest step
  // step === 0 is invalid, assign 1 if options.step is 0
  const step = this.options.step2 || 1

  if (step === 1) {
    // Simple case: round to nearest integer
    this.value = Math.round(v)
  } else {
    // Round to nearest multiple of step
    // First, determine if min value creates an offset
    const min = this.options.min ?? 0
    const offset = min % step

    // Round to nearest step, accounting for offset
    this.value = Math.round((v - offset) / step) * step + offset
  }
}

export const _for_testing = {
  onValueChange
}

export const useIntWidget = () => {
  const widgetConstructor: ComfyWidgetConstructorV2 = (
    node: LGraphNode,
    inputSpec: InputSpec
  ) => {
    if (!isIntInputSpec(inputSpec)) {
      throw new Error(`Invalid input data: ${inputSpec}`)
    }

    const settingStore = useSettingStore()
    const sliderEnabled = !settingStore.get('Comfy.DisableSliders')
    const display_type = inputSpec.display
    const widgetType =
      sliderEnabled && display_type == 'slider'
        ? 'slider'
        : display_type == 'knob'
          ? 'knob'
          : 'number'

    const step = inputSpec.step ?? 1
    /** Assertion {@link inputSpec.default} */
    const defaultValue = (inputSpec.default as number | undefined) ?? 0
    const widget = node.addWidget(
      widgetType,
      inputSpec.name,
      defaultValue,
      onValueChange,
      {
        min: inputSpec.min ?? 0,
        max: inputSpec.max ?? 2048,
        /** @deprecated Use step2 instead. The 10x value is a legacy implementation. */
        step: step * 10,
        step2: step,
        precision: 0
      }
    )

    const controlAfterGenerate =
      inputSpec.control_after_generate ??
      ['seed', 'noise_seed'].includes(inputSpec.name)

    if (controlAfterGenerate) {
      const defaultType =
        typeof inputSpec.control_after_generate === 'string'
          ? inputSpec.control_after_generate
          : 'randomize'
      const controlWidget = addValueControlWidget(
        node,
        widget,
        defaultType,
        undefined,
        undefined,
        transformInputSpecV2ToV1(inputSpec)
      )
      widget.linkedWidgets = [controlWidget]
    }

    return widget
  }
  return widgetConstructor
}
