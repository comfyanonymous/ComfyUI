import type { LGraphNode } from '@/lib/litegraph/src/litegraph'
import type { ICurveWidget } from '@/lib/litegraph/src/types/widgets'
import type {
  CurveInputSpec,
  InputSpec as InputSpecV2
} from '@/schemas/nodeDef/nodeDefSchemaV2'
import type { ComfyWidgetConstructorV2 } from '@/scripts/widgets'

export const useCurveWidget = (): ComfyWidgetConstructorV2 => {
  return (node: LGraphNode, inputSpec: InputSpecV2): ICurveWidget => {
    const spec = inputSpec as CurveInputSpec
    const defaultValue = spec.default ?? [
      [0, 0],
      [1, 1]
    ]

    const rawWidget = node.addWidget(
      'curve',
      spec.name,
      [...defaultValue],
      () => {}
    )

    if (rawWidget.type !== 'curve') {
      throw new Error(`Unexpected widget type: ${rawWidget.type}`)
    }

    return rawWidget as ICurveWidget
  }
}
