import type { LGraphNode } from '@/lib/litegraph/src/litegraph'
import type { IColorWidget } from '@/lib/litegraph/src/types/widgets'
import type {
  ColorInputSpec,
  InputSpec as InputSpecV2
} from '@/schemas/nodeDef/nodeDefSchemaV2'
import type { ComfyWidgetConstructorV2 } from '@/scripts/widgets'

export const useColorWidget = (): ComfyWidgetConstructorV2 => {
  return (node: LGraphNode, inputSpec: InputSpecV2): IColorWidget => {
    const { name, options } = inputSpec as ColorInputSpec
    const defaultValue = options?.default || '#000000'

    const widget = node.addWidget('color', name, defaultValue, () => {}, {
      serialize: true
    }) as IColorWidget

    return widget
  }
}
