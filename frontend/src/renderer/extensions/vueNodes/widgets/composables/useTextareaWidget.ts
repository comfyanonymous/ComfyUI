import type { LGraphNode } from '@/lib/litegraph/src/litegraph'
import type { ITextareaWidget } from '@/lib/litegraph/src/types/widgets'
import type {
  InputSpec as InputSpecV2,
  TextareaInputSpec
} from '@/schemas/nodeDef/nodeDefSchemaV2'
import type { ComfyWidgetConstructorV2 } from '@/scripts/widgets'

export const useTextareaWidget = (): ComfyWidgetConstructorV2 => {
  return (node: LGraphNode, inputSpec: InputSpecV2): ITextareaWidget => {
    const { name, options = {} } = inputSpec as TextareaInputSpec

    const widget = node.addWidget(
      'textarea',
      name,
      options.default || '',
      () => {},
      {
        serialize: true,
        rows: options.rows || 5,
        cols: options.cols || 50,
        ...options
      }
    ) as ITextareaWidget

    return widget
  }
}
