import type { LGraphNode } from '@/lib/litegraph/src/litegraph'
import type { IGalleriaWidget } from '@/lib/litegraph/src/types/widgets'
import type {
  GalleriaInputSpec,
  InputSpec as InputSpecV2
} from '@/schemas/nodeDef/nodeDefSchemaV2'
import type { ComfyWidgetConstructorV2 } from '@/scripts/widgets'

export const useGalleriaWidget = (): ComfyWidgetConstructorV2 => {
  return (node: LGraphNode, inputSpec: InputSpecV2): IGalleriaWidget => {
    const { name, options = {} } = inputSpec as GalleriaInputSpec

    const widget = node.addWidget(
      'galleria',
      name,
      options.images || [],
      () => {},
      {
        serialize: true,
        ...options
      }
    ) as IGalleriaWidget

    return widget
  }
}
