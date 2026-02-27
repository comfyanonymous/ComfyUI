import type { LGraphNode } from '@/lib/litegraph/src/litegraph'
import type { IChartWidget } from '@/lib/litegraph/src/types/widgets'
import { isChartInputSpec } from '@/schemas/nodeDef/nodeDefSchemaV2'
import type {
  ChartInputSpec,
  InputSpec as InputSpecV2
} from '@/schemas/nodeDef/nodeDefSchemaV2'
import type { ComfyWidgetConstructorV2 } from '@/scripts/widgets'

export const useChartWidget = (): ComfyWidgetConstructorV2 => {
  return (node: LGraphNode, inputSpec: InputSpecV2): IChartWidget => {
    if (!isChartInputSpec(inputSpec)) {
      throw new Error('Invalid input spec for chart widget')
    }

    const { name, options = {} } = inputSpec as ChartInputSpec

    const chartType = options.type || 'line'

    const widget = node.addWidget('chart', name, options.data || {}, () => {}, {
      serialize: true,
      type: chartType,
      ...options
    }) as IChartWidget

    return widget
  }
}
