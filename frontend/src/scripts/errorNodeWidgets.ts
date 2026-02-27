import { useChainCallback } from '@/composables/functional/useChainCallback'
import { LGraphNode } from '@/lib/litegraph/src/litegraph'
import type { IBaseWidget } from '@/lib/litegraph/src/types/widgets'
import { useBooleanWidget } from '@/renderer/extensions/vueNodes/widgets/composables/useBooleanWidget'
import { useFloatWidget } from '@/renderer/extensions/vueNodes/widgets/composables/useFloatWidget'
import { useStringWidget } from '@/renderer/extensions/vueNodes/widgets/composables/useStringWidget'

const StringWidget = useStringWidget()
const FloatWidget = useFloatWidget()
const BooleanWidget = useBooleanWidget()

function addWidgetFromValue(node: LGraphNode, value: unknown) {
  let widget: IBaseWidget

  if (typeof value === 'string') {
    widget = StringWidget(node, {
      type: 'STRING',
      name: 'UNKNOWN',
      multiline: value.length > 20
    })
  } else if (typeof value === 'number') {
    widget = FloatWidget(node, {
      type: 'FLOAT',
      name: 'UNKNOWN'
    })
  } else if (typeof value === 'boolean') {
    widget = BooleanWidget(node, {
      type: 'BOOLEAN',
      name: 'UNKNOWN'
    })
  } else {
    widget = StringWidget(node, {
      type: 'STRING',
      name: 'UNKNOWN',
      multiline: true
    })
    widget.value = JSON.stringify(value)
    return
  }

  widget.value = value
}

/**
 * Try add widgets to node with missing definition.
 */
LGraphNode.prototype.onConfigure = useChainCallback(
  LGraphNode.prototype.onConfigure,
  function (this: LGraphNode, info) {
    if (!this.has_errors || !info.widgets_values) return

    /**
     * Note: Some custom nodes overrides the `widgets_values` property to an
     * object that has `length` property and index access. It is not safe to call
     * any array methods on it.
     * See example in https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite/blob/8629188458dc6cb832f871ece3bd273507e8a766/web/js/VHS.core.js#L59-L84
     */
    for (let i = 0; i < info.widgets_values.length; i++) {
      const widgetValue = info.widgets_values[i]
      addWidgetFromValue(this, widgetValue)
    }

    this.serialize_widgets = true
  }
)
