import type { LGraphNode } from '@/lib/litegraph/src/LGraphNode'
import type { IAssetWidget } from '@/lib/litegraph/src/types/widgets'

import { BaseWidget } from './BaseWidget'
import type { DrawWidgetOptions } from './BaseWidget'

export class AssetWidget
  extends BaseWidget<IAssetWidget>
  implements IAssetWidget
{
  constructor(widget: IAssetWidget, node: LGraphNode) {
    super(widget, node)
    this.type ??= 'asset'
    this.value = widget.value?.toString() ?? ''
  }

  override set value(value: IAssetWidget['value']) {
    const oldValue = this.value
    super.value = value

    // Force canvas redraw when value changes to show update immediately
    if (oldValue !== value && this.node.graph?.list_of_graphcanvas) {
      for (const canvas of this.node.graph.list_of_graphcanvas) {
        canvas.setDirty(true)
      }
    }
  }

  override get value(): IAssetWidget['value'] {
    return super.value
  }

  override get _displayValue(): string {
    return String(this.value) //FIXME: Resolve asset name
  }

  override drawWidget(
    ctx: CanvasRenderingContext2D,
    options: DrawWidgetOptions
  ) {
    const { width, showText = true } = options
    // Store original context attributes
    const { fillStyle, strokeStyle, textAlign } = ctx

    this.drawWidgetShape(ctx, options)

    if (showText) {
      this.drawTruncatingText({ ctx, width, leftPadding: 0, rightPadding: 0 })
    }

    // Restore original context attributes
    Object.assign(ctx, { textAlign, strokeStyle, fillStyle })
  }

  override onClick() {
    //Open Modal
    this.options.openModal(this)
  }
}
