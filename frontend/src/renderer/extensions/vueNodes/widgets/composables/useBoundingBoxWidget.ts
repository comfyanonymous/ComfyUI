import type { LGraphNode } from '@/lib/litegraph/src/litegraph'
import type {
  IBaseWidget,
  IBoundingBoxWidget,
  IImageCropWidget,
  INumericWidget
} from '@/lib/litegraph/src/types/widgets'
import type { Bounds } from '@/renderer/core/layout/types'
import type {
  BoundingBoxInputSpec,
  InputSpec as InputSpecV2
} from '@/schemas/nodeDef/nodeDefSchemaV2'
import type { ComfyWidgetConstructorV2 } from '@/scripts/widgets'

function isBoundingBoxLikeWidget(
  widget: IBaseWidget
): widget is IBoundingBoxWidget | IImageCropWidget {
  return widget.type === 'boundingbox' || widget.type === 'imagecrop'
}

function isNumericWidget(widget: IBaseWidget): widget is INumericWidget {
  return widget.type === 'number'
}

export const useBoundingBoxWidget = (): ComfyWidgetConstructorV2 => {
  return (
    node: LGraphNode,
    inputSpec: InputSpecV2
  ): IBoundingBoxWidget | IImageCropWidget => {
    const spec = inputSpec as BoundingBoxInputSpec
    const { name, component } = spec
    const defaultValue: Bounds = spec.default ?? {
      x: 0,
      y: 0,
      width: 512,
      height: 512
    }

    const widgetType = component === 'ImageCrop' ? 'imagecrop' : 'boundingbox'

    const fields: (keyof Bounds)[] = ['x', 'y', 'width', 'height']
    const subWidgets: INumericWidget[] = []

    const rawWidget = node.addWidget(
      widgetType,
      name,
      { ...defaultValue },
      () => {
        for (let i = 0; i < fields.length; i++) {
          const field = fields[i]
          const subWidget = subWidgets[i]
          if (subWidget) {
            subWidget.value = widget.value[field]
          }
        }
      },
      {
        serialize: true,
        canvasOnly: false
      }
    )

    if (!isBoundingBoxLikeWidget(rawWidget)) {
      throw new Error(`Unexpected widget type: ${rawWidget.type}`)
    }

    const widget = rawWidget

    for (const field of fields) {
      const subWidget = node.addWidget(
        'number',
        field,
        defaultValue[field],
        function (this: INumericWidget, v: number) {
          this.value = Math.round(v)
          widget.value[field] = this.value
          widget.callback?.(widget.value)
        },
        {
          min: field === 'width' || field === 'height' ? 1 : 0,
          max: 8192,
          step: 10,
          step2: 1,
          precision: 0,
          serialize: false,
          canvasOnly: true
        }
      )

      if (!isNumericWidget(subWidget)) {
        throw new Error(`Unexpected widget type: ${subWidget.type}`)
      }

      subWidgets.push(subWidget)
    }

    widget.linkedWidgets = subWidgets

    return widget
  }
}
