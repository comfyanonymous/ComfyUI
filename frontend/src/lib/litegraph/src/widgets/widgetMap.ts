import type { LGraphNode } from '@/lib/litegraph/src/LGraphNode'
import type {
  IAssetWidget,
  IBaseWidget,
  IComboWidget,
  IWidget,
  TWidgetType
} from '@/lib/litegraph/src/types/widgets'
import { toClass } from '@/lib/litegraph/src/utils/type'

import { AssetWidget } from './AssetWidget'
import { BaseWidget } from './BaseWidget'
import { BooleanWidget } from './BooleanWidget'
import { BoundingBoxWidget } from './BoundingBoxWidget'
import { ButtonWidget } from './ButtonWidget'
import { ChartWidget } from './ChartWidget'
import { ColorWidget } from './ColorWidget'
import { ComboWidget } from './ComboWidget'
import { CurveWidget } from './CurveWidget'
import { FileUploadWidget } from './FileUploadWidget'
import { GalleriaWidget } from './GalleriaWidget'
import { GradientSliderWidget } from './GradientSliderWidget'
import { ImageCompareWidget } from './ImageCompareWidget'
import { PainterWidget } from './PainterWidget'
import { ImageCropWidget } from './ImageCropWidget'
import { KnobWidget } from './KnobWidget'
import { LegacyWidget } from './LegacyWidget'
import { MarkdownWidget } from './MarkdownWidget'
import { MultiSelectWidget } from './MultiSelectWidget'
import { NumberWidget } from './NumberWidget'
import { SelectButtonWidget } from './SelectButtonWidget'
import { SliderWidget } from './SliderWidget'
import { TextWidget } from './TextWidget'
import { TextareaWidget } from './TextareaWidget'
import { TreeSelectWidget } from './TreeSelectWidget'

export type WidgetTypeMap = {
  button: ButtonWidget
  toggle: BooleanWidget
  slider: SliderWidget
  gradientslider: GradientSliderWidget
  knob: KnobWidget
  combo: ComboWidget
  number: NumberWidget
  string: TextWidget
  text: TextWidget
  custom: LegacyWidget
  fileupload: FileUploadWidget
  color: ColorWidget
  markdown: MarkdownWidget
  treeselect: TreeSelectWidget
  multiselect: MultiSelectWidget
  chart: ChartWidget
  galleria: GalleriaWidget
  imagecompare: ImageCompareWidget
  selectbutton: SelectButtonWidget
  textarea: TextareaWidget
  asset: AssetWidget
  imagecrop: ImageCropWidget
  boundingbox: BoundingBoxWidget
  curve: CurveWidget
  painter: PainterWidget
  [key: string]: BaseWidget
}

/**
 * Convert a widget POJO to a proper widget instance.
 * @param widget The POJO to convert.
 * @param node The node the widget belongs to.
 * @param wrapLegacyWidgets Whether to wrap legacy widgets in a `LegacyWidget` instance.
 * @returns A concrete widget instance.
 */
export function toConcreteWidget<TWidget extends IWidget | IBaseWidget>(
  widget: TWidget,
  node: LGraphNode,
  wrapLegacyWidgets?: true
): WidgetTypeMap[TWidget['type']]
export function toConcreteWidget<TWidget extends IWidget | IBaseWidget>(
  widget: TWidget,
  node: LGraphNode,
  wrapLegacyWidgets: false
): WidgetTypeMap[TWidget['type']] | undefined
export function toConcreteWidget<TWidget extends IWidget | IBaseWidget>(
  widget: TWidget,
  node: LGraphNode,
  wrapLegacyWidgets = true
): WidgetTypeMap[TWidget['type']] | undefined {
  if (widget instanceof BaseWidget) return widget

  // Assertion: TypeScript has no concept of "all strings except X"
  type RemoveBaseWidgetType<T> = T extends { type: TWidgetType } ? T : never
  const narrowedWidget = widget as RemoveBaseWidgetType<TWidget>

  switch (narrowedWidget.type) {
    case 'button':
      return toClass(ButtonWidget, narrowedWidget, node)
    case 'toggle':
      return toClass(BooleanWidget, narrowedWidget, node)
    case 'slider':
      return toClass(SliderWidget, narrowedWidget, node)
    case 'gradientslider':
      return toClass(GradientSliderWidget, narrowedWidget, node)
    case 'knob':
      return toClass(KnobWidget, narrowedWidget, node)
    case 'combo':
      return toClass(ComboWidget, narrowedWidget, node)
    case 'number':
      return toClass(NumberWidget, narrowedWidget, node)
    case 'string':
      return toClass(TextWidget, narrowedWidget, node)
    case 'text':
      return toClass(TextWidget, narrowedWidget, node)
    case 'fileupload':
      return toClass(FileUploadWidget, narrowedWidget, node)
    case 'color':
      return toClass(ColorWidget, narrowedWidget, node)
    case 'markdown':
      return toClass(MarkdownWidget, narrowedWidget, node)
    case 'treeselect':
      return toClass(TreeSelectWidget, narrowedWidget, node)
    case 'multiselect':
      return toClass(MultiSelectWidget, narrowedWidget, node)
    case 'chart':
      return toClass(ChartWidget, narrowedWidget, node)
    case 'galleria':
      return toClass(GalleriaWidget, narrowedWidget, node)
    case 'imagecompare':
      return toClass(ImageCompareWidget, narrowedWidget, node)
    case 'selectbutton':
      return toClass(SelectButtonWidget, narrowedWidget, node)
    case 'textarea':
      return toClass(TextareaWidget, narrowedWidget, node)
    case 'asset':
      return toClass(AssetWidget, narrowedWidget, node)
    case 'imagecrop':
      return toClass(ImageCropWidget, narrowedWidget, node)
    case 'boundingbox':
      return toClass(BoundingBoxWidget, narrowedWidget, node)
    case 'curve':
      return toClass(CurveWidget, narrowedWidget, node)
    case 'painter':
      return toClass(PainterWidget, narrowedWidget, node)
    default: {
      if (wrapLegacyWidgets) return toClass(LegacyWidget, widget, node)
    }
  }
}

// #region Type Guards

/** Type guard: Narrow **from {@link IBaseWidget}** to {@link IComboWidget}. */
export function isComboWidget(widget: IBaseWidget): widget is IComboWidget {
  return widget.type === 'combo'
}

/**
 * Type guard: Narrow **from {@link IBaseWidget}** to {@link IAssetWidget}.
 * @knipIgnoreUnusedButUsedByCustomNodes
 */
export function isAssetWidget(widget: IBaseWidget): widget is IAssetWidget {
  return widget.type === 'asset'
}

// #endregion Type Guards
