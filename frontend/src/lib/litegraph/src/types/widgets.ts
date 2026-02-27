import type { Bounds } from '@/renderer/core/layout/types'
import type { CurvePoint } from '@/components/curve/types'

import type {
  CanvasColour,
  ColorStop,
  Point,
  RequiredProps,
  Size
} from '../interfaces'
import type {
  CanvasPointer,
  LGraphCanvas,
  LGraphNode,
  NodeId
} from '../litegraph'
import type { CanvasPointerEvent } from './events'

export interface NodeBindable {
  setNodeId(nodeId: NodeId): void
}

export interface IWidgetOptions<TValues = unknown> {
  on?: string
  off?: string
  max?: number
  min?: number
  slider_color?: CanvasColour
  marker_color?: CanvasColour
  precision?: number
  read_only?: boolean
  /**
   * @deprecated Use {@link IWidgetOptions.step2} instead.
   * The legacy step is scaled up by 10x in the legacy frontend logic.
   */
  step?: number
  /** The step value for numeric widgets. */
  step2?: number

  y?: number
  multiline?: boolean
  // TODO: Confirm this
  property?: string
  /** If `true`, an input socket will not be created for this widget. */
  socketless?: boolean
  /** If `true`, the widget will not be rendered by the Vue renderer. */
  canvasOnly?: boolean
  /** Used as a temporary override for determining the asset type in vue mode*/
  nodeType?: string

  /**
   * Whether the widget value should be included in the API prompt sent to
   * the backend for execution. Checked by {@link executionUtil} when
   * building the prompt payload.
   *
   * This is distinct from {@link IBaseWidget.serialize}, which controls
   * whether the value is persisted in the workflow JSON file.
   *
   * @default true
   * @see IBaseWidget.serialize — workflow persistence
   */
  serialize?: boolean

  values?: TValues
  /** Optional function to format values for display (e.g., hash → human-readable name) */
  getOptionLabel?: (value?: string | null) => string
  callback?: IWidget['callback']
  iconClass?: string

  // Vue widget options
  disabled?: boolean
  useGrouping?: boolean
  placeholder?: string
  showThumbnails?: boolean
  showItemNavigators?: boolean
  hidden?: boolean
}

interface IWidgetSliderOptions extends IWidgetOptions<number[]> {
  min: number
  max: number
  step2: number
  slider_color?: CanvasColour
  marker_color?: CanvasColour
}

export interface IWidgetGradientSliderOptions extends IWidgetOptions<number[]> {
  min: number
  max: number
  step2: number
  gradient_stops?: ColorStop[]
}

interface IWidgetKnobOptions extends IWidgetOptions<number[]> {
  min: number
  max: number
  step2: number
  slider_color?: CanvasColour // TODO: Replace with knob color
  marker_color?: CanvasColour
  gradient_stops?: string
}

export interface IWidgetAssetOptions extends IWidgetOptions {
  openModal: (widget: IBaseWidget) => void
}

/**
 * A widget for a node.
 * All types are based on IBaseWidget - additions can be made there or directly on individual types.
 *
 * Implemented as a discriminative union of widget types, so this type itself cannot be extended.
 * Recommend declaration merging any properties that use IWidget (e.g. {@link LGraphNode.widgets}) with a new type alias.
 * @see ICustomWidget
 */
export type IWidget =
  | IBooleanWidget
  | INumericWidget
  | IStringWidget
  | IComboWidget
  | IStringComboWidget
  | ICustomWidget
  | ISliderWidget
  | IGradientSliderWidget
  | IButtonWidget
  | IKnobWidget
  | IFileUploadWidget
  | IColorWidget
  | IMarkdownWidget
  | IImageWidget
  | ITreeSelectWidget
  | IMultiSelectWidget
  | IChartWidget
  | IGalleriaWidget
  | IImageCompareWidget
  | ISelectButtonWidget
  | ITextareaWidget
  | IAssetWidget
  | IImageCropWidget
  | IBoundingBoxWidget
  | ICurveWidget
  | IPainterWidget

export interface IBooleanWidget extends IBaseWidget<boolean, 'toggle'> {
  type: 'toggle'
  value: boolean
}

/** Any widget that uses a numeric backing */
export interface INumericWidget extends IBaseWidget<number, 'number'> {
  type: 'number'
  value: number
}

export interface ISliderWidget extends IBaseWidget<
  number,
  'slider',
  IWidgetSliderOptions
> {
  type: 'slider'
  value: number
  marker?: number
}

export interface IGradientSliderWidget extends IBaseWidget<
  number,
  'gradientslider',
  IWidgetGradientSliderOptions
> {
  type: 'gradientslider'
  value: number
}

export interface IKnobWidget extends IBaseWidget<
  number,
  'knob',
  IWidgetKnobOptions
> {
  type: 'knob'
  value: number
  options: IWidgetKnobOptions
}

/** Avoids the type issues with the legacy IComboWidget type */
export interface IStringComboWidget extends IBaseWidget<
  string,
  'combo',
  RequiredProps<IWidgetOptions<string[]>, 'values'>
> {
  type: 'combo'
  value: string
}

type ComboWidgetValues =
  | string[]
  | Record<string, string>
  | ((widget?: IComboWidget, node?: LGraphNode) => string[])

/** A combo-box widget (dropdown, select, etc) */
export interface IComboWidget extends IBaseWidget<
  string | number,
  'combo',
  RequiredProps<IWidgetOptions<ComboWidgetValues>, 'values'>
> {
  type: 'combo'
  value: string | number
}

/** A widget with a string value */
export interface IStringWidget extends IBaseWidget<
  string,
  'string' | 'text',
  IWidgetOptions<string[]>
> {
  type: 'string' | 'text'
  value: string
}

export interface IButtonWidget extends IBaseWidget<
  string | undefined,
  'button'
> {
  type: 'button'
  value: string | undefined
  clicked: boolean
}

/** A custom widget - accepts any value and has no built-in special handling */
interface ICustomWidget extends IBaseWidget<string | object, 'custom'> {
  type: 'custom'
  value: string | object
}

/** File upload widget for selecting and uploading files */
export interface IFileUploadWidget extends IBaseWidget<string, 'fileupload'> {
  type: 'fileupload'
  value: string
  label?: string
}

/** Color picker widget for selecting colors */
export interface IColorWidget extends IBaseWidget<string, 'color'> {
  type: 'color'
  value: string
}

/** Markdown widget for displaying formatted text */
export interface IMarkdownWidget extends IBaseWidget<string, 'markdown'> {
  type: 'markdown'
  value: string
}

/** Image display widget */
interface IImageWidget extends IBaseWidget<string, 'image'> {
  type: 'image'
  value: string
}

/** Tree select widget for hierarchical selection */
export interface ITreeSelectWidget extends IBaseWidget<
  string | string[],
  'treeselect'
> {
  type: 'treeselect'
  value: string | string[]
}

/** Multi-select widget for selecting multiple options */
export interface IMultiSelectWidget extends IBaseWidget<
  string[],
  'multiselect'
> {
  type: 'multiselect'
  value: string[]
}

/** Chart widget for displaying data visualizations */
export interface IChartWidget extends IBaseWidget<object, 'chart'> {
  type: 'chart'
  value: object
}

/** Gallery widget for displaying multiple images */
export interface IGalleriaWidget extends IBaseWidget<string[], 'galleria'> {
  type: 'galleria'
  value: string[]
}

/** Image comparison widget for comparing two images side by side */
export interface IImageCompareWidget extends IBaseWidget<
  string[],
  'imagecompare'
> {
  type: 'imagecompare'
  value: string[]
}

/** Select button widget for selecting from a group of buttons */
export interface ISelectButtonWidget extends IBaseWidget<
  string,
  'selectbutton',
  RequiredProps<IWidgetOptions<string[]>, 'values'>
> {
  type: 'selectbutton'
  value: string
}

/** Textarea widget for multi-line text input */
export interface ITextareaWidget extends IBaseWidget<string, 'textarea'> {
  type: 'textarea'
  value: string
}

export interface IAssetWidget extends IBaseWidget<
  string,
  'asset',
  IWidgetAssetOptions
> {
  type: 'asset'
  value: string
}

/** Image crop widget for cropping image */
export interface IImageCropWidget extends IBaseWidget<Bounds, 'imagecrop'> {
  type: 'imagecrop'
  value: Bounds
}

/** Bounding box widget for defining regions with numeric inputs */
export interface IBoundingBoxWidget extends IBaseWidget<Bounds, 'boundingbox'> {
  type: 'boundingbox'
  value: Bounds
}

export interface ICurveWidget extends IBaseWidget<CurvePoint[], 'curve'> {
  type: 'curve'
  value: CurvePoint[]
}

export interface IPainterWidget extends IBaseWidget<string, 'painter'> {
  type: 'painter'
  value: string
}

/**
 * Valid widget types.  TS cannot provide easily extensible type safety for this at present.
 * Override linkedWidgets[]
 * Values not in this list will not result in litegraph errors, however they will be treated the same as "custom".
 */
export type TWidgetType = IWidget['type']
export type TWidgetValue = IWidget['value']

/**
 * The base type for all widgets.  Should not be implemented directly.
 * @template TValue The type of value this widget holds.
 * @template TType A string designating the type of widget, e.g. "toggle" or "string".
 * @template TOptions The options for this widget.
 * @see IWidget
 */
export interface IBaseWidget<
  TValue = boolean | number | string | object | undefined,
  TType extends string = string,
  TOptions extends IWidgetOptions = IWidgetOptions
> {
  [symbol: symbol]: boolean

  linkedWidgets?: IBaseWidget[]

  name: string
  options: TOptions

  label?: string
  /** Widget type (see {@link TWidgetType}) */
  type: TType
  value?: TValue

  /**
   * Whether the widget value is persisted in the workflow JSON
   * (`widgets_values`). Checked by {@link LGraphNode.serialize} and
   * {@link LGraphNode.configure}.
   *
   * This is distinct from {@link IWidgetOptions.serialize}, which controls
   * whether the value is included in the API prompt sent for execution.
   *
   * @default true
   * @see IWidgetOptions.serialize — API prompt inclusion
   */
  serialize?: boolean

  /**
   * The computed height of the widget. Used by customized node resize logic.
   * See scripts/domWidget.ts for more details.
   * @readonly [Computed] This property is computed by the node.
   */
  computedHeight?: number

  /**
   * The starting y position of the widget after layout.
   * @readonly [Computed] This property is computed by the node.
   */
  y: number

  /**
   * The y position of the widget after drawing (rendering).
   * @readonly [Computed] This property is computed by the node.
   * @deprecated There is no longer dynamic y adjustment on rendering anymore.
   * Use {@link IBaseWidget.y} instead.
   */
  last_y?: number

  width?: number
  /**
   * Whether the widget is disabled. Disabled widgets are rendered at half opacity.
   * See also {@link IBaseWidget.computedDisabled}.
   */
  disabled?: boolean

  /**
   * The disabled state used for rendering based on various conditions including
   * {@link IBaseWidget.disabled}.
   * @readonly [Computed] This property is computed by the node.
   */
  computedDisabled?: boolean

  hidden?: boolean
  advanced?: boolean
  tooltip?: string

  // TODO: Confirm this format
  callback?(
    value: unknown,
    canvas?: LGraphCanvas,
    node?: LGraphNode,
    pos?: Point,
    e?: CanvasPointerEvent
  ): void

  /**
   * Simple callback for pointer events, allowing custom widgets to events relevant to them.
   * @param event The pointer event that triggered this callback
   * @param pointerOffset Offset of the pointer relative to {@link node.pos}
   * @param node The node this widget belongs to
   * @todo Expose CanvasPointer API to custom widgets
   */
  mouse?(
    event: CanvasPointerEvent,
    pointerOffset: Point,
    node: LGraphNode
  ): boolean
  /**
   * Draw the widget.
   * @param ctx The canvas context to draw on.
   * @param node The node this widget belongs to.
   * @param widget_width The width of the widget.
   * @param y The y position of the widget.
   * @param H The height of the widget.
   * @param lowQuality Whether to draw the widget in low quality.
   */
  draw?(
    ctx: CanvasRenderingContext2D,
    node: LGraphNode,
    widget_width: number,
    y: number,
    H: number,
    lowQuality?: boolean
  ): void

  /**
   * Compatibility method for widgets implementing the draw
   * method when displayed in non-canvas renderers.
   * Set by the current renderer implementation.
   * When called, performs a draw operation.
   */
  triggerDraw?: () => void

  /**
   * Compute the size of the widget. Overrides {@link IBaseWidget.computeSize}.
   * @param width The width of the widget.
   * @deprecated Use {@link IBaseWidget.computeLayoutSize} instead.
   * @returns The size of the widget.
   */
  computeSize?(width?: number): Size

  /**
   * Compute the layout size of the widget.
   * @param node The node this widget belongs to.
   * @returns The layout size of the widget.
   */
  computeLayoutSize?(
    this: IBaseWidget,
    node: LGraphNode
  ): {
    minHeight: number
    maxHeight?: number
    minWidth: number
    maxWidth?: number
  }

  /**
   * Callback for pointerdown events, allowing custom widgets to register callbacks to occur
   * for all {@link CanvasPointer} events.
   *
   * This callback is operated early in the pointerdown logic; actions that prevent it from firing are:
   * - `Ctrl + Drag` Multi-select
   * - `Alt + Click/Drag` Clone node
   * @param pointer The CanvasPointer handling this event
   * @param node The node this widget belongs to
   * @param canvas The LGraphCanvas where this event originated
   * @returns Returning `true` from this callback forces Litegraph to ignore the event and
   * not process it any further.
   */
  onPointerDown?(
    pointer: CanvasPointer,
    node: LGraphNode,
    canvas: LGraphCanvas
  ): boolean
}
