import type {ComfyApp, IStringWidget, IComboWidget, IWidget, LGraphNode} from "@comfyorg/frontend";

type ComfyWidgetFn<WidgetType extends IWidget> = (
  node: LGraphNode,
  inputName: string,
  inputData: any,
  app: ComfyApp,
) => {widget: WidgetType};

/**
 * A dummy ComfyWidgets that we can import from our code, which we'll rewrite later to the comfyui
 * hosted widgets.js
 */
export declare const ComfyWidgets: {
  COMBO: ComfyWidgetFn<IComboWidget>;
  STRING: ComfyWidgetFn<IStringWidget>;
  [key: string]: ComfyWidgetFn<IWidget>;
};
