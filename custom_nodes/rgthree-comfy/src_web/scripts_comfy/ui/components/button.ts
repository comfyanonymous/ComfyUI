import {ComfyApp} from "@comfyorg/frontend";

type ComfyButtonProps = {
  icon?: string;
  overIcon?: string;
  iconSize?: number;
  content?: string | HTMLElement;
  tooltip?: string;
  enabled?: boolean;
  action?: (e: Event, btn: ComfyButton) => void;
  classList?: string;
  visibilitySetting?: {id: string; showValue: any};
  app?: ComfyApp;
};

export declare class ComfyButton {
  element: HTMLElement;
  iconElement: HTMLElement;
  contentElement: HTMLElement;
  constructor(props: ComfyButtonProps);
  updateIcon(): void;
  withPopup(popup: any, mode: "click" | "hover"): this;
}
