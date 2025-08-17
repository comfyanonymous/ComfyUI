type ComfyPopupProps = {
  target: HTMLElement;
  container?: HTMLElement;
  classList?: string;
  ignoreTarget?: boolean,
  closeOnEscape?: boolean,
  position?: "absolute" | "relative",
  horizontal?: "left" | "right"
}

export declare class ComfyPopup extends EventTarget {
  element: HTMLDivElement;
  constructor(props: ComfyPopupProps, ...children: HTMLElement[]);
  toggle(): void;
  update(): void;
};
