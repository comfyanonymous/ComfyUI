import type {ComfyButton} from "scripts/ui/components/button.js";

export declare class ComfyButtonGroup {
  element: HTMLElement;
  constructor(...buttons: Array<ComfyButton|HTMLDivElement>);
  insert(button: ComfyButton, index: number): void;
  append(button: ComfyButton): void;
  remove(indexOrButton: ComfyButton|number): ComfyButton|HTMLElement|void;
  update(): void;
};
