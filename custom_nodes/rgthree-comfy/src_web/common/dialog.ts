import type { LGraphNode, LGraphNodeConstructor } from "@comfyorg/frontend";
import { createElement as $el, getClosestOrSelf, setAttributes } from "./utils_dom.js";

type RgthreeDialogButton = {
  label: string;
  className?: string;
  closes?: boolean;
  disabled?: boolean;
  callback?: (e: PointerEvent | MouseEvent) => void;
};

export type RgthreeDialogOptions = {
  content: string | HTMLElement | HTMLElement[];
  class?: string | string[];
  title?: string | HTMLElement | HTMLElement[];
  closeX?: boolean;
  closeOnEsc?: boolean;
  closeOnModalClick?: boolean;
  closeButtonLabel?: string | boolean;
  buttons?: RgthreeDialogButton[];
  onBeforeClose?: () => Promise<boolean> | boolean;
};

/**
 * A Dialog that shows content, and closes.
 */
export class RgthreeDialog extends EventTarget {
  element: HTMLDialogElement;
  contentElement: HTMLDivElement;
  titleElement: HTMLDivElement;
  options: RgthreeDialogOptions;

  constructor(options: RgthreeDialogOptions) {
    super();
    this.options = options;
    let container = $el("div.rgthree-dialog-container");
    this.element = $el("dialog", {
      classes: ["rgthree-dialog", options.class || ""],
      child: container,
      parent: document.body,
      events: {
        click: (event: MouseEvent) => {
          // Close the dialog if we've clicked outside of our container. The dialog modal will
          // report itself as the dialog itself, so we use the inner container div (and CSS to
          // remove default padding from the dialog element).
          if (
            !this.element.open ||
            event.target === container ||
            getClosestOrSelf(event.target, `.rgthree-dialog-container`) === container
          ) {
            return;
          }
          return this.close();
        },
      },
    });
    this.element.addEventListener("close", (event) => {
      this.onDialogElementClose();
    });

    this.titleElement = $el("div.rgthree-dialog-container-title", {
      parent: container,
      children: !options.title
        ? null
        : options.title instanceof Element || Array.isArray(options.title)
        ? options.title
        : typeof options.title === "string"
        ? !options.title.includes("<h2")
          ? $el("h2", { html: options.title })
          : options.title
        : options.title,
    });

    this.contentElement = $el("div.rgthree-dialog-container-content", {
      parent: container,
      child: options.content,
    });

    const footerEl = $el("footer.rgthree-dialog-container-footer", { parent: container });
    for (const button of options.buttons || []) {
      $el("button", {
        text: button.label,
        className: button.className,
        disabled: !!button.disabled,
        parent: footerEl,
        events: {
          click: (e: MouseEvent) => {
            button.callback?.(e);
          },
        },
      });
    }

    if (options.closeButtonLabel !== false) {
      $el("button", {
        text: options.closeButtonLabel || "Close",
        className: "rgthree-button",
        parent: footerEl,
        events: {
          click: (e: MouseEvent) => {
            this.close(e);
          },
        },
      });
    }
  }

  setTitle(content: string | HTMLElement | HTMLElement[]) {
    const title =
      typeof content !== "string" || content.includes("<h2")
        ? content
        : $el("h2", { html: content });
    setAttributes(this.titleElement, { children: title });
  }

  setContent(content: string | HTMLElement | HTMLElement[]) {
    setAttributes(this.contentElement, { children: content });
  }

  show() {
    document.body.classList.add("rgthree-dialog-open");
    this.element.showModal();
    this.dispatchEvent(new CustomEvent("show"));
    return this;
  }

  async close(e?: MouseEvent | PointerEvent) {
    if (this.options.onBeforeClose && !(await this.options.onBeforeClose())) {
      return;
    }
    this.element.close();
  }

  onDialogElementClose() {
    document.body.classList.remove("rgthree-dialog-open");
    this.element.remove();
    this.dispatchEvent(new CustomEvent("close", this.getCloseEventDetail()));
  }

  protected getCloseEventDetail(): { detail: any } {
    return { detail: null };
  }
}

/**
 * A help extension for the dialog class that standardizes help content.
 */
export class RgthreeHelpDialog extends RgthreeDialog {
  constructor(
    node: LGraphNode | LGraphNodeConstructor,
    content: string,
    opts: Partial<RgthreeDialogOptions> = {},
  ) {
    const title = (node.type || node.title || "").replace(
      /\s*\(rgthree\).*/,
      " <small>by rgthree</small>",
    );
    const options = Object.assign({}, opts, {
      class: "-iconed -help",
      title,
      content,
    });
    super(options);
  }
}
