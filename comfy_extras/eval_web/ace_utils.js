/**
 * Uses code adapted from https://github.com/yorkane/ComfyUI-KYNode
 *
 * MIT License
 *
 * Copyright (c) 2024 Kevin Yuan
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

// Make modal window
function makeModal({ title = "Message", text = "No text", type = "info", parent = null, stylePos = "fixed", classes = [] } = {}) {
  const overlay = document.createElement("div");
  Object.assign(overlay.style, {
    display: "none",
    position: stylePos,
    background: "rgba(0 0 0 / 0.8)",
    opacity: 0,
    top: "0",
    left: "0",
    right: "0",
    bottom: "0",
    zIndex: "500",
    transition: "all .8s",
    cursor: "pointer",
  });

  const boxModal = document.createElement("div");
  Object.assign(boxModal.style, {
    transition: "all 0.5s",
    opacity: 0,
    display: "none",
    position: stylePos,
    top: "50%",
    left: "50%",
    transform: "translate(-50%,-50%)",
    background: "#525252",
    minWidth: "300px",
    fontFamily: "sans-serif",
    zIndex: "501",
    border: "1px solid rgb(255 255 255 / 45%)",
  });

  boxModal.className = "alekpet_modal_window";
  boxModal.classList.add(...classes);

  const boxModalBody = document.createElement("div");
  Object.assign(boxModalBody.style, {
    display: "flex",
    flexDirection: "column",
    textAlign: "center",
  });

  boxModalBody.className = "alekpet_modal_body";

  const boxModalHtml = `
  <div class="alekpet_modal_header" style="display: flex;  align-items: center;  background: #222;  width: 100%;justify-content: center;">
  <div class="alekpet_modal_title" style="flex-basis: 85%; text-align: center;padding: 5px;">${title}</div>
  <div class="alekpet_modal_close">✕</div>
  </div>
  <div class="alekpet_modal_description" style="padding: 8px;">${text}</div>`;
  boxModalBody.innerHTML = boxModalHtml;

  const alekpet_modal_header = boxModalBody.querySelector(".alekpet_modal_header");
  Object.assign(alekpet_modal_header.style, {
    display: "flex",
    alignItems: "center",
  });

  const close = boxModalBody.querySelector(".alekpet_modal_close");
  Object.assign(close.style, {
    cursor: "pointer",
  });

  let parentElement = document.body;
  if (parent && parent.nodeType === 1) {
    parentElement = parent;
  }

  boxModal.append(boxModalBody);
  parentElement.append(overlay, boxModal);

  const removeEvent = new Event("removeElements");
  const remove = () => {
    animateTransitionProps(boxModal, { opacity: 0 }).then(() =>
      animateTransitionProps(overlay, { opacity: 0 }).then(() => {
        parentElement.removeChild(boxModal);
        parentElement.removeChild(overlay);
      }),
    );
  };

  boxModal.addEventListener("removeElements", remove);
  overlay.addEventListener("removeElements", remove);

  animateTransitionProps(overlay)
    .then(() => {
      overlay.addEventListener("click", () => {
        overlay.dispatchEvent(removeEvent);
      });
      animateTransitionProps(boxModal);
    })
    .then(() => boxModal.querySelector(".alekpet_modal_close").addEventListener("click", () => boxModal.dispatchEvent(removeEvent)));
}

function findWidget(node, value, attr = "name", func = "find") {
  return node?.widgets ? node.widgets[func]((w) => (Array.isArray(value) ? value.includes(w[attr]) : w[attr] === value)) : null;
}

function animateTransitionProps(el, props = { opacity: 1 }, preStyles = { display: "block" }) {
  Object.assign(el.style, preStyles);

  el.style.transition = !el.style.transition || !window.getComputedStyle(el).getPropertyValue("transition") ? "all .8s" : el.style.transition;

  return new Promise((res) => {
    setTimeout(() => {
      Object.assign(el.style, props);

      const transstart = () => (el.isAnimating = true);
      const transchancel = () => (el.isAnimating = false);
      el.addEventListener("transitionstart", transstart);
      el.addEventListener("transitioncancel", transchancel);

      el.addEventListener("transitionend", function transend() {
        el.isAnimating = false;
        el.removeEventListener("transitionend", transend);
        el.removeEventListener("transitionend", transchancel);
        el.removeEventListener("transitionend", transstart);
        res(el);
      });
    }, 100);
  });
}

function animateClick(target, params = {}) {
  const { opacityVal = 0.9, callback = () => {} } = params;
  if (target?.isAnimating) return;

  const hide = +target.style.opacity === 0;
  return animateTransitionProps(target, {
    opacity: hide ? opacityVal : 0,
  }).then((el) => {
    const isHide = hide || el.style.display === "none";
    showHide({ elements: [target], hide: !hide });
    callback();
    return isHide;
  });
}

function showHide({ elements = [], hide = null, displayProp = "block" } = {}) {
  Array.from(elements).forEach((el) => {
    if (hide !== null) {
      el.style.display = !hide ? displayProp : "none";
    } else {
      el.style.display = !el.style.display || el.style.display === "none" ? displayProp : "none";
    }
  });
}

function isEmptyObject(obj) {
  if (!obj) return true;
  return Object.keys(obj).length === 0 && obj.constructor === Object;
}

function makeElement(tag, attrs = {}) {
  if (!tag) tag = "div";
  const element = document.createElement(tag);
  Object.keys(attrs).forEach((key) => {
    const currValue = attrs[key];
    if (key === "class") {
      if (Array.isArray(currValue)) {
        element.classList.add(...currValue);
      } else if (currValue instanceof String || typeof currValue === "string") {
        element.className = currValue;
      }
    } else if (key === "dataset") {
      try {
        if (Array.isArray(currValue)) {
          currValue.forEach((datasetArr) => {
            const [prop, propval] = Object.entries(datasetArr)[0];
            element.dataset[prop] = propval;
          });
        } else {
          Object.entries(currValue).forEach((datasetArr) => {
            const [prop, propval] = datasetArr;
            element.dataset[prop] = propval;
          });
        }
      } catch (err) {
        console.log(err);
      }
    } else if (key === "style") {
      if (typeof currValue === "object" && !Array.isArray(currValue) && Object.keys(currValue).length) {
        Object.assign(element[key], currValue);
      } else if (typeof currValue === "object" && Array.isArray(currValue) && currValue.length) {
        element[key] = [...currValue];
      } else if (currValue instanceof String || typeof currValue === "string") {
        element[key] = currValue;
      }
    } else if (["for"].includes(key)) {
      element.setAttribute(key, currValue);
    } else if (key === "children") {
      element.append(...(currValue instanceof Array ? currValue : [currValue]));
    } else if (key === "parent") {
      currValue.append(element);
    } else {
      element[key] = currValue;
    }
  });
  return element;
}

function isValidStyle(opt, strColor) {
  let op = new Option().style;
  if (!op.hasOwnProperty(opt)) return { result: false, color: "", color_hex: "" };

  op[opt] = strColor;

  return {
    result: op[opt] !== "",
    color_rgb: op[opt],
    color_hex: rgbToHex(op[opt]),
  };
}

function rgbToHex(rgb) {
  const regEx = new RegExp(/\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)/);
  if (regEx.test(rgb)) {
    let [, r, g, b] = regEx.exec(rgb);
    r = parseInt(r).toString(16);
    g = parseInt(g).toString(16);
    b = parseInt(b).toString(16);

    r = r.length === 1 ? r + "0" : r;
    g = g.length === 1 ? g + "0" : g;
    b = b.length === 1 ? b + "0" : b;

    return `#${r}${g}${b}`;
  }
}

async function getDataJSON(url) {
  try {
    const response = await fetch(url);
    const jsonData = await response.json();
    return jsonData;
  } catch (err) {
    return new Error(err);
  }
}

function deepMerge(target, source) {
  if (source?.nodeType) return;
  for (let key in source) {
    if (source[key] instanceof Object && key in target) {
      Object.assign(source[key], deepMerge(target[key], source[key]));
    }
  }

  Object.assign(target || {}, source);
  return target;
}

const THEME_MODAL_WINDOW_BASE = {
  stylesTitle: {
    background: "auto",
    padding: "5px",
    borderRadius: "6px",
    marginBottom: "5px",
    alignSelf: "stretch",
  },
  stylesWrapper: {
    display: "none",
    opacity: 0,
    minWidth: "220px",
    position: "absolute",
    left: "50%",
    top: "50%",
    transform: "translate(-50%, -50%)",
    transition: "all .8s",
    fontFamily: "monospace",
    zIndex: 99999,
  },
  stylesBox: {
    display: "flex",
    flexDirection: "column",
    background: "#0e0e0e",
    padding: "6px",
    justifyContent: "center",
    alignItems: "center",
    gap: "3px",
    textAlign: "center",
    borderRadius: "6px",
    color: "white",
    border: "2px solid silver",
    boxShadow: "2px 2px 4px silver",
    maxWidth: "300px",
  },
  stylesClose: {
    position: "absolute",
    top: "-10px",
    right: "-10px",
    background: "silver",
    borderRadius: "50%",
    width: "20px",
    height: "20px",
    cursor: "pointer",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    fontSize: "0.8rem",
  },
};

const THEMES_MODAL_WINDOW = {
  error: {
    stylesTitle: {
      ...THEME_MODAL_WINDOW_BASE.stylesTitle,
      background: "#8f210f",
    },
    stylesBox: {
      ...THEME_MODAL_WINDOW_BASE.stylesBox,
      background: "#3b2222",
      boxShadow: "3px 3px 6px #141414",
      border: "1px solid #f91b1b",
    },
    stylesWrapper: { ...THEME_MODAL_WINDOW_BASE.stylesWrapper },
    stylesClose: {
      ...THEME_MODAL_WINDOW_BASE.stylesClose,
      background: "#3b2222",
    },
  },
  warning: {
    stylesTitle: {
      ...THEME_MODAL_WINDOW_BASE.stylesTitle,
      background: "#e99818",
    },
    stylesBox: {
      ...THEME_MODAL_WINDOW_BASE.stylesBox,
      background: "#594e32",
      boxShadow: "3px 3px 6px #141414",
      border: "1px solid #e99818",
    },
    stylesWrapper: { ...THEME_MODAL_WINDOW_BASE.stylesWrapper },
    stylesClose: {
      ...THEME_MODAL_WINDOW_BASE.stylesClose,
      background: "#594e32",
    },
  },
  normal: {
    stylesTitle: {
      ...THEME_MODAL_WINDOW_BASE.stylesTitle,
      background: "#108f0f",
    },
    stylesBox: {
      ...THEME_MODAL_WINDOW_BASE.stylesBox,
      background: "#223b2a",
      boxShadow: "3px 3px 6px #141414",
      border: "1px solid #108f0f",
    },
    stylesWrapper: { ...THEME_MODAL_WINDOW_BASE.stylesWrapper },
    stylesClose: {
      ...THEME_MODAL_WINDOW_BASE.stylesClose,
      background: "#223b2a",
    },
  },
};

const defaultOptions = {
  auto: {
    autohide: false,
    autoshow: false,
    autoremove: false,
    propStyles: { opacity: 0 },
    propPreStyles: {},
    timewait: 2000,
  },
  overlay: {
    overlay_enabled: false,
    overlayClasses: [],
    overlayStyles: {},
  },
  close: { closeRemove: false, showClose: true },
  parent: null,
};

function createWindowModal({ textTitle = "Message", textBody = "Hello world!", textFooter = null, classesWrapper = [], stylesWrapper = {}, classesBox = [], stylesBox = {}, classesTitle = [], stylesTitle = {}, classesBody = [], stylesBody = {}, classesClose = [], stylesClose = {}, classesFooter = [], stylesFooter = {}, options = defaultOptions } = {}) {
  // Check all options exist
  const _options = deepMerge(JSON.parse(JSON.stringify(defaultOptions)), options);

  const {
    parent,
    overlay: { overlay_enabled, overlayClasses, overlayStyles },
    close: { closeRemove, showClose },
    auto: { autohide, autoshow, autoremove, timewait, propStyles, propPreStyles },
  } = _options;

  // Function past text(html)
  function addText(text, parent) {
    if (!parent) return;

    switch (typeof text) {
      case "string":
        if (/^\<.*\/?\>$/.test(text)) {
          parent.innerHTML = text;
        } else {
          parent.textContent = text;
        }
        break;
      case "object":
      default:
        if (Array.isArray(text)) {
          text.forEach((element) => (element.nodeType === 1 || element.nodeType === 3) && parent.append(element));
        } else if (text.nodeType === 1 || text.nodeType === 3) parent.append(text);
    }
  }

  // Overlay
  let overlayElement = null;
  if (overlay_enabled) {
    overlayElement = makeElement("div", {
      class: [...overlayClasses],
      style: {
        display: "none",
        position: "fixed",
        background: "rgba(0 0 0 / 0.8)",
        opacity: 0,
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        zIndex: 99999,
        transition: "all .8s",
        cursor: "pointer",
        ...overlayStyles,
      },
    });
  }

  // Wrapper
  const wrapper_settings = makeElement("div", {
    class: ["alekpet__wrapper__window", ...classesWrapper],
  });

  Object.assign(wrapper_settings.style, {
    ...THEME_MODAL_WINDOW_BASE.stylesWrapper,
    ...stylesWrapper,
  });

  // Box
  const box__settings = makeElement("div", {
    class: ["alekpet__window__box", ...classesBox],
  });
  Object.assign(box__settings.style, {
    ...THEME_MODAL_WINDOW_BASE.stylesBox,
    ...stylesBox,
  });

  // Title
  let box_settings_title = "";
  if (textTitle) {
    box_settings_title = makeElement("div", {
      class: ["alekpet__window__title", ...classesTitle],
    });

    Object.assign(box_settings_title.style, {
      ...THEME_MODAL_WINDOW_BASE.stylesTitle,
      ...stylesTitle,
    });

    // Add text (html) to title
    addText(textTitle, box_settings_title);
  }
  // Body
  let box_settings_body = "";
  if (textBody) {
    box_settings_body = makeElement("div", {
      class: ["alekpet__window__body", ...classesBody],
    });

    Object.assign(box_settings_body.style, {
      display: "flex",
      flexDirection: "column",
      alignItems: "flex-end",
      gap: "5px",
      textWrap: "wrap",
      ...stylesBody,
    });

    // Add text (html) to body
    addText(textBody, box_settings_body);
  }

  // Close button
  const close__box__button = makeElement("div", {
    class: ["close__box__button", ...classesClose],
    textContent: "✖",
  });

  Object.assign(close__box__button.style, {
    ...THEME_MODAL_WINDOW_BASE.stylesClose,
    ...stylesClose,
  });

  if (!showClose) close__box__button.style.display = "none";

  const closeEvent = new Event("closeModal");
  const closeModalWindow = function () {
    overlay_enabled
      ? animateTransitionProps(overlayElement, {
          opacity: 0,
        })
          .then(() =>
            animateTransitionProps(wrapper_settings, {
              opacity: 0,
            }),
          )
          .then(() => {
            if (closeRemove) {
              parent.removeChild(wrapper_settings);
              parent.removeChild(overlayElement);
            } else {
              showHide({ elements: [wrapper_settings, overlayElement] });
            }
          })
      : animateTransitionProps(wrapper_settings, {
          opacity: 0,
        }).then(() => {
          showHide({ elements: [wrapper_settings] });
        });
  };

  close__box__button.addEventListener("closeModal", closeModalWindow);

  close__box__button.addEventListener("click", () => close__box__button.dispatchEvent(closeEvent));

  close__box__button.onmouseenter = () => {
    close__box__button.style.opacity = 0.8;
  };

  close__box__button.onmouseleave = () => {
    close__box__button.style.opacity = 1;
  };

  box__settings.append(box_settings_title, box_settings_body);

  // Footer
  if (textFooter) {
    const box_settings_footer = makeElement("div", {
      class: [...classesFooter],
    });
    Object.assign(box_settings_footer.style, {
      ...stylesFooter,
    });

    // Add text (html) to body
    addText(textFooter, box_settings_footer);

    box__settings.append(box_settings_footer);
  }

  wrapper_settings.append(close__box__button, box__settings);

  if (parent && parent.nodeType === 1) {
    if (overlay_enabled) parent.append(overlayElement);
    parent.append(wrapper_settings);

    if (autoshow) {
      overlay_enabled
        ? animateClick(overlayElement).then(() =>
            animateClick(wrapper_settings).then(
              () =>
                autohide &&
                setTimeout(
                  () =>
                    animateTransitionProps(wrapper_settings, { ...propStyles }, { ...propPreStyles })
                      .then(() => animateTransitionProps(overlayElement, { ...propStyles }, { ...propPreStyles }))
                      .then(() => {
                        if (autoremove) {
                          parent.removeChild(wrapper_settings);
                          parent.removeChild(overlayElement);
                        }
                      }),
                  timewait,
                ),
            ),
          )
        : animateClick(wrapper_settings).then(() => autohide && setTimeout(() => animateTransitionProps(wrapper_settings, { ...propStyles }, { ...propPreStyles }).then(() => autoremove && parent.removeChild(wrapper_settings)), timewait));
    }
  }

  return wrapper_settings;
}

// Prompt
async function comfyuiDesktopPrompt(title, message, defaultValue) {
  try {
    return await app.extensionManager.dialog.prompt({
      title,
      message,
      defaultValue,
    });
  } catch (err) {
    return prompt(title, message);
  }
}

// Alert
function comfyuiDesktopAlert(message) {
  try {
    app.extensionManager.toast.addAlert(message);
  } catch (err) {
    alert(message);
  }
}

// Confirm
function confirmModal({ title, message }) {
  return new Promise((res) => {
    const overlay = makeElement("div", {
      class: ["alekpet_confOverlay"],
      style: {
        background: "rgba(0, 0, 0, 0.7)",
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        zIndex: 9999,
        userSelect: "none",
      },
    });

    const modal = makeElement("div", {
      class: ["alekpet_confModal"],
      style: {
        ...THEME_MODAL_WINDOW_BASE.stylesBox,
        position: "fixed",
        top: "50%",
        left: "50%",
        fontFamily: "monospace",
        background: "rgb(92 186 255 / 20%)",
        transform: "translate(-50%, -50%)",
        borderColor: "rgba(92, 186, 255, 0.63)",
        boxShadow: "rgba(92, 186, 255, 0.63) 2px 2px 4px",
      },
    });

    const titleEl = makeElement("div", {
      class: ["alekpet_confTitle"],
      style: {
        ...THEME_MODAL_WINDOW_BASE.stylesTitle,
        background: "rgba(92, 186, 255, 0.63)",
      },
      textContent: title,
    });

    const messageEl = makeElement("div", {
      class: ["alekpet_confMessage"],
      style: {
        display: "flex",
        flexDirection: "column",
        alignItems: "flex-end",
        gap: "5px",
        textWrap: "wrap",
      },
      textContent: message,
    });

    const action_box = makeElement("div", {
      class: ["alekpet_confActions"],
      style: {
        display: "flex",
        gap: "5px",
        width: "100%",
        padding: "4px",
        justifyContent: "flex-end",
      },
    });

    const remove = () => {
      modal.remove();
      overlay.remove();
    };

    const ok = makeElement("div", {
      class: ["alekpet_confButtons", "alekpet_confButtonOk"],
      style: {
        background: "linear-gradient(45deg, green, limegreen) rgb(21, 100, 6)",
      },
      textContent: "Ok",
      onclick: (e) => {
        res(true);
        remove();
      },
    });

    const Cancel = makeElement("div", {
      class: ["alekpet_confButtons", "alekpet_confButtonCancel"],
      style: {
        background: "linear-gradient(45deg, #b64396, #a52a8b) rgb(135 3 161)",
      },
      textContent: "Cancel",
      onclick: (e) => {
        res(false);
        remove();
      },
    });

    action_box.append(ok, Cancel);
    modal.append(titleEl, messageEl, action_box);
    overlay.append(modal);
    document.body.append(overlay);
  });
}

async function comfyuiDesktopConfirm(message) {
  try {
    const result = await confirmModal({
      title: "Confirm",
      message: message,
    });

    // Wait update comfyui frontend! Confirm Cancel not return value! Fixed in ComfyUI_frontend ver. v1.10.8
    // https://github.com/Comfy-Org/ComfyUI_frontend/issues/2649
    // const result = await app.extensionManager.dialog.confirm({
    //   title: "Confirm",
    //   message: message,
    // });
    return result;
  } catch (err) {
    return confirm(message);
  }
}

export {
  makeModal,
  createWindowModal,
  animateTransitionProps,
  animateClick,
  showHide,
  makeElement,
  getDataJSON,
  isEmptyObject,
  isValidStyle,
  rgbToHex,
  findWidget,
  THEMES_MODAL_WINDOW,
  //
  comfyuiDesktopConfirm,
  comfyuiDesktopPrompt,
  comfyuiDesktopAlert,
};
