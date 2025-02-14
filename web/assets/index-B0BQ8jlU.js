var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });
import { bG as BaseStyle, bX as script$5, o as openBlock, f as createElementBlock, at as mergeProps, m as createBaseVNode, bH as script$6, cF as script$7, cG as script$8, cl as script$9, bO as Ripple, r as resolveDirective, y as createBlock, C as resolveDynamicComponent, F as Fragment, E as toDisplayString, cx as normalizeProps, i as withDirectives, B as createCommentVNode, dg as ToastEventBus, bZ as ZIndex, ci as isEmpty, c8 as setAttribute, ca as script$a, bR as resolveComponent, z as withCtx, k as createVNode, dh as TransitionGroup, D as renderList } from "./index-DqXp9vW4.js";
function _typeof$2(o) {
  "@babel/helpers - typeof";
  return _typeof$2 = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function(o2) {
    return typeof o2;
  } : function(o2) {
    return o2 && "function" == typeof Symbol && o2.constructor === Symbol && o2 !== Symbol.prototype ? "symbol" : typeof o2;
  }, _typeof$2(o);
}
__name(_typeof$2, "_typeof$2");
function _defineProperty$2(e, r, t) {
  return (r = _toPropertyKey$2(r)) in e ? Object.defineProperty(e, r, { value: t, enumerable: true, configurable: true, writable: true }) : e[r] = t, e;
}
__name(_defineProperty$2, "_defineProperty$2");
function _toPropertyKey$2(t) {
  var i = _toPrimitive$2(t, "string");
  return "symbol" == _typeof$2(i) ? i : i + "";
}
__name(_toPropertyKey$2, "_toPropertyKey$2");
function _toPrimitive$2(t, r) {
  if ("object" != _typeof$2(t) || !t) return t;
  var e = t[Symbol.toPrimitive];
  if (void 0 !== e) {
    var i = e.call(t, r || "default");
    if ("object" != _typeof$2(i)) return i;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return ("string" === r ? String : Number)(t);
}
__name(_toPrimitive$2, "_toPrimitive$2");
var theme = /* @__PURE__ */ __name(function theme2(_ref) {
  var dt = _ref.dt;
  return "\n.p-toast {\n    width: ".concat(dt("toast.width"), ";\n    white-space: pre-line;\n    word-break: break-word;\n}\n\n.p-toast-message {\n    margin: 0 0 1rem 0;\n}\n\n.p-toast-message-icon {\n    flex-shrink: 0;\n    font-size: ").concat(dt("toast.icon.size"), ";\n    width: ").concat(dt("toast.icon.size"), ";\n    height: ").concat(dt("toast.icon.size"), ";\n}\n\n.p-toast-message-content {\n    display: flex;\n    align-items: flex-start;\n    padding: ").concat(dt("toast.content.padding"), ";\n    gap: ").concat(dt("toast.content.gap"), ";\n}\n\n.p-toast-message-text {\n    flex: 1 1 auto;\n    display: flex;\n    flex-direction: column;\n    gap: ").concat(dt("toast.text.gap"), ";\n}\n\n.p-toast-summary {\n    font-weight: ").concat(dt("toast.summary.font.weight"), ";\n    font-size: ").concat(dt("toast.summary.font.size"), ";\n}\n\n.p-toast-detail {\n    font-weight: ").concat(dt("toast.detail.font.weight"), ";\n    font-size: ").concat(dt("toast.detail.font.size"), ";\n}\n\n.p-toast-close-button {\n    display: flex;\n    align-items: center;\n    justify-content: center;\n    overflow: hidden;\n    position: relative;\n    cursor: pointer;\n    background: transparent;\n    transition: background ").concat(dt("toast.transition.duration"), ", color ").concat(dt("toast.transition.duration"), ", outline-color ").concat(dt("toast.transition.duration"), ", box-shadow ").concat(dt("toast.transition.duration"), ";\n    outline-color: transparent;\n    color: inherit;\n    width: ").concat(dt("toast.close.button.width"), ";\n    height: ").concat(dt("toast.close.button.height"), ";\n    border-radius: ").concat(dt("toast.close.button.border.radius"), ";\n    margin: -25% 0 0 0;\n    right: -25%;\n    padding: 0;\n    border: none;\n    user-select: none;\n}\n\n.p-toast-close-button:dir(rtl) {\n    margin: -25% 0 0 auto;\n    left: -25%;\n    right: auto;\n}\n\n.p-toast-message-info,\n.p-toast-message-success,\n.p-toast-message-warn,\n.p-toast-message-error,\n.p-toast-message-secondary,\n.p-toast-message-contrast {\n    border-width: ").concat(dt("toast.border.width"), ";\n    border-style: solid;\n    backdrop-filter: blur(").concat(dt("toast.blur"), ");\n    border-radius: ").concat(dt("toast.border.radius"), ";\n}\n\n.p-toast-close-icon {\n    font-size: ").concat(dt("toast.close.icon.size"), ";\n    width: ").concat(dt("toast.close.icon.size"), ";\n    height: ").concat(dt("toast.close.icon.size"), ";\n}\n\n.p-toast-close-button:focus-visible {\n    outline-width: ").concat(dt("focus.ring.width"), ";\n    outline-style: ").concat(dt("focus.ring.style"), ";\n    outline-offset: ").concat(dt("focus.ring.offset"), ";\n}\n\n.p-toast-message-info {\n    background: ").concat(dt("toast.info.background"), ";\n    border-color: ").concat(dt("toast.info.border.color"), ";\n    color: ").concat(dt("toast.info.color"), ";\n    box-shadow: ").concat(dt("toast.info.shadow"), ";\n}\n\n.p-toast-message-info .p-toast-detail {\n    color: ").concat(dt("toast.info.detail.color"), ";\n}\n\n.p-toast-message-info .p-toast-close-button:focus-visible {\n    outline-color: ").concat(dt("toast.info.close.button.focus.ring.color"), ";\n    box-shadow: ").concat(dt("toast.info.close.button.focus.ring.shadow"), ";\n}\n\n.p-toast-message-info .p-toast-close-button:hover {\n    background: ").concat(dt("toast.info.close.button.hover.background"), ";\n}\n\n.p-toast-message-success {\n    background: ").concat(dt("toast.success.background"), ";\n    border-color: ").concat(dt("toast.success.border.color"), ";\n    color: ").concat(dt("toast.success.color"), ";\n    box-shadow: ").concat(dt("toast.success.shadow"), ";\n}\n\n.p-toast-message-success .p-toast-detail {\n    color: ").concat(dt("toast.success.detail.color"), ";\n}\n\n.p-toast-message-success .p-toast-close-button:focus-visible {\n    outline-color: ").concat(dt("toast.success.close.button.focus.ring.color"), ";\n    box-shadow: ").concat(dt("toast.success.close.button.focus.ring.shadow"), ";\n}\n\n.p-toast-message-success .p-toast-close-button:hover {\n    background: ").concat(dt("toast.success.close.button.hover.background"), ";\n}\n\n.p-toast-message-warn {\n    background: ").concat(dt("toast.warn.background"), ";\n    border-color: ").concat(dt("toast.warn.border.color"), ";\n    color: ").concat(dt("toast.warn.color"), ";\n    box-shadow: ").concat(dt("toast.warn.shadow"), ";\n}\n\n.p-toast-message-warn .p-toast-detail {\n    color: ").concat(dt("toast.warn.detail.color"), ";\n}\n\n.p-toast-message-warn .p-toast-close-button:focus-visible {\n    outline-color: ").concat(dt("toast.warn.close.button.focus.ring.color"), ";\n    box-shadow: ").concat(dt("toast.warn.close.button.focus.ring.shadow"), ";\n}\n\n.p-toast-message-warn .p-toast-close-button:hover {\n    background: ").concat(dt("toast.warn.close.button.hover.background"), ";\n}\n\n.p-toast-message-error {\n    background: ").concat(dt("toast.error.background"), ";\n    border-color: ").concat(dt("toast.error.border.color"), ";\n    color: ").concat(dt("toast.error.color"), ";\n    box-shadow: ").concat(dt("toast.error.shadow"), ";\n}\n\n.p-toast-message-error .p-toast-detail {\n    color: ").concat(dt("toast.error.detail.color"), ";\n}\n\n.p-toast-message-error .p-toast-close-button:focus-visible {\n    outline-color: ").concat(dt("toast.error.close.button.focus.ring.color"), ";\n    box-shadow: ").concat(dt("toast.error.close.button.focus.ring.shadow"), ";\n}\n\n.p-toast-message-error .p-toast-close-button:hover {\n    background: ").concat(dt("toast.error.close.button.hover.background"), ";\n}\n\n.p-toast-message-secondary {\n    background: ").concat(dt("toast.secondary.background"), ";\n    border-color: ").concat(dt("toast.secondary.border.color"), ";\n    color: ").concat(dt("toast.secondary.color"), ";\n    box-shadow: ").concat(dt("toast.secondary.shadow"), ";\n}\n\n.p-toast-message-secondary .p-toast-detail {\n    color: ").concat(dt("toast.secondary.detail.color"), ";\n}\n\n.p-toast-message-secondary .p-toast-close-button:focus-visible {\n    outline-color: ").concat(dt("toast.secondary.close.button.focus.ring.color"), ";\n    box-shadow: ").concat(dt("toast.secondary.close.button.focus.ring.shadow"), ";\n}\n\n.p-toast-message-secondary .p-toast-close-button:hover {\n    background: ").concat(dt("toast.secondary.close.button.hover.background"), ";\n}\n\n.p-toast-message-contrast {\n    background: ").concat(dt("toast.contrast.background"), ";\n    border-color: ").concat(dt("toast.contrast.border.color"), ";\n    color: ").concat(dt("toast.contrast.color"), ";\n    box-shadow: ").concat(dt("toast.contrast.shadow"), ";\n}\n\n.p-toast-message-contrast .p-toast-detail {\n    color: ").concat(dt("toast.contrast.detail.color"), ";\n}\n\n.p-toast-message-contrast .p-toast-close-button:focus-visible {\n    outline-color: ").concat(dt("toast.contrast.close.button.focus.ring.color"), ";\n    box-shadow: ").concat(dt("toast.contrast.close.button.focus.ring.shadow"), ";\n}\n\n.p-toast-message-contrast .p-toast-close-button:hover {\n    background: ").concat(dt("toast.contrast.close.button.hover.background"), ";\n}\n\n.p-toast-top-center {\n    transform: translateX(-50%);\n}\n\n.p-toast-bottom-center {\n    transform: translateX(-50%);\n}\n\n.p-toast-center {\n    min-width: 20vw;\n    transform: translate(-50%, -50%);\n}\n\n.p-toast-message-enter-from {\n    opacity: 0;\n    transform: translateY(50%);\n}\n\n.p-toast-message-leave-from {\n    max-height: 1000px;\n}\n\n.p-toast .p-toast-message.p-toast-message-leave-to {\n    max-height: 0;\n    opacity: 0;\n    margin-bottom: 0;\n    overflow: hidden;\n}\n\n.p-toast-message-enter-active {\n    transition: transform 0.3s, opacity 0.3s;\n}\n\n.p-toast-message-leave-active {\n    transition: max-height 0.45s cubic-bezier(0, 1, 0, 1), opacity 0.3s, margin-bottom 0.3s;\n}\n");
}, "theme");
var inlineStyles = {
  root: /* @__PURE__ */ __name(function root(_ref2) {
    var position = _ref2.position;
    return {
      position: "fixed",
      top: position === "top-right" || position === "top-left" || position === "top-center" ? "20px" : position === "center" ? "50%" : null,
      right: (position === "top-right" || position === "bottom-right") && "20px",
      bottom: (position === "bottom-left" || position === "bottom-right" || position === "bottom-center") && "20px",
      left: position === "top-left" || position === "bottom-left" ? "20px" : position === "center" || position === "top-center" || position === "bottom-center" ? "50%" : null
    };
  }, "root")
};
var classes = {
  root: /* @__PURE__ */ __name(function root2(_ref3) {
    var props = _ref3.props;
    return ["p-toast p-component p-toast-" + props.position];
  }, "root"),
  message: /* @__PURE__ */ __name(function message(_ref4) {
    var props = _ref4.props;
    return ["p-toast-message", {
      "p-toast-message-info": props.message.severity === "info" || props.message.severity === void 0,
      "p-toast-message-warn": props.message.severity === "warn",
      "p-toast-message-error": props.message.severity === "error",
      "p-toast-message-success": props.message.severity === "success",
      "p-toast-message-secondary": props.message.severity === "secondary",
      "p-toast-message-contrast": props.message.severity === "contrast"
    }];
  }, "message"),
  messageContent: "p-toast-message-content",
  messageIcon: /* @__PURE__ */ __name(function messageIcon(_ref5) {
    var props = _ref5.props;
    return ["p-toast-message-icon", _defineProperty$2(_defineProperty$2(_defineProperty$2(_defineProperty$2({}, props.infoIcon, props.message.severity === "info"), props.warnIcon, props.message.severity === "warn"), props.errorIcon, props.message.severity === "error"), props.successIcon, props.message.severity === "success")];
  }, "messageIcon"),
  messageText: "p-toast-message-text",
  summary: "p-toast-summary",
  detail: "p-toast-detail",
  closeButton: "p-toast-close-button",
  closeIcon: "p-toast-close-icon"
};
var ToastStyle = BaseStyle.extend({
  name: "toast",
  theme,
  classes,
  inlineStyles
});
var script$4 = {
  name: "ExclamationTriangleIcon",
  "extends": script$5
};
function render$3(_ctx, _cache, $props, $setup, $data, $options) {
  return openBlock(), createElementBlock("svg", mergeProps({
    width: "14",
    height: "14",
    viewBox: "0 0 14 14",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg"
  }, _ctx.pti()), _cache[0] || (_cache[0] = [createBaseVNode("path", {
    d: "M13.4018 13.1893H0.598161C0.49329 13.189 0.390283 13.1615 0.299143 13.1097C0.208003 13.0578 0.131826 12.9832 0.0780112 12.8932C0.0268539 12.8015 0 12.6982 0 12.5931C0 12.4881 0.0268539 12.3848 0.0780112 12.293L6.47985 1.08982C6.53679 1.00399 6.61408 0.933574 6.70484 0.884867C6.7956 0.836159 6.897 0.810669 7 0.810669C7.103 0.810669 7.2044 0.836159 7.29516 0.884867C7.38592 0.933574 7.46321 1.00399 7.52015 1.08982L13.922 12.293C13.9731 12.3848 14 12.4881 14 12.5931C14 12.6982 13.9731 12.8015 13.922 12.8932C13.8682 12.9832 13.792 13.0578 13.7009 13.1097C13.6097 13.1615 13.5067 13.189 13.4018 13.1893ZM1.63046 11.989H12.3695L7 2.59425L1.63046 11.989Z",
    fill: "currentColor"
  }, null, -1), createBaseVNode("path", {
    d: "M6.99996 8.78801C6.84143 8.78594 6.68997 8.72204 6.57787 8.60993C6.46576 8.49782 6.40186 8.34637 6.39979 8.18784V5.38703C6.39979 5.22786 6.46302 5.0752 6.57557 4.96265C6.68813 4.85009 6.84078 4.78686 6.99996 4.78686C7.15914 4.78686 7.31179 4.85009 7.42435 4.96265C7.5369 5.0752 7.60013 5.22786 7.60013 5.38703V8.18784C7.59806 8.34637 7.53416 8.49782 7.42205 8.60993C7.30995 8.72204 7.15849 8.78594 6.99996 8.78801Z",
    fill: "currentColor"
  }, null, -1), createBaseVNode("path", {
    d: "M6.99996 11.1887C6.84143 11.1866 6.68997 11.1227 6.57787 11.0106C6.46576 10.8985 6.40186 10.7471 6.39979 10.5885V10.1884C6.39979 10.0292 6.46302 9.87658 6.57557 9.76403C6.68813 9.65147 6.84078 9.58824 6.99996 9.58824C7.15914 9.58824 7.31179 9.65147 7.42435 9.76403C7.5369 9.87658 7.60013 10.0292 7.60013 10.1884V10.5885C7.59806 10.7471 7.53416 10.8985 7.42205 11.0106C7.30995 11.1227 7.15849 11.1866 6.99996 11.1887Z",
    fill: "currentColor"
  }, null, -1)]), 16);
}
__name(render$3, "render$3");
script$4.render = render$3;
var script$3 = {
  name: "InfoCircleIcon",
  "extends": script$5
};
function render$2(_ctx, _cache, $props, $setup, $data, $options) {
  return openBlock(), createElementBlock("svg", mergeProps({
    width: "14",
    height: "14",
    viewBox: "0 0 14 14",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg"
  }, _ctx.pti()), _cache[0] || (_cache[0] = [createBaseVNode("path", {
    "fill-rule": "evenodd",
    "clip-rule": "evenodd",
    d: "M3.11101 12.8203C4.26215 13.5895 5.61553 14 7 14C8.85652 14 10.637 13.2625 11.9497 11.9497C13.2625 10.637 14 8.85652 14 7C14 5.61553 13.5895 4.26215 12.8203 3.11101C12.0511 1.95987 10.9579 1.06266 9.67879 0.532846C8.3997 0.00303296 6.99224 -0.13559 5.63437 0.134506C4.2765 0.404603 3.02922 1.07129 2.05026 2.05026C1.07129 3.02922 0.404603 4.2765 0.134506 5.63437C-0.13559 6.99224 0.00303296 8.3997 0.532846 9.67879C1.06266 10.9579 1.95987 12.0511 3.11101 12.8203ZM3.75918 2.14976C4.71846 1.50879 5.84628 1.16667 7 1.16667C8.5471 1.16667 10.0308 1.78125 11.1248 2.87521C12.2188 3.96918 12.8333 5.45291 12.8333 7C12.8333 8.15373 12.4912 9.28154 11.8502 10.2408C11.2093 11.2001 10.2982 11.9478 9.23232 12.3893C8.16642 12.8308 6.99353 12.9463 5.86198 12.7212C4.73042 12.4962 3.69102 11.9406 2.87521 11.1248C2.05941 10.309 1.50384 9.26958 1.27876 8.13803C1.05367 7.00647 1.16919 5.83358 1.61071 4.76768C2.05222 3.70178 2.79989 2.79074 3.75918 2.14976ZM7.00002 4.8611C6.84594 4.85908 6.69873 4.79698 6.58977 4.68801C6.48081 4.57905 6.4187 4.43185 6.41669 4.27776V3.88888C6.41669 3.73417 6.47815 3.58579 6.58754 3.4764C6.69694 3.367 6.84531 3.30554 7.00002 3.30554C7.15473 3.30554 7.3031 3.367 7.4125 3.4764C7.52189 3.58579 7.58335 3.73417 7.58335 3.88888V4.27776C7.58134 4.43185 7.51923 4.57905 7.41027 4.68801C7.30131 4.79698 7.1541 4.85908 7.00002 4.8611ZM7.00002 10.6945C6.84594 10.6925 6.69873 10.6304 6.58977 10.5214C6.48081 10.4124 6.4187 10.2652 6.41669 10.1111V6.22225C6.41669 6.06754 6.47815 5.91917 6.58754 5.80977C6.69694 5.70037 6.84531 5.63892 7.00002 5.63892C7.15473 5.63892 7.3031 5.70037 7.4125 5.80977C7.52189 5.91917 7.58335 6.06754 7.58335 6.22225V10.1111C7.58134 10.2652 7.51923 10.4124 7.41027 10.5214C7.30131 10.6304 7.1541 10.6925 7.00002 10.6945Z",
    fill: "currentColor"
  }, null, -1)]), 16);
}
__name(render$2, "render$2");
script$3.render = render$2;
var script$2 = {
  name: "BaseToast",
  "extends": script$6,
  props: {
    group: {
      type: String,
      "default": null
    },
    position: {
      type: String,
      "default": "top-right"
    },
    autoZIndex: {
      type: Boolean,
      "default": true
    },
    baseZIndex: {
      type: Number,
      "default": 0
    },
    breakpoints: {
      type: Object,
      "default": null
    },
    closeIcon: {
      type: String,
      "default": void 0
    },
    infoIcon: {
      type: String,
      "default": void 0
    },
    warnIcon: {
      type: String,
      "default": void 0
    },
    errorIcon: {
      type: String,
      "default": void 0
    },
    successIcon: {
      type: String,
      "default": void 0
    },
    closeButtonProps: {
      type: null,
      "default": null
    }
  },
  style: ToastStyle,
  provide: /* @__PURE__ */ __name(function provide() {
    return {
      $pcToast: this,
      $parentInstance: this
    };
  }, "provide")
};
var script$1 = {
  name: "ToastMessage",
  hostName: "Toast",
  "extends": script$6,
  emits: ["close"],
  closeTimeout: null,
  props: {
    message: {
      type: null,
      "default": null
    },
    templates: {
      type: Object,
      "default": null
    },
    closeIcon: {
      type: String,
      "default": null
    },
    infoIcon: {
      type: String,
      "default": null
    },
    warnIcon: {
      type: String,
      "default": null
    },
    errorIcon: {
      type: String,
      "default": null
    },
    successIcon: {
      type: String,
      "default": null
    },
    closeButtonProps: {
      type: null,
      "default": null
    }
  },
  mounted: /* @__PURE__ */ __name(function mounted() {
    var _this = this;
    if (this.message.life) {
      this.closeTimeout = setTimeout(function() {
        _this.close({
          message: _this.message,
          type: "life-end"
        });
      }, this.message.life);
    }
  }, "mounted"),
  beforeUnmount: /* @__PURE__ */ __name(function beforeUnmount() {
    this.clearCloseTimeout();
  }, "beforeUnmount"),
  methods: {
    close: /* @__PURE__ */ __name(function close(params) {
      this.$emit("close", params);
    }, "close"),
    onCloseClick: /* @__PURE__ */ __name(function onCloseClick() {
      this.clearCloseTimeout();
      this.close({
        message: this.message,
        type: "close"
      });
    }, "onCloseClick"),
    clearCloseTimeout: /* @__PURE__ */ __name(function clearCloseTimeout() {
      if (this.closeTimeout) {
        clearTimeout(this.closeTimeout);
        this.closeTimeout = null;
      }
    }, "clearCloseTimeout")
  },
  computed: {
    iconComponent: /* @__PURE__ */ __name(function iconComponent() {
      return {
        info: !this.infoIcon && script$3,
        success: !this.successIcon && script$7,
        warn: !this.warnIcon && script$4,
        error: !this.errorIcon && script$8
      }[this.message.severity];
    }, "iconComponent"),
    closeAriaLabel: /* @__PURE__ */ __name(function closeAriaLabel() {
      return this.$primevue.config.locale.aria ? this.$primevue.config.locale.aria.close : void 0;
    }, "closeAriaLabel")
  },
  components: {
    TimesIcon: script$9,
    InfoCircleIcon: script$3,
    CheckIcon: script$7,
    ExclamationTriangleIcon: script$4,
    TimesCircleIcon: script$8
  },
  directives: {
    ripple: Ripple
  }
};
function _typeof$1(o) {
  "@babel/helpers - typeof";
  return _typeof$1 = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function(o2) {
    return typeof o2;
  } : function(o2) {
    return o2 && "function" == typeof Symbol && o2.constructor === Symbol && o2 !== Symbol.prototype ? "symbol" : typeof o2;
  }, _typeof$1(o);
}
__name(_typeof$1, "_typeof$1");
function ownKeys$1(e, r) {
  var t = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var o = Object.getOwnPropertySymbols(e);
    r && (o = o.filter(function(r2) {
      return Object.getOwnPropertyDescriptor(e, r2).enumerable;
    })), t.push.apply(t, o);
  }
  return t;
}
__name(ownKeys$1, "ownKeys$1");
function _objectSpread$1(e) {
  for (var r = 1; r < arguments.length; r++) {
    var t = null != arguments[r] ? arguments[r] : {};
    r % 2 ? ownKeys$1(Object(t), true).forEach(function(r2) {
      _defineProperty$1(e, r2, t[r2]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(t)) : ownKeys$1(Object(t)).forEach(function(r2) {
      Object.defineProperty(e, r2, Object.getOwnPropertyDescriptor(t, r2));
    });
  }
  return e;
}
__name(_objectSpread$1, "_objectSpread$1");
function _defineProperty$1(e, r, t) {
  return (r = _toPropertyKey$1(r)) in e ? Object.defineProperty(e, r, { value: t, enumerable: true, configurable: true, writable: true }) : e[r] = t, e;
}
__name(_defineProperty$1, "_defineProperty$1");
function _toPropertyKey$1(t) {
  var i = _toPrimitive$1(t, "string");
  return "symbol" == _typeof$1(i) ? i : i + "";
}
__name(_toPropertyKey$1, "_toPropertyKey$1");
function _toPrimitive$1(t, r) {
  if ("object" != _typeof$1(t) || !t) return t;
  var e = t[Symbol.toPrimitive];
  if (void 0 !== e) {
    var i = e.call(t, r || "default");
    if ("object" != _typeof$1(i)) return i;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return ("string" === r ? String : Number)(t);
}
__name(_toPrimitive$1, "_toPrimitive$1");
var _hoisted_1 = ["aria-label"];
function render$1(_ctx, _cache, $props, $setup, $data, $options) {
  var _directive_ripple = resolveDirective("ripple");
  return openBlock(), createElementBlock("div", mergeProps({
    "class": [_ctx.cx("message"), $props.message.styleClass],
    role: "alert",
    "aria-live": "assertive",
    "aria-atomic": "true"
  }, _ctx.ptm("message")), [$props.templates.container ? (openBlock(), createBlock(resolveDynamicComponent($props.templates.container), {
    key: 0,
    message: $props.message,
    closeCallback: $options.onCloseClick
  }, null, 8, ["message", "closeCallback"])) : (openBlock(), createElementBlock("div", mergeProps({
    key: 1,
    "class": [_ctx.cx("messageContent"), $props.message.contentStyleClass]
  }, _ctx.ptm("messageContent")), [!$props.templates.message ? (openBlock(), createElementBlock(Fragment, {
    key: 0
  }, [(openBlock(), createBlock(resolveDynamicComponent($props.templates.messageicon ? $props.templates.messageicon : $props.templates.icon ? $props.templates.icon : $options.iconComponent && $options.iconComponent.name ? $options.iconComponent : "span"), mergeProps({
    "class": _ctx.cx("messageIcon")
  }, _ctx.ptm("messageIcon")), null, 16, ["class"])), createBaseVNode("div", mergeProps({
    "class": _ctx.cx("messageText")
  }, _ctx.ptm("messageText")), [createBaseVNode("span", mergeProps({
    "class": _ctx.cx("summary")
  }, _ctx.ptm("summary")), toDisplayString($props.message.summary), 17), createBaseVNode("div", mergeProps({
    "class": _ctx.cx("detail")
  }, _ctx.ptm("detail")), toDisplayString($props.message.detail), 17)], 16)], 64)) : (openBlock(), createBlock(resolveDynamicComponent($props.templates.message), {
    key: 1,
    message: $props.message
  }, null, 8, ["message"])), $props.message.closable !== false ? (openBlock(), createElementBlock("div", normalizeProps(mergeProps({
    key: 2
  }, _ctx.ptm("buttonContainer"))), [withDirectives((openBlock(), createElementBlock("button", mergeProps({
    "class": _ctx.cx("closeButton"),
    type: "button",
    "aria-label": $options.closeAriaLabel,
    onClick: _cache[0] || (_cache[0] = function() {
      return $options.onCloseClick && $options.onCloseClick.apply($options, arguments);
    }),
    autofocus: ""
  }, _objectSpread$1(_objectSpread$1({}, $props.closeButtonProps), _ctx.ptm("closeButton"))), [(openBlock(), createBlock(resolveDynamicComponent($props.templates.closeicon || "TimesIcon"), mergeProps({
    "class": [_ctx.cx("closeIcon"), $props.closeIcon]
  }, _ctx.ptm("closeIcon")), null, 16, ["class"]))], 16, _hoisted_1)), [[_directive_ripple]])], 16)) : createCommentVNode("", true)], 16))], 16);
}
__name(render$1, "render$1");
script$1.render = render$1;
function _toConsumableArray(r) {
  return _arrayWithoutHoles(r) || _iterableToArray(r) || _unsupportedIterableToArray(r) || _nonIterableSpread();
}
__name(_toConsumableArray, "_toConsumableArray");
function _nonIterableSpread() {
  throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.");
}
__name(_nonIterableSpread, "_nonIterableSpread");
function _unsupportedIterableToArray(r, a) {
  if (r) {
    if ("string" == typeof r) return _arrayLikeToArray(r, a);
    var t = {}.toString.call(r).slice(8, -1);
    return "Object" === t && r.constructor && (t = r.constructor.name), "Map" === t || "Set" === t ? Array.from(r) : "Arguments" === t || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(t) ? _arrayLikeToArray(r, a) : void 0;
  }
}
__name(_unsupportedIterableToArray, "_unsupportedIterableToArray");
function _iterableToArray(r) {
  if ("undefined" != typeof Symbol && null != r[Symbol.iterator] || null != r["@@iterator"]) return Array.from(r);
}
__name(_iterableToArray, "_iterableToArray");
function _arrayWithoutHoles(r) {
  if (Array.isArray(r)) return _arrayLikeToArray(r);
}
__name(_arrayWithoutHoles, "_arrayWithoutHoles");
function _arrayLikeToArray(r, a) {
  (null == a || a > r.length) && (a = r.length);
  for (var e = 0, n = Array(a); e < a; e++) n[e] = r[e];
  return n;
}
__name(_arrayLikeToArray, "_arrayLikeToArray");
var messageIdx = 0;
var script = {
  name: "Toast",
  "extends": script$2,
  inheritAttrs: false,
  emits: ["close", "life-end"],
  data: /* @__PURE__ */ __name(function data() {
    return {
      messages: []
    };
  }, "data"),
  styleElement: null,
  mounted: /* @__PURE__ */ __name(function mounted2() {
    ToastEventBus.on("add", this.onAdd);
    ToastEventBus.on("remove", this.onRemove);
    ToastEventBus.on("remove-group", this.onRemoveGroup);
    ToastEventBus.on("remove-all-groups", this.onRemoveAllGroups);
    if (this.breakpoints) {
      this.createStyle();
    }
  }, "mounted"),
  beforeUnmount: /* @__PURE__ */ __name(function beforeUnmount2() {
    this.destroyStyle();
    if (this.$refs.container && this.autoZIndex) {
      ZIndex.clear(this.$refs.container);
    }
    ToastEventBus.off("add", this.onAdd);
    ToastEventBus.off("remove", this.onRemove);
    ToastEventBus.off("remove-group", this.onRemoveGroup);
    ToastEventBus.off("remove-all-groups", this.onRemoveAllGroups);
  }, "beforeUnmount"),
  methods: {
    add: /* @__PURE__ */ __name(function add(message2) {
      if (message2.id == null) {
        message2.id = messageIdx++;
      }
      this.messages = [].concat(_toConsumableArray(this.messages), [message2]);
    }, "add"),
    remove: /* @__PURE__ */ __name(function remove(params) {
      var index = this.messages.findIndex(function(m) {
        return m.id === params.message.id;
      });
      if (index !== -1) {
        this.messages.splice(index, 1);
        this.$emit(params.type, {
          message: params.message
        });
      }
    }, "remove"),
    onAdd: /* @__PURE__ */ __name(function onAdd(message2) {
      if (this.group == message2.group) {
        this.add(message2);
      }
    }, "onAdd"),
    onRemove: /* @__PURE__ */ __name(function onRemove(message2) {
      this.remove({
        message: message2,
        type: "close"
      });
    }, "onRemove"),
    onRemoveGroup: /* @__PURE__ */ __name(function onRemoveGroup(group) {
      if (this.group === group) {
        this.messages = [];
      }
    }, "onRemoveGroup"),
    onRemoveAllGroups: /* @__PURE__ */ __name(function onRemoveAllGroups() {
      this.messages = [];
    }, "onRemoveAllGroups"),
    onEnter: /* @__PURE__ */ __name(function onEnter() {
      if (this.autoZIndex) {
        ZIndex.set("modal", this.$refs.container, this.baseZIndex || this.$primevue.config.zIndex.modal);
      }
    }, "onEnter"),
    onLeave: /* @__PURE__ */ __name(function onLeave() {
      var _this = this;
      if (this.$refs.container && this.autoZIndex && isEmpty(this.messages)) {
        setTimeout(function() {
          ZIndex.clear(_this.$refs.container);
        }, 200);
      }
    }, "onLeave"),
    createStyle: /* @__PURE__ */ __name(function createStyle() {
      if (!this.styleElement && !this.isUnstyled) {
        var _this$$primevue;
        this.styleElement = document.createElement("style");
        this.styleElement.type = "text/css";
        setAttribute(this.styleElement, "nonce", (_this$$primevue = this.$primevue) === null || _this$$primevue === void 0 || (_this$$primevue = _this$$primevue.config) === null || _this$$primevue === void 0 || (_this$$primevue = _this$$primevue.csp) === null || _this$$primevue === void 0 ? void 0 : _this$$primevue.nonce);
        document.head.appendChild(this.styleElement);
        var innerHTML = "";
        for (var breakpoint in this.breakpoints) {
          var breakpointStyle = "";
          for (var styleProp in this.breakpoints[breakpoint]) {
            breakpointStyle += styleProp + ":" + this.breakpoints[breakpoint][styleProp] + "!important;";
          }
          innerHTML += "\n                        @media screen and (max-width: ".concat(breakpoint, ") {\n                            .p-toast[").concat(this.$attrSelector, "] {\n                                ").concat(breakpointStyle, "\n                            }\n                        }\n                    ");
        }
        this.styleElement.innerHTML = innerHTML;
      }
    }, "createStyle"),
    destroyStyle: /* @__PURE__ */ __name(function destroyStyle() {
      if (this.styleElement) {
        document.head.removeChild(this.styleElement);
        this.styleElement = null;
      }
    }, "destroyStyle")
  },
  components: {
    ToastMessage: script$1,
    Portal: script$a
  }
};
function _typeof(o) {
  "@babel/helpers - typeof";
  return _typeof = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function(o2) {
    return typeof o2;
  } : function(o2) {
    return o2 && "function" == typeof Symbol && o2.constructor === Symbol && o2 !== Symbol.prototype ? "symbol" : typeof o2;
  }, _typeof(o);
}
__name(_typeof, "_typeof");
function ownKeys(e, r) {
  var t = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var o = Object.getOwnPropertySymbols(e);
    r && (o = o.filter(function(r2) {
      return Object.getOwnPropertyDescriptor(e, r2).enumerable;
    })), t.push.apply(t, o);
  }
  return t;
}
__name(ownKeys, "ownKeys");
function _objectSpread(e) {
  for (var r = 1; r < arguments.length; r++) {
    var t = null != arguments[r] ? arguments[r] : {};
    r % 2 ? ownKeys(Object(t), true).forEach(function(r2) {
      _defineProperty(e, r2, t[r2]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(t)) : ownKeys(Object(t)).forEach(function(r2) {
      Object.defineProperty(e, r2, Object.getOwnPropertyDescriptor(t, r2));
    });
  }
  return e;
}
__name(_objectSpread, "_objectSpread");
function _defineProperty(e, r, t) {
  return (r = _toPropertyKey(r)) in e ? Object.defineProperty(e, r, { value: t, enumerable: true, configurable: true, writable: true }) : e[r] = t, e;
}
__name(_defineProperty, "_defineProperty");
function _toPropertyKey(t) {
  var i = _toPrimitive(t, "string");
  return "symbol" == _typeof(i) ? i : i + "";
}
__name(_toPropertyKey, "_toPropertyKey");
function _toPrimitive(t, r) {
  if ("object" != _typeof(t) || !t) return t;
  var e = t[Symbol.toPrimitive];
  if (void 0 !== e) {
    var i = e.call(t, r || "default");
    if ("object" != _typeof(i)) return i;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return ("string" === r ? String : Number)(t);
}
__name(_toPrimitive, "_toPrimitive");
function render(_ctx, _cache, $props, $setup, $data, $options) {
  var _component_ToastMessage = resolveComponent("ToastMessage");
  var _component_Portal = resolveComponent("Portal");
  return openBlock(), createBlock(_component_Portal, null, {
    "default": withCtx(function() {
      return [createBaseVNode("div", mergeProps({
        ref: "container",
        "class": _ctx.cx("root"),
        style: _ctx.sx("root", true, {
          position: _ctx.position
        })
      }, _ctx.ptmi("root")), [createVNode(TransitionGroup, mergeProps({
        name: "p-toast-message",
        tag: "div",
        onEnter: $options.onEnter,
        onLeave: $options.onLeave
      }, _objectSpread({}, _ctx.ptm("transition"))), {
        "default": withCtx(function() {
          return [(openBlock(true), createElementBlock(Fragment, null, renderList($data.messages, function(msg) {
            return openBlock(), createBlock(_component_ToastMessage, {
              key: msg.id,
              message: msg,
              templates: _ctx.$slots,
              closeIcon: _ctx.closeIcon,
              infoIcon: _ctx.infoIcon,
              warnIcon: _ctx.warnIcon,
              errorIcon: _ctx.errorIcon,
              successIcon: _ctx.successIcon,
              closeButtonProps: _ctx.closeButtonProps,
              unstyled: _ctx.unstyled,
              onClose: _cache[0] || (_cache[0] = function($event) {
                return $options.remove($event);
              }),
              pt: _ctx.pt
            }, null, 8, ["message", "templates", "closeIcon", "infoIcon", "warnIcon", "errorIcon", "successIcon", "closeButtonProps", "unstyled", "pt"]);
          }), 128))];
        }),
        _: 1
      }, 16, ["onEnter", "onLeave"])], 16)];
    }),
    _: 1
  });
}
__name(render, "render");
script.render = render;
export {
  script$3 as a,
  script$4 as b,
  script as s
};
//# sourceMappingURL=index-B0BQ8jlU.js.map
