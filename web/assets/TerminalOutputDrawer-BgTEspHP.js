var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });
import { bG as BaseStyle, bH as script$2, bZ as ZIndex, bU as addClass, bL as focus, cy as blockBodyScroll, cA as unblockBodyScroll, cB as FocusTrap, l as script$3, ca as script$4, cl as script$5, bR as resolveComponent, r as resolveDirective, o as openBlock, y as createBlock, z as withCtx, f as createElementBlock, at as mergeProps, k as createVNode, bI as Transition, i as withDirectives, A as renderSlot, F as Fragment, m as createBaseVNode, aj as normalizeClass, E as toDisplayString, B as createCommentVNode, C as resolveDynamicComponent, dd as commonjsGlobal, de as getDefaultExportFromCjs, H as markRaw, df as xtermExports, p as onMounted, d8 as onUnmounted, d as defineComponent, bw as mergeModels, bq as useModel, bp as BaseTerminal, j as unref, b9 as electronAPI } from "./index-DqXp9vW4.js";
var theme = /* @__PURE__ */ __name(function theme2(_ref) {
  var dt = _ref.dt;
  return "\n.p-drawer {\n    display: flex;\n    flex-direction: column;\n    transform: translate3d(0px, 0px, 0px);\n    position: relative;\n    transition: transform 0.3s;\n    background: ".concat(dt("drawer.background"), ";\n    color: ").concat(dt("drawer.color"), ";\n    border: 1px solid ").concat(dt("drawer.border.color"), ";\n    box-shadow: ").concat(dt("drawer.shadow"), ";\n}\n\n.p-drawer-content {\n    overflow-y: auto;\n    flex-grow: 1;\n    padding: ").concat(dt("drawer.content.padding"), ";\n}\n\n.p-drawer-header {\n    display: flex;\n    align-items: center;\n    justify-content: space-between;\n    flex-shrink: 0;\n    padding: ").concat(dt("drawer.header.padding"), ";\n}\n\n.p-drawer-footer {\n    padding: ").concat(dt("drawer.footer.padding"), ";\n}\n\n.p-drawer-title {\n    font-weight: ").concat(dt("drawer.title.font.weight"), ";\n    font-size: ").concat(dt("drawer.title.font.size"), ";\n}\n\n.p-drawer-full .p-drawer {\n    transition: none;\n    transform: none;\n    width: 100vw !important;\n    height: 100vh !important;\n    max-height: 100%;\n    top: 0px !important;\n    left: 0px !important;\n    border-width: 1px;\n}\n\n.p-drawer-left .p-drawer-enter-from,\n.p-drawer-left .p-drawer-leave-to {\n    transform: translateX(-100%);\n}\n\n.p-drawer-right .p-drawer-enter-from,\n.p-drawer-right .p-drawer-leave-to {\n    transform: translateX(100%);\n}\n\n.p-drawer-top .p-drawer-enter-from,\n.p-drawer-top .p-drawer-leave-to {\n    transform: translateY(-100%);\n}\n\n.p-drawer-bottom .p-drawer-enter-from,\n.p-drawer-bottom .p-drawer-leave-to {\n    transform: translateY(100%);\n}\n\n.p-drawer-full .p-drawer-enter-from,\n.p-drawer-full .p-drawer-leave-to {\n    opacity: 0;\n}\n\n.p-drawer-full .p-drawer-enter-active,\n.p-drawer-full .p-drawer-leave-active {\n    transition: opacity 400ms cubic-bezier(0.25, 0.8, 0.25, 1);\n}\n\n.p-drawer-left .p-drawer {\n    width: 20rem;\n    height: 100%;\n    border-inline-end-width: 1px;\n}\n\n.p-drawer-right .p-drawer {\n    width: 20rem;\n    height: 100%;\n    border-inline-start-width: 1px;\n}\n\n.p-drawer-top .p-drawer {\n    height: 10rem;\n    width: 100%;\n    border-block-end-width: 1px;\n}\n\n.p-drawer-bottom .p-drawer {\n    height: 10rem;\n    width: 100%;\n    border-block-start-width: 1px;\n}\n\n.p-drawer-left .p-drawer-content,\n.p-drawer-right .p-drawer-content,\n.p-drawer-top .p-drawer-content,\n.p-drawer-bottom .p-drawer-content {\n    width: 100%;\n    height: 100%;\n}\n\n.p-drawer-open {\n    display: flex;\n}\n\n.p-drawer-mask:dir(rtl) {\n    flex-direction: row-reverse;\n}\n");
}, "theme");
var inlineStyles = {
  mask: /* @__PURE__ */ __name(function mask(_ref2) {
    var position = _ref2.position, modal = _ref2.modal;
    return {
      position: "fixed",
      height: "100%",
      width: "100%",
      left: 0,
      top: 0,
      display: "flex",
      justifyContent: position === "left" ? "flex-start" : position === "right" ? "flex-end" : "center",
      alignItems: position === "top" ? "flex-start" : position === "bottom" ? "flex-end" : "center",
      pointerEvents: modal ? "auto" : "none"
    };
  }, "mask"),
  root: {
    pointerEvents: "auto"
  }
};
var classes = {
  mask: /* @__PURE__ */ __name(function mask2(_ref3) {
    var instance = _ref3.instance, props = _ref3.props;
    var positions = ["left", "right", "top", "bottom"];
    var pos = positions.find(function(item) {
      return item === props.position;
    });
    return ["p-drawer-mask", {
      "p-overlay-mask p-overlay-mask-enter": props.modal,
      "p-drawer-open": instance.containerVisible,
      "p-drawer-full": instance.fullScreen
    }, pos ? "p-drawer-".concat(pos) : ""];
  }, "mask"),
  root: /* @__PURE__ */ __name(function root(_ref4) {
    var instance = _ref4.instance;
    return ["p-drawer p-component", {
      "p-drawer-full": instance.fullScreen
    }];
  }, "root"),
  header: "p-drawer-header",
  title: "p-drawer-title",
  pcCloseButton: "p-drawer-close-button",
  content: "p-drawer-content",
  footer: "p-drawer-footer"
};
var DrawerStyle = BaseStyle.extend({
  name: "drawer",
  theme,
  classes,
  inlineStyles
});
var script$1 = {
  name: "BaseDrawer",
  "extends": script$2,
  props: {
    visible: {
      type: Boolean,
      "default": false
    },
    position: {
      type: String,
      "default": "left"
    },
    header: {
      type: null,
      "default": null
    },
    baseZIndex: {
      type: Number,
      "default": 0
    },
    autoZIndex: {
      type: Boolean,
      "default": true
    },
    dismissable: {
      type: Boolean,
      "default": true
    },
    showCloseIcon: {
      type: Boolean,
      "default": true
    },
    closeButtonProps: {
      type: Object,
      "default": /* @__PURE__ */ __name(function _default() {
        return {
          severity: "secondary",
          text: true,
          rounded: true
        };
      }, "_default")
    },
    closeIcon: {
      type: String,
      "default": void 0
    },
    modal: {
      type: Boolean,
      "default": true
    },
    blockScroll: {
      type: Boolean,
      "default": false
    }
  },
  style: DrawerStyle,
  provide: /* @__PURE__ */ __name(function provide() {
    return {
      $pcDrawer: this,
      $parentInstance: this
    };
  }, "provide")
};
var script = {
  name: "Drawer",
  "extends": script$1,
  inheritAttrs: false,
  emits: ["update:visible", "show", "after-show", "hide", "after-hide"],
  data: /* @__PURE__ */ __name(function data() {
    return {
      containerVisible: this.visible
    };
  }, "data"),
  container: null,
  mask: null,
  content: null,
  headerContainer: null,
  footerContainer: null,
  closeButton: null,
  outsideClickListener: null,
  documentKeydownListener: null,
  watch: {
    dismissable: /* @__PURE__ */ __name(function dismissable(newValue) {
      if (newValue) {
        this.enableDocumentSettings();
      } else {
        this.disableDocumentSettings();
      }
    }, "dismissable")
  },
  updated: /* @__PURE__ */ __name(function updated() {
    if (this.visible) {
      this.containerVisible = this.visible;
    }
  }, "updated"),
  beforeUnmount: /* @__PURE__ */ __name(function beforeUnmount() {
    this.disableDocumentSettings();
    if (this.mask && this.autoZIndex) {
      ZIndex.clear(this.mask);
    }
    this.container = null;
    this.mask = null;
  }, "beforeUnmount"),
  methods: {
    hide: /* @__PURE__ */ __name(function hide() {
      this.$emit("update:visible", false);
    }, "hide"),
    onEnter: /* @__PURE__ */ __name(function onEnter() {
      this.$emit("show");
      this.focus();
      this.bindDocumentKeyDownListener();
      if (this.autoZIndex) {
        ZIndex.set("modal", this.mask, this.baseZIndex || this.$primevue.config.zIndex.modal);
      }
    }, "onEnter"),
    onAfterEnter: /* @__PURE__ */ __name(function onAfterEnter() {
      this.enableDocumentSettings();
      this.$emit("after-show");
    }, "onAfterEnter"),
    onBeforeLeave: /* @__PURE__ */ __name(function onBeforeLeave() {
      if (this.modal) {
        !this.isUnstyled && addClass(this.mask, "p-overlay-mask-leave");
      }
    }, "onBeforeLeave"),
    onLeave: /* @__PURE__ */ __name(function onLeave() {
      this.$emit("hide");
    }, "onLeave"),
    onAfterLeave: /* @__PURE__ */ __name(function onAfterLeave() {
      if (this.autoZIndex) {
        ZIndex.clear(this.mask);
      }
      this.unbindDocumentKeyDownListener();
      this.containerVisible = false;
      this.disableDocumentSettings();
      this.$emit("after-hide");
    }, "onAfterLeave"),
    onMaskClick: /* @__PURE__ */ __name(function onMaskClick(event) {
      if (this.dismissable && this.modal && this.mask === event.target) {
        this.hide();
      }
    }, "onMaskClick"),
    focus: /* @__PURE__ */ __name(function focus$1() {
      var findFocusableElement = /* @__PURE__ */ __name(function findFocusableElement2(container) {
        return container && container.querySelector("[autofocus]");
      }, "findFocusableElement");
      var focusTarget = this.$slots.header && findFocusableElement(this.headerContainer);
      if (!focusTarget) {
        focusTarget = this.$slots["default"] && findFocusableElement(this.container);
        if (!focusTarget) {
          focusTarget = this.$slots.footer && findFocusableElement(this.footerContainer);
          if (!focusTarget) {
            focusTarget = this.closeButton;
          }
        }
      }
      focusTarget && focus(focusTarget);
    }, "focus$1"),
    enableDocumentSettings: /* @__PURE__ */ __name(function enableDocumentSettings() {
      if (this.dismissable && !this.modal) {
        this.bindOutsideClickListener();
      }
      if (this.blockScroll) {
        blockBodyScroll();
      }
    }, "enableDocumentSettings"),
    disableDocumentSettings: /* @__PURE__ */ __name(function disableDocumentSettings() {
      this.unbindOutsideClickListener();
      if (this.blockScroll) {
        unblockBodyScroll();
      }
    }, "disableDocumentSettings"),
    onKeydown: /* @__PURE__ */ __name(function onKeydown(event) {
      if (event.code === "Escape") {
        this.hide();
      }
    }, "onKeydown"),
    containerRef: /* @__PURE__ */ __name(function containerRef(el) {
      this.container = el;
    }, "containerRef"),
    maskRef: /* @__PURE__ */ __name(function maskRef(el) {
      this.mask = el;
    }, "maskRef"),
    contentRef: /* @__PURE__ */ __name(function contentRef(el) {
      this.content = el;
    }, "contentRef"),
    headerContainerRef: /* @__PURE__ */ __name(function headerContainerRef(el) {
      this.headerContainer = el;
    }, "headerContainerRef"),
    footerContainerRef: /* @__PURE__ */ __name(function footerContainerRef(el) {
      this.footerContainer = el;
    }, "footerContainerRef"),
    closeButtonRef: /* @__PURE__ */ __name(function closeButtonRef(el) {
      this.closeButton = el ? el.$el : void 0;
    }, "closeButtonRef"),
    bindDocumentKeyDownListener: /* @__PURE__ */ __name(function bindDocumentKeyDownListener() {
      if (!this.documentKeydownListener) {
        this.documentKeydownListener = this.onKeydown;
        document.addEventListener("keydown", this.documentKeydownListener);
      }
    }, "bindDocumentKeyDownListener"),
    unbindDocumentKeyDownListener: /* @__PURE__ */ __name(function unbindDocumentKeyDownListener() {
      if (this.documentKeydownListener) {
        document.removeEventListener("keydown", this.documentKeydownListener);
        this.documentKeydownListener = null;
      }
    }, "unbindDocumentKeyDownListener"),
    bindOutsideClickListener: /* @__PURE__ */ __name(function bindOutsideClickListener() {
      var _this = this;
      if (!this.outsideClickListener) {
        this.outsideClickListener = function(event) {
          if (_this.isOutsideClicked(event)) {
            _this.hide();
          }
        };
        document.addEventListener("click", this.outsideClickListener);
      }
    }, "bindOutsideClickListener"),
    unbindOutsideClickListener: /* @__PURE__ */ __name(function unbindOutsideClickListener() {
      if (this.outsideClickListener) {
        document.removeEventListener("click", this.outsideClickListener);
        this.outsideClickListener = null;
      }
    }, "unbindOutsideClickListener"),
    isOutsideClicked: /* @__PURE__ */ __name(function isOutsideClicked(event) {
      return this.container && !this.container.contains(event.target);
    }, "isOutsideClicked")
  },
  computed: {
    fullScreen: /* @__PURE__ */ __name(function fullScreen() {
      return this.position === "full";
    }, "fullScreen"),
    closeAriaLabel: /* @__PURE__ */ __name(function closeAriaLabel() {
      return this.$primevue.config.locale.aria ? this.$primevue.config.locale.aria.close : void 0;
    }, "closeAriaLabel")
  },
  directives: {
    focustrap: FocusTrap
  },
  components: {
    Button: script$3,
    Portal: script$4,
    TimesIcon: script$5
  }
};
var _hoisted_1 = ["aria-modal"];
function render(_ctx, _cache, $props, $setup, $data, $options) {
  var _component_Button = resolveComponent("Button");
  var _component_Portal = resolveComponent("Portal");
  var _directive_focustrap = resolveDirective("focustrap");
  return openBlock(), createBlock(_component_Portal, null, {
    "default": withCtx(function() {
      return [$data.containerVisible ? (openBlock(), createElementBlock("div", mergeProps({
        key: 0,
        ref: $options.maskRef,
        onMousedown: _cache[0] || (_cache[0] = function() {
          return $options.onMaskClick && $options.onMaskClick.apply($options, arguments);
        }),
        "class": _ctx.cx("mask"),
        style: _ctx.sx("mask", true, {
          position: _ctx.position,
          modal: _ctx.modal
        })
      }, _ctx.ptm("mask")), [createVNode(Transition, mergeProps({
        name: "p-drawer",
        onEnter: $options.onEnter,
        onAfterEnter: $options.onAfterEnter,
        onBeforeLeave: $options.onBeforeLeave,
        onLeave: $options.onLeave,
        onAfterLeave: $options.onAfterLeave,
        appear: ""
      }, _ctx.ptm("transition")), {
        "default": withCtx(function() {
          return [_ctx.visible ? withDirectives((openBlock(), createElementBlock("div", mergeProps({
            key: 0,
            ref: $options.containerRef,
            "class": _ctx.cx("root"),
            style: _ctx.sx("root"),
            role: "complementary",
            "aria-modal": _ctx.modal
          }, _ctx.ptmi("root")), [_ctx.$slots.container ? renderSlot(_ctx.$slots, "container", {
            key: 0,
            closeCallback: $options.hide
          }) : (openBlock(), createElementBlock(Fragment, {
            key: 1
          }, [createBaseVNode("div", mergeProps({
            ref: $options.headerContainerRef,
            "class": _ctx.cx("header")
          }, _ctx.ptm("header")), [renderSlot(_ctx.$slots, "header", {
            "class": normalizeClass(_ctx.cx("title"))
          }, function() {
            return [_ctx.header ? (openBlock(), createElementBlock("div", mergeProps({
              key: 0,
              "class": _ctx.cx("title")
            }, _ctx.ptm("title")), toDisplayString(_ctx.header), 17)) : createCommentVNode("", true)];
          }), _ctx.showCloseIcon ? (openBlock(), createBlock(_component_Button, mergeProps({
            key: 0,
            ref: $options.closeButtonRef,
            type: "button",
            "class": _ctx.cx("pcCloseButton"),
            "aria-label": $options.closeAriaLabel,
            unstyled: _ctx.unstyled,
            onClick: $options.hide
          }, _ctx.closeButtonProps, {
            pt: _ctx.ptm("pcCloseButton"),
            "data-pc-group-section": "iconcontainer"
          }), {
            icon: withCtx(function(slotProps) {
              return [renderSlot(_ctx.$slots, "closeicon", {}, function() {
                return [(openBlock(), createBlock(resolveDynamicComponent(_ctx.closeIcon ? "span" : "TimesIcon"), mergeProps({
                  "class": [_ctx.closeIcon, slotProps["class"]]
                }, _ctx.ptm("pcCloseButton")["icon"]), null, 16, ["class"]))];
              })];
            }),
            _: 3
          }, 16, ["class", "aria-label", "unstyled", "onClick", "pt"])) : createCommentVNode("", true)], 16), createBaseVNode("div", mergeProps({
            ref: $options.contentRef,
            "class": _ctx.cx("content")
          }, _ctx.ptm("content")), [renderSlot(_ctx.$slots, "default")], 16), _ctx.$slots.footer ? (openBlock(), createElementBlock("div", mergeProps({
            key: 0,
            ref: $options.footerContainerRef,
            "class": _ctx.cx("footer")
          }, _ctx.ptm("footer")), [renderSlot(_ctx.$slots, "footer")], 16)) : createCommentVNode("", true)], 64))], 16, _hoisted_1)), [[_directive_focustrap]]) : createCommentVNode("", true)];
        }),
        _: 3
      }, 16, ["onEnter", "onAfterEnter", "onBeforeLeave", "onLeave", "onAfterLeave"])], 16)) : createCommentVNode("", true)];
    }),
    _: 3
  });
}
__name(render, "render");
script.render = render;
var addonSerialize$2 = { exports: {} };
var addonSerialize = addonSerialize$2.exports;
(function(module, exports) {
  !function(e, t) {
    true ? module.exports = t() : false ? (void 0)([], t) : true ? exports.SerializeAddon = t() : e.SerializeAddon = t();
  }(commonjsGlobal, () => (() => {
    "use strict";
    var e = { 930: (e2, t2, s2) => {
      Object.defineProperty(t2, "__esModule", { value: true }), t2.ColorContrastCache = void 0;
      const r2 = s2(485);
      t2.ColorContrastCache = class {
        constructor() {
          this._color = new r2.TwoKeyMap(), this._css = new r2.TwoKeyMap();
        }
        setCss(e3, t3, s3) {
          this._css.set(e3, t3, s3);
        }
        getCss(e3, t3) {
          return this._css.get(e3, t3);
        }
        setColor(e3, t3, s3) {
          this._color.set(e3, t3, s3);
        }
        getColor(e3, t3) {
          return this._color.get(e3, t3);
        }
        clear() {
          this._color.clear(), this._css.clear();
        }
      };
    }, 997: function(e2, t2, s2) {
      var r2 = this && this.__decorate || function(e3, t3, s3, r3) {
        var o2, i2 = arguments.length, n2 = i2 < 3 ? t3 : null === r3 ? r3 = Object.getOwnPropertyDescriptor(t3, s3) : r3;
        if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) n2 = Reflect.decorate(e3, t3, s3, r3);
        else for (var l2 = e3.length - 1; l2 >= 0; l2--) (o2 = e3[l2]) && (n2 = (i2 < 3 ? o2(n2) : i2 > 3 ? o2(t3, s3, n2) : o2(t3, s3)) || n2);
        return i2 > 3 && n2 && Object.defineProperty(t3, s3, n2), n2;
      }, o = this && this.__param || function(e3, t3) {
        return function(s3, r3) {
          t3(s3, r3, e3);
        };
      };
      Object.defineProperty(t2, "__esModule", { value: true }), t2.ThemeService = t2.DEFAULT_ANSI_COLORS = void 0;
      const i = s2(930), n = s2(160), l = s2(345), a = s2(859), c = s2(97), h = n.css.toColor("#ffffff"), u = n.css.toColor("#000000"), _ = n.css.toColor("#ffffff"), d = n.css.toColor("#000000"), C = { css: "rgba(255, 255, 255, 0.3)", rgba: 4294967117 };
      t2.DEFAULT_ANSI_COLORS = Object.freeze((() => {
        const e3 = [n.css.toColor("#2e3436"), n.css.toColor("#cc0000"), n.css.toColor("#4e9a06"), n.css.toColor("#c4a000"), n.css.toColor("#3465a4"), n.css.toColor("#75507b"), n.css.toColor("#06989a"), n.css.toColor("#d3d7cf"), n.css.toColor("#555753"), n.css.toColor("#ef2929"), n.css.toColor("#8ae234"), n.css.toColor("#fce94f"), n.css.toColor("#729fcf"), n.css.toColor("#ad7fa8"), n.css.toColor("#34e2e2"), n.css.toColor("#eeeeec")], t3 = [0, 95, 135, 175, 215, 255];
        for (let s3 = 0; s3 < 216; s3++) {
          const r3 = t3[s3 / 36 % 6 | 0], o2 = t3[s3 / 6 % 6 | 0], i2 = t3[s3 % 6];
          e3.push({ css: n.channels.toCss(r3, o2, i2), rgba: n.channels.toRgba(r3, o2, i2) });
        }
        for (let t4 = 0; t4 < 24; t4++) {
          const s3 = 8 + 10 * t4;
          e3.push({ css: n.channels.toCss(s3, s3, s3), rgba: n.channels.toRgba(s3, s3, s3) });
        }
        return e3;
      })());
      let f = t2.ThemeService = class extends a.Disposable {
        get colors() {
          return this._colors;
        }
        constructor(e3) {
          super(), this._optionsService = e3, this._contrastCache = new i.ColorContrastCache(), this._halfContrastCache = new i.ColorContrastCache(), this._onChangeColors = this.register(new l.EventEmitter()), this.onChangeColors = this._onChangeColors.event, this._colors = { foreground: h, background: u, cursor: _, cursorAccent: d, selectionForeground: void 0, selectionBackgroundTransparent: C, selectionBackgroundOpaque: n.color.blend(u, C), selectionInactiveBackgroundTransparent: C, selectionInactiveBackgroundOpaque: n.color.blend(u, C), ansi: t2.DEFAULT_ANSI_COLORS.slice(), contrastCache: this._contrastCache, halfContrastCache: this._halfContrastCache }, this._updateRestoreColors(), this._setTheme(this._optionsService.rawOptions.theme), this.register(this._optionsService.onSpecificOptionChange("minimumContrastRatio", () => this._contrastCache.clear())), this.register(this._optionsService.onSpecificOptionChange("theme", () => this._setTheme(this._optionsService.rawOptions.theme)));
        }
        _setTheme(e3 = {}) {
          const s3 = this._colors;
          if (s3.foreground = g(e3.foreground, h), s3.background = g(e3.background, u), s3.cursor = g(e3.cursor, _), s3.cursorAccent = g(e3.cursorAccent, d), s3.selectionBackgroundTransparent = g(e3.selectionBackground, C), s3.selectionBackgroundOpaque = n.color.blend(s3.background, s3.selectionBackgroundTransparent), s3.selectionInactiveBackgroundTransparent = g(e3.selectionInactiveBackground, s3.selectionBackgroundTransparent), s3.selectionInactiveBackgroundOpaque = n.color.blend(s3.background, s3.selectionInactiveBackgroundTransparent), s3.selectionForeground = e3.selectionForeground ? g(e3.selectionForeground, n.NULL_COLOR) : void 0, s3.selectionForeground === n.NULL_COLOR && (s3.selectionForeground = void 0), n.color.isOpaque(s3.selectionBackgroundTransparent)) {
            const e4 = 0.3;
            s3.selectionBackgroundTransparent = n.color.opacity(s3.selectionBackgroundTransparent, e4);
          }
          if (n.color.isOpaque(s3.selectionInactiveBackgroundTransparent)) {
            const e4 = 0.3;
            s3.selectionInactiveBackgroundTransparent = n.color.opacity(s3.selectionInactiveBackgroundTransparent, e4);
          }
          if (s3.ansi = t2.DEFAULT_ANSI_COLORS.slice(), s3.ansi[0] = g(e3.black, t2.DEFAULT_ANSI_COLORS[0]), s3.ansi[1] = g(e3.red, t2.DEFAULT_ANSI_COLORS[1]), s3.ansi[2] = g(e3.green, t2.DEFAULT_ANSI_COLORS[2]), s3.ansi[3] = g(e3.yellow, t2.DEFAULT_ANSI_COLORS[3]), s3.ansi[4] = g(e3.blue, t2.DEFAULT_ANSI_COLORS[4]), s3.ansi[5] = g(e3.magenta, t2.DEFAULT_ANSI_COLORS[5]), s3.ansi[6] = g(e3.cyan, t2.DEFAULT_ANSI_COLORS[6]), s3.ansi[7] = g(e3.white, t2.DEFAULT_ANSI_COLORS[7]), s3.ansi[8] = g(e3.brightBlack, t2.DEFAULT_ANSI_COLORS[8]), s3.ansi[9] = g(e3.brightRed, t2.DEFAULT_ANSI_COLORS[9]), s3.ansi[10] = g(e3.brightGreen, t2.DEFAULT_ANSI_COLORS[10]), s3.ansi[11] = g(e3.brightYellow, t2.DEFAULT_ANSI_COLORS[11]), s3.ansi[12] = g(e3.brightBlue, t2.DEFAULT_ANSI_COLORS[12]), s3.ansi[13] = g(e3.brightMagenta, t2.DEFAULT_ANSI_COLORS[13]), s3.ansi[14] = g(e3.brightCyan, t2.DEFAULT_ANSI_COLORS[14]), s3.ansi[15] = g(e3.brightWhite, t2.DEFAULT_ANSI_COLORS[15]), e3.extendedAnsi) {
            const r3 = Math.min(s3.ansi.length - 16, e3.extendedAnsi.length);
            for (let o2 = 0; o2 < r3; o2++) s3.ansi[o2 + 16] = g(e3.extendedAnsi[o2], t2.DEFAULT_ANSI_COLORS[o2 + 16]);
          }
          this._contrastCache.clear(), this._halfContrastCache.clear(), this._updateRestoreColors(), this._onChangeColors.fire(this.colors);
        }
        restoreColor(e3) {
          this._restoreColor(e3), this._onChangeColors.fire(this.colors);
        }
        _restoreColor(e3) {
          if (void 0 !== e3) switch (e3) {
            case 256:
              this._colors.foreground = this._restoreColors.foreground;
              break;
            case 257:
              this._colors.background = this._restoreColors.background;
              break;
            case 258:
              this._colors.cursor = this._restoreColors.cursor;
              break;
            default:
              this._colors.ansi[e3] = this._restoreColors.ansi[e3];
          }
          else for (let e4 = 0; e4 < this._restoreColors.ansi.length; ++e4) this._colors.ansi[e4] = this._restoreColors.ansi[e4];
        }
        modifyColors(e3) {
          e3(this._colors), this._onChangeColors.fire(this.colors);
        }
        _updateRestoreColors() {
          this._restoreColors = { foreground: this._colors.foreground, background: this._colors.background, cursor: this._colors.cursor, ansi: this._colors.ansi.slice() };
        }
      };
      function g(e3, t3) {
        if (void 0 !== e3) try {
          return n.css.toColor(e3);
        } catch {
        }
        return t3;
      }
      __name(g, "g");
      t2.ThemeService = f = r2([o(0, c.IOptionsService)], f);
    }, 160: (e2, t2) => {
      Object.defineProperty(t2, "__esModule", { value: true }), t2.contrastRatio = t2.toPaddedHex = t2.rgba = t2.rgb = t2.css = t2.color = t2.channels = t2.NULL_COLOR = void 0;
      let s2 = 0, r2 = 0, o = 0, i = 0;
      var n, l, a, c, h;
      function u(e3) {
        const t3 = e3.toString(16);
        return t3.length < 2 ? "0" + t3 : t3;
      }
      __name(u, "u");
      function _(e3, t3) {
        return e3 < t3 ? (t3 + 0.05) / (e3 + 0.05) : (e3 + 0.05) / (t3 + 0.05);
      }
      __name(_, "_");
      t2.NULL_COLOR = { css: "#00000000", rgba: 0 }, function(e3) {
        e3.toCss = function(e4, t3, s3, r3) {
          return void 0 !== r3 ? `#${u(e4)}${u(t3)}${u(s3)}${u(r3)}` : `#${u(e4)}${u(t3)}${u(s3)}`;
        }, e3.toRgba = function(e4, t3, s3, r3 = 255) {
          return (e4 << 24 | t3 << 16 | s3 << 8 | r3) >>> 0;
        }, e3.toColor = function(t3, s3, r3, o2) {
          return { css: e3.toCss(t3, s3, r3, o2), rgba: e3.toRgba(t3, s3, r3, o2) };
        };
      }(n || (t2.channels = n = {})), function(e3) {
        function t3(e4, t4) {
          return i = Math.round(255 * t4), [s2, r2, o] = h.toChannels(e4.rgba), { css: n.toCss(s2, r2, o, i), rgba: n.toRgba(s2, r2, o, i) };
        }
        __name(t3, "t");
        e3.blend = function(e4, t4) {
          if (i = (255 & t4.rgba) / 255, 1 === i) return { css: t4.css, rgba: t4.rgba };
          const l2 = t4.rgba >> 24 & 255, a2 = t4.rgba >> 16 & 255, c2 = t4.rgba >> 8 & 255, h2 = e4.rgba >> 24 & 255, u2 = e4.rgba >> 16 & 255, _2 = e4.rgba >> 8 & 255;
          return s2 = h2 + Math.round((l2 - h2) * i), r2 = u2 + Math.round((a2 - u2) * i), o = _2 + Math.round((c2 - _2) * i), { css: n.toCss(s2, r2, o), rgba: n.toRgba(s2, r2, o) };
        }, e3.isOpaque = function(e4) {
          return 255 == (255 & e4.rgba);
        }, e3.ensureContrastRatio = function(e4, t4, s3) {
          const r3 = h.ensureContrastRatio(e4.rgba, t4.rgba, s3);
          if (r3) return n.toColor(r3 >> 24 & 255, r3 >> 16 & 255, r3 >> 8 & 255);
        }, e3.opaque = function(e4) {
          const t4 = (255 | e4.rgba) >>> 0;
          return [s2, r2, o] = h.toChannels(t4), { css: n.toCss(s2, r2, o), rgba: t4 };
        }, e3.opacity = t3, e3.multiplyOpacity = function(e4, s3) {
          return i = 255 & e4.rgba, t3(e4, i * s3 / 255);
        }, e3.toColorRGB = function(e4) {
          return [e4.rgba >> 24 & 255, e4.rgba >> 16 & 255, e4.rgba >> 8 & 255];
        };
      }(l || (t2.color = l = {})), function(e3) {
        let t3, l2;
        try {
          const e4 = document.createElement("canvas");
          e4.width = 1, e4.height = 1;
          const s3 = e4.getContext("2d", { willReadFrequently: true });
          s3 && (t3 = s3, t3.globalCompositeOperation = "copy", l2 = t3.createLinearGradient(0, 0, 1, 1));
        } catch {
        }
        e3.toColor = function(e4) {
          if (e4.match(/#[\da-f]{3,8}/i)) switch (e4.length) {
            case 4:
              return s2 = parseInt(e4.slice(1, 2).repeat(2), 16), r2 = parseInt(e4.slice(2, 3).repeat(2), 16), o = parseInt(e4.slice(3, 4).repeat(2), 16), n.toColor(s2, r2, o);
            case 5:
              return s2 = parseInt(e4.slice(1, 2).repeat(2), 16), r2 = parseInt(e4.slice(2, 3).repeat(2), 16), o = parseInt(e4.slice(3, 4).repeat(2), 16), i = parseInt(e4.slice(4, 5).repeat(2), 16), n.toColor(s2, r2, o, i);
            case 7:
              return { css: e4, rgba: (parseInt(e4.slice(1), 16) << 8 | 255) >>> 0 };
            case 9:
              return { css: e4, rgba: parseInt(e4.slice(1), 16) >>> 0 };
          }
          const a2 = e4.match(/rgba?\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*(,\s*(0|1|\d?\.(\d+))\s*)?\)/);
          if (a2) return s2 = parseInt(a2[1]), r2 = parseInt(a2[2]), o = parseInt(a2[3]), i = Math.round(255 * (void 0 === a2[5] ? 1 : parseFloat(a2[5]))), n.toColor(s2, r2, o, i);
          if (!t3 || !l2) throw new Error("css.toColor: Unsupported css format");
          if (t3.fillStyle = l2, t3.fillStyle = e4, "string" != typeof t3.fillStyle) throw new Error("css.toColor: Unsupported css format");
          if (t3.fillRect(0, 0, 1, 1), [s2, r2, o, i] = t3.getImageData(0, 0, 1, 1).data, 255 !== i) throw new Error("css.toColor: Unsupported css format");
          return { rgba: n.toRgba(s2, r2, o, i), css: e4 };
        };
      }(a || (t2.css = a = {})), function(e3) {
        function t3(e4, t4, s3) {
          const r3 = e4 / 255, o2 = t4 / 255, i2 = s3 / 255;
          return 0.2126 * (r3 <= 0.03928 ? r3 / 12.92 : Math.pow((r3 + 0.055) / 1.055, 2.4)) + 0.7152 * (o2 <= 0.03928 ? o2 / 12.92 : Math.pow((o2 + 0.055) / 1.055, 2.4)) + 0.0722 * (i2 <= 0.03928 ? i2 / 12.92 : Math.pow((i2 + 0.055) / 1.055, 2.4));
        }
        __name(t3, "t");
        e3.relativeLuminance = function(e4) {
          return t3(e4 >> 16 & 255, e4 >> 8 & 255, 255 & e4);
        }, e3.relativeLuminance2 = t3;
      }(c || (t2.rgb = c = {})), function(e3) {
        function t3(e4, t4, s3) {
          const r3 = e4 >> 24 & 255, o2 = e4 >> 16 & 255, i2 = e4 >> 8 & 255;
          let n2 = t4 >> 24 & 255, l3 = t4 >> 16 & 255, a2 = t4 >> 8 & 255, h2 = _(c.relativeLuminance2(n2, l3, a2), c.relativeLuminance2(r3, o2, i2));
          for (; h2 < s3 && (n2 > 0 || l3 > 0 || a2 > 0); ) n2 -= Math.max(0, Math.ceil(0.1 * n2)), l3 -= Math.max(0, Math.ceil(0.1 * l3)), a2 -= Math.max(0, Math.ceil(0.1 * a2)), h2 = _(c.relativeLuminance2(n2, l3, a2), c.relativeLuminance2(r3, o2, i2));
          return (n2 << 24 | l3 << 16 | a2 << 8 | 255) >>> 0;
        }
        __name(t3, "t");
        function l2(e4, t4, s3) {
          const r3 = e4 >> 24 & 255, o2 = e4 >> 16 & 255, i2 = e4 >> 8 & 255;
          let n2 = t4 >> 24 & 255, l3 = t4 >> 16 & 255, a2 = t4 >> 8 & 255, h2 = _(c.relativeLuminance2(n2, l3, a2), c.relativeLuminance2(r3, o2, i2));
          for (; h2 < s3 && (n2 < 255 || l3 < 255 || a2 < 255); ) n2 = Math.min(255, n2 + Math.ceil(0.1 * (255 - n2))), l3 = Math.min(255, l3 + Math.ceil(0.1 * (255 - l3))), a2 = Math.min(255, a2 + Math.ceil(0.1 * (255 - a2))), h2 = _(c.relativeLuminance2(n2, l3, a2), c.relativeLuminance2(r3, o2, i2));
          return (n2 << 24 | l3 << 16 | a2 << 8 | 255) >>> 0;
        }
        __name(l2, "l");
        e3.blend = function(e4, t4) {
          if (i = (255 & t4) / 255, 1 === i) return t4;
          const l3 = t4 >> 24 & 255, a2 = t4 >> 16 & 255, c2 = t4 >> 8 & 255, h2 = e4 >> 24 & 255, u2 = e4 >> 16 & 255, _2 = e4 >> 8 & 255;
          return s2 = h2 + Math.round((l3 - h2) * i), r2 = u2 + Math.round((a2 - u2) * i), o = _2 + Math.round((c2 - _2) * i), n.toRgba(s2, r2, o);
        }, e3.ensureContrastRatio = function(e4, s3, r3) {
          const o2 = c.relativeLuminance(e4 >> 8), i2 = c.relativeLuminance(s3 >> 8);
          if (_(o2, i2) < r3) {
            if (i2 < o2) {
              const i3 = t3(e4, s3, r3), n3 = _(o2, c.relativeLuminance(i3 >> 8));
              if (n3 < r3) {
                const t4 = l2(e4, s3, r3);
                return n3 > _(o2, c.relativeLuminance(t4 >> 8)) ? i3 : t4;
              }
              return i3;
            }
            const n2 = l2(e4, s3, r3), a2 = _(o2, c.relativeLuminance(n2 >> 8));
            if (a2 < r3) {
              const i3 = t3(e4, s3, r3);
              return a2 > _(o2, c.relativeLuminance(i3 >> 8)) ? n2 : i3;
            }
            return n2;
          }
        }, e3.reduceLuminance = t3, e3.increaseLuminance = l2, e3.toChannels = function(e4) {
          return [e4 >> 24 & 255, e4 >> 16 & 255, e4 >> 8 & 255, 255 & e4];
        };
      }(h || (t2.rgba = h = {})), t2.toPaddedHex = u, t2.contrastRatio = _;
    }, 345: (e2, t2) => {
      Object.defineProperty(t2, "__esModule", { value: true }), t2.runAndSubscribe = t2.forwardEvent = t2.EventEmitter = void 0, t2.EventEmitter = class {
        constructor() {
          this._listeners = [], this._disposed = false;
        }
        get event() {
          return this._event || (this._event = (e3) => (this._listeners.push(e3), { dispose: /* @__PURE__ */ __name(() => {
            if (!this._disposed) {
              for (let t3 = 0; t3 < this._listeners.length; t3++) if (this._listeners[t3] === e3) return void this._listeners.splice(t3, 1);
            }
          }, "dispose") })), this._event;
        }
        fire(e3, t3) {
          const s2 = [];
          for (let e4 = 0; e4 < this._listeners.length; e4++) s2.push(this._listeners[e4]);
          for (let r2 = 0; r2 < s2.length; r2++) s2[r2].call(void 0, e3, t3);
        }
        dispose() {
          this.clearListeners(), this._disposed = true;
        }
        clearListeners() {
          this._listeners && (this._listeners.length = 0);
        }
      }, t2.forwardEvent = function(e3, t3) {
        return e3((e4) => t3.fire(e4));
      }, t2.runAndSubscribe = function(e3, t3) {
        return t3(void 0), e3((e4) => t3(e4));
      };
    }, 859: (e2, t2) => {
      function s2(e3) {
        for (const t3 of e3) t3.dispose();
        e3.length = 0;
      }
      __name(s2, "s");
      Object.defineProperty(t2, "__esModule", { value: true }), t2.getDisposeArrayDisposable = t2.disposeArray = t2.toDisposable = t2.MutableDisposable = t2.Disposable = void 0, t2.Disposable = class {
        constructor() {
          this._disposables = [], this._isDisposed = false;
        }
        dispose() {
          this._isDisposed = true;
          for (const e3 of this._disposables) e3.dispose();
          this._disposables.length = 0;
        }
        register(e3) {
          return this._disposables.push(e3), e3;
        }
        unregister(e3) {
          const t3 = this._disposables.indexOf(e3);
          -1 !== t3 && this._disposables.splice(t3, 1);
        }
      }, t2.MutableDisposable = class {
        constructor() {
          this._isDisposed = false;
        }
        get value() {
          return this._isDisposed ? void 0 : this._value;
        }
        set value(e3) {
          this._isDisposed || e3 === this._value || (this._value?.dispose(), this._value = e3);
        }
        clear() {
          this.value = void 0;
        }
        dispose() {
          this._isDisposed = true, this._value?.dispose(), this._value = void 0;
        }
      }, t2.toDisposable = function(e3) {
        return { dispose: e3 };
      }, t2.disposeArray = s2, t2.getDisposeArrayDisposable = function(e3) {
        return { dispose: /* @__PURE__ */ __name(() => s2(e3), "dispose") };
      };
    }, 485: (e2, t2) => {
      Object.defineProperty(t2, "__esModule", { value: true }), t2.FourKeyMap = t2.TwoKeyMap = void 0;
      class s2 {
        static {
          __name(this, "s");
        }
        constructor() {
          this._data = {};
        }
        set(e3, t3, s3) {
          this._data[e3] || (this._data[e3] = {}), this._data[e3][t3] = s3;
        }
        get(e3, t3) {
          return this._data[e3] ? this._data[e3][t3] : void 0;
        }
        clear() {
          this._data = {};
        }
      }
      t2.TwoKeyMap = s2, t2.FourKeyMap = class {
        constructor() {
          this._data = new s2();
        }
        set(e3, t3, r2, o, i) {
          this._data.get(e3, t3) || this._data.set(e3, t3, new s2()), this._data.get(e3, t3).set(r2, o, i);
        }
        get(e3, t3, s3, r2) {
          return this._data.get(e3, t3)?.get(s3, r2);
        }
        clear() {
          this._data.clear();
        }
      };
    }, 726: (e2, t2) => {
      Object.defineProperty(t2, "__esModule", { value: true }), t2.createDecorator = t2.getServiceDependencies = t2.serviceRegistry = void 0;
      const s2 = "di$target", r2 = "di$dependencies";
      t2.serviceRegistry = /* @__PURE__ */ new Map(), t2.getServiceDependencies = function(e3) {
        return e3[r2] || [];
      }, t2.createDecorator = function(e3) {
        if (t2.serviceRegistry.has(e3)) return t2.serviceRegistry.get(e3);
        const o = /* @__PURE__ */ __name(function(e4, t3, i) {
          if (3 !== arguments.length) throw new Error("@IServiceName-decorator can only be used to decorate a parameter");
          !function(e5, t4, o2) {
            t4[s2] === t4 ? t4[r2].push({ id: e5, index: o2 }) : (t4[r2] = [{ id: e5, index: o2 }], t4[s2] = t4);
          }(o, e4, i);
        }, "o");
        return o.toString = () => e3, t2.serviceRegistry.set(e3, o), o;
      };
    }, 97: (e2, t2, s2) => {
      Object.defineProperty(t2, "__esModule", { value: true }), t2.IDecorationService = t2.IUnicodeService = t2.IOscLinkService = t2.IOptionsService = t2.ILogService = t2.LogLevelEnum = t2.IInstantiationService = t2.ICharsetService = t2.ICoreService = t2.ICoreMouseService = t2.IBufferService = void 0;
      const r2 = s2(726);
      var o;
      t2.IBufferService = (0, r2.createDecorator)("BufferService"), t2.ICoreMouseService = (0, r2.createDecorator)("CoreMouseService"), t2.ICoreService = (0, r2.createDecorator)("CoreService"), t2.ICharsetService = (0, r2.createDecorator)("CharsetService"), t2.IInstantiationService = (0, r2.createDecorator)("InstantiationService"), function(e3) {
        e3[e3.TRACE = 0] = "TRACE", e3[e3.DEBUG = 1] = "DEBUG", e3[e3.INFO = 2] = "INFO", e3[e3.WARN = 3] = "WARN", e3[e3.ERROR = 4] = "ERROR", e3[e3.OFF = 5] = "OFF";
      }(o || (t2.LogLevelEnum = o = {})), t2.ILogService = (0, r2.createDecorator)("LogService"), t2.IOptionsService = (0, r2.createDecorator)("OptionsService"), t2.IOscLinkService = (0, r2.createDecorator)("OscLinkService"), t2.IUnicodeService = (0, r2.createDecorator)("UnicodeService"), t2.IDecorationService = (0, r2.createDecorator)("DecorationService");
    } }, t = {};
    function s(r2) {
      var o = t[r2];
      if (void 0 !== o) return o.exports;
      var i = t[r2] = { exports: {} };
      return e[r2].call(i.exports, i, i.exports, s), i.exports;
    }
    __name(s, "s");
    var r = {};
    return (() => {
      var e2 = r;
      Object.defineProperty(e2, "__esModule", { value: true }), e2.HTMLSerializeHandler = e2.SerializeAddon = void 0;
      const t2 = s(997);
      function o(e3, t3, s2) {
        return Math.max(t3, Math.min(e3, s2));
      }
      __name(o, "o");
      class i {
        static {
          __name(this, "i");
        }
        constructor(e3) {
          this._buffer = e3;
        }
        serialize(e3, t3) {
          const s2 = this._buffer.getNullCell(), r2 = this._buffer.getNullCell();
          let o2 = s2;
          const i2 = e3.start.y, n2 = e3.end.y, l2 = e3.start.x, a2 = e3.end.x;
          this._beforeSerialize(n2 - i2, i2, n2);
          for (let t4 = i2; t4 <= n2; t4++) {
            const i3 = this._buffer.getLine(t4);
            if (i3) {
              const n3 = t4 === e3.start.y ? l2 : 0, c2 = t4 === e3.end.y ? a2 : i3.length;
              for (let e4 = n3; e4 < c2; e4++) {
                const n4 = i3.getCell(e4, o2 === s2 ? r2 : s2);
                n4 ? (this._nextCell(n4, o2, t4, e4), o2 = n4) : console.warn(`Can't get cell at row=${t4}, col=${e4}`);
              }
            }
            this._rowEnd(t4, t4 === n2);
          }
          return this._afterSerialize(), this._serializeString(t3);
        }
        _nextCell(e3, t3, s2, r2) {
        }
        _rowEnd(e3, t3) {
        }
        _beforeSerialize(e3, t3, s2) {
        }
        _afterSerialize() {
        }
        _serializeString(e3) {
          return "";
        }
      }
      function n(e3, t3) {
        return e3.getFgColorMode() === t3.getFgColorMode() && e3.getFgColor() === t3.getFgColor();
      }
      __name(n, "n");
      function l(e3, t3) {
        return e3.getBgColorMode() === t3.getBgColorMode() && e3.getBgColor() === t3.getBgColor();
      }
      __name(l, "l");
      function a(e3, t3) {
        return e3.isInverse() === t3.isInverse() && e3.isBold() === t3.isBold() && e3.isUnderline() === t3.isUnderline() && e3.isOverline() === t3.isOverline() && e3.isBlink() === t3.isBlink() && e3.isInvisible() === t3.isInvisible() && e3.isItalic() === t3.isItalic() && e3.isDim() === t3.isDim() && e3.isStrikethrough() === t3.isStrikethrough();
      }
      __name(a, "a");
      class c extends i {
        static {
          __name(this, "c");
        }
        constructor(e3, t3) {
          super(e3), this._terminal = t3, this._rowIndex = 0, this._allRows = new Array(), this._allRowSeparators = new Array(), this._currentRow = "", this._nullCellCount = 0, this._cursorStyle = this._buffer.getNullCell(), this._cursorStyleRow = 0, this._cursorStyleCol = 0, this._backgroundCell = this._buffer.getNullCell(), this._firstRow = 0, this._lastCursorRow = 0, this._lastCursorCol = 0, this._lastContentCursorRow = 0, this._lastContentCursorCol = 0, this._thisRowLastChar = this._buffer.getNullCell(), this._thisRowLastSecondChar = this._buffer.getNullCell(), this._nextRowFirstChar = this._buffer.getNullCell();
        }
        _beforeSerialize(e3, t3, s2) {
          this._allRows = new Array(e3), this._lastContentCursorRow = t3, this._lastCursorRow = t3, this._firstRow = t3;
        }
        _rowEnd(e3, t3) {
          this._nullCellCount > 0 && !l(this._cursorStyle, this._backgroundCell) && (this._currentRow += `\x1B[${this._nullCellCount}X`);
          let s2 = "";
          if (!t3) {
            e3 - this._firstRow >= this._terminal.rows && this._buffer.getLine(this._cursorStyleRow)?.getCell(this._cursorStyleCol, this._backgroundCell);
            const t4 = this._buffer.getLine(e3), r2 = this._buffer.getLine(e3 + 1);
            if (r2.isWrapped) {
              s2 = "";
              const o2 = t4.getCell(t4.length - 1, this._thisRowLastChar), i2 = t4.getCell(t4.length - 2, this._thisRowLastSecondChar), n2 = r2.getCell(0, this._nextRowFirstChar), a2 = n2.getWidth() > 1;
              let c2 = false;
              (n2.getChars() && a2 ? this._nullCellCount <= 1 : this._nullCellCount <= 0) && ((o2.getChars() || 0 === o2.getWidth()) && l(o2, n2) && (c2 = true), a2 && (i2.getChars() || 0 === i2.getWidth()) && l(o2, n2) && l(i2, n2) && (c2 = true)), c2 || (s2 = "-".repeat(this._nullCellCount + 1), s2 += "\x1B[1D\x1B[1X", this._nullCellCount > 0 && (s2 += "\x1B[A", s2 += `\x1B[${t4.length - this._nullCellCount}C`, s2 += `\x1B[${this._nullCellCount}X`, s2 += `\x1B[${t4.length - this._nullCellCount}D`, s2 += "\x1B[B"), this._lastContentCursorRow = e3 + 1, this._lastContentCursorCol = 0, this._lastCursorRow = e3 + 1, this._lastCursorCol = 0);
            } else s2 = "\r\n", this._lastCursorRow = e3 + 1, this._lastCursorCol = 0;
          }
          this._allRows[this._rowIndex] = this._currentRow, this._allRowSeparators[this._rowIndex++] = s2, this._currentRow = "", this._nullCellCount = 0;
        }
        _diffStyle(e3, t3) {
          const s2 = [], r2 = !n(e3, t3), o2 = !l(e3, t3), i2 = !a(e3, t3);
          if (r2 || o2 || i2) if (e3.isAttributeDefault()) t3.isAttributeDefault() || s2.push(0);
          else {
            if (r2) {
              const t4 = e3.getFgColor();
              e3.isFgRGB() ? s2.push(38, 2, t4 >>> 16 & 255, t4 >>> 8 & 255, 255 & t4) : e3.isFgPalette() ? t4 >= 16 ? s2.push(38, 5, t4) : s2.push(8 & t4 ? 90 + (7 & t4) : 30 + (7 & t4)) : s2.push(39);
            }
            if (o2) {
              const t4 = e3.getBgColor();
              e3.isBgRGB() ? s2.push(48, 2, t4 >>> 16 & 255, t4 >>> 8 & 255, 255 & t4) : e3.isBgPalette() ? t4 >= 16 ? s2.push(48, 5, t4) : s2.push(8 & t4 ? 100 + (7 & t4) : 40 + (7 & t4)) : s2.push(49);
            }
            i2 && (e3.isInverse() !== t3.isInverse() && s2.push(e3.isInverse() ? 7 : 27), e3.isBold() !== t3.isBold() && s2.push(e3.isBold() ? 1 : 22), e3.isUnderline() !== t3.isUnderline() && s2.push(e3.isUnderline() ? 4 : 24), e3.isOverline() !== t3.isOverline() && s2.push(e3.isOverline() ? 53 : 55), e3.isBlink() !== t3.isBlink() && s2.push(e3.isBlink() ? 5 : 25), e3.isInvisible() !== t3.isInvisible() && s2.push(e3.isInvisible() ? 8 : 28), e3.isItalic() !== t3.isItalic() && s2.push(e3.isItalic() ? 3 : 23), e3.isDim() !== t3.isDim() && s2.push(e3.isDim() ? 2 : 22), e3.isStrikethrough() !== t3.isStrikethrough() && s2.push(e3.isStrikethrough() ? 9 : 29));
          }
          return s2;
        }
        _nextCell(e3, t3, s2, r2) {
          if (0 === e3.getWidth()) return;
          const o2 = "" === e3.getChars(), i2 = this._diffStyle(e3, this._cursorStyle);
          if (o2 ? !l(this._cursorStyle, e3) : i2.length > 0) {
            this._nullCellCount > 0 && (l(this._cursorStyle, this._backgroundCell) || (this._currentRow += `\x1B[${this._nullCellCount}X`), this._currentRow += `\x1B[${this._nullCellCount}C`, this._nullCellCount = 0), this._lastContentCursorRow = this._lastCursorRow = s2, this._lastContentCursorCol = this._lastCursorCol = r2, this._currentRow += `\x1B[${i2.join(";")}m`;
            const e4 = this._buffer.getLine(s2);
            void 0 !== e4 && (e4.getCell(r2, this._cursorStyle), this._cursorStyleRow = s2, this._cursorStyleCol = r2);
          }
          o2 ? this._nullCellCount += e3.getWidth() : (this._nullCellCount > 0 && (l(this._cursorStyle, this._backgroundCell) || (this._currentRow += `\x1B[${this._nullCellCount}X`), this._currentRow += `\x1B[${this._nullCellCount}C`, this._nullCellCount = 0), this._currentRow += e3.getChars(), this._lastContentCursorRow = this._lastCursorRow = s2, this._lastContentCursorCol = this._lastCursorCol = r2 + e3.getWidth());
        }
        _serializeString(e3) {
          let t3 = this._allRows.length;
          this._buffer.length - this._firstRow <= this._terminal.rows && (t3 = this._lastContentCursorRow + 1 - this._firstRow, this._lastCursorCol = this._lastContentCursorCol, this._lastCursorRow = this._lastContentCursorRow);
          let s2 = "";
          for (let e4 = 0; e4 < t3; e4++) s2 += this._allRows[e4], e4 + 1 < t3 && (s2 += this._allRowSeparators[e4]);
          if (!e3) {
            const e4 = this._buffer.baseY + this._buffer.cursorY, t4 = this._buffer.cursorX, o3 = /* @__PURE__ */ __name((e5) => {
              e5 > 0 ? s2 += `\x1B[${e5}C` : e5 < 0 && (s2 += `\x1B[${-e5}D`);
            }, "o");
            (e4 !== this._lastCursorRow || t4 !== this._lastCursorCol) && ((r2 = e4 - this._lastCursorRow) > 0 ? s2 += `\x1B[${r2}B` : r2 < 0 && (s2 += `\x1B[${-r2}A`), o3(t4 - this._lastCursorCol));
          }
          var r2;
          const o2 = this._terminal._core._inputHandler._curAttrData, i2 = this._diffStyle(o2, this._cursorStyle);
          return i2.length > 0 && (s2 += `\x1B[${i2.join(";")}m`), s2;
        }
      }
      e2.SerializeAddon = class {
        activate(e3) {
          this._terminal = e3;
        }
        _serializeBufferByScrollback(e3, t3, s2) {
          const r2 = t3.length, i2 = void 0 === s2 ? r2 : o(s2 + e3.rows, 0, r2);
          return this._serializeBufferByRange(e3, t3, { start: r2 - i2, end: r2 - 1 }, false);
        }
        _serializeBufferByRange(e3, t3, s2, r2) {
          return new c(t3, e3).serialize({ start: { x: 0, y: "number" == typeof s2.start ? s2.start : s2.start.line }, end: { x: e3.cols, y: "number" == typeof s2.end ? s2.end : s2.end.line } }, r2);
        }
        _serializeBufferAsHTML(e3, t3) {
          const s2 = e3.buffer.active, r2 = new h(s2, e3, t3);
          if (!t3.onlySelection) {
            const i3 = s2.length, n2 = t3.scrollback, l2 = void 0 === n2 ? i3 : o(n2 + e3.rows, 0, i3);
            return r2.serialize({ start: { x: 0, y: i3 - l2 }, end: { x: e3.cols, y: i3 - 1 } });
          }
          const i2 = this._terminal?.getSelectionPosition();
          return void 0 !== i2 ? r2.serialize({ start: { x: i2.start.x, y: i2.start.y }, end: { x: i2.end.x, y: i2.end.y } }) : "";
        }
        _serializeModes(e3) {
          let t3 = "";
          const s2 = e3.modes;
          if (s2.applicationCursorKeysMode && (t3 += "\x1B[?1h"), s2.applicationKeypadMode && (t3 += "\x1B[?66h"), s2.bracketedPasteMode && (t3 += "\x1B[?2004h"), s2.insertMode && (t3 += "\x1B[4h"), s2.originMode && (t3 += "\x1B[?6h"), s2.reverseWraparoundMode && (t3 += "\x1B[?45h"), s2.sendFocusMode && (t3 += "\x1B[?1004h"), false === s2.wraparoundMode && (t3 += "\x1B[?7l"), "none" !== s2.mouseTrackingMode) switch (s2.mouseTrackingMode) {
            case "x10":
              t3 += "\x1B[?9h";
              break;
            case "vt200":
              t3 += "\x1B[?1000h";
              break;
            case "drag":
              t3 += "\x1B[?1002h";
              break;
            case "any":
              t3 += "\x1B[?1003h";
          }
          return t3;
        }
        serialize(e3) {
          if (!this._terminal) throw new Error("Cannot use addon until it has been loaded");
          let t3 = e3?.range ? this._serializeBufferByRange(this._terminal, this._terminal.buffer.normal, e3.range, true) : this._serializeBufferByScrollback(this._terminal, this._terminal.buffer.normal, e3?.scrollback);
          return e3?.excludeAltBuffer || "alternate" !== this._terminal.buffer.active.type || (t3 += `\x1B[?1049h\x1B[H${this._serializeBufferByScrollback(this._terminal, this._terminal.buffer.alternate, void 0)}`), e3?.excludeModes || (t3 += this._serializeModes(this._terminal)), t3;
        }
        serializeAsHTML(e3) {
          if (!this._terminal) throw new Error("Cannot use addon until it has been loaded");
          return this._serializeBufferAsHTML(this._terminal, e3 || {});
        }
        dispose() {
        }
      };
      class h extends i {
        static {
          __name(this, "h");
        }
        constructor(e3, s2, r2) {
          super(e3), this._terminal = s2, this._options = r2, this._currentRow = "", this._htmlContent = "", s2._core._themeService ? this._ansiColors = s2._core._themeService.colors.ansi : this._ansiColors = t2.DEFAULT_ANSI_COLORS;
        }
        _padStart(e3, t3, s2) {
          return t3 >>= 0, s2 = s2 ?? " ", e3.length > t3 ? e3 : ((t3 -= e3.length) > s2.length && (s2 += s2.repeat(t3 / s2.length)), s2.slice(0, t3) + e3);
        }
        _beforeSerialize(e3, t3, s2) {
          this._htmlContent += "<html><body><!--StartFragment--><pre>";
          let r2 = "#000000", o2 = "#ffffff";
          this._options.includeGlobalBackground && (r2 = this._terminal.options.theme?.foreground ?? "#ffffff", o2 = this._terminal.options.theme?.background ?? "#000000");
          const i2 = [];
          i2.push("color: " + r2 + ";"), i2.push("background-color: " + o2 + ";"), i2.push("font-family: " + this._terminal.options.fontFamily + ";"), i2.push("font-size: " + this._terminal.options.fontSize + "px;"), this._htmlContent += "<div style='" + i2.join(" ") + "'>";
        }
        _afterSerialize() {
          this._htmlContent += "</div>", this._htmlContent += "</pre><!--EndFragment--></body></html>";
        }
        _rowEnd(e3, t3) {
          this._htmlContent += "<div><span>" + this._currentRow + "</span></div>", this._currentRow = "";
        }
        _getHexColor(e3, t3) {
          const s2 = t3 ? e3.getFgColor() : e3.getBgColor();
          return (t3 ? e3.isFgRGB() : e3.isBgRGB()) ? "#" + [s2 >> 16 & 255, s2 >> 8 & 255, 255 & s2].map((e4) => this._padStart(e4.toString(16), 2, "0")).join("") : (t3 ? e3.isFgPalette() : e3.isBgPalette()) ? this._ansiColors[s2].css : void 0;
        }
        _diffStyle(e3, t3) {
          const s2 = [], r2 = !n(e3, t3), o2 = !l(e3, t3), i2 = !a(e3, t3);
          if (r2 || o2 || i2) {
            const t4 = this._getHexColor(e3, true);
            t4 && s2.push("color: " + t4 + ";");
            const r3 = this._getHexColor(e3, false);
            return r3 && s2.push("background-color: " + r3 + ";"), e3.isInverse() && s2.push("color: #000000; background-color: #BFBFBF;"), e3.isBold() && s2.push("font-weight: bold;"), e3.isUnderline() && e3.isOverline() ? s2.push("text-decoration: overline underline;") : e3.isUnderline() ? s2.push("text-decoration: underline;") : e3.isOverline() && s2.push("text-decoration: overline;"), e3.isBlink() && s2.push("text-decoration: blink;"), e3.isInvisible() && s2.push("visibility: hidden;"), e3.isItalic() && s2.push("font-style: italic;"), e3.isDim() && s2.push("opacity: 0.5;"), e3.isStrikethrough() && s2.push("text-decoration: line-through;"), s2;
          }
        }
        _nextCell(e3, t3, s2, r2) {
          if (0 === e3.getWidth()) return;
          const o2 = "" === e3.getChars(), i2 = this._diffStyle(e3, t3);
          i2 && (this._currentRow += 0 === i2.length ? "</span><span>" : "</span><span style='" + i2.join(" ") + "'>"), this._currentRow += o2 ? " " : e3.getChars();
        }
        _serializeString() {
          return this._htmlContent;
        }
      }
      e2.HTMLSerializeHandler = h;
    })(), r;
  })());
})(addonSerialize$2, addonSerialize$2.exports);
var addonSerializeExports = addonSerialize$2.exports;
const addonSerialize$1 = /* @__PURE__ */ getDefaultExportFromCjs(addonSerializeExports);
function useTerminalBuffer() {
  const serializeAddon = new addonSerializeExports.SerializeAddon();
  const terminal = markRaw(new xtermExports.Terminal({ convertEol: true }));
  const copyTo = /* @__PURE__ */ __name((destinationTerminal) => {
    destinationTerminal.write(serializeAddon.serialize());
  }, "copyTo");
  const write = /* @__PURE__ */ __name((message) => terminal.write(message), "write");
  const serialize = /* @__PURE__ */ __name(() => serializeAddon.serialize(), "serialize");
  onMounted(() => {
    terminal.loadAddon(serializeAddon);
  });
  onUnmounted(() => {
    terminal.dispose();
  });
  return {
    copyTo,
    serialize,
    write
  };
}
__name(useTerminalBuffer, "useTerminalBuffer");
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "TerminalOutputDrawer",
  props: /* @__PURE__ */ mergeModels({
    header: {},
    defaultMessage: {}
  }, {
    "modelValue": { type: Boolean, ...{ required: true } },
    "modelModifiers": {}
  }),
  emits: ["update:modelValue"],
  setup(__props) {
    const terminalVisible = useModel(__props, "modelValue");
    const props = __props;
    const electron = electronAPI();
    const buffer = useTerminalBuffer();
    let xterm = null;
    const terminalCreated = /* @__PURE__ */ __name(({ terminal, useAutoSize }, root2) => {
      xterm = terminal;
      useAutoSize({ root: root2, autoRows: true, autoCols: true });
      terminal.write(props.defaultMessage);
      buffer.copyTo(terminal);
      terminal.options.cursorBlink = false;
      terminal.options.cursorStyle = "bar";
      terminal.options.cursorInactiveStyle = "bar";
      terminal.options.disableStdin = true;
    }, "terminalCreated");
    const terminalUnmounted = /* @__PURE__ */ __name(() => {
      xterm = null;
    }, "terminalUnmounted");
    onMounted(async () => {
      electron.onLogMessage((message) => {
        buffer.write(message);
        xterm?.write(message);
      });
    });
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(script), {
        visible: terminalVisible.value,
        "onUpdate:visible": _cache[0] || (_cache[0] = ($event) => terminalVisible.value = $event),
        header: _ctx.header,
        position: "bottom",
        style: { "height": "max(50vh, 34rem)" }
      }, {
        default: withCtx(() => [
          createVNode(BaseTerminal, {
            onCreated: terminalCreated,
            onUnmounted: terminalUnmounted
          })
        ]),
        _: 1
      }, 8, ["visible", "header"]);
    };
  }
});
export {
  _sfc_main as _,
  script as s
};
//# sourceMappingURL=TerminalOutputDrawer-BgTEspHP.js.map
