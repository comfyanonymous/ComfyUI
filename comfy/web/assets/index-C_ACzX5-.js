var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });
import { ce as BaseStyle, cf as script$2, cw as ZIndex, cr as addClass, ci as focus, d4 as blockBodyScroll, d6 as unblockBodyScroll, d7 as FocusTrap, l as script$3, cJ as script$4, cU as script$5, co as resolveComponent, r as resolveDirective, o as openBlock, y as createBlock, z as withCtx, f as createElementBlock, aE as mergeProps, k as createVNode, c8 as Transition, i as withDirectives, A as renderSlot, F as Fragment, m as createBaseVNode, Z as normalizeClass, E as toDisplayString, B as createCommentVNode, C as resolveDynamicComponent } from "./index-DIgj6hpb.js";
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
export {
  script as s
};
//# sourceMappingURL=index-C_ACzX5-.js.map
