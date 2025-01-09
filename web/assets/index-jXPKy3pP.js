var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });
import { B as BaseStyle, q as script$2, ak as UniqueComponentId, c9 as script$4, l as script$5, S as Ripple, aB as resolveComponent, o as openBlock, f as createElementBlock, D as mergeProps, H as createBaseVNode, J as renderSlot, T as normalizeClass, X as toDisplayString, I as createCommentVNode, k as createBlock, M as withCtx, G as resolveDynamicComponent, N as createVNode, aC as Transition, i as withDirectives, v as vShow } from "./index-DjNHn37O.js";
import { s as script$3 } from "./index-5HFeZax4.js";
var theme = /* @__PURE__ */ __name(function theme2(_ref) {
  var dt = _ref.dt;
  return "\n.p-panel {\n    border: 1px solid ".concat(dt("panel.border.color"), ";\n    border-radius: ").concat(dt("panel.border.radius"), ";\n    background: ").concat(dt("panel.background"), ";\n    color: ").concat(dt("panel.color"), ";\n}\n\n.p-panel-header {\n    display: flex;\n    justify-content: space-between;\n    align-items: center;\n    padding: ").concat(dt("panel.header.padding"), ";\n    background: ").concat(dt("panel.header.background"), ";\n    color: ").concat(dt("panel.header.color"), ";\n    border-style: solid;\n    border-width: ").concat(dt("panel.header.border.width"), ";\n    border-color: ").concat(dt("panel.header.border.color"), ";\n    border-radius: ").concat(dt("panel.header.border.radius"), ";\n}\n\n.p-panel-toggleable .p-panel-header {\n    padding: ").concat(dt("panel.toggleable.header.padding"), ";\n}\n\n.p-panel-title {\n    line-height: 1;\n    font-weight: ").concat(dt("panel.title.font.weight"), ";\n}\n\n.p-panel-content {\n    padding: ").concat(dt("panel.content.padding"), ";\n}\n\n.p-panel-footer {\n    padding: ").concat(dt("panel.footer.padding"), ";\n}\n");
}, "theme");
var classes = {
  root: /* @__PURE__ */ __name(function root(_ref2) {
    var props = _ref2.props;
    return ["p-panel p-component", {
      "p-panel-toggleable": props.toggleable
    }];
  }, "root"),
  header: "p-panel-header",
  title: "p-panel-title",
  headerActions: "p-panel-header-actions",
  pcToggleButton: "p-panel-toggle-button",
  contentContainer: "p-panel-content-container",
  content: "p-panel-content",
  footer: "p-panel-footer"
};
var PanelStyle = BaseStyle.extend({
  name: "panel",
  theme,
  classes
});
var script$1 = {
  name: "BasePanel",
  "extends": script$2,
  props: {
    header: String,
    toggleable: Boolean,
    collapsed: Boolean,
    toggleButtonProps: {
      type: Object,
      "default": /* @__PURE__ */ __name(function _default() {
        return {
          severity: "secondary",
          text: true,
          rounded: true
        };
      }, "_default")
    }
  },
  style: PanelStyle,
  provide: /* @__PURE__ */ __name(function provide() {
    return {
      $pcPanel: this,
      $parentInstance: this
    };
  }, "provide")
};
var script = {
  name: "Panel",
  "extends": script$1,
  inheritAttrs: false,
  emits: ["update:collapsed", "toggle"],
  data: /* @__PURE__ */ __name(function data() {
    return {
      id: this.$attrs.id,
      d_collapsed: this.collapsed
    };
  }, "data"),
  watch: {
    "$attrs.id": /* @__PURE__ */ __name(function $attrsId(newValue) {
      this.id = newValue || UniqueComponentId();
    }, "$attrsId"),
    collapsed: /* @__PURE__ */ __name(function collapsed(newValue) {
      this.d_collapsed = newValue;
    }, "collapsed")
  },
  mounted: /* @__PURE__ */ __name(function mounted() {
    this.id = this.id || UniqueComponentId();
  }, "mounted"),
  methods: {
    toggle: /* @__PURE__ */ __name(function toggle(event) {
      this.d_collapsed = !this.d_collapsed;
      this.$emit("update:collapsed", this.d_collapsed);
      this.$emit("toggle", {
        originalEvent: event,
        value: this.d_collapsed
      });
    }, "toggle"),
    onKeyDown: /* @__PURE__ */ __name(function onKeyDown(event) {
      if (event.code === "Enter" || event.code === "NumpadEnter" || event.code === "Space") {
        this.toggle(event);
        event.preventDefault();
      }
    }, "onKeyDown")
  },
  computed: {
    buttonAriaLabel: /* @__PURE__ */ __name(function buttonAriaLabel() {
      return this.toggleButtonProps && this.toggleButtonProps.ariaLabel ? this.toggleButtonProps.ariaLabel : this.header;
    }, "buttonAriaLabel")
  },
  components: {
    PlusIcon: script$3,
    MinusIcon: script$4,
    Button: script$5
  },
  directives: {
    ripple: Ripple
  }
};
var _hoisted_1 = ["id"];
var _hoisted_2 = ["id", "aria-labelledby"];
function render(_ctx, _cache, $props, $setup, $data, $options) {
  var _component_Button = resolveComponent("Button");
  return openBlock(), createElementBlock("div", mergeProps({
    "class": _ctx.cx("root")
  }, _ctx.ptmi("root")), [createBaseVNode("div", mergeProps({
    "class": _ctx.cx("header")
  }, _ctx.ptm("header")), [renderSlot(_ctx.$slots, "header", {
    id: $data.id + "_header",
    "class": normalizeClass(_ctx.cx("title"))
  }, function() {
    return [_ctx.header ? (openBlock(), createElementBlock("span", mergeProps({
      key: 0,
      id: $data.id + "_header",
      "class": _ctx.cx("title")
    }, _ctx.ptm("title")), toDisplayString(_ctx.header), 17, _hoisted_1)) : createCommentVNode("", true)];
  }), createBaseVNode("div", mergeProps({
    "class": _ctx.cx("headerActions")
  }, _ctx.ptm("headerActions")), [renderSlot(_ctx.$slots, "icons"), _ctx.toggleable ? (openBlock(), createBlock(_component_Button, mergeProps({
    key: 0,
    id: $data.id + "_header",
    "class": _ctx.cx("pcToggleButton"),
    "aria-label": $options.buttonAriaLabel,
    "aria-controls": $data.id + "_content",
    "aria-expanded": !$data.d_collapsed,
    unstyled: _ctx.unstyled,
    onClick: $options.toggle,
    onKeydown: $options.onKeyDown
  }, _ctx.toggleButtonProps, {
    pt: _ctx.ptm("pcToggleButton")
  }), {
    icon: withCtx(function(slotProps) {
      return [renderSlot(_ctx.$slots, _ctx.$slots.toggleicon ? "toggleicon" : "togglericon", {
        collapsed: $data.d_collapsed
      }, function() {
        return [(openBlock(), createBlock(resolveDynamicComponent($data.d_collapsed ? "PlusIcon" : "MinusIcon"), mergeProps({
          "class": slotProps["class"]
        }, _ctx.ptm("pcToggleButton")["icon"]), null, 16, ["class"]))];
      })];
    }),
    _: 3
  }, 16, ["id", "class", "aria-label", "aria-controls", "aria-expanded", "unstyled", "onClick", "onKeydown", "pt"])) : createCommentVNode("", true)], 16)], 16), createVNode(Transition, mergeProps({
    name: "p-toggleable-content"
  }, _ctx.ptm("transition")), {
    "default": withCtx(function() {
      return [withDirectives(createBaseVNode("div", mergeProps({
        id: $data.id + "_content",
        "class": _ctx.cx("contentContainer"),
        role: "region",
        "aria-labelledby": $data.id + "_header"
      }, _ctx.ptm("contentContainer")), [createBaseVNode("div", mergeProps({
        "class": _ctx.cx("content")
      }, _ctx.ptm("content")), [renderSlot(_ctx.$slots, "default")], 16), _ctx.$slots.footer ? (openBlock(), createElementBlock("div", mergeProps({
        key: 0,
        "class": _ctx.cx("footer")
      }, _ctx.ptm("footer")), [renderSlot(_ctx.$slots, "footer")], 16)) : createCommentVNode("", true)], 16, _hoisted_2), [[vShow, !$data.d_collapsed]])];
    }),
    _: 3
  }, 16)], 16);
}
__name(render, "render");
script.render = render;
export {
  script as s
};
//# sourceMappingURL=index-jXPKy3pP.js.map
