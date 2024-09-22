var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });
import { d as defineComponent, u as useSettingStore, r as ref, a as useTitleEditorStore, b as useCanvasStore, w as watch, L as LGraphGroup, c as app, e as LGraphNode, o as onMounted, f as onUnmounted, g as openBlock, h as createElementBlock, i as createVNode, E as EditableText, n as normalizeStyle, j as createCommentVNode, k as LiteGraph, _ as _export_sfc, B as BaseStyle, s as script$5, l as resolveComponent, m as mergeProps, p as renderSlot, q as computed, t as resolveDirective, v as withDirectives, x as createBlock, y as withCtx, z as unref, A as createBaseVNode, C as normalizeClass, D as script$6, F as useDialogStore, S as SettingDialogHeader, G as SettingDialogContent, H as useWorkspaceStore, I as onBeforeUnmount, J as Fragment, K as renderList, T as Teleport, M as resolveDynamicComponent, N as pushScopeId, O as popScopeId, P as script$7, Q as getWidth, R as getHeight, U as getOuterWidth, V as getOuterHeight, W as getVNodeProp, X as isArray, Y as vShow, Z as isNotEmpty, $ as UniqueComponentId, a0 as ZIndex, a1 as resolveFieldData, a2 as focus, a3 as OverlayEventBus, a4 as isEmpty, a5 as addStyle, a6 as relativePosition, a7 as absolutePosition, a8 as ConnectedOverlayScrollHandler, a9 as isTouchDevice, aa as equals, ab as findLastIndex, ac as findSingle, ad as script$8, ae as script$9, af as script$a, ag as script$b, ah as script$c, ai as script$d, aj as Ripple, ak as toDisplayString, al as Transition, am as createSlots, an as createTextVNode, ao as useNodeFrequencyStore, ap as useNodeBookmarkStore, aq as highlightQuery, ar as script$e, as as formatNumberWithSuffix, at as NodeSourceType, au as useI18n, av as useNodeDefStore, aw as NodePreview, ax as NodeSearchFilter, ay as script$f, az as SearchFilterChip, aA as watchEffect, aB as toRaw, aC as LinkReleaseTriggerAction, aD as nextTick, aE as LGraph, aF as LLink, aG as DragAndScale, aH as LGraphCanvas, aI as ContextMenu, aJ as LGraphBadge$1, aK as dropTargetForElements, aL as ComfyNodeDefImpl } from "./index-Drc_oD2f.js";
const _sfc_main$c = /* @__PURE__ */ defineComponent({
  __name: "TitleEditor",
  setup(__props) {
    const settingStore = useSettingStore();
    const showInput = ref(false);
    const editedTitle = ref("");
    const inputStyle = ref({
      position: "fixed",
      left: "0px",
      top: "0px",
      width: "200px",
      height: "20px",
      fontSize: "12px"
    });
    const titleEditorStore = useTitleEditorStore();
    const canvasStore = useCanvasStore();
    const previousCanvasDraggable = ref(true);
    const onEdit = /* @__PURE__ */ __name((newValue) => {
      if (titleEditorStore.titleEditorTarget && newValue.trim() !== "") {
        titleEditorStore.titleEditorTarget.title = newValue.trim();
        app.graph.setDirtyCanvas(true, true);
      }
      showInput.value = false;
      titleEditorStore.titleEditorTarget = null;
      canvasStore.canvas.allow_dragcanvas = previousCanvasDraggable.value;
    }, "onEdit");
    watch(
      () => titleEditorStore.titleEditorTarget,
      (target) => {
        if (target === null) {
          return;
        }
        editedTitle.value = target.title;
        showInput.value = true;
        previousCanvasDraggable.value = canvasStore.canvas.allow_dragcanvas;
        canvasStore.canvas.allow_dragcanvas = false;
        if (target instanceof LGraphGroup) {
          const group = target;
          const [x, y] = group.pos;
          const [w, h] = group.size;
          const [left, top] = app.canvasPosToClientPos([x, y]);
          inputStyle.value.left = `${left}px`;
          inputStyle.value.top = `${top}px`;
          const width = w * app.canvas.ds.scale;
          const height = group.titleHeight * app.canvas.ds.scale;
          inputStyle.value.width = `${width}px`;
          inputStyle.value.height = `${height}px`;
          const fontSize = group.font_size * app.canvas.ds.scale;
          inputStyle.value.fontSize = `${fontSize}px`;
        } else if (target instanceof LGraphNode) {
          const node = target;
          const [x, y] = node.getBounding();
          const canvasWidth = node.width;
          const canvasHeight = LiteGraph.NODE_TITLE_HEIGHT;
          const [left, top] = app.canvasPosToClientPos([x, y]);
          inputStyle.value.left = `${left}px`;
          inputStyle.value.top = `${top}px`;
          const width = canvasWidth * app.canvas.ds.scale;
          const height = canvasHeight * app.canvas.ds.scale;
          inputStyle.value.width = `${width}px`;
          inputStyle.value.height = `${height}px`;
          const fontSize = 12 * app.canvas.ds.scale;
          inputStyle.value.fontSize = `${fontSize}px`;
        }
      }
    );
    const canvasEventHandler = /* @__PURE__ */ __name((event) => {
      if (!settingStore.get("Comfy.Group.DoubleClickTitleToEdit")) {
        return;
      }
      if (event.detail.subType === "group-double-click") {
        const group = event.detail.group;
        const [x, y] = group.pos;
        const e = event.detail.originalEvent;
        const relativeY = e.canvasY - y;
        if (relativeY > group.titleHeight) {
          return;
        }
        titleEditorStore.titleEditorTarget = group;
      }
    }, "canvasEventHandler");
    const extension = {
      name: "Comfy.NodeTitleEditor",
      nodeCreated(node) {
        const originalCallback = node.onNodeTitleDblClick;
        node.onNodeTitleDblClick = function(e, ...args) {
          if (!settingStore.get("Comfy.Node.DoubleClickTitleToEdit")) {
            return;
          }
          titleEditorStore.titleEditorTarget = this;
          if (typeof originalCallback === "function") {
            originalCallback.call(this, e, ...args);
          }
        };
      }
    };
    onMounted(() => {
      document.addEventListener("litegraph:canvas", canvasEventHandler);
      app.registerExtension(extension);
    });
    onUnmounted(() => {
      document.removeEventListener("litegraph:canvas", canvasEventHandler);
    });
    return (_ctx, _cache) => {
      return showInput.value ? (openBlock(), createElementBlock("div", {
        key: 0,
        class: "group-title-editor node-title-editor",
        style: normalizeStyle(inputStyle.value)
      }, [
        createVNode(EditableText, {
          isEditing: showInput.value,
          modelValue: editedTitle.value,
          onEdit
        }, null, 8, ["isEditing", "modelValue"])
      ], 4)) : createCommentVNode("", true);
    };
  }
});
const TitleEditor = /* @__PURE__ */ _export_sfc(_sfc_main$c, [["__scopeId", "data-v-fc3f26e3"]]);
var theme$2 = /* @__PURE__ */ __name(function theme(_ref) {
  var dt = _ref.dt;
  return "\n.p-overlaybadge {\n    position: relative;\n}\n\n.p-overlaybadge .p-badge {\n    position: absolute;\n    top: 0;\n    right: 0;\n    transform: translate(50%, -50%);\n    transform-origin: 100% 0;\n    margin: 0;\n    outline-width: ".concat(dt("overlaybadge.outline.width"), ";\n    outline-style: solid;\n    outline-color: ").concat(dt("overlaybadge.outline.color"), ";\n}\n");
}, "theme");
var classes$3 = {
  root: "p-overlaybadge"
};
var OverlayBadgeStyle = BaseStyle.extend({
  name: "overlaybadge",
  theme: theme$2,
  classes: classes$3
});
var script$1$3 = {
  name: "OverlayBadge",
  "extends": script$5,
  style: OverlayBadgeStyle,
  provide: /* @__PURE__ */ __name(function provide() {
    return {
      $pcOverlayBadge: this,
      $parentInstance: this
    };
  }, "provide")
};
var script$4 = {
  name: "OverlayBadge",
  "extends": script$1$3,
  inheritAttrs: false,
  components: {
    Badge: script$5
  }
};
function render$3(_ctx, _cache, $props, $setup, $data, $options) {
  var _component_Badge = resolveComponent("Badge");
  return openBlock(), createElementBlock("div", mergeProps({
    "class": _ctx.cx("root")
  }, _ctx.ptmi("root")), [renderSlot(_ctx.$slots, "default"), createVNode(_component_Badge, mergeProps(_ctx.$props, {
    pt: _ctx.ptm("pcBadge")
  }), null, 16, ["pt"])], 16);
}
__name(render$3, "render$3");
script$4.render = render$3;
const _sfc_main$b = /* @__PURE__ */ defineComponent({
  __name: "SidebarIcon",
  props: {
    icon: String,
    selected: Boolean,
    tooltip: {
      type: String,
      default: ""
    },
    class: {
      type: String,
      default: ""
    },
    iconBadge: {
      type: [String, Function],
      default: ""
    }
  },
  emits: ["click"],
  setup(__props, { emit: __emit }) {
    const props = __props;
    const emit = __emit;
    const overlayValue = computed(
      () => typeof props.iconBadge === "function" ? props.iconBadge() || "" : props.iconBadge
    );
    const shouldShowBadge = computed(() => !!overlayValue.value);
    return (_ctx, _cache) => {
      const _directive_tooltip = resolveDirective("tooltip");
      return withDirectives((openBlock(), createBlock(unref(script$6), {
        class: normalizeClass(props.class),
        text: "",
        pt: {
          root: {
            class: `side-bar-button ${props.selected ? "p-button-primary side-bar-button-selected" : "p-button-secondary"}`,
            "aria-label": props.tooltip
          }
        },
        onClick: _cache[0] || (_cache[0] = ($event) => emit("click", $event))
      }, {
        icon: withCtx(() => [
          shouldShowBadge.value ? (openBlock(), createBlock(unref(script$4), {
            key: 0,
            value: overlayValue.value
          }, {
            default: withCtx(() => [
              createBaseVNode("i", {
                class: normalizeClass(props.icon + " side-bar-button-icon")
              }, null, 2)
            ]),
            _: 1
          }, 8, ["value"])) : (openBlock(), createElementBlock("i", {
            key: 1,
            class: normalizeClass(props.icon + " side-bar-button-icon")
          }, null, 2))
        ]),
        _: 1
      }, 8, ["class", "pt"])), [
        [_directive_tooltip, { value: props.tooltip, showDelay: 300, hideDelay: 300 }]
      ]);
    };
  }
});
const SidebarIcon = /* @__PURE__ */ _export_sfc(_sfc_main$b, [["__scopeId", "data-v-caa3ee9c"]]);
const _sfc_main$a = /* @__PURE__ */ defineComponent({
  __name: "SidebarThemeToggleIcon",
  setup(__props) {
    const previousDarkTheme = ref("dark");
    const currentTheme = computed(() => useSettingStore().get("Comfy.ColorPalette"));
    const isDarkMode = computed(() => currentTheme.value !== "light");
    const icon = computed(() => isDarkMode.value ? "pi pi-moon" : "pi pi-sun");
    const toggleTheme = /* @__PURE__ */ __name(() => {
      if (isDarkMode.value) {
        previousDarkTheme.value = currentTheme.value;
        useSettingStore().set("Comfy.ColorPalette", "light");
      } else {
        useSettingStore().set("Comfy.ColorPalette", previousDarkTheme.value);
      }
    }, "toggleTheme");
    return (_ctx, _cache) => {
      return openBlock(), createBlock(SidebarIcon, {
        icon: icon.value,
        onClick: toggleTheme,
        tooltip: _ctx.$t("sideToolbar.themeToggle"),
        class: "comfy-vue-theme-toggle"
      }, null, 8, ["icon", "tooltip"]);
    };
  }
});
const _sfc_main$9 = /* @__PURE__ */ defineComponent({
  __name: "SidebarSettingsToggleIcon",
  setup(__props) {
    const dialogStore = useDialogStore();
    const showSetting = /* @__PURE__ */ __name(() => {
      dialogStore.showDialog({
        headerComponent: SettingDialogHeader,
        component: SettingDialogContent
      });
    }, "showSetting");
    return (_ctx, _cache) => {
      return openBlock(), createBlock(SidebarIcon, {
        icon: "pi pi-cog",
        onClick: showSetting,
        tooltip: _ctx.$t("settings")
      }, null, 8, ["tooltip"]);
    };
  }
});
const _withScopeId$3 = /* @__PURE__ */ __name((n) => (pushScopeId("data-v-ed7a1148"), n = n(), popScopeId(), n), "_withScopeId$3");
const _hoisted_1$5 = { class: "side-tool-bar-end" };
const _hoisted_2$4 = {
  key: 0,
  class: "sidebar-content-container"
};
const _sfc_main$8 = /* @__PURE__ */ defineComponent({
  __name: "SideToolbar",
  setup(__props) {
    const workspaceStore = useWorkspaceStore();
    const settingStore = useSettingStore();
    const teleportTarget = computed(
      () => settingStore.get("Comfy.Sidebar.Location") === "left" ? ".comfyui-body-left" : ".comfyui-body-right"
    );
    const isSmall = computed(
      () => settingStore.get("Comfy.Sidebar.Size") === "small"
    );
    const tabs = computed(() => workspaceStore.getSidebarTabs());
    const selectedTab = computed(() => {
      const tabId = workspaceStore.activeSidebarTab;
      return tabs.value.find((tab) => tab.id === tabId) || null;
    });
    const mountCustomTab = /* @__PURE__ */ __name((tab, el) => {
      tab.render(el);
    }, "mountCustomTab");
    const onTabClick = /* @__PURE__ */ __name((item) => {
      workspaceStore.updateActiveSidebarTab(
        workspaceStore.activeSidebarTab === item.id ? null : item.id
      );
    }, "onTabClick");
    onBeforeUnmount(() => {
      tabs.value.forEach((tab) => {
        if (tab.type === "custom" && tab.destroy) {
          tab.destroy();
        }
      });
    });
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock(Fragment, null, [
        (openBlock(), createBlock(Teleport, { to: teleportTarget.value }, [
          createBaseVNode("nav", {
            class: normalizeClass("side-tool-bar-container" + (isSmall.value ? " small-sidebar" : ""))
          }, [
            (openBlock(true), createElementBlock(Fragment, null, renderList(tabs.value, (tab) => {
              return openBlock(), createBlock(SidebarIcon, {
                key: tab.id,
                icon: tab.icon,
                iconBadge: tab.iconBadge,
                tooltip: tab.tooltip,
                selected: tab === selectedTab.value,
                class: normalizeClass(tab.id + "-tab-button"),
                onClick: /* @__PURE__ */ __name(($event) => onTabClick(tab), "onClick")
              }, null, 8, ["icon", "iconBadge", "tooltip", "selected", "class", "onClick"]);
            }), 128)),
            createBaseVNode("div", _hoisted_1$5, [
              createVNode(_sfc_main$a),
              createVNode(_sfc_main$9)
            ])
          ], 2)
        ], 8, ["to"])),
        selectedTab.value ? (openBlock(), createElementBlock("div", _hoisted_2$4, [
          selectedTab.value.type === "vue" ? (openBlock(), createBlock(resolveDynamicComponent(selectedTab.value.component), { key: 0 })) : (openBlock(), createElementBlock("div", {
            key: 1,
            ref: /* @__PURE__ */ __name((el) => {
              if (el)
                mountCustomTab(
                  selectedTab.value,
                  el
                );
            }, "ref")
          }, null, 512))
        ])) : createCommentVNode("", true)
      ], 64);
    };
  }
});
const SideToolbar = /* @__PURE__ */ _export_sfc(_sfc_main$8, [["__scopeId", "data-v-ed7a1148"]]);
var theme$1 = /* @__PURE__ */ __name(function theme2(_ref) {
  var dt = _ref.dt;
  return "\n.p-splitter {\n    display: flex;\n    flex-wrap: nowrap;\n    border: 1px solid ".concat(dt("splitter.border.color"), ";\n    background: ").concat(dt("splitter.background"), ";\n    border-radius: ").concat(dt("border.radius.md"), ";\n    color: ").concat(dt("splitter.color"), ";\n}\n\n.p-splitter-vertical {\n    flex-direction: column;\n}\n\n.p-splitter-gutter {\n    flex-grow: 0;\n    flex-shrink: 0;\n    display: flex;\n    align-items: center;\n    justify-content: center;\n    z-index: 1;\n    background: ").concat(dt("splitter.gutter.background"), ";\n}\n\n.p-splitter-gutter-handle {\n    border-radius: ").concat(dt("splitter.handle.border.radius"), ";\n    background: ").concat(dt("splitter.handle.background"), ";\n    transition: outline-color ").concat(dt("splitter.transition.duration"), ", box-shadow ").concat(dt("splitter.transition.duration"), ";\n    outline-color: transparent;\n}\n\n.p-splitter-gutter-handle:focus-visible {\n    box-shadow: ").concat(dt("splitter.handle.focus.ring.shadow"), ";\n    outline: ").concat(dt("splitter.handle.focus.ring.width"), " ").concat(dt("splitter.handle.focus.ring.style"), " ").concat(dt("splitter.handle.focus.ring.color"), ";\n    outline-offset: ").concat(dt("splitter.handle.focus.ring.offset"), ";\n}\n\n.p-splitter-horizontal.p-splitter-resizing {\n    cursor: col-resize;\n    user-select: none;\n}\n\n.p-splitter-vertical.p-splitter-resizing {\n    cursor: row-resize;\n    user-select: none;\n}\n\n.p-splitter-horizontal > .p-splitter-gutter > .p-splitter-gutter-handle {\n    height: ").concat(dt("splitter.handle.size"), ";\n    width: 100%;\n}\n\n.p-splitter-vertical > .p-splitter-gutter > .p-splitter-gutter-handle {\n    width: ").concat(dt("splitter.handle.size"), ";\n    height: 100%;\n}\n\n.p-splitter-horizontal > .p-splitter-gutter {\n    cursor: col-resize;\n}\n\n.p-splitter-vertical > .p-splitter-gutter {\n    cursor: row-resize;\n}\n\n.p-splitterpanel {\n    flex-grow: 1;\n    overflow: hidden;\n}\n\n.p-splitterpanel-nested {\n    display: flex;\n}\n\n.p-splitterpanel .p-splitter {\n    flex-grow: 1;\n    border: 0 none;\n}\n");
}, "theme");
var classes$2 = {
  root: /* @__PURE__ */ __name(function root(_ref2) {
    var props = _ref2.props;
    return ["p-splitter p-component", "p-splitter-" + props.layout];
  }, "root"),
  gutter: "p-splitter-gutter",
  gutterHandle: "p-splitter-gutter-handle"
};
var inlineStyles$1 = {
  root: /* @__PURE__ */ __name(function root2(_ref3) {
    var props = _ref3.props;
    return [{
      display: "flex",
      "flex-wrap": "nowrap"
    }, props.layout === "vertical" ? {
      "flex-direction": "column"
    } : ""];
  }, "root")
};
var SplitterStyle = BaseStyle.extend({
  name: "splitter",
  theme: theme$1,
  classes: classes$2,
  inlineStyles: inlineStyles$1
});
var script$1$2 = {
  name: "BaseSplitter",
  "extends": script$7,
  props: {
    layout: {
      type: String,
      "default": "horizontal"
    },
    gutterSize: {
      type: Number,
      "default": 4
    },
    stateKey: {
      type: String,
      "default": null
    },
    stateStorage: {
      type: String,
      "default": "session"
    },
    step: {
      type: Number,
      "default": 5
    }
  },
  style: SplitterStyle,
  provide: /* @__PURE__ */ __name(function provide2() {
    return {
      $pcSplitter: this,
      $parentInstance: this
    };
  }, "provide")
};
function _toConsumableArray$1(r) {
  return _arrayWithoutHoles$1(r) || _iterableToArray$1(r) || _unsupportedIterableToArray$1(r) || _nonIterableSpread$1();
}
__name(_toConsumableArray$1, "_toConsumableArray$1");
function _nonIterableSpread$1() {
  throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.");
}
__name(_nonIterableSpread$1, "_nonIterableSpread$1");
function _unsupportedIterableToArray$1(r, a) {
  if (r) {
    if ("string" == typeof r) return _arrayLikeToArray$1(r, a);
    var t = {}.toString.call(r).slice(8, -1);
    return "Object" === t && r.constructor && (t = r.constructor.name), "Map" === t || "Set" === t ? Array.from(r) : "Arguments" === t || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(t) ? _arrayLikeToArray$1(r, a) : void 0;
  }
}
__name(_unsupportedIterableToArray$1, "_unsupportedIterableToArray$1");
function _iterableToArray$1(r) {
  if ("undefined" != typeof Symbol && null != r[Symbol.iterator] || null != r["@@iterator"]) return Array.from(r);
}
__name(_iterableToArray$1, "_iterableToArray$1");
function _arrayWithoutHoles$1(r) {
  if (Array.isArray(r)) return _arrayLikeToArray$1(r);
}
__name(_arrayWithoutHoles$1, "_arrayWithoutHoles$1");
function _arrayLikeToArray$1(r, a) {
  (null == a || a > r.length) && (a = r.length);
  for (var e = 0, n = Array(a); e < a; e++) n[e] = r[e];
  return n;
}
__name(_arrayLikeToArray$1, "_arrayLikeToArray$1");
var script$3 = {
  name: "Splitter",
  "extends": script$1$2,
  inheritAttrs: false,
  emits: ["resizestart", "resizeend", "resize"],
  dragging: false,
  mouseMoveListener: null,
  mouseUpListener: null,
  touchMoveListener: null,
  touchEndListener: null,
  size: null,
  gutterElement: null,
  startPos: null,
  prevPanelElement: null,
  nextPanelElement: null,
  nextPanelSize: null,
  prevPanelSize: null,
  panelSizes: null,
  prevPanelIndex: null,
  timer: null,
  data: /* @__PURE__ */ __name(function data() {
    return {
      prevSize: null
    };
  }, "data"),
  mounted: /* @__PURE__ */ __name(function mounted() {
    var _this = this;
    if (this.panels && this.panels.length) {
      var initialized = false;
      if (this.isStateful()) {
        initialized = this.restoreState();
      }
      if (!initialized) {
        var children = _toConsumableArray$1(this.$el.children).filter(function(child) {
          return child.getAttribute("data-pc-name") === "splitterpanel";
        });
        var _panelSizes = [];
        this.panels.map(function(panel, i) {
          var panelInitialSize = panel.props && panel.props.size ? panel.props.size : null;
          var panelSize = panelInitialSize || 100 / _this.panels.length;
          _panelSizes[i] = panelSize;
          children[i].style.flexBasis = "calc(" + panelSize + "% - " + (_this.panels.length - 1) * _this.gutterSize + "px)";
        });
        this.panelSizes = _panelSizes;
        this.prevSize = parseFloat(_panelSizes[0]).toFixed(4);
      }
    }
  }, "mounted"),
  beforeUnmount: /* @__PURE__ */ __name(function beforeUnmount() {
    this.clear();
    this.unbindMouseListeners();
  }, "beforeUnmount"),
  methods: {
    isSplitterPanel: /* @__PURE__ */ __name(function isSplitterPanel(child) {
      return child.type.name === "SplitterPanel";
    }, "isSplitterPanel"),
    onResizeStart: /* @__PURE__ */ __name(function onResizeStart(event, index, isKeyDown) {
      this.gutterElement = event.currentTarget || event.target.parentElement;
      this.size = this.horizontal ? getWidth(this.$el) : getHeight(this.$el);
      if (!isKeyDown) {
        this.dragging = true;
        this.startPos = this.layout === "horizontal" ? event.pageX || event.changedTouches[0].pageX : event.pageY || event.changedTouches[0].pageY;
      }
      this.prevPanelElement = this.gutterElement.previousElementSibling;
      this.nextPanelElement = this.gutterElement.nextElementSibling;
      if (isKeyDown) {
        this.prevPanelSize = this.horizontal ? getOuterWidth(this.prevPanelElement, true) : getOuterHeight(this.prevPanelElement, true);
        this.nextPanelSize = this.horizontal ? getOuterWidth(this.nextPanelElement, true) : getOuterHeight(this.nextPanelElement, true);
      } else {
        this.prevPanelSize = 100 * (this.horizontal ? getOuterWidth(this.prevPanelElement, true) : getOuterHeight(this.prevPanelElement, true)) / this.size;
        this.nextPanelSize = 100 * (this.horizontal ? getOuterWidth(this.nextPanelElement, true) : getOuterHeight(this.nextPanelElement, true)) / this.size;
      }
      this.prevPanelIndex = index;
      this.$emit("resizestart", {
        originalEvent: event,
        sizes: this.panelSizes
      });
      this.$refs.gutter[index].setAttribute("data-p-gutter-resizing", true);
      this.$el.setAttribute("data-p-resizing", true);
    }, "onResizeStart"),
    onResize: /* @__PURE__ */ __name(function onResize(event, step, isKeyDown) {
      var newPos, newPrevPanelSize, newNextPanelSize;
      if (isKeyDown) {
        if (this.horizontal) {
          newPrevPanelSize = 100 * (this.prevPanelSize + step) / this.size;
          newNextPanelSize = 100 * (this.nextPanelSize - step) / this.size;
        } else {
          newPrevPanelSize = 100 * (this.prevPanelSize - step) / this.size;
          newNextPanelSize = 100 * (this.nextPanelSize + step) / this.size;
        }
      } else {
        if (this.horizontal) newPos = event.pageX * 100 / this.size - this.startPos * 100 / this.size;
        else newPos = event.pageY * 100 / this.size - this.startPos * 100 / this.size;
        newPrevPanelSize = this.prevPanelSize + newPos;
        newNextPanelSize = this.nextPanelSize - newPos;
      }
      if (this.validateResize(newPrevPanelSize, newNextPanelSize)) {
        this.prevPanelElement.style.flexBasis = "calc(" + newPrevPanelSize + "% - " + (this.panels.length - 1) * this.gutterSize + "px)";
        this.nextPanelElement.style.flexBasis = "calc(" + newNextPanelSize + "% - " + (this.panels.length - 1) * this.gutterSize + "px)";
        this.panelSizes[this.prevPanelIndex] = newPrevPanelSize;
        this.panelSizes[this.prevPanelIndex + 1] = newNextPanelSize;
        this.prevSize = parseFloat(newPrevPanelSize).toFixed(4);
      }
      this.$emit("resize", {
        originalEvent: event,
        sizes: this.panelSizes
      });
    }, "onResize"),
    onResizeEnd: /* @__PURE__ */ __name(function onResizeEnd(event) {
      if (this.isStateful()) {
        this.saveState();
      }
      this.$emit("resizeend", {
        originalEvent: event,
        sizes: this.panelSizes
      });
      this.$refs.gutter.forEach(function(gutter) {
        return gutter.setAttribute("data-p-gutter-resizing", false);
      });
      this.$el.setAttribute("data-p-resizing", false);
      this.clear();
    }, "onResizeEnd"),
    repeat: /* @__PURE__ */ __name(function repeat(event, index, step) {
      this.onResizeStart(event, index, true);
      this.onResize(event, step, true);
    }, "repeat"),
    setTimer: /* @__PURE__ */ __name(function setTimer(event, index, step) {
      var _this2 = this;
      if (!this.timer) {
        this.timer = setInterval(function() {
          _this2.repeat(event, index, step);
        }, 40);
      }
    }, "setTimer"),
    clearTimer: /* @__PURE__ */ __name(function clearTimer() {
      if (this.timer) {
        clearInterval(this.timer);
        this.timer = null;
      }
    }, "clearTimer"),
    onGutterKeyUp: /* @__PURE__ */ __name(function onGutterKeyUp() {
      this.clearTimer();
      this.onResizeEnd();
    }, "onGutterKeyUp"),
    onGutterKeyDown: /* @__PURE__ */ __name(function onGutterKeyDown(event, index) {
      switch (event.code) {
        case "ArrowLeft": {
          if (this.layout === "horizontal") {
            this.setTimer(event, index, this.step * -1);
          }
          event.preventDefault();
          break;
        }
        case "ArrowRight": {
          if (this.layout === "horizontal") {
            this.setTimer(event, index, this.step);
          }
          event.preventDefault();
          break;
        }
        case "ArrowDown": {
          if (this.layout === "vertical") {
            this.setTimer(event, index, this.step * -1);
          }
          event.preventDefault();
          break;
        }
        case "ArrowUp": {
          if (this.layout === "vertical") {
            this.setTimer(event, index, this.step);
          }
          event.preventDefault();
          break;
        }
      }
    }, "onGutterKeyDown"),
    onGutterMouseDown: /* @__PURE__ */ __name(function onGutterMouseDown(event, index) {
      this.onResizeStart(event, index);
      this.bindMouseListeners();
    }, "onGutterMouseDown"),
    onGutterTouchStart: /* @__PURE__ */ __name(function onGutterTouchStart(event, index) {
      this.onResizeStart(event, index);
      this.bindTouchListeners();
      event.preventDefault();
    }, "onGutterTouchStart"),
    onGutterTouchMove: /* @__PURE__ */ __name(function onGutterTouchMove(event) {
      this.onResize(event);
      event.preventDefault();
    }, "onGutterTouchMove"),
    onGutterTouchEnd: /* @__PURE__ */ __name(function onGutterTouchEnd(event) {
      this.onResizeEnd(event);
      this.unbindTouchListeners();
      event.preventDefault();
    }, "onGutterTouchEnd"),
    bindMouseListeners: /* @__PURE__ */ __name(function bindMouseListeners() {
      var _this3 = this;
      if (!this.mouseMoveListener) {
        this.mouseMoveListener = function(event) {
          return _this3.onResize(event);
        };
        document.addEventListener("mousemove", this.mouseMoveListener);
      }
      if (!this.mouseUpListener) {
        this.mouseUpListener = function(event) {
          _this3.onResizeEnd(event);
          _this3.unbindMouseListeners();
        };
        document.addEventListener("mouseup", this.mouseUpListener);
      }
    }, "bindMouseListeners"),
    bindTouchListeners: /* @__PURE__ */ __name(function bindTouchListeners() {
      var _this4 = this;
      if (!this.touchMoveListener) {
        this.touchMoveListener = function(event) {
          return _this4.onResize(event.changedTouches[0]);
        };
        document.addEventListener("touchmove", this.touchMoveListener);
      }
      if (!this.touchEndListener) {
        this.touchEndListener = function(event) {
          _this4.resizeEnd(event);
          _this4.unbindTouchListeners();
        };
        document.addEventListener("touchend", this.touchEndListener);
      }
    }, "bindTouchListeners"),
    validateResize: /* @__PURE__ */ __name(function validateResize(newPrevPanelSize, newNextPanelSize) {
      if (newPrevPanelSize > 100 || newPrevPanelSize < 0) return false;
      if (newNextPanelSize > 100 || newNextPanelSize < 0) return false;
      var prevPanelMinSize = getVNodeProp(this.panels[this.prevPanelIndex], "minSize");
      if (this.panels[this.prevPanelIndex].props && prevPanelMinSize && prevPanelMinSize > newPrevPanelSize) {
        return false;
      }
      var newPanelMinSize = getVNodeProp(this.panels[this.prevPanelIndex + 1], "minSize");
      if (this.panels[this.prevPanelIndex + 1].props && newPanelMinSize && newPanelMinSize > newNextPanelSize) {
        return false;
      }
      return true;
    }, "validateResize"),
    unbindMouseListeners: /* @__PURE__ */ __name(function unbindMouseListeners() {
      if (this.mouseMoveListener) {
        document.removeEventListener("mousemove", this.mouseMoveListener);
        this.mouseMoveListener = null;
      }
      if (this.mouseUpListener) {
        document.removeEventListener("mouseup", this.mouseUpListener);
        this.mouseUpListener = null;
      }
    }, "unbindMouseListeners"),
    unbindTouchListeners: /* @__PURE__ */ __name(function unbindTouchListeners() {
      if (this.touchMoveListener) {
        document.removeEventListener("touchmove", this.touchMoveListener);
        this.touchMoveListener = null;
      }
      if (this.touchEndListener) {
        document.removeEventListener("touchend", this.touchEndListener);
        this.touchEndListener = null;
      }
    }, "unbindTouchListeners"),
    clear: /* @__PURE__ */ __name(function clear() {
      this.dragging = false;
      this.size = null;
      this.startPos = null;
      this.prevPanelElement = null;
      this.nextPanelElement = null;
      this.prevPanelSize = null;
      this.nextPanelSize = null;
      this.gutterElement = null;
      this.prevPanelIndex = null;
    }, "clear"),
    isStateful: /* @__PURE__ */ __name(function isStateful() {
      return this.stateKey != null;
    }, "isStateful"),
    getStorage: /* @__PURE__ */ __name(function getStorage() {
      switch (this.stateStorage) {
        case "local":
          return window.localStorage;
        case "session":
          return window.sessionStorage;
        default:
          throw new Error(this.stateStorage + ' is not a valid value for the state storage, supported values are "local" and "session".');
      }
    }, "getStorage"),
    saveState: /* @__PURE__ */ __name(function saveState() {
      if (isArray(this.panelSizes)) {
        this.getStorage().setItem(this.stateKey, JSON.stringify(this.panelSizes));
      }
    }, "saveState"),
    restoreState: /* @__PURE__ */ __name(function restoreState() {
      var _this5 = this;
      var storage = this.getStorage();
      var stateString = storage.getItem(this.stateKey);
      if (stateString) {
        this.panelSizes = JSON.parse(stateString);
        var children = _toConsumableArray$1(this.$el.children).filter(function(child) {
          return child.getAttribute("data-pc-name") === "splitterpanel";
        });
        children.forEach(function(child, i) {
          child.style.flexBasis = "calc(" + _this5.panelSizes[i] + "% - " + (_this5.panels.length - 1) * _this5.gutterSize + "px)";
        });
        return true;
      }
      return false;
    }, "restoreState")
  },
  computed: {
    panels: /* @__PURE__ */ __name(function panels() {
      var _this6 = this;
      var panels2 = [];
      this.$slots["default"]().forEach(function(child) {
        if (_this6.isSplitterPanel(child)) {
          panels2.push(child);
        } else if (child.children instanceof Array) {
          child.children.forEach(function(nestedChild) {
            if (_this6.isSplitterPanel(nestedChild)) {
              panels2.push(nestedChild);
            }
          });
        }
      });
      return panels2;
    }, "panels"),
    gutterStyle: /* @__PURE__ */ __name(function gutterStyle() {
      if (this.horizontal) return {
        width: this.gutterSize + "px"
      };
      else return {
        height: this.gutterSize + "px"
      };
    }, "gutterStyle"),
    horizontal: /* @__PURE__ */ __name(function horizontal() {
      return this.layout === "horizontal";
    }, "horizontal"),
    getPTOptions: /* @__PURE__ */ __name(function getPTOptions() {
      var _this$$parentInstance;
      return {
        context: {
          nested: (_this$$parentInstance = this.$parentInstance) === null || _this$$parentInstance === void 0 ? void 0 : _this$$parentInstance.nestedState
        }
      };
    }, "getPTOptions")
  }
};
var _hoisted_1$4 = ["onMousedown", "onTouchstart", "onTouchmove", "onTouchend"];
var _hoisted_2$3 = ["aria-orientation", "aria-valuenow", "onKeydown"];
function render$2(_ctx, _cache, $props, $setup, $data, $options) {
  return openBlock(), createElementBlock("div", mergeProps({
    "class": _ctx.cx("root"),
    style: _ctx.sx("root"),
    "data-p-resizing": false
  }, _ctx.ptmi("root", $options.getPTOptions)), [(openBlock(true), createElementBlock(Fragment, null, renderList($options.panels, function(panel, i) {
    return openBlock(), createElementBlock(Fragment, {
      key: i
    }, [(openBlock(), createBlock(resolveDynamicComponent(panel), {
      tabindex: "-1"
    })), i !== $options.panels.length - 1 ? (openBlock(), createElementBlock("div", mergeProps({
      key: 0,
      ref_for: true,
      ref: "gutter",
      "class": _ctx.cx("gutter"),
      role: "separator",
      tabindex: "-1",
      onMousedown: /* @__PURE__ */ __name(function onMousedown($event) {
        return $options.onGutterMouseDown($event, i);
      }, "onMousedown"),
      onTouchstart: /* @__PURE__ */ __name(function onTouchstart($event) {
        return $options.onGutterTouchStart($event, i);
      }, "onTouchstart"),
      onTouchmove: /* @__PURE__ */ __name(function onTouchmove($event) {
        return $options.onGutterTouchMove($event, i);
      }, "onTouchmove"),
      onTouchend: /* @__PURE__ */ __name(function onTouchend($event) {
        return $options.onGutterTouchEnd($event, i);
      }, "onTouchend"),
      "data-p-gutter-resizing": false
    }, _ctx.ptm("gutter")), [createBaseVNode("div", mergeProps({
      "class": _ctx.cx("gutterHandle"),
      tabindex: "0",
      style: [$options.gutterStyle],
      "aria-orientation": _ctx.layout,
      "aria-valuenow": $data.prevSize,
      onKeyup: _cache[0] || (_cache[0] = function() {
        return $options.onGutterKeyUp && $options.onGutterKeyUp.apply($options, arguments);
      }),
      onKeydown: /* @__PURE__ */ __name(function onKeydown($event) {
        return $options.onGutterKeyDown($event, i);
      }, "onKeydown"),
      ref_for: true
    }, _ctx.ptm("gutterHandle")), null, 16, _hoisted_2$3)], 16, _hoisted_1$4)) : createCommentVNode("", true)], 64);
  }), 128))], 16);
}
__name(render$2, "render$2");
script$3.render = render$2;
var classes$1 = {
  root: /* @__PURE__ */ __name(function root3(_ref) {
    var instance = _ref.instance;
    return ["p-splitterpanel", {
      "p-splitterpanel-nested": instance.isNested
    }];
  }, "root")
};
var SplitterPanelStyle = BaseStyle.extend({
  name: "splitterpanel",
  classes: classes$1
});
var script$1$1 = {
  name: "BaseSplitterPanel",
  "extends": script$7,
  props: {
    size: {
      type: Number,
      "default": null
    },
    minSize: {
      type: Number,
      "default": null
    }
  },
  style: SplitterPanelStyle,
  provide: /* @__PURE__ */ __name(function provide3() {
    return {
      $pcSplitterPanel: this,
      $parentInstance: this
    };
  }, "provide")
};
var script$2 = {
  name: "SplitterPanel",
  "extends": script$1$1,
  inheritAttrs: false,
  data: /* @__PURE__ */ __name(function data2() {
    return {
      nestedState: null
    };
  }, "data"),
  computed: {
    isNested: /* @__PURE__ */ __name(function isNested() {
      var _this = this;
      return this.$slots["default"]().some(function(child) {
        _this.nestedState = child.type.name === "Splitter" ? true : null;
        return _this.nestedState;
      });
    }, "isNested"),
    getPTOptions: /* @__PURE__ */ __name(function getPTOptions2() {
      return {
        context: {
          nested: this.isNested
        }
      };
    }, "getPTOptions")
  }
};
function render$1(_ctx, _cache, $props, $setup, $data, $options) {
  return openBlock(), createElementBlock("div", mergeProps({
    ref: "container",
    "class": _ctx.cx("root")
  }, _ctx.ptmi("root", $options.getPTOptions)), [renderSlot(_ctx.$slots, "default")], 16);
}
__name(render$1, "render$1");
script$2.render = render$1;
const _withScopeId$2 = /* @__PURE__ */ __name((n) => (pushScopeId("data-v-edca8328"), n = n(), popScopeId(), n), "_withScopeId$2");
const _hoisted_1$3 = /* @__PURE__ */ _withScopeId$2(() => /* @__PURE__ */ createBaseVNode("div", null, null, -1));
const _sfc_main$7 = /* @__PURE__ */ defineComponent({
  __name: "LiteGraphCanvasSplitterOverlay",
  setup(__props) {
    const settingStore = useSettingStore();
    const sidebarLocation = computed(
      () => settingStore.get("Comfy.Sidebar.Location")
    );
    const sidebarPanelVisible = computed(
      () => useWorkspaceStore().activeSidebarTab !== null
    );
    const gutterClass = computed(() => {
      return sidebarPanelVisible.value ? "" : "gutter-hidden";
    });
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(script$3), {
        class: "splitter-overlay",
        "pt:gutter": gutterClass.value
      }, {
        default: withCtx(() => [
          sidebarLocation.value === "left" ? withDirectives((openBlock(), createBlock(unref(script$2), {
            key: 0,
            class: "side-bar-panel",
            minSize: 10,
            size: 20
          }, {
            default: withCtx(() => [
              renderSlot(_ctx.$slots, "side-bar-panel", {}, void 0, true)
            ]),
            _: 3
          }, 512)), [
            [vShow, sidebarPanelVisible.value]
          ]) : createCommentVNode("", true),
          createVNode(unref(script$2), {
            class: "graph-canvas-panel",
            size: 100
          }, {
            default: withCtx(() => [
              _hoisted_1$3
            ]),
            _: 1
          }),
          sidebarLocation.value === "right" ? withDirectives((openBlock(), createBlock(unref(script$2), {
            key: 1,
            class: "side-bar-panel",
            minSize: 10,
            size: 20
          }, {
            default: withCtx(() => [
              renderSlot(_ctx.$slots, "side-bar-panel", {}, void 0, true)
            ]),
            _: 3
          }, 512)), [
            [vShow, sidebarPanelVisible.value]
          ]) : createCommentVNode("", true)
        ]),
        _: 3
      }, 8, ["pt:gutter"]);
    };
  }
});
const LiteGraphCanvasSplitterOverlay = /* @__PURE__ */ _export_sfc(_sfc_main$7, [["__scopeId", "data-v-edca8328"]]);
var theme3 = /* @__PURE__ */ __name(function theme4(_ref) {
  var dt = _ref.dt;
  return "\n.p-autocomplete {\n    display: inline-flex;\n}\n\n.p-autocomplete-loader {\n    position: absolute;\n    top: 50%;\n    margin-top: -0.5rem;\n    right: ".concat(dt("autocomplete.padding.x"), ";\n}\n\n.p-autocomplete:has(.p-autocomplete-dropdown) .p-autocomplete-loader {\n    right: calc(").concat(dt("autocomplete.dropdown.width"), " + ").concat(dt("autocomplete.padding.x"), ");\n}\n\n.p-autocomplete:has(.p-autocomplete-dropdown) .p-autocomplete-input {\n    flex: 1 1 auto;\n    width: 1%;\n}\n\n.p-autocomplete:has(.p-autocomplete-dropdown) .p-autocomplete-input,\n.p-autocomplete:has(.p-autocomplete-dropdown) .p-autocomplete-input-multiple {\n    border-top-right-radius: 0;\n    border-bottom-right-radius: 0;\n}\n\n.p-autocomplete-dropdown {\n    cursor: pointer;\n    display: inline-flex;\n    cursor: pointer;\n    user-select: none;\n    align-items: center;\n    justify-content: center;\n    overflow: hidden;\n    position: relative;\n    width: ").concat(dt("autocomplete.dropdown.width"), ";\n    border-top-right-radius: ").concat(dt("autocomplete.dropdown.border.radius"), ";\n    border-bottom-right-radius: ").concat(dt("autocomplete.dropdown.border.radius"), ";\n    background: ").concat(dt("autocomplete.dropdown.background"), ";\n    border: 1px solid ").concat(dt("autocomplete.dropdown.border.color"), ";\n    border-left: 0 none;\n    color: ").concat(dt("autocomplete.dropdown.color"), ";\n    transition: background ").concat(dt("autocomplete.transition.duration"), ", color ").concat(dt("autocomplete.transition.duration"), ", border-color ").concat(dt("autocomplete.transition.duration"), ", outline-color ").concat(dt("autocomplete.transition.duration"), ", box-shadow ").concat(dt("autocomplete.transition.duration"), ";\n    outline-color: transparent;\n}\n\n.p-autocomplete-dropdown:not(:disabled):hover {\n    background: ").concat(dt("autocomplete.dropdown.hover.background"), ";\n    border-color: ").concat(dt("autocomplete.dropdown.hover.border.color"), ";\n    color: ").concat(dt("autocomplete.dropdown.hover.color"), ";\n}\n\n.p-autocomplete-dropdown:not(:disabled):active {\n    background: ").concat(dt("autocomplete.dropdown.active.background"), ";\n    border-color: ").concat(dt("autocomplete.dropdown.active.border.color"), ";\n    color: ").concat(dt("autocomplete.dropdown.active.color"), ";\n}\n\n.p-autocomplete-dropdown:focus-visible {\n    box-shadow: ").concat(dt("autocomplete.dropdown.focus.ring.shadow"), ";\n    outline: ").concat(dt("autocomplete.dropdown.focus.ring.width"), " ").concat(dt("autocomplete.dropdown.focus.ring.style"), " ").concat(dt("autocomplete.dropdown.focus.ring.color"), ";\n    outline-offset: ").concat(dt("autocomplete.dropdown.focus.ring.offset"), ";\n}\n\n.p-autocomplete .p-autocomplete-overlay {\n    min-width: 100%;\n}\n\n.p-autocomplete-overlay {\n    position: absolute;\n    overflow: auto;\n    top: 0;\n    left: 0;\n    background: ").concat(dt("autocomplete.overlay.background"), ";\n    color: ").concat(dt("autocomplete.overlay.color"), ";\n    border: 1px solid ").concat(dt("autocomplete.overlay.border.color"), ";\n    border-radius: ").concat(dt("autocomplete.overlay.border.radius"), ";\n    box-shadow: ").concat(dt("autocomplete.overlay.shadow"), ";\n}\n\n.p-autocomplete-list {\n    margin: 0;\n    padding: 0;\n    list-style-type: none;\n    display: flex;\n    flex-direction: column;\n    gap: ").concat(dt("autocomplete.list.gap"), ";\n    padding: ").concat(dt("autocomplete.list.padding"), ";\n}\n\n.p-autocomplete-option {\n    cursor: pointer;\n    white-space: nowrap;\n    position: relative;\n    overflow: hidden;\n    display: flex;\n    align-items: center;\n    padding: ").concat(dt("autocomplete.option.padding"), ";\n    border: 0 none;\n    color: ").concat(dt("autocomplete.option.color"), ";\n    background: transparent;\n    transition: background ").concat(dt("autocomplete.transition.duration"), ", color ").concat(dt("autocomplete.transition.duration"), ", border-color ").concat(dt("autocomplete.transition.duration"), ";\n    border-radius: ").concat(dt("autocomplete.option.border.radius"), ";\n}\n\n.p-autocomplete-option:not(.p-autocomplete-option-selected):not(.p-disabled).p-focus {\n    background: ").concat(dt("autocomplete.option.focus.background"), ";\n    color: ").concat(dt("autocomplete.option.focus.color"), ";\n}\n\n.p-autocomplete-option-selected {\n    background: ").concat(dt("autocomplete.option.selected.background"), ";\n    color: ").concat(dt("autocomplete.option.selected.color"), ";\n}\n\n.p-autocomplete-option-selected.p-focus {\n    background: ").concat(dt("autocomplete.option.selected.focus.background"), ";\n    color: ").concat(dt("autocomplete.option.selected.focus.color"), ";\n}\n\n.p-autocomplete-option-group {\n    margin: 0;\n    padding: ").concat(dt("autocomplete.option.group.padding"), ";\n    color: ").concat(dt("autocomplete.option.group.color"), ";\n    background: ").concat(dt("autocomplete.option.group.background"), ";\n    font-weight: ").concat(dt("autocomplete.option.group.font.weight"), ";\n}\n\n.p-autocomplete-input-multiple {\n    margin: 0;\n    list-style-type: none;\n    cursor: text;\n    overflow: hidden;\n    display: flex;\n    align-items: center;\n    flex-wrap: wrap;\n    padding: calc(").concat(dt("autocomplete.padding.y"), " / 2) ").concat(dt("autocomplete.padding.x"), ";\n    gap: calc(").concat(dt("autocomplete.padding.y"), " / 2);\n    color: ").concat(dt("autocomplete.color"), ";\n    background: ").concat(dt("autocomplete.background"), ";\n    border: 1px solid ").concat(dt("autocomplete.border.color"), ";\n    border-radius: ").concat(dt("autocomplete.border.radius"), ";\n    width: 100%;\n    transition: background ").concat(dt("autocomplete.transition.duration"), ", color ").concat(dt("autocomplete.transition.duration"), ", border-color ").concat(dt("autocomplete.transition.duration"), ", outline-color ").concat(dt("autocomplete.transition.duration"), ", box-shadow ").concat(dt("autocomplete.transition.duration"), ";\n    outline-color: transparent;\n    box-shadow: ").concat(dt("autocomplete.shadow"), ";\n}\n\n.p-autocomplete:not(.p-disabled):hover .p-autocomplete-input-multiple {\n    border-color: ").concat(dt("autocomplete.hover.border.color"), ";\n}\n\n.p-autocomplete:not(.p-disabled).p-focus .p-autocomplete-input-multiple {\n    border-color: ").concat(dt("autocomplete.focus.border.color"), ";\n    box-shadow: ").concat(dt("autocomplete.focus.ring.shadow"), ";\n    outline: ").concat(dt("autocomplete.focus.ring.width"), " ").concat(dt("autocomplete.focus.ring.style"), " ").concat(dt("autocomplete.focus.ring.color"), ";\n    outline-offset: ").concat(dt("autocomplete.focus.ring.offset"), ";\n}\n\n.p-autocomplete.p-invalid .p-autocomplete-input-multiple {\n    border-color: ").concat(dt("autocomplete.invalid.border.color"), ";\n}\n\n.p-variant-filled.p-autocomplete-input-multiple {\n    background: ").concat(dt("autocomplete.filled.background"), ";\n}\n\n.p-autocomplete:not(.p-disabled).p-focus .p-variant-filled.p-autocomplete-input-multiple  {\n    background: ").concat(dt("autocomplete.filled.focus.background"), ";\n}\n\n.p-autocomplete.p-disabled .p-autocomplete-input-multiple {\n    opacity: 1;\n    background: ").concat(dt("autocomplete.disabled.background"), ";\n    color: ").concat(dt("autocomplete.disabled.color"), ";\n}\n\n.p-autocomplete-chip.p-chip {\n    padding-top: calc(").concat(dt("autocomplete.padding.y"), " / 2);\n    padding-bottom: calc(").concat(dt("autocomplete.padding.y"), " / 2);\n    border-radius: ").concat(dt("autocomplete.chip.border.radius"), ";\n}\n\n.p-autocomplete-input-multiple:has(.p-autocomplete-chip) {\n    padding-left: calc(").concat(dt("autocomplete.padding.y"), " / 2);\n    padding-right: calc(").concat(dt("autocomplete.padding.y"), " / 2);\n}\n\n.p-autocomplete-chip-item.p-focus .p-autocomplete-chip {\n    background: ").concat(dt("inputchips.chip.focus.background"), ";\n    color: ").concat(dt("inputchips.chip.focus.color"), ";\n}\n\n.p-autocomplete-input-chip {\n    flex: 1 1 auto;\n    display: inline-flex;\n    padding-top: calc(").concat(dt("autocomplete.padding.y"), " / 2);\n    padding-bottom: calc(").concat(dt("autocomplete.padding.y"), " / 2);\n}\n\n.p-autocomplete-input-chip input {\n    border: 0 none;\n    outline: 0 none;\n    background: transparent;\n    margin: 0;\n    padding: 0;\n    box-shadow: none;\n    border-radius: 0;\n    width: 100%;\n    font-family: inherit;\n    font-feature-settings: inherit;\n    font-size: 1rem;\n    color: inherit;\n}\n\n.p-autocomplete-input-chip input::placeholder {\n    color: ").concat(dt("autocomplete.placeholder.color"), ";\n}\n\n.p-autocomplete-empty-message {\n    padding: ").concat(dt("autocomplete.empty.message.padding"), ";\n}\n\n.p-autocomplete-fluid {\n    display: flex;\n}\n\n.p-autocomplete-fluid:has(.p-autocomplete-dropdown) .p-autocomplete-input {\n    width: 1%;\n}\n");
}, "theme");
var inlineStyles = {
  root: {
    position: "relative"
  }
};
var classes = {
  root: /* @__PURE__ */ __name(function root4(_ref2) {
    var instance = _ref2.instance, props = _ref2.props;
    return ["p-autocomplete p-component p-inputwrapper", {
      "p-disabled": props.disabled,
      "p-invalid": props.invalid,
      "p-focus": instance.focused,
      "p-inputwrapper-filled": props.modelValue || isNotEmpty(instance.inputValue),
      "p-inputwrapper-focus": instance.focused,
      "p-autocomplete-open": instance.overlayVisible,
      "p-autocomplete-fluid": instance.hasFluid
    }];
  }, "root"),
  pcInput: "p-autocomplete-input",
  inputMultiple: /* @__PURE__ */ __name(function inputMultiple(_ref3) {
    var props = _ref3.props, instance = _ref3.instance;
    return ["p-autocomplete-input-multiple", {
      "p-variant-filled": props.variant ? props.variant === "filled" : instance.$primevue.config.inputStyle === "filled" || instance.$primevue.config.inputVariant === "filled"
    }];
  }, "inputMultiple"),
  chipItem: /* @__PURE__ */ __name(function chipItem(_ref4) {
    var instance = _ref4.instance, i = _ref4.i;
    return ["p-autocomplete-chip-item", {
      "p-focus": instance.focusedMultipleOptionIndex === i
    }];
  }, "chipItem"),
  pcChip: "p-autocomplete-chip",
  chipIcon: "p-autocomplete-chip-icon",
  inputChip: "p-autocomplete-input-chip",
  loader: "p-autocomplete-loader",
  dropdown: "p-autocomplete-dropdown",
  overlay: "p-autocomplete-overlay p-component",
  list: "p-autocomplete-list",
  optionGroup: "p-autocomplete-option-group",
  option: /* @__PURE__ */ __name(function option(_ref5) {
    var instance = _ref5.instance, _option = _ref5.option, i = _ref5.i, getItemOptions = _ref5.getItemOptions;
    return ["p-autocomplete-option", {
      "p-autocomplete-option-selected": instance.isSelected(_option),
      "p-focus": instance.focusedOptionIndex === instance.getOptionIndex(i, getItemOptions),
      "p-disabled": instance.isOptionDisabled(_option)
    }];
  }, "option"),
  emptyMessage: "p-autocomplete-empty-message"
};
var AutoCompleteStyle = BaseStyle.extend({
  name: "autocomplete",
  theme: theme3,
  classes,
  inlineStyles
});
var script$1 = {
  name: "BaseAutoComplete",
  "extends": script$7,
  props: {
    modelValue: null,
    suggestions: {
      type: Array,
      "default": null
    },
    optionLabel: null,
    optionDisabled: null,
    optionGroupLabel: null,
    optionGroupChildren: null,
    scrollHeight: {
      type: String,
      "default": "14rem"
    },
    dropdown: {
      type: Boolean,
      "default": false
    },
    dropdownMode: {
      type: String,
      "default": "blank"
    },
    multiple: {
      type: Boolean,
      "default": false
    },
    loading: {
      type: Boolean,
      "default": false
    },
    variant: {
      type: String,
      "default": null
    },
    invalid: {
      type: Boolean,
      "default": false
    },
    disabled: {
      type: Boolean,
      "default": false
    },
    placeholder: {
      type: String,
      "default": null
    },
    dataKey: {
      type: String,
      "default": null
    },
    minLength: {
      type: Number,
      "default": 1
    },
    delay: {
      type: Number,
      "default": 300
    },
    appendTo: {
      type: [String, Object],
      "default": "body"
    },
    forceSelection: {
      type: Boolean,
      "default": false
    },
    completeOnFocus: {
      type: Boolean,
      "default": false
    },
    inputId: {
      type: String,
      "default": null
    },
    inputStyle: {
      type: Object,
      "default": null
    },
    inputClass: {
      type: [String, Object],
      "default": null
    },
    panelStyle: {
      type: Object,
      "default": null
    },
    panelClass: {
      type: [String, Object],
      "default": null
    },
    overlayStyle: {
      type: Object,
      "default": null
    },
    overlayClass: {
      type: [String, Object],
      "default": null
    },
    dropdownIcon: {
      type: String,
      "default": null
    },
    dropdownClass: {
      type: [String, Object],
      "default": null
    },
    loader: {
      type: String,
      "default": null
    },
    loadingIcon: {
      type: String,
      "default": null
    },
    removeTokenIcon: {
      type: String,
      "default": null
    },
    chipIcon: {
      type: String,
      "default": null
    },
    virtualScrollerOptions: {
      type: Object,
      "default": null
    },
    autoOptionFocus: {
      type: Boolean,
      "default": false
    },
    selectOnFocus: {
      type: Boolean,
      "default": false
    },
    focusOnHover: {
      type: Boolean,
      "default": true
    },
    searchLocale: {
      type: String,
      "default": void 0
    },
    searchMessage: {
      type: String,
      "default": null
    },
    selectionMessage: {
      type: String,
      "default": null
    },
    emptySelectionMessage: {
      type: String,
      "default": null
    },
    emptySearchMessage: {
      type: String,
      "default": null
    },
    tabindex: {
      type: Number,
      "default": 0
    },
    typeahead: {
      type: Boolean,
      "default": true
    },
    ariaLabel: {
      type: String,
      "default": null
    },
    ariaLabelledby: {
      type: String,
      "default": null
    },
    fluid: {
      type: Boolean,
      "default": null
    }
  },
  style: AutoCompleteStyle,
  provide: /* @__PURE__ */ __name(function provide4() {
    return {
      $pcAutoComplete: this,
      $parentInstance: this
    };
  }, "provide")
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
var script = {
  name: "AutoComplete",
  "extends": script$1,
  inheritAttrs: false,
  emits: ["update:modelValue", "change", "focus", "blur", "item-select", "item-unselect", "option-select", "option-unselect", "dropdown-click", "clear", "complete", "before-show", "before-hide", "show", "hide"],
  inject: {
    $pcFluid: {
      "default": null
    }
  },
  outsideClickListener: null,
  resizeListener: null,
  scrollHandler: null,
  overlay: null,
  virtualScroller: null,
  searchTimeout: null,
  dirty: false,
  data: /* @__PURE__ */ __name(function data3() {
    return {
      id: this.$attrs.id,
      clicked: false,
      focused: false,
      focusedOptionIndex: -1,
      focusedMultipleOptionIndex: -1,
      overlayVisible: false,
      searching: false
    };
  }, "data"),
  watch: {
    "$attrs.id": /* @__PURE__ */ __name(function $attrsId(newValue) {
      this.id = newValue || UniqueComponentId();
    }, "$attrsId"),
    suggestions: /* @__PURE__ */ __name(function suggestions() {
      if (this.searching) {
        this.show();
        this.focusedOptionIndex = this.overlayVisible && this.autoOptionFocus ? this.findFirstFocusedOptionIndex() : -1;
        this.searching = false;
      }
      this.autoUpdateModel();
    }, "suggestions")
  },
  mounted: /* @__PURE__ */ __name(function mounted2() {
    this.id = this.id || UniqueComponentId();
    this.autoUpdateModel();
  }, "mounted"),
  updated: /* @__PURE__ */ __name(function updated() {
    if (this.overlayVisible) {
      this.alignOverlay();
    }
  }, "updated"),
  beforeUnmount: /* @__PURE__ */ __name(function beforeUnmount2() {
    this.unbindOutsideClickListener();
    this.unbindResizeListener();
    if (this.scrollHandler) {
      this.scrollHandler.destroy();
      this.scrollHandler = null;
    }
    if (this.overlay) {
      ZIndex.clear(this.overlay);
      this.overlay = null;
    }
  }, "beforeUnmount"),
  methods: {
    getOptionIndex: /* @__PURE__ */ __name(function getOptionIndex(index, fn) {
      return this.virtualScrollerDisabled ? index : fn && fn(index)["index"];
    }, "getOptionIndex"),
    getOptionLabel: /* @__PURE__ */ __name(function getOptionLabel(option2) {
      return this.optionLabel ? resolveFieldData(option2, this.optionLabel) : option2;
    }, "getOptionLabel"),
    getOptionValue: /* @__PURE__ */ __name(function getOptionValue(option2) {
      return option2;
    }, "getOptionValue"),
    getOptionRenderKey: /* @__PURE__ */ __name(function getOptionRenderKey(option2, index) {
      return (this.dataKey ? resolveFieldData(option2, this.dataKey) : this.getOptionLabel(option2)) + "_" + index;
    }, "getOptionRenderKey"),
    getPTOptions: /* @__PURE__ */ __name(function getPTOptions3(option2, itemOptions, index, key) {
      return this.ptm(key, {
        context: {
          selected: this.isSelected(option2),
          focused: this.focusedOptionIndex === this.getOptionIndex(index, itemOptions),
          disabled: this.isOptionDisabled(option2)
        }
      });
    }, "getPTOptions"),
    isOptionDisabled: /* @__PURE__ */ __name(function isOptionDisabled(option2) {
      return this.optionDisabled ? resolveFieldData(option2, this.optionDisabled) : false;
    }, "isOptionDisabled"),
    isOptionGroup: /* @__PURE__ */ __name(function isOptionGroup(option2) {
      return this.optionGroupLabel && option2.optionGroup && option2.group;
    }, "isOptionGroup"),
    getOptionGroupLabel: /* @__PURE__ */ __name(function getOptionGroupLabel(optionGroup) {
      return resolveFieldData(optionGroup, this.optionGroupLabel);
    }, "getOptionGroupLabel"),
    getOptionGroupChildren: /* @__PURE__ */ __name(function getOptionGroupChildren(optionGroup) {
      return resolveFieldData(optionGroup, this.optionGroupChildren);
    }, "getOptionGroupChildren"),
    getAriaPosInset: /* @__PURE__ */ __name(function getAriaPosInset(index) {
      var _this = this;
      return (this.optionGroupLabel ? index - this.visibleOptions.slice(0, index).filter(function(option2) {
        return _this.isOptionGroup(option2);
      }).length : index) + 1;
    }, "getAriaPosInset"),
    show: /* @__PURE__ */ __name(function show(isFocus) {
      this.$emit("before-show");
      this.dirty = true;
      this.overlayVisible = true;
      this.focusedOptionIndex = this.focusedOptionIndex !== -1 ? this.focusedOptionIndex : this.autoOptionFocus ? this.findFirstFocusedOptionIndex() : -1;
      isFocus && focus(this.multiple ? this.$refs.focusInput : this.$refs.focusInput.$el);
    }, "show"),
    hide: /* @__PURE__ */ __name(function hide(isFocus) {
      var _this2 = this;
      var _hide = /* @__PURE__ */ __name(function _hide2() {
        _this2.$emit("before-hide");
        _this2.dirty = isFocus;
        _this2.overlayVisible = false;
        _this2.clicked = false;
        _this2.focusedOptionIndex = -1;
        isFocus && focus(_this2.multiple ? _this2.$refs.focusInput : _this2.$refs.focusInput.$el);
      }, "_hide");
      setTimeout(function() {
        _hide();
      }, 0);
    }, "hide"),
    onFocus: /* @__PURE__ */ __name(function onFocus(event) {
      if (this.disabled) {
        return;
      }
      if (!this.dirty && this.completeOnFocus) {
        this.search(event, event.target.value, "focus");
      }
      this.dirty = true;
      this.focused = true;
      if (this.overlayVisible) {
        this.focusedOptionIndex = this.focusedOptionIndex !== -1 ? this.focusedOptionIndex : this.overlayVisible && this.autoOptionFocus ? this.findFirstFocusedOptionIndex() : -1;
        this.scrollInView(this.focusedOptionIndex);
      }
      this.$emit("focus", event);
    }, "onFocus"),
    onBlur: /* @__PURE__ */ __name(function onBlur(event) {
      this.dirty = false;
      this.focused = false;
      this.focusedOptionIndex = -1;
      this.$emit("blur", event);
    }, "onBlur"),
    onKeyDown: /* @__PURE__ */ __name(function onKeyDown(event) {
      if (this.disabled) {
        event.preventDefault();
        return;
      }
      switch (event.code) {
        case "ArrowDown":
          this.onArrowDownKey(event);
          break;
        case "ArrowUp":
          this.onArrowUpKey(event);
          break;
        case "ArrowLeft":
          this.onArrowLeftKey(event);
          break;
        case "ArrowRight":
          this.onArrowRightKey(event);
          break;
        case "Home":
          this.onHomeKey(event);
          break;
        case "End":
          this.onEndKey(event);
          break;
        case "PageDown":
          this.onPageDownKey(event);
          break;
        case "PageUp":
          this.onPageUpKey(event);
          break;
        case "Enter":
        case "NumpadEnter":
          this.onEnterKey(event);
          break;
        case "Escape":
          this.onEscapeKey(event);
          break;
        case "Tab":
          this.onTabKey(event);
          break;
        case "Backspace":
          this.onBackspaceKey(event);
          break;
      }
      this.clicked = false;
    }, "onKeyDown"),
    onInput: /* @__PURE__ */ __name(function onInput(event) {
      var _this3 = this;
      if (this.typeahead) {
        if (this.searchTimeout) {
          clearTimeout(this.searchTimeout);
        }
        var query = event.target.value;
        if (!this.multiple) {
          this.updateModel(event, query);
        }
        if (query.length === 0) {
          this.hide();
          this.$emit("clear");
        } else {
          if (query.length >= this.minLength) {
            this.focusedOptionIndex = -1;
            this.searchTimeout = setTimeout(function() {
              _this3.search(event, query, "input");
            }, this.delay);
          } else {
            this.hide();
          }
        }
      }
    }, "onInput"),
    onChange: /* @__PURE__ */ __name(function onChange(event) {
      var _this4 = this;
      if (this.forceSelection) {
        var valid = false;
        if (this.visibleOptions && !this.multiple) {
          var value = this.multiple ? this.$refs.focusInput.value : this.$refs.focusInput.$el.value;
          var matchedValue = this.visibleOptions.find(function(option2) {
            return _this4.isOptionMatched(option2, value || "");
          });
          if (matchedValue !== void 0) {
            valid = true;
            !this.isSelected(matchedValue) && this.onOptionSelect(event, matchedValue);
          }
        }
        if (!valid) {
          if (this.multiple) this.$refs.focusInput.value = "";
          else this.$refs.focusInput.$el.value = "";
          this.$emit("clear");
          !this.multiple && this.updateModel(event, null);
        }
      }
    }, "onChange"),
    onMultipleContainerFocus: /* @__PURE__ */ __name(function onMultipleContainerFocus() {
      if (this.disabled) {
        return;
      }
      this.focused = true;
    }, "onMultipleContainerFocus"),
    onMultipleContainerBlur: /* @__PURE__ */ __name(function onMultipleContainerBlur() {
      this.focusedMultipleOptionIndex = -1;
      this.focused = false;
    }, "onMultipleContainerBlur"),
    onMultipleContainerKeyDown: /* @__PURE__ */ __name(function onMultipleContainerKeyDown(event) {
      if (this.disabled) {
        event.preventDefault();
        return;
      }
      switch (event.code) {
        case "ArrowLeft":
          this.onArrowLeftKeyOnMultiple(event);
          break;
        case "ArrowRight":
          this.onArrowRightKeyOnMultiple(event);
          break;
        case "Backspace":
          this.onBackspaceKeyOnMultiple(event);
          break;
      }
    }, "onMultipleContainerKeyDown"),
    onContainerClick: /* @__PURE__ */ __name(function onContainerClick(event) {
      this.clicked = true;
      if (this.disabled || this.searching || this.loading || this.isInputClicked(event) || this.isDropdownClicked(event)) {
        return;
      }
      if (!this.overlay || !this.overlay.contains(event.target)) {
        focus(this.multiple ? this.$refs.focusInput : this.$refs.focusInput.$el);
      }
    }, "onContainerClick"),
    onDropdownClick: /* @__PURE__ */ __name(function onDropdownClick(event) {
      var query = void 0;
      if (this.overlayVisible) {
        this.hide(true);
      } else {
        var target = this.multiple ? this.$refs.focusInput : this.$refs.focusInput.$el;
        focus(target);
        query = target.value;
        if (this.dropdownMode === "blank") this.search(event, "", "dropdown");
        else if (this.dropdownMode === "current") this.search(event, query, "dropdown");
      }
      this.$emit("dropdown-click", {
        originalEvent: event,
        query
      });
    }, "onDropdownClick"),
    onOptionSelect: /* @__PURE__ */ __name(function onOptionSelect(event, option2) {
      var isHide = arguments.length > 2 && arguments[2] !== void 0 ? arguments[2] : true;
      var value = this.getOptionValue(option2);
      if (this.multiple) {
        this.$refs.focusInput.value = "";
        if (!this.isSelected(option2)) {
          this.updateModel(event, [].concat(_toConsumableArray(this.modelValue || []), [value]));
        }
      } else {
        this.updateModel(event, value);
      }
      this.$emit("item-select", {
        originalEvent: event,
        value: option2
      });
      this.$emit("option-select", {
        originalEvent: event,
        value: option2
      });
      isHide && this.hide(true);
    }, "onOptionSelect"),
    onOptionMouseMove: /* @__PURE__ */ __name(function onOptionMouseMove(event, index) {
      if (this.focusOnHover) {
        this.changeFocusedOptionIndex(event, index);
      }
    }, "onOptionMouseMove"),
    onOverlayClick: /* @__PURE__ */ __name(function onOverlayClick(event) {
      OverlayEventBus.emit("overlay-click", {
        originalEvent: event,
        target: this.$el
      });
    }, "onOverlayClick"),
    onOverlayKeyDown: /* @__PURE__ */ __name(function onOverlayKeyDown(event) {
      switch (event.code) {
        case "Escape":
          this.onEscapeKey(event);
          break;
      }
    }, "onOverlayKeyDown"),
    onArrowDownKey: /* @__PURE__ */ __name(function onArrowDownKey(event) {
      if (!this.overlayVisible) {
        return;
      }
      var optionIndex = this.focusedOptionIndex !== -1 ? this.findNextOptionIndex(this.focusedOptionIndex) : this.clicked ? this.findFirstOptionIndex() : this.findFirstFocusedOptionIndex();
      this.changeFocusedOptionIndex(event, optionIndex);
      event.preventDefault();
    }, "onArrowDownKey"),
    onArrowUpKey: /* @__PURE__ */ __name(function onArrowUpKey(event) {
      if (!this.overlayVisible) {
        return;
      }
      if (event.altKey) {
        if (this.focusedOptionIndex !== -1) {
          this.onOptionSelect(event, this.visibleOptions[this.focusedOptionIndex]);
        }
        this.overlayVisible && this.hide();
        event.preventDefault();
      } else {
        var optionIndex = this.focusedOptionIndex !== -1 ? this.findPrevOptionIndex(this.focusedOptionIndex) : this.clicked ? this.findLastOptionIndex() : this.findLastFocusedOptionIndex();
        this.changeFocusedOptionIndex(event, optionIndex);
        event.preventDefault();
      }
    }, "onArrowUpKey"),
    onArrowLeftKey: /* @__PURE__ */ __name(function onArrowLeftKey(event) {
      var target = event.currentTarget;
      this.focusedOptionIndex = -1;
      if (this.multiple) {
        if (isEmpty(target.value) && this.hasSelectedOption) {
          focus(this.$refs.multiContainer);
          this.focusedMultipleOptionIndex = this.modelValue.length;
        } else {
          event.stopPropagation();
        }
      }
    }, "onArrowLeftKey"),
    onArrowRightKey: /* @__PURE__ */ __name(function onArrowRightKey(event) {
      this.focusedOptionIndex = -1;
      this.multiple && event.stopPropagation();
    }, "onArrowRightKey"),
    onHomeKey: /* @__PURE__ */ __name(function onHomeKey(event) {
      var currentTarget = event.currentTarget;
      var len = currentTarget.value.length;
      currentTarget.setSelectionRange(0, event.shiftKey ? len : 0);
      this.focusedOptionIndex = -1;
      event.preventDefault();
    }, "onHomeKey"),
    onEndKey: /* @__PURE__ */ __name(function onEndKey(event) {
      var currentTarget = event.currentTarget;
      var len = currentTarget.value.length;
      currentTarget.setSelectionRange(event.shiftKey ? 0 : len, len);
      this.focusedOptionIndex = -1;
      event.preventDefault();
    }, "onEndKey"),
    onPageUpKey: /* @__PURE__ */ __name(function onPageUpKey(event) {
      this.scrollInView(0);
      event.preventDefault();
    }, "onPageUpKey"),
    onPageDownKey: /* @__PURE__ */ __name(function onPageDownKey(event) {
      this.scrollInView(this.visibleOptions.length - 1);
      event.preventDefault();
    }, "onPageDownKey"),
    onEnterKey: /* @__PURE__ */ __name(function onEnterKey(event) {
      if (!this.typeahead) {
        if (this.multiple) {
          this.updateModel(event, [].concat(_toConsumableArray(this.modelValue || []), [event.target.value]));
          this.$refs.focusInput.value = "";
        }
      } else {
        if (!this.overlayVisible) {
          this.focusedOptionIndex = -1;
          this.onArrowDownKey(event);
        } else {
          if (this.focusedOptionIndex !== -1) {
            this.onOptionSelect(event, this.visibleOptions[this.focusedOptionIndex]);
          }
          this.hide();
        }
      }
    }, "onEnterKey"),
    onEscapeKey: /* @__PURE__ */ __name(function onEscapeKey(event) {
      this.overlayVisible && this.hide(true);
      event.preventDefault();
    }, "onEscapeKey"),
    onTabKey: /* @__PURE__ */ __name(function onTabKey(event) {
      if (this.focusedOptionIndex !== -1) {
        this.onOptionSelect(event, this.visibleOptions[this.focusedOptionIndex]);
      }
      this.overlayVisible && this.hide();
    }, "onTabKey"),
    onBackspaceKey: /* @__PURE__ */ __name(function onBackspaceKey(event) {
      if (this.multiple) {
        if (isNotEmpty(this.modelValue) && !this.$refs.focusInput.value) {
          var removedValue = this.modelValue[this.modelValue.length - 1];
          var newValue = this.modelValue.slice(0, -1);
          this.$emit("update:modelValue", newValue);
          this.$emit("item-unselect", {
            originalEvent: event,
            value: removedValue
          });
          this.$emit("option-unselect", {
            originalEvent: event,
            value: removedValue
          });
        }
        event.stopPropagation();
      }
    }, "onBackspaceKey"),
    onArrowLeftKeyOnMultiple: /* @__PURE__ */ __name(function onArrowLeftKeyOnMultiple() {
      this.focusedMultipleOptionIndex = this.focusedMultipleOptionIndex < 1 ? 0 : this.focusedMultipleOptionIndex - 1;
    }, "onArrowLeftKeyOnMultiple"),
    onArrowRightKeyOnMultiple: /* @__PURE__ */ __name(function onArrowRightKeyOnMultiple() {
      this.focusedMultipleOptionIndex++;
      if (this.focusedMultipleOptionIndex > this.modelValue.length - 1) {
        this.focusedMultipleOptionIndex = -1;
        focus(this.$refs.focusInput);
      }
    }, "onArrowRightKeyOnMultiple"),
    onBackspaceKeyOnMultiple: /* @__PURE__ */ __name(function onBackspaceKeyOnMultiple(event) {
      if (this.focusedMultipleOptionIndex !== -1) {
        this.removeOption(event, this.focusedMultipleOptionIndex);
      }
    }, "onBackspaceKeyOnMultiple"),
    onOverlayEnter: /* @__PURE__ */ __name(function onOverlayEnter(el) {
      ZIndex.set("overlay", el, this.$primevue.config.zIndex.overlay);
      addStyle(el, {
        position: "absolute",
        top: "0",
        left: "0"
      });
      this.alignOverlay();
    }, "onOverlayEnter"),
    onOverlayAfterEnter: /* @__PURE__ */ __name(function onOverlayAfterEnter() {
      this.bindOutsideClickListener();
      this.bindScrollListener();
      this.bindResizeListener();
      this.$emit("show");
    }, "onOverlayAfterEnter"),
    onOverlayLeave: /* @__PURE__ */ __name(function onOverlayLeave() {
      this.unbindOutsideClickListener();
      this.unbindScrollListener();
      this.unbindResizeListener();
      this.$emit("hide");
      this.overlay = null;
    }, "onOverlayLeave"),
    onOverlayAfterLeave: /* @__PURE__ */ __name(function onOverlayAfterLeave(el) {
      ZIndex.clear(el);
    }, "onOverlayAfterLeave"),
    alignOverlay: /* @__PURE__ */ __name(function alignOverlay() {
      var target = this.multiple ? this.$refs.multiContainer : this.$refs.focusInput.$el;
      if (this.appendTo === "self") {
        relativePosition(this.overlay, target);
      } else {
        this.overlay.style.minWidth = getOuterWidth(target) + "px";
        absolutePosition(this.overlay, target);
      }
    }, "alignOverlay"),
    bindOutsideClickListener: /* @__PURE__ */ __name(function bindOutsideClickListener() {
      var _this5 = this;
      if (!this.outsideClickListener) {
        this.outsideClickListener = function(event) {
          if (_this5.overlayVisible && _this5.overlay && _this5.isOutsideClicked(event)) {
            _this5.hide();
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
    bindScrollListener: /* @__PURE__ */ __name(function bindScrollListener() {
      var _this6 = this;
      if (!this.scrollHandler) {
        this.scrollHandler = new ConnectedOverlayScrollHandler(this.$refs.container, function() {
          if (_this6.overlayVisible) {
            _this6.hide();
          }
        });
      }
      this.scrollHandler.bindScrollListener();
    }, "bindScrollListener"),
    unbindScrollListener: /* @__PURE__ */ __name(function unbindScrollListener() {
      if (this.scrollHandler) {
        this.scrollHandler.unbindScrollListener();
      }
    }, "unbindScrollListener"),
    bindResizeListener: /* @__PURE__ */ __name(function bindResizeListener() {
      var _this7 = this;
      if (!this.resizeListener) {
        this.resizeListener = function() {
          if (_this7.overlayVisible && !isTouchDevice()) {
            _this7.hide();
          }
        };
        window.addEventListener("resize", this.resizeListener);
      }
    }, "bindResizeListener"),
    unbindResizeListener: /* @__PURE__ */ __name(function unbindResizeListener() {
      if (this.resizeListener) {
        window.removeEventListener("resize", this.resizeListener);
        this.resizeListener = null;
      }
    }, "unbindResizeListener"),
    isOutsideClicked: /* @__PURE__ */ __name(function isOutsideClicked(event) {
      return !this.overlay.contains(event.target) && !this.isInputClicked(event) && !this.isDropdownClicked(event);
    }, "isOutsideClicked"),
    isInputClicked: /* @__PURE__ */ __name(function isInputClicked(event) {
      if (this.multiple) return event.target === this.$refs.multiContainer || this.$refs.multiContainer.contains(event.target);
      else return event.target === this.$refs.focusInput.$el;
    }, "isInputClicked"),
    isDropdownClicked: /* @__PURE__ */ __name(function isDropdownClicked(event) {
      return this.$refs.dropdownButton ? event.target === this.$refs.dropdownButton || this.$refs.dropdownButton.contains(event.target) : false;
    }, "isDropdownClicked"),
    isOptionMatched: /* @__PURE__ */ __name(function isOptionMatched(option2, value) {
      var _this$getOptionLabel;
      return this.isValidOption(option2) && ((_this$getOptionLabel = this.getOptionLabel(option2)) === null || _this$getOptionLabel === void 0 ? void 0 : _this$getOptionLabel.toLocaleLowerCase(this.searchLocale)) === value.toLocaleLowerCase(this.searchLocale);
    }, "isOptionMatched"),
    isValidOption: /* @__PURE__ */ __name(function isValidOption(option2) {
      return isNotEmpty(option2) && !(this.isOptionDisabled(option2) || this.isOptionGroup(option2));
    }, "isValidOption"),
    isValidSelectedOption: /* @__PURE__ */ __name(function isValidSelectedOption(option2) {
      return this.isValidOption(option2) && this.isSelected(option2);
    }, "isValidSelectedOption"),
    isEquals: /* @__PURE__ */ __name(function isEquals(value1, value2) {
      return equals(value1, value2, this.equalityKey);
    }, "isEquals"),
    isSelected: /* @__PURE__ */ __name(function isSelected(option2) {
      var _this8 = this;
      var optionValue = this.getOptionValue(option2);
      return this.multiple ? (this.modelValue || []).some(function(value) {
        return _this8.isEquals(value, optionValue);
      }) : this.isEquals(this.modelValue, this.getOptionValue(option2));
    }, "isSelected"),
    findFirstOptionIndex: /* @__PURE__ */ __name(function findFirstOptionIndex() {
      var _this9 = this;
      return this.visibleOptions.findIndex(function(option2) {
        return _this9.isValidOption(option2);
      });
    }, "findFirstOptionIndex"),
    findLastOptionIndex: /* @__PURE__ */ __name(function findLastOptionIndex() {
      var _this10 = this;
      return findLastIndex(this.visibleOptions, function(option2) {
        return _this10.isValidOption(option2);
      });
    }, "findLastOptionIndex"),
    findNextOptionIndex: /* @__PURE__ */ __name(function findNextOptionIndex(index) {
      var _this11 = this;
      var matchedOptionIndex = index < this.visibleOptions.length - 1 ? this.visibleOptions.slice(index + 1).findIndex(function(option2) {
        return _this11.isValidOption(option2);
      }) : -1;
      return matchedOptionIndex > -1 ? matchedOptionIndex + index + 1 : index;
    }, "findNextOptionIndex"),
    findPrevOptionIndex: /* @__PURE__ */ __name(function findPrevOptionIndex(index) {
      var _this12 = this;
      var matchedOptionIndex = index > 0 ? findLastIndex(this.visibleOptions.slice(0, index), function(option2) {
        return _this12.isValidOption(option2);
      }) : -1;
      return matchedOptionIndex > -1 ? matchedOptionIndex : index;
    }, "findPrevOptionIndex"),
    findSelectedOptionIndex: /* @__PURE__ */ __name(function findSelectedOptionIndex() {
      var _this13 = this;
      return this.hasSelectedOption ? this.visibleOptions.findIndex(function(option2) {
        return _this13.isValidSelectedOption(option2);
      }) : -1;
    }, "findSelectedOptionIndex"),
    findFirstFocusedOptionIndex: /* @__PURE__ */ __name(function findFirstFocusedOptionIndex() {
      var selectedIndex = this.findSelectedOptionIndex();
      return selectedIndex < 0 ? this.findFirstOptionIndex() : selectedIndex;
    }, "findFirstFocusedOptionIndex"),
    findLastFocusedOptionIndex: /* @__PURE__ */ __name(function findLastFocusedOptionIndex() {
      var selectedIndex = this.findSelectedOptionIndex();
      return selectedIndex < 0 ? this.findLastOptionIndex() : selectedIndex;
    }, "findLastFocusedOptionIndex"),
    search: /* @__PURE__ */ __name(function search(event, query, source) {
      if (query === void 0 || query === null) {
        return;
      }
      if (source === "input" && query.trim().length === 0) {
        return;
      }
      this.searching = true;
      this.$emit("complete", {
        originalEvent: event,
        query
      });
    }, "search"),
    removeOption: /* @__PURE__ */ __name(function removeOption(event, index) {
      var _this14 = this;
      var removedOption = this.modelValue[index];
      var value = this.modelValue.filter(function(_, i) {
        return i !== index;
      }).map(function(option2) {
        return _this14.getOptionValue(option2);
      });
      this.updateModel(event, value);
      this.$emit("item-unselect", {
        originalEvent: event,
        value: removedOption
      });
      this.$emit("option-unselect", {
        originalEvent: event,
        value: removedOption
      });
      this.dirty = true;
      focus(this.multiple ? this.$refs.focusInput : this.$refs.focusInput.$el);
    }, "removeOption"),
    changeFocusedOptionIndex: /* @__PURE__ */ __name(function changeFocusedOptionIndex(event, index) {
      if (this.focusedOptionIndex !== index) {
        this.focusedOptionIndex = index;
        this.scrollInView();
        if (this.selectOnFocus) {
          this.onOptionSelect(event, this.visibleOptions[index], false);
        }
      }
    }, "changeFocusedOptionIndex"),
    scrollInView: /* @__PURE__ */ __name(function scrollInView() {
      var _this15 = this;
      var index = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : -1;
      this.$nextTick(function() {
        var id = index !== -1 ? "".concat(_this15.id, "_").concat(index) : _this15.focusedOptionId;
        var element = findSingle(_this15.list, 'li[id="'.concat(id, '"]'));
        if (element) {
          element.scrollIntoView && element.scrollIntoView({
            block: "nearest",
            inline: "start"
          });
        } else if (!_this15.virtualScrollerDisabled) {
          _this15.virtualScroller && _this15.virtualScroller.scrollToIndex(index !== -1 ? index : _this15.focusedOptionIndex);
        }
      });
    }, "scrollInView"),
    autoUpdateModel: /* @__PURE__ */ __name(function autoUpdateModel() {
      if (this.selectOnFocus && this.autoOptionFocus && !this.hasSelectedOption) {
        this.focusedOptionIndex = this.findFirstFocusedOptionIndex();
        this.onOptionSelect(null, this.visibleOptions[this.focusedOptionIndex], false);
      }
    }, "autoUpdateModel"),
    updateModel: /* @__PURE__ */ __name(function updateModel(event, value) {
      this.$emit("update:modelValue", value);
      this.$emit("change", {
        originalEvent: event,
        value
      });
    }, "updateModel"),
    flatOptions: /* @__PURE__ */ __name(function flatOptions(options) {
      var _this16 = this;
      return (options || []).reduce(function(result, option2, index) {
        result.push({
          optionGroup: option2,
          group: true,
          index
        });
        var optionGroupChildren = _this16.getOptionGroupChildren(option2);
        optionGroupChildren && optionGroupChildren.forEach(function(o) {
          return result.push(o);
        });
        return result;
      }, []);
    }, "flatOptions"),
    overlayRef: /* @__PURE__ */ __name(function overlayRef(el) {
      this.overlay = el;
    }, "overlayRef"),
    listRef: /* @__PURE__ */ __name(function listRef(el, contentRef) {
      this.list = el;
      contentRef && contentRef(el);
    }, "listRef"),
    virtualScrollerRef: /* @__PURE__ */ __name(function virtualScrollerRef(el) {
      this.virtualScroller = el;
    }, "virtualScrollerRef")
  },
  computed: {
    visibleOptions: /* @__PURE__ */ __name(function visibleOptions() {
      return this.optionGroupLabel ? this.flatOptions(this.suggestions) : this.suggestions || [];
    }, "visibleOptions"),
    inputValue: /* @__PURE__ */ __name(function inputValue() {
      if (isNotEmpty(this.modelValue)) {
        if (_typeof$1(this.modelValue) === "object") {
          var label = this.getOptionLabel(this.modelValue);
          return label != null ? label : this.modelValue;
        } else {
          return this.modelValue;
        }
      } else {
        return "";
      }
    }, "inputValue"),
    hasSelectedOption: /* @__PURE__ */ __name(function hasSelectedOption() {
      return isNotEmpty(this.modelValue);
    }, "hasSelectedOption"),
    equalityKey: /* @__PURE__ */ __name(function equalityKey() {
      return this.dataKey;
    }, "equalityKey"),
    searchResultMessageText: /* @__PURE__ */ __name(function searchResultMessageText() {
      return isNotEmpty(this.visibleOptions) && this.overlayVisible ? this.searchMessageText.replaceAll("{0}", this.visibleOptions.length) : this.emptySearchMessageText;
    }, "searchResultMessageText"),
    searchMessageText: /* @__PURE__ */ __name(function searchMessageText() {
      return this.searchMessage || this.$primevue.config.locale.searchMessage || "";
    }, "searchMessageText"),
    emptySearchMessageText: /* @__PURE__ */ __name(function emptySearchMessageText() {
      return this.emptySearchMessage || this.$primevue.config.locale.emptySearchMessage || "";
    }, "emptySearchMessageText"),
    selectionMessageText: /* @__PURE__ */ __name(function selectionMessageText() {
      return this.selectionMessage || this.$primevue.config.locale.selectionMessage || "";
    }, "selectionMessageText"),
    emptySelectionMessageText: /* @__PURE__ */ __name(function emptySelectionMessageText() {
      return this.emptySelectionMessage || this.$primevue.config.locale.emptySelectionMessage || "";
    }, "emptySelectionMessageText"),
    selectedMessageText: /* @__PURE__ */ __name(function selectedMessageText() {
      return this.hasSelectedOption ? this.selectionMessageText.replaceAll("{0}", this.multiple ? this.modelValue.length : "1") : this.emptySelectionMessageText;
    }, "selectedMessageText"),
    listAriaLabel: /* @__PURE__ */ __name(function listAriaLabel() {
      return this.$primevue.config.locale.aria ? this.$primevue.config.locale.aria.listLabel : void 0;
    }, "listAriaLabel"),
    focusedOptionId: /* @__PURE__ */ __name(function focusedOptionId() {
      return this.focusedOptionIndex !== -1 ? "".concat(this.id, "_").concat(this.focusedOptionIndex) : null;
    }, "focusedOptionId"),
    focusedMultipleOptionId: /* @__PURE__ */ __name(function focusedMultipleOptionId() {
      return this.focusedMultipleOptionIndex !== -1 ? "".concat(this.id, "_multiple_option_").concat(this.focusedMultipleOptionIndex) : null;
    }, "focusedMultipleOptionId"),
    ariaSetSize: /* @__PURE__ */ __name(function ariaSetSize() {
      var _this17 = this;
      return this.visibleOptions.filter(function(option2) {
        return !_this17.isOptionGroup(option2);
      }).length;
    }, "ariaSetSize"),
    virtualScrollerDisabled: /* @__PURE__ */ __name(function virtualScrollerDisabled() {
      return !this.virtualScrollerOptions;
    }, "virtualScrollerDisabled"),
    panelId: /* @__PURE__ */ __name(function panelId() {
      return this.id + "_panel";
    }, "panelId"),
    hasFluid: /* @__PURE__ */ __name(function hasFluid() {
      return isEmpty(this.fluid) ? !!this.$pcFluid : this.fluid;
    }, "hasFluid")
  },
  components: {
    InputText: script$8,
    VirtualScroller: script$9,
    Portal: script$a,
    ChevronDownIcon: script$b,
    SpinnerIcon: script$c,
    Chip: script$d
  },
  directives: {
    ripple: Ripple
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
var _hoisted_1$2 = ["aria-activedescendant"];
var _hoisted_2$2 = ["id", "aria-label", "aria-setsize", "aria-posinset"];
var _hoisted_3$2 = ["id", "placeholder", "tabindex", "disabled", "aria-label", "aria-labelledby", "aria-expanded", "aria-controls", "aria-activedescendant", "aria-invalid"];
var _hoisted_4$2 = ["disabled", "aria-expanded", "aria-controls"];
var _hoisted_5$1 = ["id"];
var _hoisted_6$1 = ["id", "aria-label"];
var _hoisted_7$1 = ["id"];
var _hoisted_8$1 = ["id", "aria-label", "aria-selected", "aria-disabled", "aria-setsize", "aria-posinset", "onClick", "onMousemove", "data-p-selected", "data-p-focus", "data-p-disabled"];
function render(_ctx, _cache, $props, $setup, $data, $options) {
  var _component_InputText = resolveComponent("InputText");
  var _component_Chip = resolveComponent("Chip");
  var _component_SpinnerIcon = resolveComponent("SpinnerIcon");
  var _component_VirtualScroller = resolveComponent("VirtualScroller");
  var _component_Portal = resolveComponent("Portal");
  var _directive_ripple = resolveDirective("ripple");
  return openBlock(), createElementBlock("div", mergeProps({
    ref: "container",
    "class": _ctx.cx("root"),
    style: _ctx.sx("root"),
    onClick: _cache[11] || (_cache[11] = function() {
      return $options.onContainerClick && $options.onContainerClick.apply($options, arguments);
    })
  }, _ctx.ptmi("root")), [!_ctx.multiple ? (openBlock(), createBlock(_component_InputText, {
    key: 0,
    ref: "focusInput",
    id: _ctx.inputId,
    type: "text",
    "class": normalizeClass([_ctx.cx("pcInput"), _ctx.inputClass]),
    style: normalizeStyle(_ctx.inputStyle),
    value: $options.inputValue,
    placeholder: _ctx.placeholder,
    tabindex: !_ctx.disabled ? _ctx.tabindex : -1,
    fluid: $options.hasFluid,
    disabled: _ctx.disabled,
    invalid: _ctx.invalid,
    variant: _ctx.variant,
    autocomplete: "off",
    role: "combobox",
    "aria-label": _ctx.ariaLabel,
    "aria-labelledby": _ctx.ariaLabelledby,
    "aria-haspopup": "listbox",
    "aria-autocomplete": "list",
    "aria-expanded": $data.overlayVisible,
    "aria-controls": $options.panelId,
    "aria-activedescendant": $data.focused ? $options.focusedOptionId : void 0,
    onFocus: $options.onFocus,
    onBlur: $options.onBlur,
    onKeydown: $options.onKeyDown,
    onInput: $options.onInput,
    onChange: $options.onChange,
    unstyled: _ctx.unstyled,
    pt: _ctx.ptm("pcInput")
  }, null, 8, ["id", "class", "style", "value", "placeholder", "tabindex", "fluid", "disabled", "invalid", "variant", "aria-label", "aria-labelledby", "aria-expanded", "aria-controls", "aria-activedescendant", "onFocus", "onBlur", "onKeydown", "onInput", "onChange", "unstyled", "pt"])) : createCommentVNode("", true), _ctx.multiple ? (openBlock(), createElementBlock("ul", mergeProps({
    key: 1,
    ref: "multiContainer",
    "class": _ctx.cx("inputMultiple"),
    tabindex: "-1",
    role: "listbox",
    "aria-orientation": "horizontal",
    "aria-activedescendant": $data.focused ? $options.focusedMultipleOptionId : void 0,
    onFocus: _cache[5] || (_cache[5] = function() {
      return $options.onMultipleContainerFocus && $options.onMultipleContainerFocus.apply($options, arguments);
    }),
    onBlur: _cache[6] || (_cache[6] = function() {
      return $options.onMultipleContainerBlur && $options.onMultipleContainerBlur.apply($options, arguments);
    }),
    onKeydown: _cache[7] || (_cache[7] = function() {
      return $options.onMultipleContainerKeyDown && $options.onMultipleContainerKeyDown.apply($options, arguments);
    })
  }, _ctx.ptm("inputMultiple")), [(openBlock(true), createElementBlock(Fragment, null, renderList(_ctx.modelValue, function(option2, i) {
    return openBlock(), createElementBlock("li", mergeProps({
      key: "".concat(i, "_").concat($options.getOptionLabel(option2)),
      id: $data.id + "_multiple_option_" + i,
      "class": _ctx.cx("chipItem", {
        i
      }),
      role: "option",
      "aria-label": $options.getOptionLabel(option2),
      "aria-selected": true,
      "aria-setsize": _ctx.modelValue.length,
      "aria-posinset": i + 1,
      ref_for: true
    }, _ctx.ptm("chipItem")), [renderSlot(_ctx.$slots, "chip", mergeProps({
      "class": _ctx.cx("pcChip"),
      value: option2,
      index: i,
      removeCallback: /* @__PURE__ */ __name(function removeCallback(event) {
        return $options.removeOption(event, i);
      }, "removeCallback"),
      ref_for: true
    }, _ctx.ptm("pcChip")), function() {
      return [createVNode(_component_Chip, {
        "class": normalizeClass(_ctx.cx("pcChip")),
        label: $options.getOptionLabel(option2),
        removeIcon: _ctx.chipIcon || _ctx.removeTokenIcon,
        removable: "",
        unstyled: _ctx.unstyled,
        onRemove: /* @__PURE__ */ __name(function onRemove($event) {
          return $options.removeOption($event, i);
        }, "onRemove"),
        pt: _ctx.ptm("pcChip")
      }, {
        removeicon: withCtx(function() {
          return [renderSlot(_ctx.$slots, _ctx.$slots.chipicon ? "chipicon" : "removetokenicon", {
            "class": normalizeClass(_ctx.cx("chipIcon")),
            index: i,
            removeCallback: /* @__PURE__ */ __name(function removeCallback(event) {
              return $options.removeOption(event, i);
            }, "removeCallback")
          })];
        }),
        _: 2
      }, 1032, ["class", "label", "removeIcon", "unstyled", "onRemove", "pt"])];
    })], 16, _hoisted_2$2);
  }), 128)), createBaseVNode("li", mergeProps({
    "class": _ctx.cx("inputChip"),
    role: "option"
  }, _ctx.ptm("inputChip")), [createBaseVNode("input", mergeProps({
    ref: "focusInput",
    id: _ctx.inputId,
    type: "text",
    style: _ctx.inputStyle,
    "class": _ctx.inputClass,
    placeholder: _ctx.placeholder,
    tabindex: !_ctx.disabled ? _ctx.tabindex : -1,
    disabled: _ctx.disabled,
    autocomplete: "off",
    role: "combobox",
    "aria-label": _ctx.ariaLabel,
    "aria-labelledby": _ctx.ariaLabelledby,
    "aria-haspopup": "listbox",
    "aria-autocomplete": "list",
    "aria-expanded": $data.overlayVisible,
    "aria-controls": $data.id + "_list",
    "aria-activedescendant": $data.focused ? $options.focusedOptionId : void 0,
    "aria-invalid": _ctx.invalid || void 0,
    onFocus: _cache[0] || (_cache[0] = function() {
      return $options.onFocus && $options.onFocus.apply($options, arguments);
    }),
    onBlur: _cache[1] || (_cache[1] = function() {
      return $options.onBlur && $options.onBlur.apply($options, arguments);
    }),
    onKeydown: _cache[2] || (_cache[2] = function() {
      return $options.onKeyDown && $options.onKeyDown.apply($options, arguments);
    }),
    onInput: _cache[3] || (_cache[3] = function() {
      return $options.onInput && $options.onInput.apply($options, arguments);
    }),
    onChange: _cache[4] || (_cache[4] = function() {
      return $options.onChange && $options.onChange.apply($options, arguments);
    })
  }, _ctx.ptm("input")), null, 16, _hoisted_3$2)], 16)], 16, _hoisted_1$2)) : createCommentVNode("", true), $data.searching || _ctx.loading ? renderSlot(_ctx.$slots, _ctx.$slots.loader ? "loader" : "loadingicon", {
    key: 2,
    "class": normalizeClass(_ctx.cx("loader"))
  }, function() {
    return [_ctx.loader || _ctx.loadingIcon ? (openBlock(), createElementBlock("i", mergeProps({
      key: 0,
      "class": ["pi-spin", _ctx.cx("loader"), _ctx.loader, _ctx.loadingIcon],
      "aria-hidden": "true"
    }, _ctx.ptm("loader")), null, 16)) : (openBlock(), createBlock(_component_SpinnerIcon, mergeProps({
      key: 1,
      "class": _ctx.cx("loader"),
      spin: "",
      "aria-hidden": "true"
    }, _ctx.ptm("loader")), null, 16, ["class"]))];
  }) : createCommentVNode("", true), renderSlot(_ctx.$slots, _ctx.$slots.dropdown ? "dropdown" : "dropdownbutton", {
    toggleCallback: /* @__PURE__ */ __name(function toggleCallback(event) {
      return $options.onDropdownClick(event);
    }, "toggleCallback")
  }, function() {
    return [_ctx.dropdown ? (openBlock(), createElementBlock("button", mergeProps({
      key: 0,
      ref: "dropdownButton",
      type: "button",
      "class": [_ctx.cx("dropdown"), _ctx.dropdownClass],
      disabled: _ctx.disabled,
      "aria-haspopup": "listbox",
      "aria-expanded": $data.overlayVisible,
      "aria-controls": $options.panelId,
      onClick: _cache[8] || (_cache[8] = function() {
        return $options.onDropdownClick && $options.onDropdownClick.apply($options, arguments);
      })
    }, _ctx.ptm("dropdown")), [renderSlot(_ctx.$slots, "dropdownicon", {
      "class": normalizeClass(_ctx.dropdownIcon)
    }, function() {
      return [(openBlock(), createBlock(resolveDynamicComponent(_ctx.dropdownIcon ? "span" : "ChevronDownIcon"), mergeProps({
        "class": _ctx.dropdownIcon
      }, _ctx.ptm("dropdownIcon")), null, 16, ["class"]))];
    })], 16, _hoisted_4$2)) : createCommentVNode("", true)];
  }), createBaseVNode("span", mergeProps({
    role: "status",
    "aria-live": "polite",
    "class": "p-hidden-accessible"
  }, _ctx.ptm("hiddenSearchResult"), {
    "data-p-hidden-accessible": true
  }), toDisplayString($options.searchResultMessageText), 17), createVNode(_component_Portal, {
    appendTo: _ctx.appendTo
  }, {
    "default": withCtx(function() {
      return [createVNode(Transition, mergeProps({
        name: "p-connected-overlay",
        onEnter: $options.onOverlayEnter,
        onAfterEnter: $options.onOverlayAfterEnter,
        onLeave: $options.onOverlayLeave,
        onAfterLeave: $options.onOverlayAfterLeave
      }, _ctx.ptm("transition")), {
        "default": withCtx(function() {
          return [$data.overlayVisible ? (openBlock(), createElementBlock("div", mergeProps({
            key: 0,
            ref: $options.overlayRef,
            id: $options.panelId,
            "class": [_ctx.cx("overlay"), _ctx.panelClass, _ctx.overlayClass],
            style: _objectSpread(_objectSpread(_objectSpread({}, _ctx.panelStyle), _ctx.overlayStyle), {}, {
              "max-height": $options.virtualScrollerDisabled ? _ctx.scrollHeight : ""
            }),
            onClick: _cache[9] || (_cache[9] = function() {
              return $options.onOverlayClick && $options.onOverlayClick.apply($options, arguments);
            }),
            onKeydown: _cache[10] || (_cache[10] = function() {
              return $options.onOverlayKeyDown && $options.onOverlayKeyDown.apply($options, arguments);
            })
          }, _ctx.ptm("overlay")), [renderSlot(_ctx.$slots, "header", {
            value: _ctx.modelValue,
            suggestions: $options.visibleOptions
          }), createVNode(_component_VirtualScroller, mergeProps({
            ref: $options.virtualScrollerRef
          }, _ctx.virtualScrollerOptions, {
            style: {
              height: _ctx.scrollHeight
            },
            items: $options.visibleOptions,
            tabindex: -1,
            disabled: $options.virtualScrollerDisabled,
            pt: _ctx.ptm("virtualScroller")
          }), createSlots({
            content: withCtx(function(_ref) {
              var styleClass = _ref.styleClass, contentRef = _ref.contentRef, items = _ref.items, getItemOptions = _ref.getItemOptions, contentStyle = _ref.contentStyle, itemSize = _ref.itemSize;
              return [createBaseVNode("ul", mergeProps({
                ref: /* @__PURE__ */ __name(function ref2(el) {
                  return $options.listRef(el, contentRef);
                }, "ref"),
                id: $data.id + "_list",
                "class": [_ctx.cx("list"), styleClass],
                style: contentStyle,
                role: "listbox",
                "aria-label": $options.listAriaLabel
              }, _ctx.ptm("list")), [(openBlock(true), createElementBlock(Fragment, null, renderList(items, function(option2, i) {
                return openBlock(), createElementBlock(Fragment, {
                  key: $options.getOptionRenderKey(option2, $options.getOptionIndex(i, getItemOptions))
                }, [$options.isOptionGroup(option2) ? (openBlock(), createElementBlock("li", mergeProps({
                  key: 0,
                  id: $data.id + "_" + $options.getOptionIndex(i, getItemOptions),
                  style: {
                    height: itemSize ? itemSize + "px" : void 0
                  },
                  "class": _ctx.cx("optionGroup"),
                  role: "option",
                  ref_for: true
                }, _ctx.ptm("optionGroup")), [renderSlot(_ctx.$slots, "optiongroup", {
                  option: option2.optionGroup,
                  index: $options.getOptionIndex(i, getItemOptions)
                }, function() {
                  return [createTextVNode(toDisplayString($options.getOptionGroupLabel(option2.optionGroup)), 1)];
                })], 16, _hoisted_7$1)) : withDirectives((openBlock(), createElementBlock("li", mergeProps({
                  key: 1,
                  id: $data.id + "_" + $options.getOptionIndex(i, getItemOptions),
                  style: {
                    height: itemSize ? itemSize + "px" : void 0
                  },
                  "class": _ctx.cx("option", {
                    option: option2,
                    i,
                    getItemOptions
                  }),
                  role: "option",
                  "aria-label": $options.getOptionLabel(option2),
                  "aria-selected": $options.isSelected(option2),
                  "aria-disabled": $options.isOptionDisabled(option2),
                  "aria-setsize": $options.ariaSetSize,
                  "aria-posinset": $options.getAriaPosInset($options.getOptionIndex(i, getItemOptions)),
                  onClick: /* @__PURE__ */ __name(function onClick($event) {
                    return $options.onOptionSelect($event, option2);
                  }, "onClick"),
                  onMousemove: /* @__PURE__ */ __name(function onMousemove($event) {
                    return $options.onOptionMouseMove($event, $options.getOptionIndex(i, getItemOptions));
                  }, "onMousemove"),
                  "data-p-selected": $options.isSelected(option2),
                  "data-p-focus": $data.focusedOptionIndex === $options.getOptionIndex(i, getItemOptions),
                  "data-p-disabled": $options.isOptionDisabled(option2),
                  ref_for: true
                }, $options.getPTOptions(option2, getItemOptions, i, "option")), [renderSlot(_ctx.$slots, "option", {
                  option: option2,
                  index: $options.getOptionIndex(i, getItemOptions)
                }, function() {
                  return [createTextVNode(toDisplayString($options.getOptionLabel(option2)), 1)];
                })], 16, _hoisted_8$1)), [[_directive_ripple]])], 64);
              }), 128)), !items || items && items.length === 0 ? (openBlock(), createElementBlock("li", mergeProps({
                key: 0,
                "class": _ctx.cx("emptyMessage"),
                role: "option"
              }, _ctx.ptm("emptyMessage")), [renderSlot(_ctx.$slots, "empty", {}, function() {
                return [createTextVNode(toDisplayString($options.searchResultMessageText), 1)];
              })], 16)) : createCommentVNode("", true)], 16, _hoisted_6$1)];
            }),
            _: 2
          }, [_ctx.$slots.loader ? {
            name: "loader",
            fn: withCtx(function(_ref2) {
              var options = _ref2.options;
              return [renderSlot(_ctx.$slots, "loader", {
                options
              })];
            }),
            key: "0"
          } : void 0]), 1040, ["style", "items", "disabled", "pt"]), renderSlot(_ctx.$slots, "footer", {
            value: _ctx.modelValue,
            suggestions: $options.visibleOptions
          }), createBaseVNode("span", mergeProps({
            role: "status",
            "aria-live": "polite",
            "class": "p-hidden-accessible"
          }, _ctx.ptm("hiddenSelectedMessage"), {
            "data-p-hidden-accessible": true
          }), toDisplayString($options.selectedMessageText), 17)], 16, _hoisted_5$1)) : createCommentVNode("", true)];
        }),
        _: 3
      }, 16, ["onEnter", "onAfterEnter", "onLeave", "onAfterLeave"])];
    }),
    _: 3
  }, 8, ["appendTo"])], 16);
}
__name(render, "render");
script.render = render;
const _sfc_main$6 = {
  name: "AutoCompletePlus",
  extends: script,
  emits: ["focused-option-changed"],
  mounted() {
    if (typeof script.mounted === "function") {
      script.mounted.call(this);
    }
    this.$watch(
      () => this.focusedOptionIndex,
      (newVal, oldVal) => {
        this.$emit("focused-option-changed", newVal);
      }
    );
  }
};
const _withScopeId$1 = /* @__PURE__ */ __name((n) => (pushScopeId("data-v-37f672ab"), n = n(), popScopeId(), n), "_withScopeId$1");
const _hoisted_1$1 = { class: "option-container flex justify-between items-center px-2 py-0 cursor-pointer overflow-hidden w-full" };
const _hoisted_2$1 = { class: "option-display-name font-semibold flex flex-col" };
const _hoisted_3$1 = { key: 0 };
const _hoisted_4$1 = /* @__PURE__ */ _withScopeId$1(() => /* @__PURE__ */ createBaseVNode("i", { class: "pi pi-bookmark-fill text-sm mr-1" }, null, -1));
const _hoisted_5 = [
  _hoisted_4$1
];
const _hoisted_6 = ["innerHTML"];
const _hoisted_7 = /* @__PURE__ */ _withScopeId$1(() => /* @__PURE__ */ createBaseVNode("span", null, " ", -1));
const _hoisted_8 = ["innerHTML"];
const _hoisted_9 = {
  key: 0,
  class: "option-category font-light text-sm text-gray-400 overflow-hidden text-ellipsis whitespace-nowrap"
};
const _hoisted_10 = { class: "option-badges" };
const _sfc_main$5 = /* @__PURE__ */ defineComponent({
  __name: "NodeSearchItem",
  props: {
    nodeDef: {},
    currentQuery: {}
  },
  setup(__props) {
    const settingStore = useSettingStore();
    const showCategory = computed(
      () => settingStore.get("Comfy.NodeSearchBoxImpl.ShowCategory")
    );
    const showIdName = computed(
      () => settingStore.get("Comfy.NodeSearchBoxImpl.ShowIdName")
    );
    const showNodeFrequency = computed(
      () => settingStore.get("Comfy.NodeSearchBoxImpl.ShowNodeFrequency")
    );
    const nodeFrequencyStore = useNodeFrequencyStore();
    const nodeFrequency = computed(
      () => nodeFrequencyStore.getNodeFrequency(props.nodeDef)
    );
    const nodeBookmarkStore = useNodeBookmarkStore();
    const isBookmarked = computed(
      () => nodeBookmarkStore.isBookmarked(props.nodeDef)
    );
    const props = __props;
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", _hoisted_1$1, [
        createBaseVNode("div", _hoisted_2$1, [
          createBaseVNode("div", null, [
            isBookmarked.value ? (openBlock(), createElementBlock("span", _hoisted_3$1, _hoisted_5)) : createCommentVNode("", true),
            createBaseVNode("span", {
              innerHTML: unref(highlightQuery)(_ctx.nodeDef.display_name, _ctx.currentQuery)
            }, null, 8, _hoisted_6),
            _hoisted_7,
            showIdName.value ? (openBlock(), createBlock(unref(script$e), {
              key: 1,
              severity: "secondary"
            }, {
              default: withCtx(() => [
                createBaseVNode("span", {
                  innerHTML: unref(highlightQuery)(_ctx.nodeDef.name, _ctx.currentQuery)
                }, null, 8, _hoisted_8)
              ]),
              _: 1
            })) : createCommentVNode("", true)
          ]),
          showCategory.value ? (openBlock(), createElementBlock("div", _hoisted_9, toDisplayString(_ctx.nodeDef.category.replaceAll("/", " > ")), 1)) : createCommentVNode("", true)
        ]),
        createBaseVNode("div", _hoisted_10, [
          _ctx.nodeDef.experimental ? (openBlock(), createBlock(unref(script$e), {
            key: 0,
            value: _ctx.$t("experimental"),
            severity: "primary"
          }, null, 8, ["value"])) : createCommentVNode("", true),
          _ctx.nodeDef.deprecated ? (openBlock(), createBlock(unref(script$e), {
            key: 1,
            value: _ctx.$t("deprecated"),
            severity: "danger"
          }, null, 8, ["value"])) : createCommentVNode("", true),
          showNodeFrequency.value && nodeFrequency.value > 0 ? (openBlock(), createBlock(unref(script$e), {
            key: 2,
            value: unref(formatNumberWithSuffix)(nodeFrequency.value, { roundToInt: true }),
            severity: "secondary"
          }, null, 8, ["value"])) : createCommentVNode("", true),
          _ctx.nodeDef.nodeSource.type !== unref(NodeSourceType).Unknown ? (openBlock(), createBlock(unref(script$d), {
            key: 3,
            class: "text-sm font-light"
          }, {
            default: withCtx(() => [
              createTextVNode(toDisplayString(_ctx.nodeDef.nodeSource.displayText), 1)
            ]),
            _: 1
          })) : createCommentVNode("", true)
        ])
      ]);
    };
  }
});
const NodeSearchItem = /* @__PURE__ */ _export_sfc(_sfc_main$5, [["__scopeId", "data-v-37f672ab"]]);
const _withScopeId = /* @__PURE__ */ __name((n) => (pushScopeId("data-v-2d409367"), n = n(), popScopeId(), n), "_withScopeId");
const _hoisted_1 = { class: "comfy-vue-node-search-container" };
const _hoisted_2 = {
  key: 0,
  class: "comfy-vue-node-preview-container"
};
const _hoisted_3 = /* @__PURE__ */ _withScopeId(() => /* @__PURE__ */ createBaseVNode("h3", null, "Add node filter condition", -1));
const _hoisted_4 = { class: "_dialog-body" };
const _sfc_main$4 = /* @__PURE__ */ defineComponent({
  __name: "NodeSearchBox",
  props: {
    filters: {},
    searchLimit: { default: 64 }
  },
  emits: ["addFilter", "removeFilter", "addNode"],
  setup(__props, { emit: __emit }) {
    const settingStore = useSettingStore();
    const { t } = useI18n();
    const enableNodePreview = computed(
      () => settingStore.get("Comfy.NodeSearchBoxImpl.NodePreview")
    );
    const props = __props;
    const nodeSearchFilterVisible = ref(false);
    const inputId = `comfy-vue-node-search-box-input-${Math.random()}`;
    const suggestions2 = ref([]);
    const hoveredSuggestion = ref(null);
    const currentQuery = ref("");
    const placeholder = computed(() => {
      return props.filters.length === 0 ? t("searchNodes") + "..." : "";
    });
    const nodeDefStore = useNodeDefStore();
    const nodeFrequencyStore = useNodeFrequencyStore();
    const search2 = /* @__PURE__ */ __name((query) => {
      const queryIsEmpty = query === "" && props.filters.length === 0;
      currentQuery.value = query;
      suggestions2.value = queryIsEmpty ? nodeFrequencyStore.topNodeDefs : [
        ...nodeDefStore.nodeSearchService.searchNode(query, props.filters, {
          limit: props.searchLimit
        })
      ];
    }, "search");
    const emit = __emit;
    const reFocusInput = /* @__PURE__ */ __name(() => {
      const inputElement = document.getElementById(inputId);
      if (inputElement) {
        inputElement.blur();
        inputElement.focus();
      }
    }, "reFocusInput");
    onMounted(reFocusInput);
    const onAddFilter = /* @__PURE__ */ __name((filterAndValue) => {
      nodeSearchFilterVisible.value = false;
      emit("addFilter", filterAndValue);
      reFocusInput();
    }, "onAddFilter");
    const onRemoveFilter = /* @__PURE__ */ __name((event, filterAndValue) => {
      event.stopPropagation();
      event.preventDefault();
      emit("removeFilter", filterAndValue);
      reFocusInput();
    }, "onRemoveFilter");
    const setHoverSuggestion = /* @__PURE__ */ __name((index) => {
      if (index === -1) {
        hoveredSuggestion.value = null;
        return;
      }
      const value = suggestions2.value[index];
      hoveredSuggestion.value = value;
    }, "setHoverSuggestion");
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", _hoisted_1, [
        enableNodePreview.value ? (openBlock(), createElementBlock("div", _hoisted_2, [
          hoveredSuggestion.value ? (openBlock(), createBlock(NodePreview, {
            nodeDef: hoveredSuggestion.value,
            key: hoveredSuggestion.value?.name || ""
          }, null, 8, ["nodeDef"])) : createCommentVNode("", true)
        ])) : createCommentVNode("", true),
        createVNode(unref(script$6), {
          icon: "pi pi-filter",
          severity: "secondary",
          class: "_filter-button",
          onClick: _cache[0] || (_cache[0] = ($event) => nodeSearchFilterVisible.value = true)
        }),
        createVNode(unref(script$f), {
          visible: nodeSearchFilterVisible.value,
          "onUpdate:visible": _cache[1] || (_cache[1] = ($event) => nodeSearchFilterVisible.value = $event),
          class: "_dialog"
        }, {
          header: withCtx(() => [
            _hoisted_3
          ]),
          default: withCtx(() => [
            createBaseVNode("div", _hoisted_4, [
              createVNode(NodeSearchFilter, { onAddFilter })
            ])
          ]),
          _: 1
        }, 8, ["visible"]),
        createVNode(_sfc_main$6, {
          "model-value": props.filters,
          class: "comfy-vue-node-search-box",
          scrollHeight: "40vh",
          placeholder: placeholder.value,
          "input-id": inputId,
          "append-to": "self",
          suggestions: suggestions2.value,
          "min-length": 0,
          delay: 100,
          loading: !unref(nodeFrequencyStore).isLoaded,
          onComplete: _cache[2] || (_cache[2] = ($event) => search2($event.query)),
          onOptionSelect: _cache[3] || (_cache[3] = ($event) => emit("addNode", $event.value)),
          onFocusedOptionChanged: _cache[4] || (_cache[4] = ($event) => setHoverSuggestion($event)),
          "complete-on-focus": "",
          "auto-option-focus": "",
          "force-selection": "",
          multiple: "",
          optionLabel: "display_name"
        }, {
          option: withCtx(({ option: option2 }) => [
            createVNode(NodeSearchItem, {
              nodeDef: option2,
              currentQuery: currentQuery.value
            }, null, 8, ["nodeDef", "currentQuery"])
          ]),
          chip: withCtx(({ value }) => [
            createVNode(SearchFilterChip, {
              onRemove: /* @__PURE__ */ __name(($event) => onRemoveFilter($event, value), "onRemove"),
              text: value[1],
              badge: value[0].invokeSequence.toUpperCase(),
              "badge-class": value[0].invokeSequence + "-badge"
            }, null, 8, ["onRemove", "text", "badge", "badge-class"])
          ]),
          _: 1
        }, 8, ["model-value", "placeholder", "suggestions", "loading"])
      ]);
    };
  }
});
const NodeSearchBox = /* @__PURE__ */ _export_sfc(_sfc_main$4, [["__scopeId", "data-v-2d409367"]]);
class ConnectingLinkImpl {
  static {
    __name(this, "ConnectingLinkImpl");
  }
  node;
  slot;
  input;
  output;
  pos;
  constructor(node, slot, input, output, pos) {
    this.node = node;
    this.slot = slot;
    this.input = input;
    this.output = output;
    this.pos = pos;
  }
  static createFromPlainObject(obj) {
    return new ConnectingLinkImpl(
      obj.node,
      obj.slot,
      obj.input,
      obj.output,
      obj.pos
    );
  }
  get type() {
    const result = this.input ? this.input.type : this.output.type;
    return result === -1 ? null : result;
  }
  /**
   * Which slot type is release and need to be reconnected.
   * - 'output' means we need a new node's outputs slot to connect with this link
   */
  get releaseSlotType() {
    return this.output ? "input" : "output";
  }
  connectTo(newNode) {
    const newNodeSlots = this.releaseSlotType === "output" ? newNode.outputs : newNode.inputs;
    if (!newNodeSlots) return;
    const newNodeSlot = newNodeSlots.findIndex(
      (slot) => LiteGraph.isValidConnection(slot.type, this.type)
    );
    if (newNodeSlot === -1) {
      console.warn(
        `Could not find slot with type ${this.type} on node ${newNode.title}. This should never happen`
      );
      return;
    }
    if (this.releaseSlotType === "input") {
      this.node.connect(this.slot, newNode, newNodeSlot);
    } else {
      newNode.connect(newNodeSlot, this.node, this.slot);
    }
  }
}
const _sfc_main$3 = /* @__PURE__ */ defineComponent({
  __name: "NodeSearchBoxPopover",
  setup(__props) {
    const settingStore = useSettingStore();
    const visible = ref(false);
    const dismissable = ref(true);
    const triggerEvent = ref(null);
    const getNewNodeLocation = /* @__PURE__ */ __name(() => {
      if (triggerEvent.value === null) {
        return [100, 100];
      }
      const originalEvent = triggerEvent.value.detail.originalEvent;
      return [originalEvent.canvasX, originalEvent.canvasY];
    }, "getNewNodeLocation");
    const nodeFilters = ref([]);
    const addFilter = /* @__PURE__ */ __name((filter) => {
      nodeFilters.value.push(filter);
    }, "addFilter");
    const removeFilter = /* @__PURE__ */ __name((filter) => {
      nodeFilters.value = nodeFilters.value.filter(
        (f) => toRaw(f) !== toRaw(filter)
      );
    }, "removeFilter");
    const clearFilters = /* @__PURE__ */ __name(() => {
      nodeFilters.value = [];
    }, "clearFilters");
    const closeDialog = /* @__PURE__ */ __name(() => {
      visible.value = false;
    }, "closeDialog");
    const addNode = /* @__PURE__ */ __name((nodeDef) => {
      const node = app.addNodeOnGraph(nodeDef, { pos: getNewNodeLocation() });
      const eventDetail = triggerEvent.value.detail;
      if (eventDetail.subType === "empty-release") {
        eventDetail.linkReleaseContext.links.forEach((link) => {
          ConnectingLinkImpl.createFromPlainObject(link).connectTo(node);
        });
      }
      window.setTimeout(() => {
        closeDialog();
      }, 100);
    }, "addNode");
    const newSearchBoxEnabled = computed(
      () => settingStore.get("Comfy.NodeSearchBoxImpl") === "default"
    );
    const showSearchBox = /* @__PURE__ */ __name((e) => {
      if (newSearchBoxEnabled.value) {
        if (e.detail.originalEvent?.pointerType === "touch") {
          setTimeout(() => {
            showNewSearchBox(e);
          }, 128);
        } else {
          showNewSearchBox(e);
        }
      } else {
        canvasStore.canvas.showSearchBox(e.detail.originalEvent);
      }
    }, "showSearchBox");
    const nodeDefStore = useNodeDefStore();
    const showNewSearchBox = /* @__PURE__ */ __name((e) => {
      if (e.detail.linkReleaseContext) {
        const links = e.detail.linkReleaseContext.links;
        if (links.length === 0) {
          console.warn("Empty release with no links! This should never happen");
          return;
        }
        const firstLink = ConnectingLinkImpl.createFromPlainObject(links[0]);
        const filter = nodeDefStore.nodeSearchService.getFilterById(
          firstLink.releaseSlotType
        );
        const dataType = firstLink.type;
        addFilter([filter, dataType]);
      }
      visible.value = true;
      triggerEvent.value = e;
      dismissable.value = false;
      setTimeout(() => {
        dismissable.value = true;
      }, 300);
    }, "showNewSearchBox");
    const showContextMenu = /* @__PURE__ */ __name((e) => {
      const links = e.detail.linkReleaseContext.links;
      if (links.length === 0) {
        console.warn("Empty release with no links! This should never happen");
        return;
      }
      const firstLink = ConnectingLinkImpl.createFromPlainObject(links[0]);
      const mouseEvent = e.detail.originalEvent;
      const commonOptions = {
        e: mouseEvent,
        allow_searchbox: true,
        showSearchBox: /* @__PURE__ */ __name(() => showSearchBox(e), "showSearchBox")
      };
      const connectionOptions = firstLink.output ? { nodeFrom: firstLink.node, slotFrom: firstLink.output } : { nodeTo: firstLink.node, slotTo: firstLink.input };
      canvasStore.canvas.showConnectionMenu({
        ...connectionOptions,
        ...commonOptions
      });
    }, "showContextMenu");
    const canvasStore = useCanvasStore();
    watchEffect(() => {
      if (canvasStore.canvas) {
        LiteGraph.release_link_on_empty_shows_menu = false;
        canvasStore.canvas.allow_searchbox = false;
      }
    });
    const canvasEventHandler = /* @__PURE__ */ __name((e) => {
      if (e.detail.subType === "empty-double-click") {
        showSearchBox(e);
      } else if (e.detail.subType === "empty-release") {
        handleCanvasEmptyRelease(e);
      } else if (e.detail.subType === "group-double-click") {
        const group = e.detail.group;
        const [x, y] = group.pos;
        const relativeY = e.detail.originalEvent.canvasY - y;
        if (relativeY > group.titleHeight) {
          showSearchBox(e);
        }
      }
    }, "canvasEventHandler");
    const linkReleaseAction = computed(() => {
      return settingStore.get("Comfy.LinkRelease.Action");
    });
    const linkReleaseActionShift = computed(() => {
      return settingStore.get("Comfy.LinkRelease.ActionShift");
    });
    const handleCanvasEmptyRelease = /* @__PURE__ */ __name((e) => {
      const originalEvent = e.detail.originalEvent;
      const shiftPressed = originalEvent.shiftKey;
      const action = shiftPressed ? linkReleaseActionShift.value : linkReleaseAction.value;
      switch (action) {
        case LinkReleaseTriggerAction.SEARCH_BOX:
          showSearchBox(e);
          break;
        case LinkReleaseTriggerAction.CONTEXT_MENU:
          showContextMenu(e);
          break;
        case LinkReleaseTriggerAction.NO_ACTION:
        default:
          break;
      }
    }, "handleCanvasEmptyRelease");
    onMounted(() => {
      document.addEventListener("litegraph:canvas", canvasEventHandler);
    });
    onUnmounted(() => {
      document.removeEventListener("litegraph:canvas", canvasEventHandler);
    });
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", null, [
        createVNode(unref(script$f), {
          visible: visible.value,
          "onUpdate:visible": _cache[0] || (_cache[0] = ($event) => visible.value = $event),
          modal: "",
          "dismissable-mask": dismissable.value,
          onHide: clearFilters,
          pt: {
            root: {
              class: "invisible-dialog-root",
              role: "search"
            },
            mask: { class: "node-search-box-dialog-mask" },
            transition: {
              enterFromClass: "opacity-0 scale-75",
              // 100ms is the duration of the transition in the dialog component
              enterActiveClass: "transition-all duration-100 ease-out",
              leaveActiveClass: "transition-all duration-100 ease-in",
              leaveToClass: "opacity-0 scale-75"
            }
          }
        }, {
          container: withCtx(() => [
            createVNode(NodeSearchBox, {
              filters: nodeFilters.value,
              onAddFilter: addFilter,
              onRemoveFilter: removeFilter,
              onAddNode: addNode
            }, null, 8, ["filters"])
          ]),
          _: 1
        }, 8, ["visible", "dismissable-mask"])
      ]);
    };
  }
});
const _sfc_main$2 = /* @__PURE__ */ defineComponent({
  __name: "NodeTooltip",
  setup(__props) {
    let idleTimeout;
    const nodeDefStore = useNodeDefStore();
    const settingStore = useSettingStore();
    const tooltipRef = ref();
    const tooltipText = ref("");
    const left = ref();
    const top = ref();
    const getHoveredWidget = /* @__PURE__ */ __name(() => {
      const node = app.canvas.node_over;
      if (!node.widgets) return;
      const graphPos = app.canvas.graph_mouse;
      const x = graphPos[0] - node.pos[0];
      const y = graphPos[1] - node.pos[1];
      for (const w of node.widgets) {
        let widgetWidth, widgetHeight;
        if (w.computeSize) {
          ;
          [widgetWidth, widgetHeight] = w.computeSize(node.size[0]);
        } else {
          widgetWidth = w.width || node.size[0];
          widgetHeight = LiteGraph.NODE_WIDGET_HEIGHT;
        }
        if (w.last_y !== void 0 && x >= 6 && x <= widgetWidth - 12 && y >= w.last_y && y <= w.last_y + widgetHeight) {
          return w;
        }
      }
    }, "getHoveredWidget");
    const hideTooltip = /* @__PURE__ */ __name(() => tooltipText.value = null, "hideTooltip");
    const showTooltip = /* @__PURE__ */ __name(async (tooltip) => {
      if (!tooltip) return;
      left.value = app.canvas.mouse[0] + "px";
      top.value = app.canvas.mouse[1] + "px";
      tooltipText.value = tooltip;
      await nextTick();
      const rect = tooltipRef.value.getBoundingClientRect();
      if (rect.right > window.innerWidth) {
        left.value = app.canvas.mouse[0] - rect.width + "px";
      }
      if (rect.top < 0) {
        top.value = app.canvas.mouse[1] + rect.height + "px";
      }
    }, "showTooltip");
    const onIdle = /* @__PURE__ */ __name(() => {
      const { canvas } = app;
      const node = canvas.node_over;
      if (!node) return;
      const ctor = node.constructor;
      const nodeDef = nodeDefStore.nodeDefsByName[node.type];
      if (ctor.title_mode !== LiteGraph.NO_TITLE && canvas.graph_mouse[1] < node.pos[1]) {
        return showTooltip(nodeDef.description);
      }
      if (node.flags?.collapsed) return;
      const inputSlot = canvas.isOverNodeInput(
        node,
        canvas.graph_mouse[0],
        canvas.graph_mouse[1],
        [0, 0]
      );
      if (inputSlot !== -1) {
        const inputName = node.inputs[inputSlot].name;
        return showTooltip(nodeDef.input.getInput(inputName)?.tooltip);
      }
      const outputSlot = canvas.isOverNodeOutput(
        node,
        canvas.graph_mouse[0],
        canvas.graph_mouse[1],
        [0, 0]
      );
      if (outputSlot !== -1) {
        return showTooltip(nodeDef.output.all?.[outputSlot].tooltip);
      }
      const widget = getHoveredWidget();
      if (widget && !widget.element) {
        return showTooltip(
          widget.tooltip ?? nodeDef.input.getInput(widget.name)?.tooltip
        );
      }
    }, "onIdle");
    const onMouseMove = /* @__PURE__ */ __name((e) => {
      hideTooltip();
      clearTimeout(idleTimeout);
      if (e.target.nodeName !== "CANVAS") return;
      idleTimeout = window.setTimeout(onIdle, 500);
    }, "onMouseMove");
    watch(
      () => settingStore.get("Comfy.EnableTooltips"),
      (enabled) => {
        if (enabled) {
          window.addEventListener("mousemove", onMouseMove);
          window.addEventListener("click", hideTooltip);
        } else {
          window.removeEventListener("mousemove", onMouseMove);
          window.removeEventListener("click", hideTooltip);
        }
      },
      { immediate: true }
    );
    onBeforeUnmount(() => {
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("click", hideTooltip);
    });
    return (_ctx, _cache) => {
      return tooltipText.value ? (openBlock(), createElementBlock("div", {
        key: 0,
        ref_key: "tooltipRef",
        ref: tooltipRef,
        class: "node-tooltip",
        style: normalizeStyle({ left: left.value, top: top.value })
      }, toDisplayString(tooltipText.value), 5)) : createCommentVNode("", true);
    };
  }
});
const NodeTooltip = /* @__PURE__ */ _export_sfc(_sfc_main$2, [["__scopeId", "data-v-e0597bf9"]]);
const _sfc_main$1 = /* @__PURE__ */ defineComponent({
  __name: "GraphCanvas",
  emits: ["ready"],
  setup(__props, { emit: __emit }) {
    const emit = __emit;
    const canvasRef = ref(null);
    const settingStore = useSettingStore();
    const nodeDefStore = useNodeDefStore();
    const workspaceStore = useWorkspaceStore();
    const canvasStore = useCanvasStore();
    const betaMenuEnabled = computed(
      () => settingStore.get("Comfy.UseNewMenu") !== "Disabled"
    );
    watchEffect(() => {
      const canvasInfoEnabled = settingStore.get("Comfy.Graph.CanvasInfo");
      if (canvasStore.canvas) {
        canvasStore.canvas.show_info = canvasInfoEnabled;
      }
    });
    watchEffect(() => {
      const zoomSpeed = settingStore.get("Comfy.Graph.ZoomSpeed");
      if (canvasStore.canvas) {
        canvasStore.canvas.zoom_speed = zoomSpeed;
      }
    });
    watchEffect(() => {
      nodeDefStore.showDeprecated = settingStore.get("Comfy.Node.ShowDeprecated");
    });
    watchEffect(() => {
      nodeDefStore.showExperimental = settingStore.get(
        "Comfy.Node.ShowExperimental"
      );
    });
    watchEffect(() => {
      const spellcheckEnabled = settingStore.get("Comfy.TextareaWidget.Spellcheck");
      const textareas = document.querySelectorAll("textarea.comfy-multiline-input");
      textareas.forEach((textarea) => {
        textarea.spellcheck = spellcheckEnabled;
        textarea.focus();
        textarea.blur();
      });
    });
    let dropTargetCleanup = /* @__PURE__ */ __name(() => {
    }, "dropTargetCleanup");
    onMounted(async () => {
      window["LiteGraph"] = LiteGraph;
      window["LGraph"] = LGraph;
      window["LLink"] = LLink;
      window["LGraphNode"] = LGraphNode;
      window["LGraphGroup"] = LGraphGroup;
      window["DragAndScale"] = DragAndScale;
      window["LGraphCanvas"] = LGraphCanvas;
      window["ContextMenu"] = ContextMenu;
      window["LGraphBadge"] = LGraphBadge$1;
      app.vueAppReady = true;
      workspaceStore.spinner = true;
      await app.setup(canvasRef.value);
      canvasStore.canvas = app.canvas;
      workspaceStore.spinner = false;
      window["app"] = app;
      window["graph"] = app.graph;
      dropTargetCleanup = dropTargetForElements({
        element: canvasRef.value,
        onDrop: /* @__PURE__ */ __name((event) => {
          const loc = event.location.current.input;
          const dndData = event.source.data;
          if (dndData.type === "tree-explorer-node") {
            const node = dndData.data;
            if (node.data instanceof ComfyNodeDefImpl) {
              const nodeDef = node.data;
              const pos = app.clientPosToCanvasPos([
                loc.clientX - 20,
                loc.clientY
              ]);
              app.addNodeOnGraph(nodeDef, { pos });
            }
          }
        }, "onDrop")
      });
      useNodeBookmarkStore().migrateLegacyBookmarks();
      useNodeDefStore().nodeSearchService.endsWithFilterStartSequence("");
      useNodeFrequencyStore().loadNodeFrequencies();
      emit("ready");
    });
    onUnmounted(() => {
      dropTargetCleanup();
    });
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock(Fragment, null, [
        (openBlock(), createBlock(Teleport, { to: ".graph-canvas-container" }, [
          betaMenuEnabled.value ? (openBlock(), createBlock(LiteGraphCanvasSplitterOverlay, { key: 0 }, {
            "side-bar-panel": withCtx(() => [
              createVNode(SideToolbar)
            ]),
            _: 1
          })) : createCommentVNode("", true),
          createVNode(TitleEditor),
          createBaseVNode("canvas", {
            ref_key: "canvasRef",
            ref: canvasRef,
            id: "graph-canvas",
            tabindex: "1"
          }, null, 512)
        ])),
        createVNode(_sfc_main$3),
        createVNode(NodeTooltip)
      ], 64);
    };
  }
});
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "GraphView",
  setup(__props) {
    return (_ctx, _cache) => {
      return openBlock(), createBlock(_sfc_main$1);
    };
  }
});
export {
  _sfc_main as default
};
//# sourceMappingURL=GraphView-DN9xGvF3.js.map
