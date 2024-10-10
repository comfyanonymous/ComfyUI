var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });
import { d as defineComponent, r as ref, w as watch, n as nextTick, o as openBlock, c as createElementBlock, t as toDisplayString, a as withDirectives, b as createBlock, e as withKeys, f as withModifiers, u as unref, s as script$m, p as pushScopeId, g as popScopeId, _ as _export_sfc, h as useSettingStore, i as useTitleEditorStore, j as useCanvasStore, L as LGraphGroup, k as app, l as LGraphNode, m as onMounted, q as onUnmounted, v as createVNode, x as normalizeStyle, y as createCommentVNode, z as LiteGraph, B as BaseStyle, A as script$n, C as resolveComponent, D as mergeProps, E as renderSlot, F as computed, G as resolveDirective, H as withCtx, I as createBaseVNode, J as normalizeClass, K as script$o, M as useDialogStore, S as SettingDialogHeader, N as SettingDialogContent, O as useWorkspaceStore, P as onBeforeUnmount, Q as Fragment, R as renderList, T as Teleport, U as resolveDynamicComponent, V as script$p, W as getWidth, X as getHeight, Y as getOuterWidth, Z as getOuterHeight, $ as getVNodeProp, a0 as isArray, a1 as vShow, a2 as isNotEmpty, a3 as UniqueComponentId, a4 as ZIndex, a5 as resolveFieldData, a6 as focus, a7 as OverlayEventBus, a8 as isEmpty, a9 as addStyle, aa as relativePosition, ab as absolutePosition, ac as ConnectedOverlayScrollHandler, ad as isTouchDevice, ae as equals, af as findLastIndex, ag as findSingle, ah as script$q, ai as script$r, aj as script$s, ak as script$t, al as script$u, am as Ripple, an as Transition, ao as createSlots, ap as createTextVNode, aq as useNodeDefStore, ar as script$v, as as defineStore, at as createDummyFolderNodeDef, au as _, av as buildNodeDefTree, aw as useNodeFrequencyStore, ax as highlightQuery, ay as script$w, az as formatNumberWithSuffix, aA as NodeSourceType, aB as ComfyNodeDefImpl, aC as useI18n, aD as script$x, aE as SearchFilterChip, aF as watchEffect, aG as toRaw, aH as LinkReleaseTriggerAction, aI as commonjsGlobal, aJ as getDefaultExportFromCjs, aK as markRaw, aL as useCommandStore, aM as LGraph, aN as LLink, aO as DragAndScale, aP as LGraphCanvas, aQ as ContextMenu, aR as LGraphBadge, aS as ComfyModelDef, aT as useKeybindingStore, aU as ConfirmationEventBus, aV as getOffset, aW as $dt, aX as addClass, aY as FocusTrap, aZ as resolve, a_ as nestedPosition, a$ as script$y, b0 as isPrintableCharacter, b1 as getHiddenElementOuterWidth, b2 as getHiddenElementOuterHeight, b3 as getViewport, b4 as api, b5 as TaskItemDisplayStatus, b6 as script$z, b7 as find, b8 as getAttribute, b9 as script$A, ba as script$B, bb as removeClass, bc as setAttribute, bd as localeComparator, be as sort, bf as script$C, bg as unblockBodyScroll, bh as blockBodyScroll, bi as useConfirm, bj as useToast, bk as useQueueStore, bl as useInfiniteScroll, bm as useResizeObserver, bn as script$D, bo as NoResultsPlaceholder, bp as isClient, bq as script$E, br as script$F, bs as script$G, bt as script$H, bu as script$I, bv as script$J, bw as inject, bx as useErrorHandling, by as mergeModels, bz as useModel, bA as provide, bB as script$K, bC as findNodeByKey, bD as sortedTree, bE as SearchBox, bF as useModelStore, bG as buildTree, bH as toRef, bI as script$L, bJ as script$M, bK as script$N, bL as normalizeProps, bM as ToastEventBus, bN as TransitionGroup, bO as useToastStore, bP as useExecutionStore, bQ as useWorkflowStore, bR as useTitle, bS as useWorkflowBookmarkStore, bT as script$O, bU as script$P, bV as guardReactiveProps, bW as useMenuItemStore, bX as script$Q, bY as useQueueSettingsStore, bZ as storeToRefs, b_ as isRef, b$ as script$R, c0 as useQueuePendingTaskCountStore, c1 as useLocalStorage, c2 as useDraggable, c3 as watchDebounced, c4 as useEventListener, c5 as useElementBounding, c6 as lodashExports, c7 as useEventBus, c8 as i18n } from "./index-DGAbdBYF.js";
import { g as getColorPalette, d as defaultColorPalette } from "./colorPalette-D5oi2-2V.js";
const _withScopeId$l = /* @__PURE__ */ __name((n) => (pushScopeId("data-v-54da6fc9"), n = n(), popScopeId(), n), "_withScopeId$l");
const _hoisted_1$F = { class: "editable-text" };
const _hoisted_2$v = { key: 0 };
const _sfc_main$I = /* @__PURE__ */ defineComponent({
  __name: "EditableText",
  props: {
    modelValue: {},
    isEditing: { type: Boolean, default: false }
  },
  emits: ["update:modelValue", "edit"],
  setup(__props, { emit: __emit }) {
    const props = __props;
    const emit = __emit;
    const inputValue2 = ref(props.modelValue);
    const isEditingFinished = ref(false);
    const inputRef = ref(null);
    const finishEditing = /* @__PURE__ */ __name(() => {
      if (isEditingFinished.value) {
        return;
      }
      isEditingFinished.value = true;
      emit("edit", inputValue2.value);
    }, "finishEditing");
    watch(
      () => props.isEditing,
      (newVal) => {
        if (newVal) {
          inputValue2.value = props.modelValue;
          isEditingFinished.value = false;
          nextTick(() => {
            if (!inputRef.value) return;
            const fileName = inputValue2.value.includes(".") ? inputValue2.value.split(".").slice(0, -1).join(".") : inputValue2.value;
            const start2 = 0;
            const end = fileName.length;
            const inputElement = inputRef.value.$el;
            inputElement.setSelectionRange?.(start2, end);
          });
        }
      },
      { immediate: true }
    );
    const vFocus = {
      mounted: /* @__PURE__ */ __name((el) => el.focus(), "mounted")
    };
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", _hoisted_1$F, [
        !props.isEditing ? (openBlock(), createElementBlock("span", _hoisted_2$v, toDisplayString(_ctx.modelValue), 1)) : withDirectives((openBlock(), createBlock(unref(script$m), {
          key: 1,
          type: "text",
          size: "small",
          fluid: "",
          modelValue: inputValue2.value,
          "onUpdate:modelValue": _cache[0] || (_cache[0] = ($event) => inputValue2.value = $event),
          ref_key: "inputRef",
          ref: inputRef,
          onKeyup: withKeys(finishEditing, ["enter"]),
          onClick: _cache[1] || (_cache[1] = withModifiers(() => {
          }, ["stop"])),
          pt: {
            root: {
              onBlur: finishEditing
            }
          }
        }, null, 8, ["modelValue", "pt"])), [
          [vFocus]
        ])
      ]);
    };
  }
});
const EditableText = /* @__PURE__ */ _export_sfc(_sfc_main$I, [["__scopeId", "data-v-54da6fc9"]]);
const _sfc_main$H = /* @__PURE__ */ defineComponent({
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
          const node2 = target;
          const [x, y] = node2.getBounding();
          const canvasWidth = node2.width;
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
      nodeCreated(node2) {
        const originalCallback = node2.onNodeTitleDblClick;
        node2.onNodeTitleDblClick = function(e, ...args) {
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
const TitleEditor = /* @__PURE__ */ _export_sfc(_sfc_main$H, [["__scopeId", "data-v-fc3f26e3"]]);
var theme$h = /* @__PURE__ */ __name(function theme(_ref) {
  var dt = _ref.dt;
  return "\n.p-overlaybadge {\n    position: relative;\n}\n\n.p-overlaybadge .p-badge {\n    position: absolute;\n    top: 0;\n    right: 0;\n    transform: translate(50%, -50%);\n    transform-origin: 100% 0;\n    margin: 0;\n    outline-width: ".concat(dt("overlaybadge.outline.width"), ";\n    outline-style: solid;\n    outline-color: ").concat(dt("overlaybadge.outline.color"), ";\n}\n");
}, "theme");
var classes$i = {
  root: "p-overlaybadge"
};
var OverlayBadgeStyle = BaseStyle.extend({
  name: "overlaybadge",
  theme: theme$h,
  classes: classes$i
});
var script$1$i = {
  name: "OverlayBadge",
  "extends": script$n,
  style: OverlayBadgeStyle,
  provide: /* @__PURE__ */ __name(function provide2() {
    return {
      $pcOverlayBadge: this,
      $parentInstance: this
    };
  }, "provide")
};
var script$l = {
  name: "OverlayBadge",
  "extends": script$1$i,
  inheritAttrs: false,
  components: {
    Badge: script$n
  }
};
function render$m(_ctx, _cache, $props, $setup, $data, $options) {
  var _component_Badge = resolveComponent("Badge");
  return openBlock(), createElementBlock("div", mergeProps({
    "class": _ctx.cx("root")
  }, _ctx.ptmi("root")), [renderSlot(_ctx.$slots, "default"), createVNode(_component_Badge, mergeProps(_ctx.$props, {
    pt: _ctx.ptm("pcBadge")
  }), null, 16, ["pt"])], 16);
}
__name(render$m, "render$m");
script$l.render = render$m;
const _sfc_main$G = /* @__PURE__ */ defineComponent({
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
      return withDirectives((openBlock(), createBlock(unref(script$o), {
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
          shouldShowBadge.value ? (openBlock(), createBlock(unref(script$l), {
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
const SidebarIcon = /* @__PURE__ */ _export_sfc(_sfc_main$G, [["__scopeId", "data-v-caa3ee9c"]]);
const _sfc_main$F = /* @__PURE__ */ defineComponent({
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
const _sfc_main$E = /* @__PURE__ */ defineComponent({
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
        class: "comfy-settings-btn",
        onClick: showSetting,
        tooltip: _ctx.$t("settings")
      }, null, 8, ["tooltip"]);
    };
  }
});
const _withScopeId$k = /* @__PURE__ */ __name((n) => (pushScopeId("data-v-4da64512"), n = n(), popScopeId(), n), "_withScopeId$k");
const _hoisted_1$E = { class: "side-tool-bar-end" };
const _hoisted_2$u = {
  key: 0,
  class: "sidebar-content-container"
};
const _sfc_main$D = /* @__PURE__ */ defineComponent({
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
    const selectedTab = computed(() => workspaceStore.sidebarTab.activeSidebarTab);
    const mountCustomTab = /* @__PURE__ */ __name((tab, el) => {
      tab.render(el);
    }, "mountCustomTab");
    const onTabClick = /* @__PURE__ */ __name((item4) => {
      workspaceStore.sidebarTab.toggleSidebarTab(item4.id);
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
                selected: tab.id === selectedTab.value?.id,
                class: normalizeClass(tab.id + "-tab-button"),
                onClick: /* @__PURE__ */ __name(($event) => onTabClick(tab), "onClick")
              }, null, 8, ["icon", "iconBadge", "tooltip", "selected", "class", "onClick"]);
            }), 128)),
            createBaseVNode("div", _hoisted_1$E, [
              createVNode(_sfc_main$F),
              createVNode(_sfc_main$E)
            ])
          ], 2)
        ], 8, ["to"])),
        selectedTab.value ? (openBlock(), createElementBlock("div", _hoisted_2$u, [
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
const SideToolbar = /* @__PURE__ */ _export_sfc(_sfc_main$D, [["__scopeId", "data-v-4da64512"]]);
var theme$g = /* @__PURE__ */ __name(function theme2(_ref) {
  var dt = _ref.dt;
  return "\n.p-splitter {\n    display: flex;\n    flex-wrap: nowrap;\n    border: 1px solid ".concat(dt("splitter.border.color"), ";\n    background: ").concat(dt("splitter.background"), ";\n    border-radius: ").concat(dt("border.radius.md"), ";\n    color: ").concat(dt("splitter.color"), ";\n}\n\n.p-splitter-vertical {\n    flex-direction: column;\n}\n\n.p-splitter-gutter {\n    flex-grow: 0;\n    flex-shrink: 0;\n    display: flex;\n    align-items: center;\n    justify-content: center;\n    z-index: 1;\n    background: ").concat(dt("splitter.gutter.background"), ";\n}\n\n.p-splitter-gutter-handle {\n    border-radius: ").concat(dt("splitter.handle.border.radius"), ";\n    background: ").concat(dt("splitter.handle.background"), ";\n    transition: outline-color ").concat(dt("splitter.transition.duration"), ", box-shadow ").concat(dt("splitter.transition.duration"), ";\n    outline-color: transparent;\n}\n\n.p-splitter-gutter-handle:focus-visible {\n    box-shadow: ").concat(dt("splitter.handle.focus.ring.shadow"), ";\n    outline: ").concat(dt("splitter.handle.focus.ring.width"), " ").concat(dt("splitter.handle.focus.ring.style"), " ").concat(dt("splitter.handle.focus.ring.color"), ";\n    outline-offset: ").concat(dt("splitter.handle.focus.ring.offset"), ";\n}\n\n.p-splitter-horizontal.p-splitter-resizing {\n    cursor: col-resize;\n    user-select: none;\n}\n\n.p-splitter-vertical.p-splitter-resizing {\n    cursor: row-resize;\n    user-select: none;\n}\n\n.p-splitter-horizontal > .p-splitter-gutter > .p-splitter-gutter-handle {\n    height: ").concat(dt("splitter.handle.size"), ";\n    width: 100%;\n}\n\n.p-splitter-vertical > .p-splitter-gutter > .p-splitter-gutter-handle {\n    width: ").concat(dt("splitter.handle.size"), ";\n    height: 100%;\n}\n\n.p-splitter-horizontal > .p-splitter-gutter {\n    cursor: col-resize;\n}\n\n.p-splitter-vertical > .p-splitter-gutter {\n    cursor: row-resize;\n}\n\n.p-splitterpanel {\n    flex-grow: 1;\n    overflow: hidden;\n}\n\n.p-splitterpanel-nested {\n    display: flex;\n}\n\n.p-splitterpanel .p-splitter {\n    flex-grow: 1;\n    border: 0 none;\n}\n");
}, "theme");
var classes$h = {
  root: /* @__PURE__ */ __name(function root(_ref2) {
    var props = _ref2.props;
    return ["p-splitter p-component", "p-splitter-" + props.layout];
  }, "root"),
  gutter: "p-splitter-gutter",
  gutterHandle: "p-splitter-gutter-handle"
};
var inlineStyles$4 = {
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
  theme: theme$g,
  classes: classes$h,
  inlineStyles: inlineStyles$4
});
var script$1$h = {
  name: "BaseSplitter",
  "extends": script$p,
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
  provide: /* @__PURE__ */ __name(function provide3() {
    return {
      $pcSplitter: this,
      $parentInstance: this
    };
  }, "provide")
};
function _toConsumableArray$7(r) {
  return _arrayWithoutHoles$7(r) || _iterableToArray$7(r) || _unsupportedIterableToArray$9(r) || _nonIterableSpread$7();
}
__name(_toConsumableArray$7, "_toConsumableArray$7");
function _nonIterableSpread$7() {
  throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.");
}
__name(_nonIterableSpread$7, "_nonIterableSpread$7");
function _unsupportedIterableToArray$9(r, a) {
  if (r) {
    if ("string" == typeof r) return _arrayLikeToArray$9(r, a);
    var t = {}.toString.call(r).slice(8, -1);
    return "Object" === t && r.constructor && (t = r.constructor.name), "Map" === t || "Set" === t ? Array.from(r) : "Arguments" === t || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(t) ? _arrayLikeToArray$9(r, a) : void 0;
  }
}
__name(_unsupportedIterableToArray$9, "_unsupportedIterableToArray$9");
function _iterableToArray$7(r) {
  if ("undefined" != typeof Symbol && null != r[Symbol.iterator] || null != r["@@iterator"]) return Array.from(r);
}
__name(_iterableToArray$7, "_iterableToArray$7");
function _arrayWithoutHoles$7(r) {
  if (Array.isArray(r)) return _arrayLikeToArray$9(r);
}
__name(_arrayWithoutHoles$7, "_arrayWithoutHoles$7");
function _arrayLikeToArray$9(r, a) {
  (null == a || a > r.length) && (a = r.length);
  for (var e = 0, n = Array(a); e < a; e++) n[e] = r[e];
  return n;
}
__name(_arrayLikeToArray$9, "_arrayLikeToArray$9");
var script$k = {
  name: "Splitter",
  "extends": script$1$h,
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
        var children = _toConsumableArray$7(this.$el.children).filter(function(child) {
          return child.getAttribute("data-pc-name") === "splitterpanel";
        });
        var _panelSizes = [];
        this.panels.map(function(panel2, i) {
          var panelInitialSize = panel2.props && panel2.props.size ? panel2.props.size : null;
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
    onResizeStart: /* @__PURE__ */ __name(function onResizeStart(event, index2, isKeyDown) {
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
      this.prevPanelIndex = index2;
      this.$emit("resizestart", {
        originalEvent: event,
        sizes: this.panelSizes
      });
      this.$refs.gutter[index2].setAttribute("data-p-gutter-resizing", true);
      this.$el.setAttribute("data-p-resizing", true);
    }, "onResizeStart"),
    onResize: /* @__PURE__ */ __name(function onResize(event, step2, isKeyDown) {
      var newPos, newPrevPanelSize, newNextPanelSize;
      if (isKeyDown) {
        if (this.horizontal) {
          newPrevPanelSize = 100 * (this.prevPanelSize + step2) / this.size;
          newNextPanelSize = 100 * (this.nextPanelSize - step2) / this.size;
        } else {
          newPrevPanelSize = 100 * (this.prevPanelSize - step2) / this.size;
          newNextPanelSize = 100 * (this.nextPanelSize + step2) / this.size;
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
    repeat: /* @__PURE__ */ __name(function repeat(event, index2, step2) {
      this.onResizeStart(event, index2, true);
      this.onResize(event, step2, true);
    }, "repeat"),
    setTimer: /* @__PURE__ */ __name(function setTimer(event, index2, step2) {
      var _this2 = this;
      if (!this.timer) {
        this.timer = setInterval(function() {
          _this2.repeat(event, index2, step2);
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
    onGutterKeyDown: /* @__PURE__ */ __name(function onGutterKeyDown(event, index2) {
      switch (event.code) {
        case "ArrowLeft": {
          if (this.layout === "horizontal") {
            this.setTimer(event, index2, this.step * -1);
          }
          event.preventDefault();
          break;
        }
        case "ArrowRight": {
          if (this.layout === "horizontal") {
            this.setTimer(event, index2, this.step);
          }
          event.preventDefault();
          break;
        }
        case "ArrowDown": {
          if (this.layout === "vertical") {
            this.setTimer(event, index2, this.step * -1);
          }
          event.preventDefault();
          break;
        }
        case "ArrowUp": {
          if (this.layout === "vertical") {
            this.setTimer(event, index2, this.step);
          }
          event.preventDefault();
          break;
        }
      }
    }, "onGutterKeyDown"),
    onGutterMouseDown: /* @__PURE__ */ __name(function onGutterMouseDown(event, index2) {
      this.onResizeStart(event, index2);
      this.bindMouseListeners();
    }, "onGutterMouseDown"),
    onGutterTouchStart: /* @__PURE__ */ __name(function onGutterTouchStart(event, index2) {
      this.onResizeStart(event, index2);
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
        var children = _toConsumableArray$7(this.$el.children).filter(function(child) {
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
var _hoisted_1$D = ["onMousedown", "onTouchstart", "onTouchmove", "onTouchend"];
var _hoisted_2$t = ["aria-orientation", "aria-valuenow", "onKeydown"];
function render$l(_ctx, _cache, $props, $setup, $data, $options) {
  return openBlock(), createElementBlock("div", mergeProps({
    "class": _ctx.cx("root"),
    style: _ctx.sx("root"),
    "data-p-resizing": false
  }, _ctx.ptmi("root", $options.getPTOptions)), [(openBlock(true), createElementBlock(Fragment, null, renderList($options.panels, function(panel2, i) {
    return openBlock(), createElementBlock(Fragment, {
      key: i
    }, [(openBlock(), createBlock(resolveDynamicComponent(panel2), {
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
    }, _ctx.ptm("gutterHandle")), null, 16, _hoisted_2$t)], 16, _hoisted_1$D)) : createCommentVNode("", true)], 64);
  }), 128))], 16);
}
__name(render$l, "render$l");
script$k.render = render$l;
var classes$g = {
  root: /* @__PURE__ */ __name(function root3(_ref) {
    var instance = _ref.instance;
    return ["p-splitterpanel", {
      "p-splitterpanel-nested": instance.isNested
    }];
  }, "root")
};
var SplitterPanelStyle = BaseStyle.extend({
  name: "splitterpanel",
  classes: classes$g
});
var script$1$g = {
  name: "BaseSplitterPanel",
  "extends": script$p,
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
  provide: /* @__PURE__ */ __name(function provide4() {
    return {
      $pcSplitterPanel: this,
      $parentInstance: this
    };
  }, "provide")
};
var script$j = {
  name: "SplitterPanel",
  "extends": script$1$g,
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
function render$k(_ctx, _cache, $props, $setup, $data, $options) {
  return openBlock(), createElementBlock("div", mergeProps({
    ref: "container",
    "class": _ctx.cx("root")
  }, _ctx.ptmi("root", $options.getPTOptions)), [renderSlot(_ctx.$slots, "default")], 16);
}
__name(render$k, "render$k");
script$j.render = render$k;
const _sfc_main$C = /* @__PURE__ */ defineComponent({
  __name: "LiteGraphCanvasSplitterOverlay",
  setup(__props) {
    const settingStore = useSettingStore();
    const sidebarLocation = computed(
      () => settingStore.get("Comfy.Sidebar.Location")
    );
    const sidebarPanelVisible = computed(
      () => useWorkspaceStore().sidebarTab.activeSidebarTab !== null
    );
    const gutterClass = computed(() => {
      return sidebarPanelVisible.value ? "" : "gutter-hidden";
    });
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(script$k), {
        class: "splitter-overlay",
        "pt:gutter": gutterClass.value
      }, {
        default: withCtx(() => [
          sidebarLocation.value === "left" ? withDirectives((openBlock(), createBlock(unref(script$j), {
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
          createVNode(unref(script$j), {
            class: "graph-canvas-panel relative",
            size: 100
          }, {
            default: withCtx(() => [
              renderSlot(_ctx.$slots, "graph-canvas-panel", {}, void 0, true)
            ]),
            _: 3
          }),
          sidebarLocation.value === "right" ? withDirectives((openBlock(), createBlock(unref(script$j), {
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
const LiteGraphCanvasSplitterOverlay = /* @__PURE__ */ _export_sfc(_sfc_main$C, [["__scopeId", "data-v-b9df3042"]]);
var theme$f = /* @__PURE__ */ __name(function theme3(_ref) {
  var dt = _ref.dt;
  return "\n.p-autocomplete {\n    display: inline-flex;\n}\n\n.p-autocomplete-loader {\n    position: absolute;\n    top: 50%;\n    margin-top: -0.5rem;\n    right: ".concat(dt("autocomplete.padding.x"), ";\n}\n\n.p-autocomplete:has(.p-autocomplete-dropdown) .p-autocomplete-loader {\n    right: calc(").concat(dt("autocomplete.dropdown.width"), " + ").concat(dt("autocomplete.padding.x"), ");\n}\n\n.p-autocomplete:has(.p-autocomplete-dropdown) .p-autocomplete-input {\n    flex: 1 1 auto;\n    width: 1%;\n}\n\n.p-autocomplete:has(.p-autocomplete-dropdown) .p-autocomplete-input,\n.p-autocomplete:has(.p-autocomplete-dropdown) .p-autocomplete-input-multiple {\n    border-top-right-radius: 0;\n    border-bottom-right-radius: 0;\n}\n\n.p-autocomplete-dropdown {\n    cursor: pointer;\n    display: inline-flex;\n    cursor: pointer;\n    user-select: none;\n    align-items: center;\n    justify-content: center;\n    overflow: hidden;\n    position: relative;\n    width: ").concat(dt("autocomplete.dropdown.width"), ";\n    border-top-right-radius: ").concat(dt("autocomplete.dropdown.border.radius"), ";\n    border-bottom-right-radius: ").concat(dt("autocomplete.dropdown.border.radius"), ";\n    background: ").concat(dt("autocomplete.dropdown.background"), ";\n    border: 1px solid ").concat(dt("autocomplete.dropdown.border.color"), ";\n    border-left: 0 none;\n    color: ").concat(dt("autocomplete.dropdown.color"), ";\n    transition: background ").concat(dt("autocomplete.transition.duration"), ", color ").concat(dt("autocomplete.transition.duration"), ", border-color ").concat(dt("autocomplete.transition.duration"), ", outline-color ").concat(dt("autocomplete.transition.duration"), ", box-shadow ").concat(dt("autocomplete.transition.duration"), ";\n    outline-color: transparent;\n}\n\n.p-autocomplete-dropdown:not(:disabled):hover {\n    background: ").concat(dt("autocomplete.dropdown.hover.background"), ";\n    border-color: ").concat(dt("autocomplete.dropdown.hover.border.color"), ";\n    color: ").concat(dt("autocomplete.dropdown.hover.color"), ";\n}\n\n.p-autocomplete-dropdown:not(:disabled):active {\n    background: ").concat(dt("autocomplete.dropdown.active.background"), ";\n    border-color: ").concat(dt("autocomplete.dropdown.active.border.color"), ";\n    color: ").concat(dt("autocomplete.dropdown.active.color"), ";\n}\n\n.p-autocomplete-dropdown:focus-visible {\n    box-shadow: ").concat(dt("autocomplete.dropdown.focus.ring.shadow"), ";\n    outline: ").concat(dt("autocomplete.dropdown.focus.ring.width"), " ").concat(dt("autocomplete.dropdown.focus.ring.style"), " ").concat(dt("autocomplete.dropdown.focus.ring.color"), ";\n    outline-offset: ").concat(dt("autocomplete.dropdown.focus.ring.offset"), ";\n}\n\n.p-autocomplete .p-autocomplete-overlay {\n    min-width: 100%;\n}\n\n.p-autocomplete-overlay {\n    position: absolute;\n    overflow: auto;\n    top: 0;\n    left: 0;\n    background: ").concat(dt("autocomplete.overlay.background"), ";\n    color: ").concat(dt("autocomplete.overlay.color"), ";\n    border: 1px solid ").concat(dt("autocomplete.overlay.border.color"), ";\n    border-radius: ").concat(dt("autocomplete.overlay.border.radius"), ";\n    box-shadow: ").concat(dt("autocomplete.overlay.shadow"), ";\n}\n\n.p-autocomplete-list {\n    margin: 0;\n    padding: 0;\n    list-style-type: none;\n    display: flex;\n    flex-direction: column;\n    gap: ").concat(dt("autocomplete.list.gap"), ";\n    padding: ").concat(dt("autocomplete.list.padding"), ";\n}\n\n.p-autocomplete-option {\n    cursor: pointer;\n    white-space: nowrap;\n    position: relative;\n    overflow: hidden;\n    display: flex;\n    align-items: center;\n    padding: ").concat(dt("autocomplete.option.padding"), ";\n    border: 0 none;\n    color: ").concat(dt("autocomplete.option.color"), ";\n    background: transparent;\n    transition: background ").concat(dt("autocomplete.transition.duration"), ", color ").concat(dt("autocomplete.transition.duration"), ", border-color ").concat(dt("autocomplete.transition.duration"), ";\n    border-radius: ").concat(dt("autocomplete.option.border.radius"), ";\n}\n\n.p-autocomplete-option:not(.p-autocomplete-option-selected):not(.p-disabled).p-focus {\n    background: ").concat(dt("autocomplete.option.focus.background"), ";\n    color: ").concat(dt("autocomplete.option.focus.color"), ";\n}\n\n.p-autocomplete-option-selected {\n    background: ").concat(dt("autocomplete.option.selected.background"), ";\n    color: ").concat(dt("autocomplete.option.selected.color"), ";\n}\n\n.p-autocomplete-option-selected.p-focus {\n    background: ").concat(dt("autocomplete.option.selected.focus.background"), ";\n    color: ").concat(dt("autocomplete.option.selected.focus.color"), ";\n}\n\n.p-autocomplete-option-group {\n    margin: 0;\n    padding: ").concat(dt("autocomplete.option.group.padding"), ";\n    color: ").concat(dt("autocomplete.option.group.color"), ";\n    background: ").concat(dt("autocomplete.option.group.background"), ";\n    font-weight: ").concat(dt("autocomplete.option.group.font.weight"), ";\n}\n\n.p-autocomplete-input-multiple {\n    margin: 0;\n    list-style-type: none;\n    cursor: text;\n    overflow: hidden;\n    display: flex;\n    align-items: center;\n    flex-wrap: wrap;\n    padding: calc(").concat(dt("autocomplete.padding.y"), " / 2) ").concat(dt("autocomplete.padding.x"), ";\n    gap: calc(").concat(dt("autocomplete.padding.y"), " / 2);\n    color: ").concat(dt("autocomplete.color"), ";\n    background: ").concat(dt("autocomplete.background"), ";\n    border: 1px solid ").concat(dt("autocomplete.border.color"), ";\n    border-radius: ").concat(dt("autocomplete.border.radius"), ";\n    width: 100%;\n    transition: background ").concat(dt("autocomplete.transition.duration"), ", color ").concat(dt("autocomplete.transition.duration"), ", border-color ").concat(dt("autocomplete.transition.duration"), ", outline-color ").concat(dt("autocomplete.transition.duration"), ", box-shadow ").concat(dt("autocomplete.transition.duration"), ";\n    outline-color: transparent;\n    box-shadow: ").concat(dt("autocomplete.shadow"), ";\n}\n\n.p-autocomplete:not(.p-disabled):hover .p-autocomplete-input-multiple {\n    border-color: ").concat(dt("autocomplete.hover.border.color"), ";\n}\n\n.p-autocomplete:not(.p-disabled).p-focus .p-autocomplete-input-multiple {\n    border-color: ").concat(dt("autocomplete.focus.border.color"), ";\n    box-shadow: ").concat(dt("autocomplete.focus.ring.shadow"), ";\n    outline: ").concat(dt("autocomplete.focus.ring.width"), " ").concat(dt("autocomplete.focus.ring.style"), " ").concat(dt("autocomplete.focus.ring.color"), ";\n    outline-offset: ").concat(dt("autocomplete.focus.ring.offset"), ";\n}\n\n.p-autocomplete.p-invalid .p-autocomplete-input-multiple {\n    border-color: ").concat(dt("autocomplete.invalid.border.color"), ";\n}\n\n.p-variant-filled.p-autocomplete-input-multiple {\n    background: ").concat(dt("autocomplete.filled.background"), ";\n}\n\n.p-autocomplete:not(.p-disabled).p-focus .p-variant-filled.p-autocomplete-input-multiple  {\n    background: ").concat(dt("autocomplete.filled.focus.background"), ";\n}\n\n.p-autocomplete.p-disabled .p-autocomplete-input-multiple {\n    opacity: 1;\n    background: ").concat(dt("autocomplete.disabled.background"), ";\n    color: ").concat(dt("autocomplete.disabled.color"), ";\n}\n\n.p-autocomplete-chip.p-chip {\n    padding-top: calc(").concat(dt("autocomplete.padding.y"), " / 2);\n    padding-bottom: calc(").concat(dt("autocomplete.padding.y"), " / 2);\n    border-radius: ").concat(dt("autocomplete.chip.border.radius"), ";\n}\n\n.p-autocomplete-input-multiple:has(.p-autocomplete-chip) {\n    padding-left: calc(").concat(dt("autocomplete.padding.y"), " / 2);\n    padding-right: calc(").concat(dt("autocomplete.padding.y"), " / 2);\n}\n\n.p-autocomplete-chip-item.p-focus .p-autocomplete-chip {\n    background: ").concat(dt("inputchips.chip.focus.background"), ";\n    color: ").concat(dt("inputchips.chip.focus.color"), ";\n}\n\n.p-autocomplete-input-chip {\n    flex: 1 1 auto;\n    display: inline-flex;\n    padding-top: calc(").concat(dt("autocomplete.padding.y"), " / 2);\n    padding-bottom: calc(").concat(dt("autocomplete.padding.y"), " / 2);\n}\n\n.p-autocomplete-input-chip input {\n    border: 0 none;\n    outline: 0 none;\n    background: transparent;\n    margin: 0;\n    padding: 0;\n    box-shadow: none;\n    border-radius: 0;\n    width: 100%;\n    font-family: inherit;\n    font-feature-settings: inherit;\n    font-size: 1rem;\n    color: inherit;\n}\n\n.p-autocomplete-input-chip input::placeholder {\n    color: ").concat(dt("autocomplete.placeholder.color"), ";\n}\n\n.p-autocomplete-empty-message {\n    padding: ").concat(dt("autocomplete.empty.message.padding"), ";\n}\n\n.p-autocomplete-fluid {\n    display: flex;\n}\n\n.p-autocomplete-fluid:has(.p-autocomplete-dropdown) .p-autocomplete-input {\n    width: 1%;\n}\n");
}, "theme");
var inlineStyles$3 = {
  root: {
    position: "relative"
  }
};
var classes$f = {
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
  theme: theme$f,
  classes: classes$f,
  inlineStyles: inlineStyles$3
});
var script$1$f = {
  name: "BaseAutoComplete",
  "extends": script$p,
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
  provide: /* @__PURE__ */ __name(function provide5() {
    return {
      $pcAutoComplete: this,
      $parentInstance: this
    };
  }, "provide")
};
function _typeof$1$3(o) {
  "@babel/helpers - typeof";
  return _typeof$1$3 = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function(o2) {
    return typeof o2;
  } : function(o2) {
    return o2 && "function" == typeof Symbol && o2.constructor === Symbol && o2 !== Symbol.prototype ? "symbol" : typeof o2;
  }, _typeof$1$3(o);
}
__name(_typeof$1$3, "_typeof$1$3");
function _toConsumableArray$6(r) {
  return _arrayWithoutHoles$6(r) || _iterableToArray$6(r) || _unsupportedIterableToArray$8(r) || _nonIterableSpread$6();
}
__name(_toConsumableArray$6, "_toConsumableArray$6");
function _nonIterableSpread$6() {
  throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.");
}
__name(_nonIterableSpread$6, "_nonIterableSpread$6");
function _unsupportedIterableToArray$8(r, a) {
  if (r) {
    if ("string" == typeof r) return _arrayLikeToArray$8(r, a);
    var t = {}.toString.call(r).slice(8, -1);
    return "Object" === t && r.constructor && (t = r.constructor.name), "Map" === t || "Set" === t ? Array.from(r) : "Arguments" === t || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(t) ? _arrayLikeToArray$8(r, a) : void 0;
  }
}
__name(_unsupportedIterableToArray$8, "_unsupportedIterableToArray$8");
function _iterableToArray$6(r) {
  if ("undefined" != typeof Symbol && null != r[Symbol.iterator] || null != r["@@iterator"]) return Array.from(r);
}
__name(_iterableToArray$6, "_iterableToArray$6");
function _arrayWithoutHoles$6(r) {
  if (Array.isArray(r)) return _arrayLikeToArray$8(r);
}
__name(_arrayWithoutHoles$6, "_arrayWithoutHoles$6");
function _arrayLikeToArray$8(r, a) {
  (null == a || a > r.length) && (a = r.length);
  for (var e = 0, n = Array(a); e < a; e++) n[e] = r[e];
  return n;
}
__name(_arrayLikeToArray$8, "_arrayLikeToArray$8");
var script$i = {
  name: "AutoComplete",
  "extends": script$1$f,
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
    getOptionIndex: /* @__PURE__ */ __name(function getOptionIndex(index2, fn) {
      return this.virtualScrollerDisabled ? index2 : fn && fn(index2)["index"];
    }, "getOptionIndex"),
    getOptionLabel: /* @__PURE__ */ __name(function getOptionLabel(option2) {
      return this.optionLabel ? resolveFieldData(option2, this.optionLabel) : option2;
    }, "getOptionLabel"),
    getOptionValue: /* @__PURE__ */ __name(function getOptionValue(option2) {
      return option2;
    }, "getOptionValue"),
    getOptionRenderKey: /* @__PURE__ */ __name(function getOptionRenderKey(option2, index2) {
      return (this.dataKey ? resolveFieldData(option2, this.dataKey) : this.getOptionLabel(option2)) + "_" + index2;
    }, "getOptionRenderKey"),
    getPTOptions: /* @__PURE__ */ __name(function getPTOptions3(option2, itemOptions, index2, key) {
      return this.ptm(key, {
        context: {
          selected: this.isSelected(option2),
          focused: this.focusedOptionIndex === this.getOptionIndex(index2, itemOptions),
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
    getAriaPosInset: /* @__PURE__ */ __name(function getAriaPosInset(index2) {
      var _this = this;
      return (this.optionGroupLabel ? index2 - this.visibleOptions.slice(0, index2).filter(function(option2) {
        return _this.isOptionGroup(option2);
      }).length : index2) + 1;
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
          this.updateModel(event, [].concat(_toConsumableArray$6(this.modelValue || []), [value]));
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
    onOptionMouseMove: /* @__PURE__ */ __name(function onOptionMouseMove(event, index2) {
      if (this.focusOnHover) {
        this.changeFocusedOptionIndex(event, index2);
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
          this.updateModel(event, [].concat(_toConsumableArray$6(this.modelValue || []), [event.target.value]));
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
    findNextOptionIndex: /* @__PURE__ */ __name(function findNextOptionIndex(index2) {
      var _this11 = this;
      var matchedOptionIndex = index2 < this.visibleOptions.length - 1 ? this.visibleOptions.slice(index2 + 1).findIndex(function(option2) {
        return _this11.isValidOption(option2);
      }) : -1;
      return matchedOptionIndex > -1 ? matchedOptionIndex + index2 + 1 : index2;
    }, "findNextOptionIndex"),
    findPrevOptionIndex: /* @__PURE__ */ __name(function findPrevOptionIndex(index2) {
      var _this12 = this;
      var matchedOptionIndex = index2 > 0 ? findLastIndex(this.visibleOptions.slice(0, index2), function(option2) {
        return _this12.isValidOption(option2);
      }) : -1;
      return matchedOptionIndex > -1 ? matchedOptionIndex : index2;
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
    removeOption: /* @__PURE__ */ __name(function removeOption(event, index2) {
      var _this14 = this;
      var removedOption = this.modelValue[index2];
      var value = this.modelValue.filter(function(_2, i) {
        return i !== index2;
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
    changeFocusedOptionIndex: /* @__PURE__ */ __name(function changeFocusedOptionIndex(event, index2) {
      if (this.focusedOptionIndex !== index2) {
        this.focusedOptionIndex = index2;
        this.scrollInView();
        if (this.selectOnFocus) {
          this.onOptionSelect(event, this.visibleOptions[index2], false);
        }
      }
    }, "changeFocusedOptionIndex"),
    scrollInView: /* @__PURE__ */ __name(function scrollInView() {
      var _this15 = this;
      var index2 = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : -1;
      this.$nextTick(function() {
        var id = index2 !== -1 ? "".concat(_this15.id, "_").concat(index2) : _this15.focusedOptionId;
        var element = findSingle(_this15.list, 'li[id="'.concat(id, '"]'));
        if (element) {
          element.scrollIntoView && element.scrollIntoView({
            block: "nearest",
            inline: "start"
          });
        } else if (!_this15.virtualScrollerDisabled) {
          _this15.virtualScroller && _this15.virtualScroller.scrollToIndex(index2 !== -1 ? index2 : _this15.focusedOptionIndex);
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
      return (options || []).reduce(function(result, option2, index2) {
        result.push({
          optionGroup: option2,
          group: true,
          index: index2
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
        if (_typeof$1$3(this.modelValue) === "object") {
          var label3 = this.getOptionLabel(this.modelValue);
          return label3 != null ? label3 : this.modelValue;
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
    InputText: script$m,
    VirtualScroller: script$q,
    Portal: script$r,
    ChevronDownIcon: script$s,
    SpinnerIcon: script$t,
    Chip: script$u
  },
  directives: {
    ripple: Ripple
  }
};
function _typeof$7(o) {
  "@babel/helpers - typeof";
  return _typeof$7 = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function(o2) {
    return typeof o2;
  } : function(o2) {
    return o2 && "function" == typeof Symbol && o2.constructor === Symbol && o2 !== Symbol.prototype ? "symbol" : typeof o2;
  }, _typeof$7(o);
}
__name(_typeof$7, "_typeof$7");
function ownKeys$8(e, r) {
  var t = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var o = Object.getOwnPropertySymbols(e);
    r && (o = o.filter(function(r2) {
      return Object.getOwnPropertyDescriptor(e, r2).enumerable;
    })), t.push.apply(t, o);
  }
  return t;
}
__name(ownKeys$8, "ownKeys$8");
function _objectSpread$8(e) {
  for (var r = 1; r < arguments.length; r++) {
    var t = null != arguments[r] ? arguments[r] : {};
    r % 2 ? ownKeys$8(Object(t), true).forEach(function(r2) {
      _defineProperty$7(e, r2, t[r2]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(t)) : ownKeys$8(Object(t)).forEach(function(r2) {
      Object.defineProperty(e, r2, Object.getOwnPropertyDescriptor(t, r2));
    });
  }
  return e;
}
__name(_objectSpread$8, "_objectSpread$8");
function _defineProperty$7(e, r, t) {
  return (r = _toPropertyKey$6(r)) in e ? Object.defineProperty(e, r, { value: t, enumerable: true, configurable: true, writable: true }) : e[r] = t, e;
}
__name(_defineProperty$7, "_defineProperty$7");
function _toPropertyKey$6(t) {
  var i = _toPrimitive$6(t, "string");
  return "symbol" == _typeof$7(i) ? i : i + "";
}
__name(_toPropertyKey$6, "_toPropertyKey$6");
function _toPrimitive$6(t, r) {
  if ("object" != _typeof$7(t) || !t) return t;
  var e = t[Symbol.toPrimitive];
  if (void 0 !== e) {
    var i = e.call(t, r || "default");
    if ("object" != _typeof$7(i)) return i;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return ("string" === r ? String : Number)(t);
}
__name(_toPrimitive$6, "_toPrimitive$6");
var _hoisted_1$C = ["aria-activedescendant"];
var _hoisted_2$s = ["id", "aria-label", "aria-setsize", "aria-posinset"];
var _hoisted_3$l = ["id", "placeholder", "tabindex", "disabled", "aria-label", "aria-labelledby", "aria-expanded", "aria-controls", "aria-activedescendant", "aria-invalid"];
var _hoisted_4$e = ["disabled", "aria-expanded", "aria-controls"];
var _hoisted_5$a = ["id"];
var _hoisted_6$8 = ["id", "aria-label"];
var _hoisted_7$5 = ["id"];
var _hoisted_8$4 = ["id", "aria-label", "aria-selected", "aria-disabled", "aria-setsize", "aria-posinset", "onClick", "onMousemove", "data-p-selected", "data-p-focus", "data-p-disabled"];
function render$j(_ctx, _cache, $props, $setup, $data, $options) {
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
        onRemove: /* @__PURE__ */ __name(function onRemove2($event) {
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
    })], 16, _hoisted_2$s);
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
  }, _ctx.ptm("input")), null, 16, _hoisted_3$l)], 16)], 16, _hoisted_1$C)) : createCommentVNode("", true), $data.searching || _ctx.loading ? renderSlot(_ctx.$slots, _ctx.$slots.loader ? "loader" : "loadingicon", {
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
    })], 16, _hoisted_4$e)) : createCommentVNode("", true)];
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
            style: _objectSpread$8(_objectSpread$8(_objectSpread$8({}, _ctx.panelStyle), _ctx.overlayStyle), {}, {
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
                })], 16, _hoisted_7$5)) : withDirectives((openBlock(), createElementBlock("li", mergeProps({
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
                  onClick: /* @__PURE__ */ __name(function onClick2($event) {
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
                })], 16, _hoisted_8$4)), [[_directive_ripple]])], 64);
              }), 128)), !items || items && items.length === 0 ? (openBlock(), createElementBlock("li", mergeProps({
                key: 0,
                "class": _ctx.cx("emptyMessage"),
                role: "option"
              }, _ctx.ptm("emptyMessage")), [renderSlot(_ctx.$slots, "empty", {}, function() {
                return [createTextVNode(toDisplayString($options.searchResultMessageText), 1)];
              })], 16)) : createCommentVNode("", true)], 16, _hoisted_6$8)];
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
          }), toDisplayString($options.selectedMessageText), 17)], 16, _hoisted_5$a)) : createCommentVNode("", true)];
        }),
        _: 3
      }, 16, ["onEnter", "onAfterEnter", "onLeave", "onAfterLeave"])];
    }),
    _: 3
  }, 8, ["appendTo"])], 16);
}
__name(render$j, "render$j");
script$i.render = render$j;
const _sfc_main$B = {
  name: "AutoCompletePlus",
  extends: script$i,
  emits: ["focused-option-changed"],
  mounted() {
    if (typeof script$i.mounted === "function") {
      script$i.mounted.call(this);
    }
    this.$watch(
      () => this.focusedOptionIndex,
      (newVal, oldVal) => {
        this.$emit("focused-option-changed", newVal);
      }
    );
  }
};
var theme$e = /* @__PURE__ */ __name(function theme4(_ref) {
  var dt = _ref.dt;
  return "\n.p-togglebutton {\n    display: inline-flex;\n    cursor: pointer;\n    user-select: none;\n    align-items: center;\n    justify-content: center;\n    overflow: hidden;\n    position: relative;\n    color: ".concat(dt("togglebutton.color"), ";\n    background: ").concat(dt("togglebutton.background"), ";\n    border: 1px solid ").concat(dt("togglebutton.border.color"), ";\n    padding: ").concat(dt("togglebutton.padding"), ";\n    font-size: 1rem;\n    font-family: inherit;\n    font-feature-settings: inherit;\n    transition: background ").concat(dt("togglebutton.transition.duration"), ", color ").concat(dt("togglebutton.transition.duration"), ", border-color ").concat(dt("togglebutton.transition.duration"), ",\n        outline-color ").concat(dt("togglebutton.transition.duration"), ", box-shadow ").concat(dt("togglebutton.transition.duration"), ";\n    border-radius: ").concat(dt("togglebutton.border.radius"), ";\n    outline-color: transparent;\n    font-weight: ").concat(dt("togglebutton.font.weight"), ";\n}\n\n.p-togglebutton-content {\n    position: relative;\n    display: inline-flex;\n    align-items: center;\n    justify-content: center;\n    gap: ").concat(dt("togglebutton.gap"), ';\n}\n\n.p-togglebutton-label,\n.p-togglebutton-icon {\n    position: relative;\n    transition: none;\n}\n\n.p-togglebutton::before {\n    content: "";\n    background: transparent;\n    transition: background ').concat(dt("togglebutton.transition.duration"), ", color ").concat(dt("togglebutton.transition.duration"), ", border-color ").concat(dt("togglebutton.transition.duration"), ",\n            outline-color ").concat(dt("togglebutton.transition.duration"), ", box-shadow ").concat(dt("togglebutton.transition.duration"), ";\n    position: absolute;\n    left: ").concat(dt("togglebutton.content.left"), ";\n    top: ").concat(dt("togglebutton.content.top"), ";\n    width: calc(100% - calc(2 *  ").concat(dt("togglebutton.content.left"), "));\n    height: calc(100% - calc(2 *  ").concat(dt("togglebutton.content.top"), "));\n    border-radius: ").concat(dt("togglebutton.border.radius"), ";\n}\n\n.p-togglebutton.p-togglebutton-checked::before {\n    background: ").concat(dt("togglebutton.content.checked.background"), ";\n    box-shadow: ").concat(dt("togglebutton.content.checked.shadow"), ";\n}\n\n.p-togglebutton:not(:disabled):not(.p-togglebutton-checked):hover {\n    background: ").concat(dt("togglebutton.hover.background"), ";\n    color: ").concat(dt("togglebutton.hover.color"), ";\n}\n\n.p-togglebutton.p-togglebutton-checked {\n    background: ").concat(dt("togglebutton.checked.background"), ";\n    border-color: ").concat(dt("togglebutton.checked.border.color"), ";\n    color: ").concat(dt("togglebutton.checked.color"), ";\n}\n\n.p-togglebutton:focus-visible {\n    box-shadow: ").concat(dt("togglebutton.focus.ring.shadow"), ";\n    outline: ").concat(dt("togglebutton.focus.ring.width"), " ").concat(dt("togglebutton.focus.ring.style"), " ").concat(dt("togglebutton.focus.ring.color"), ";\n    outline-offset: ").concat(dt("togglebutton.focus.ring.offset"), ";\n}\n\n.p-togglebutton.p-invalid {\n    border-color: ").concat(dt("togglebutton.invalid.border.color"), ";\n}\n\n.p-togglebutton:disabled {\n    opacity: 1;\n    cursor: default;\n    background: ").concat(dt("togglebutton.disabled.background"), ";\n    border-color: ").concat(dt("togglebutton.disabled.border.color"), ";\n    color: ").concat(dt("togglebutton.disabled.color"), ";\n}\n\n.p-togglebutton-icon {\n    color: ").concat(dt("togglebutton.icon.color"), ";\n}\n\n.p-togglebutton:not(:disabled):not(.p-togglebutton-checked):hover .p-togglebutton-icon {\n    color: ").concat(dt("togglebutton.icon.hover.color"), ";\n}\n\n.p-togglebutton.p-togglebutton-checked .p-togglebutton-icon {\n    color: ").concat(dt("togglebutton.icon.checked.color"), ";\n}\n\n.p-togglebutton:disabled .p-togglebutton-icon {\n    color: ").concat(dt("togglebutton.icon.disabled.color"), ";\n}\n");
}, "theme");
var classes$e = {
  root: /* @__PURE__ */ __name(function root5(_ref2) {
    var instance = _ref2.instance, props = _ref2.props;
    return ["p-togglebutton p-component", {
      "p-togglebutton-checked": instance.active,
      "p-invalid": props.invalid
    }];
  }, "root"),
  content: "p-togglebutton-content",
  icon: "p-togglebutton-icon",
  label: "p-togglebutton-label"
};
var ToggleButtonStyle = BaseStyle.extend({
  name: "togglebutton",
  theme: theme$e,
  classes: classes$e
});
var script$1$e = {
  name: "BaseToggleButton",
  "extends": script$p,
  props: {
    modelValue: Boolean,
    onIcon: String,
    offIcon: String,
    onLabel: {
      type: String,
      "default": "Yes"
    },
    offLabel: {
      type: String,
      "default": "No"
    },
    iconPos: {
      type: String,
      "default": "left"
    },
    invalid: {
      type: Boolean,
      "default": false
    },
    disabled: {
      type: Boolean,
      "default": false
    },
    readonly: {
      type: Boolean,
      "default": false
    },
    tabindex: {
      type: Number,
      "default": null
    },
    ariaLabelledby: {
      type: String,
      "default": null
    },
    ariaLabel: {
      type: String,
      "default": null
    }
  },
  style: ToggleButtonStyle,
  provide: /* @__PURE__ */ __name(function provide6() {
    return {
      $pcToggleButton: this,
      $parentInstance: this
    };
  }, "provide")
};
var script$h = {
  name: "ToggleButton",
  "extends": script$1$e,
  inheritAttrs: false,
  emits: ["update:modelValue", "change"],
  methods: {
    getPTOptions: /* @__PURE__ */ __name(function getPTOptions4(key) {
      var _ptm = key === "root" ? this.ptmi : this.ptm;
      return _ptm(key, {
        context: {
          active: this.active,
          disabled: this.disabled
        }
      });
    }, "getPTOptions"),
    onChange: /* @__PURE__ */ __name(function onChange2(event) {
      if (!this.disabled && !this.readonly) {
        this.$emit("update:modelValue", !this.modelValue);
        this.$emit("change", event);
      }
    }, "onChange")
  },
  computed: {
    active: /* @__PURE__ */ __name(function active() {
      return this.modelValue === true;
    }, "active"),
    hasLabel: /* @__PURE__ */ __name(function hasLabel() {
      return isNotEmpty(this.onLabel) && isNotEmpty(this.offLabel);
    }, "hasLabel"),
    label: /* @__PURE__ */ __name(function label() {
      return this.hasLabel ? this.modelValue ? this.onLabel : this.offLabel : "&nbsp;";
    }, "label")
  },
  directives: {
    ripple: Ripple
  }
};
var _hoisted_1$B = ["tabindex", "disabled", "aria-pressed", "data-p-checked", "data-p-disabled"];
function render$i(_ctx, _cache, $props, $setup, $data, $options) {
  var _directive_ripple = resolveDirective("ripple");
  return withDirectives((openBlock(), createElementBlock("button", mergeProps({
    type: "button",
    "class": _ctx.cx("root"),
    tabindex: _ctx.tabindex,
    disabled: _ctx.disabled,
    "aria-pressed": _ctx.modelValue,
    onClick: _cache[0] || (_cache[0] = function() {
      return $options.onChange && $options.onChange.apply($options, arguments);
    })
  }, $options.getPTOptions("root"), {
    "data-p-checked": $options.active,
    "data-p-disabled": _ctx.disabled
  }), [createBaseVNode("span", mergeProps({
    "class": _ctx.cx("content")
  }, $options.getPTOptions("content")), [renderSlot(_ctx.$slots, "default", {}, function() {
    return [renderSlot(_ctx.$slots, "icon", {
      value: _ctx.modelValue,
      "class": normalizeClass(_ctx.cx("icon"))
    }, function() {
      return [_ctx.onIcon || _ctx.offIcon ? (openBlock(), createElementBlock("span", mergeProps({
        key: 0,
        "class": [_ctx.cx("icon"), _ctx.modelValue ? _ctx.onIcon : _ctx.offIcon]
      }, $options.getPTOptions("icon")), null, 16)) : createCommentVNode("", true)];
    }), createBaseVNode("span", mergeProps({
      "class": _ctx.cx("label")
    }, $options.getPTOptions("label")), toDisplayString($options.label), 17)];
  })], 16)], 16, _hoisted_1$B)), [[_directive_ripple]]);
}
__name(render$i, "render$i");
script$h.render = render$i;
var theme$d = /* @__PURE__ */ __name(function theme5(_ref) {
  var dt = _ref.dt;
  return "\n.p-selectbutton {\n    display: inline-flex;\n    user-select: none;\n    vertical-align: bottom;\n    outline-color: transparent;\n    border-radius: ".concat(dt("selectbutton.border.radius"), ";\n}\n\n.p-selectbutton .p-togglebutton {\n    border-radius: 0;\n    border-width: 1px 1px 1px 0;\n}\n\n.p-selectbutton .p-togglebutton:focus-visible {\n    position: relative;\n    z-index: 1;\n}\n\n.p-selectbutton .p-togglebutton:first-child {\n    border-left-width: 1px;\n    border-top-left-radius: ").concat(dt("selectbutton.border.radius"), ";\n    border-bottom-left-radius: ").concat(dt("selectbutton.border.radius"), ";\n}\n\n.p-selectbutton .p-togglebutton:last-child {\n    border-top-right-radius: ").concat(dt("selectbutton.border.radius"), ";\n    border-bottom-right-radius: ").concat(dt("selectbutton.border.radius"), ";\n}\n\n.p-selectbutton.p-invalid {\n    outline: 1px solid ").concat(dt("selectbutton.invalid.border.color"), ";\n    outline-offset: 0;\n}\n");
}, "theme");
var classes$d = {
  root: /* @__PURE__ */ __name(function root6(_ref2) {
    var props = _ref2.props;
    return ["p-selectbutton p-component", {
      "p-invalid": props.invalid
    }];
  }, "root")
};
var SelectButtonStyle = BaseStyle.extend({
  name: "selectbutton",
  theme: theme$d,
  classes: classes$d
});
var script$1$d = {
  name: "BaseSelectButton",
  "extends": script$p,
  props: {
    modelValue: null,
    options: Array,
    optionLabel: null,
    optionValue: null,
    optionDisabled: null,
    multiple: Boolean,
    allowEmpty: {
      type: Boolean,
      "default": true
    },
    invalid: {
      type: Boolean,
      "default": false
    },
    disabled: Boolean,
    dataKey: null,
    ariaLabelledby: {
      type: String,
      "default": null
    }
  },
  style: SelectButtonStyle,
  provide: /* @__PURE__ */ __name(function provide7() {
    return {
      $pcSelectButton: this,
      $parentInstance: this
    };
  }, "provide")
};
function _createForOfIteratorHelper$4(r, e) {
  var t = "undefined" != typeof Symbol && r[Symbol.iterator] || r["@@iterator"];
  if (!t) {
    if (Array.isArray(r) || (t = _unsupportedIterableToArray$7(r)) || e) {
      t && (r = t);
      var _n = 0, F = /* @__PURE__ */ __name(function F2() {
      }, "F");
      return { s: F, n: /* @__PURE__ */ __name(function n() {
        return _n >= r.length ? { done: true } : { done: false, value: r[_n++] };
      }, "n"), e: /* @__PURE__ */ __name(function e2(r2) {
        throw r2;
      }, "e"), f: F };
    }
    throw new TypeError("Invalid attempt to iterate non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.");
  }
  var o, a = true, u = false;
  return { s: /* @__PURE__ */ __name(function s() {
    t = t.call(r);
  }, "s"), n: /* @__PURE__ */ __name(function n() {
    var r2 = t.next();
    return a = r2.done, r2;
  }, "n"), e: /* @__PURE__ */ __name(function e2(r2) {
    u = true, o = r2;
  }, "e"), f: /* @__PURE__ */ __name(function f() {
    try {
      a || null == t["return"] || t["return"]();
    } finally {
      if (u) throw o;
    }
  }, "f") };
}
__name(_createForOfIteratorHelper$4, "_createForOfIteratorHelper$4");
function _toConsumableArray$5(r) {
  return _arrayWithoutHoles$5(r) || _iterableToArray$5(r) || _unsupportedIterableToArray$7(r) || _nonIterableSpread$5();
}
__name(_toConsumableArray$5, "_toConsumableArray$5");
function _nonIterableSpread$5() {
  throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.");
}
__name(_nonIterableSpread$5, "_nonIterableSpread$5");
function _unsupportedIterableToArray$7(r, a) {
  if (r) {
    if ("string" == typeof r) return _arrayLikeToArray$7(r, a);
    var t = {}.toString.call(r).slice(8, -1);
    return "Object" === t && r.constructor && (t = r.constructor.name), "Map" === t || "Set" === t ? Array.from(r) : "Arguments" === t || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(t) ? _arrayLikeToArray$7(r, a) : void 0;
  }
}
__name(_unsupportedIterableToArray$7, "_unsupportedIterableToArray$7");
function _iterableToArray$5(r) {
  if ("undefined" != typeof Symbol && null != r[Symbol.iterator] || null != r["@@iterator"]) return Array.from(r);
}
__name(_iterableToArray$5, "_iterableToArray$5");
function _arrayWithoutHoles$5(r) {
  if (Array.isArray(r)) return _arrayLikeToArray$7(r);
}
__name(_arrayWithoutHoles$5, "_arrayWithoutHoles$5");
function _arrayLikeToArray$7(r, a) {
  (null == a || a > r.length) && (a = r.length);
  for (var e = 0, n = Array(a); e < a; e++) n[e] = r[e];
  return n;
}
__name(_arrayLikeToArray$7, "_arrayLikeToArray$7");
var script$g = {
  name: "SelectButton",
  "extends": script$1$d,
  inheritAttrs: false,
  emits: ["update:modelValue", "change"],
  methods: {
    getOptionLabel: /* @__PURE__ */ __name(function getOptionLabel2(option2) {
      return this.optionLabel ? resolveFieldData(option2, this.optionLabel) : option2;
    }, "getOptionLabel"),
    getOptionValue: /* @__PURE__ */ __name(function getOptionValue2(option2) {
      return this.optionValue ? resolveFieldData(option2, this.optionValue) : option2;
    }, "getOptionValue"),
    getOptionRenderKey: /* @__PURE__ */ __name(function getOptionRenderKey2(option2) {
      return this.dataKey ? resolveFieldData(option2, this.dataKey) : this.getOptionLabel(option2);
    }, "getOptionRenderKey"),
    getPTOptions: /* @__PURE__ */ __name(function getPTOptions5(option2, key) {
      return this.ptm(key, {
        context: {
          active: this.isSelected(option2),
          disabled: this.isOptionDisabled(option2),
          option: option2
        }
      });
    }, "getPTOptions"),
    isOptionDisabled: /* @__PURE__ */ __name(function isOptionDisabled2(option2) {
      return this.optionDisabled ? resolveFieldData(option2, this.optionDisabled) : false;
    }, "isOptionDisabled"),
    onOptionSelect: /* @__PURE__ */ __name(function onOptionSelect2(event, option2, index2) {
      var _this = this;
      if (this.disabled || this.isOptionDisabled(option2)) {
        return;
      }
      var selected2 = this.isSelected(option2);
      if (selected2 && !this.allowEmpty) {
        return;
      }
      var optionValue = this.getOptionValue(option2);
      var newValue;
      if (this.multiple) {
        if (selected2) newValue = this.modelValue.filter(function(val) {
          return !equals(val, optionValue, _this.equalityKey);
        });
        else newValue = this.modelValue ? [].concat(_toConsumableArray$5(this.modelValue), [optionValue]) : [optionValue];
      } else {
        newValue = selected2 ? null : optionValue;
      }
      this.focusedIndex = index2;
      this.$emit("update:modelValue", newValue);
      this.$emit("change", {
        event,
        value: newValue
      });
    }, "onOptionSelect"),
    isSelected: /* @__PURE__ */ __name(function isSelected2(option2) {
      var selected2 = false;
      var optionValue = this.getOptionValue(option2);
      if (this.multiple) {
        if (this.modelValue) {
          var _iterator = _createForOfIteratorHelper$4(this.modelValue), _step;
          try {
            for (_iterator.s(); !(_step = _iterator.n()).done; ) {
              var val = _step.value;
              if (equals(val, optionValue, this.equalityKey)) {
                selected2 = true;
                break;
              }
            }
          } catch (err) {
            _iterator.e(err);
          } finally {
            _iterator.f();
          }
        }
      } else {
        selected2 = equals(this.modelValue, optionValue, this.equalityKey);
      }
      return selected2;
    }, "isSelected")
  },
  computed: {
    equalityKey: /* @__PURE__ */ __name(function equalityKey2() {
      return this.optionValue ? null : this.dataKey;
    }, "equalityKey")
  },
  directives: {
    ripple: Ripple
  },
  components: {
    ToggleButton: script$h
  }
};
var _hoisted_1$A = ["aria-labelledby"];
function render$h(_ctx, _cache, $props, $setup, $data, $options) {
  var _component_ToggleButton = resolveComponent("ToggleButton");
  return openBlock(), createElementBlock("div", mergeProps({
    "class": _ctx.cx("root"),
    role: "group",
    "aria-labelledby": _ctx.ariaLabelledby
  }, _ctx.ptmi("root")), [(openBlock(true), createElementBlock(Fragment, null, renderList(_ctx.options, function(option2, index2) {
    return openBlock(), createBlock(_component_ToggleButton, {
      key: $options.getOptionRenderKey(option2),
      modelValue: $options.isSelected(option2),
      onLabel: $options.getOptionLabel(option2),
      offLabel: $options.getOptionLabel(option2),
      disabled: _ctx.disabled || $options.isOptionDisabled(option2),
      unstyled: _ctx.unstyled,
      onChange: /* @__PURE__ */ __name(function onChange3($event) {
        return $options.onOptionSelect($event, option2, index2);
      }, "onChange"),
      pt: _ctx.ptm("pcButton")
    }, createSlots({
      _: 2
    }, [_ctx.$slots.option ? {
      name: "default",
      fn: withCtx(function() {
        return [renderSlot(_ctx.$slots, "option", {
          option: option2,
          index: index2
        }, function() {
          return [createBaseVNode("span", mergeProps({
            ref_for: true
          }, _ctx.ptm("pcButton")["label"]), toDisplayString($options.getOptionLabel(option2)), 17)];
        })];
      }),
      key: "0"
    } : void 0]), 1032, ["modelValue", "onLabel", "offLabel", "disabled", "unstyled", "onChange", "pt"]);
  }), 128))], 16, _hoisted_1$A);
}
__name(render$h, "render$h");
script$g.render = render$h;
const _withScopeId$j = /* @__PURE__ */ __name((n) => (pushScopeId("data-v-e7b35fd9"), n = n(), popScopeId(), n), "_withScopeId$j");
const _hoisted_1$z = { class: "_content" };
const _hoisted_2$r = { class: "_footer" };
const _sfc_main$A = /* @__PURE__ */ defineComponent({
  __name: "NodeSearchFilter",
  emits: ["addFilter"],
  setup(__props, { emit: __emit }) {
    const filters = computed(() => nodeDefStore.nodeSearchService.nodeFilters);
    const selectedFilter = ref();
    const filterValues = computed(() => selectedFilter.value?.fuseSearch.data ?? []);
    const selectedFilterValue = ref("");
    const nodeDefStore = useNodeDefStore();
    onMounted(() => {
      selectedFilter.value = nodeDefStore.nodeSearchService.nodeFilters[0];
      updateSelectedFilterValue();
    });
    const emit = __emit;
    const updateSelectedFilterValue = /* @__PURE__ */ __name(() => {
      if (filterValues.value.includes(selectedFilterValue.value)) {
        return;
      }
      selectedFilterValue.value = filterValues.value[0];
    }, "updateSelectedFilterValue");
    const submit = /* @__PURE__ */ __name(() => {
      emit("addFilter", [
        selectedFilter.value,
        selectedFilterValue.value
      ]);
    }, "submit");
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock(Fragment, null, [
        createBaseVNode("div", _hoisted_1$z, [
          createVNode(unref(script$g), {
            class: "filter-type-select",
            modelValue: selectedFilter.value,
            "onUpdate:modelValue": _cache[0] || (_cache[0] = ($event) => selectedFilter.value = $event),
            options: filters.value,
            allowEmpty: false,
            optionLabel: "name",
            onChange: updateSelectedFilterValue
          }, null, 8, ["modelValue", "options"]),
          createVNode(unref(script$v), {
            class: "filter-value-select",
            modelValue: selectedFilterValue.value,
            "onUpdate:modelValue": _cache[1] || (_cache[1] = ($event) => selectedFilterValue.value = $event),
            options: filterValues.value,
            filter: ""
          }, null, 8, ["modelValue", "options"])
        ]),
        createBaseVNode("div", _hoisted_2$r, [
          createVNode(unref(script$o), {
            type: "button",
            label: _ctx.$t("add"),
            onClick: submit
          }, null, 8, ["label"])
        ])
      ], 64);
    };
  }
});
const NodeSearchFilter = /* @__PURE__ */ _export_sfc(_sfc_main$A, [["__scopeId", "data-v-e7b35fd9"]]);
const BOOKMARK_SETTING_ID = "Comfy.NodeLibrary.Bookmarks.V2";
const useNodeBookmarkStore = defineStore("nodeBookmark", () => {
  const settingStore = useSettingStore();
  const nodeDefStore = useNodeDefStore();
  const migrateLegacyBookmarks = /* @__PURE__ */ __name(() => {
    settingStore.get("Comfy.NodeLibrary.Bookmarks").forEach((bookmark) => {
      if (bookmark.endsWith("/")) {
        addBookmark(bookmark);
        return;
      }
      const category = bookmark.split("/").slice(0, -1).join("/");
      const displayName = bookmark.split("/").pop();
      const nodeDef = nodeDefStore.nodeDefsByDisplayName[displayName];
      if (!nodeDef) return;
      addBookmark(`${category === "" ? "" : category + "/"}${nodeDef.name}`);
    });
    settingStore.set("Comfy.NodeLibrary.Bookmarks", []);
  }, "migrateLegacyBookmarks");
  const bookmarks = computed(
    () => settingStore.get(BOOKMARK_SETTING_ID)
  );
  const bookmarksSet = computed(() => new Set(bookmarks.value));
  const bookmarkedRoot = computed(
    () => buildBookmarkTree(bookmarks.value)
  );
  const isBookmarked = /* @__PURE__ */ __name((node2) => bookmarksSet.value.has(node2.nodePath) || bookmarksSet.value.has(node2.name), "isBookmarked");
  const toggleBookmark = /* @__PURE__ */ __name((node2) => {
    if (isBookmarked(node2)) {
      deleteBookmark(node2.nodePath);
      deleteBookmark(node2.name);
    } else {
      addBookmark(node2.name);
    }
  }, "toggleBookmark");
  const buildBookmarkTree = /* @__PURE__ */ __name((bookmarks2) => {
    const bookmarkNodes = bookmarks2.map((bookmark) => {
      if (bookmark.endsWith("/")) return createDummyFolderNodeDef(bookmark);
      const parts = bookmark.split("/");
      const name = parts.pop();
      const category = parts.join("/");
      const srcNodeDef = nodeDefStore.nodeDefsByName[name];
      if (!srcNodeDef) {
        return null;
      }
      const nodeDef = _.clone(srcNodeDef);
      nodeDef.category = category;
      return nodeDef;
    }).filter((nodeDef) => nodeDef !== null);
    return buildNodeDefTree(bookmarkNodes);
  }, "buildBookmarkTree");
  const addBookmark = /* @__PURE__ */ __name((nodePath) => {
    settingStore.set(BOOKMARK_SETTING_ID, [...bookmarks.value, nodePath]);
  }, "addBookmark");
  const deleteBookmark = /* @__PURE__ */ __name((nodePath) => {
    settingStore.set(
      BOOKMARK_SETTING_ID,
      bookmarks.value.filter((b) => b !== nodePath)
    );
  }, "deleteBookmark");
  const addNewBookmarkFolder = /* @__PURE__ */ __name((parent) => {
    const parentPath = parent ? parent.nodePath : "";
    let newFolderPath = parentPath + "New Folder/";
    let suffix = 1;
    while (bookmarks.value.some((b) => b.startsWith(newFolderPath))) {
      newFolderPath = parentPath + `New Folder ${suffix}/`;
      suffix++;
    }
    addBookmark(newFolderPath);
    return newFolderPath;
  }, "addNewBookmarkFolder");
  const renameBookmarkFolder = /* @__PURE__ */ __name((folderNode, newName) => {
    if (!folderNode.isDummyFolder) {
      throw new Error("Cannot rename non-folder node");
    }
    const newNodePath = folderNode.category.split("/").slice(0, -1).concat(newName).join("/") + "/";
    if (newNodePath === folderNode.nodePath) {
      return;
    }
    if (bookmarks.value.some((b) => b.startsWith(newNodePath))) {
      throw new Error(`Folder name "${newNodePath}" already exists`);
    }
    settingStore.set(
      BOOKMARK_SETTING_ID,
      bookmarks.value.map(
        (b) => b.startsWith(folderNode.nodePath) ? b.replace(folderNode.nodePath, newNodePath) : b
      )
    );
    renameBookmarkCustomization(folderNode.nodePath, newNodePath);
  }, "renameBookmarkFolder");
  const deleteBookmarkFolder = /* @__PURE__ */ __name((folderNode) => {
    if (!folderNode.isDummyFolder) {
      throw new Error("Cannot delete non-folder node");
    }
    settingStore.set(
      BOOKMARK_SETTING_ID,
      bookmarks.value.filter(
        (b) => b !== folderNode.nodePath && !b.startsWith(folderNode.nodePath)
      )
    );
    deleteBookmarkCustomization(folderNode.nodePath);
  }, "deleteBookmarkFolder");
  const bookmarksCustomization = computed(() => settingStore.get("Comfy.NodeLibrary.BookmarksCustomization"));
  const updateBookmarkCustomization = /* @__PURE__ */ __name((nodePath, customization) => {
    const currentCustomization = bookmarksCustomization.value[nodePath] || {};
    const newCustomization = { ...currentCustomization, ...customization };
    if (newCustomization.icon === defaultBookmarkIcon) {
      delete newCustomization.icon;
    }
    if (newCustomization.color === defaultBookmarkColor) {
      delete newCustomization.color;
    }
    if (Object.keys(newCustomization).length === 0) {
      deleteBookmarkCustomization(nodePath);
    } else {
      settingStore.set("Comfy.NodeLibrary.BookmarksCustomization", {
        ...bookmarksCustomization.value,
        [nodePath]: newCustomization
      });
    }
  }, "updateBookmarkCustomization");
  const deleteBookmarkCustomization = /* @__PURE__ */ __name((nodePath) => {
    settingStore.set("Comfy.NodeLibrary.BookmarksCustomization", {
      ...bookmarksCustomization.value,
      [nodePath]: void 0
    });
  }, "deleteBookmarkCustomization");
  const renameBookmarkCustomization = /* @__PURE__ */ __name((oldNodePath, newNodePath) => {
    const updatedCustomization = { ...bookmarksCustomization.value };
    if (updatedCustomization[oldNodePath]) {
      updatedCustomization[newNodePath] = updatedCustomization[oldNodePath];
      delete updatedCustomization[oldNodePath];
    }
    settingStore.set(
      "Comfy.NodeLibrary.BookmarksCustomization",
      updatedCustomization
    );
  }, "renameBookmarkCustomization");
  const defaultBookmarkIcon = "pi-bookmark-fill";
  const defaultBookmarkColor = "#a1a1aa";
  return {
    bookmarks,
    bookmarkedRoot,
    isBookmarked,
    toggleBookmark,
    addBookmark,
    addNewBookmarkFolder,
    renameBookmarkFolder,
    deleteBookmarkFolder,
    bookmarksCustomization,
    updateBookmarkCustomization,
    deleteBookmarkCustomization,
    renameBookmarkCustomization,
    defaultBookmarkIcon,
    defaultBookmarkColor,
    migrateLegacyBookmarks
  };
});
const _withScopeId$i = /* @__PURE__ */ __name((n) => (pushScopeId("data-v-37f672ab"), n = n(), popScopeId(), n), "_withScopeId$i");
const _hoisted_1$y = { class: "option-container flex justify-between items-center px-2 py-0 cursor-pointer overflow-hidden w-full" };
const _hoisted_2$q = { class: "option-display-name font-semibold flex flex-col" };
const _hoisted_3$k = { key: 0 };
const _hoisted_4$d = /* @__PURE__ */ _withScopeId$i(() => /* @__PURE__ */ createBaseVNode("i", { class: "pi pi-bookmark-fill text-sm mr-1" }, null, -1));
const _hoisted_5$9 = [
  _hoisted_4$d
];
const _hoisted_6$7 = ["innerHTML"];
const _hoisted_7$4 = /* @__PURE__ */ _withScopeId$i(() => /* @__PURE__ */ createBaseVNode("span", null, " ", -1));
const _hoisted_8$3 = ["innerHTML"];
const _hoisted_9$3 = {
  key: 0,
  class: "option-category font-light text-sm text-gray-400 overflow-hidden text-ellipsis whitespace-nowrap"
};
const _hoisted_10$3 = { class: "option-badges" };
const _sfc_main$z = /* @__PURE__ */ defineComponent({
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
      return openBlock(), createElementBlock("div", _hoisted_1$y, [
        createBaseVNode("div", _hoisted_2$q, [
          createBaseVNode("div", null, [
            isBookmarked.value ? (openBlock(), createElementBlock("span", _hoisted_3$k, _hoisted_5$9)) : createCommentVNode("", true),
            createBaseVNode("span", {
              innerHTML: unref(highlightQuery)(_ctx.nodeDef.display_name, _ctx.currentQuery)
            }, null, 8, _hoisted_6$7),
            _hoisted_7$4,
            showIdName.value ? (openBlock(), createBlock(unref(script$w), {
              key: 1,
              severity: "secondary"
            }, {
              default: withCtx(() => [
                createBaseVNode("span", {
                  innerHTML: unref(highlightQuery)(_ctx.nodeDef.name, _ctx.currentQuery)
                }, null, 8, _hoisted_8$3)
              ]),
              _: 1
            })) : createCommentVNode("", true)
          ]),
          showCategory.value ? (openBlock(), createElementBlock("div", _hoisted_9$3, toDisplayString(_ctx.nodeDef.category.replaceAll("/", " > ")), 1)) : createCommentVNode("", true)
        ]),
        createBaseVNode("div", _hoisted_10$3, [
          _ctx.nodeDef.experimental ? (openBlock(), createBlock(unref(script$w), {
            key: 0,
            value: _ctx.$t("experimental"),
            severity: "primary"
          }, null, 8, ["value"])) : createCommentVNode("", true),
          _ctx.nodeDef.deprecated ? (openBlock(), createBlock(unref(script$w), {
            key: 1,
            value: _ctx.$t("deprecated"),
            severity: "danger"
          }, null, 8, ["value"])) : createCommentVNode("", true),
          showNodeFrequency.value && nodeFrequency.value > 0 ? (openBlock(), createBlock(unref(script$w), {
            key: 2,
            value: unref(formatNumberWithSuffix)(nodeFrequency.value, { roundToInt: true }),
            severity: "secondary"
          }, null, 8, ["value"])) : createCommentVNode("", true),
          _ctx.nodeDef.nodeSource.type !== unref(NodeSourceType).Unknown ? (openBlock(), createBlock(unref(script$u), {
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
const NodeSearchItem = /* @__PURE__ */ _export_sfc(_sfc_main$z, [["__scopeId", "data-v-37f672ab"]]);
const _withScopeId$h = /* @__PURE__ */ __name((n) => (pushScopeId("data-v-ff07c900"), n = n(), popScopeId(), n), "_withScopeId$h");
const _hoisted_1$x = { class: "_sb_node_preview" };
const _hoisted_2$p = { class: "_sb_table" };
const _hoisted_3$j = /* @__PURE__ */ _withScopeId$h(() => /* @__PURE__ */ createBaseVNode("div", { class: "_sb_dot headdot" }, null, -1));
const _hoisted_4$c = /* @__PURE__ */ _withScopeId$h(() => /* @__PURE__ */ createBaseVNode("div", { class: "_sb_preview_badge" }, "PREVIEW", -1));
const _hoisted_5$8 = { class: "_sb_col" };
const _hoisted_6$6 = { class: "_sb_col" };
const _hoisted_7$3 = /* @__PURE__ */ _withScopeId$h(() => /* @__PURE__ */ createBaseVNode("div", { class: "_sb_col middle-column" }, null, -1));
const _hoisted_8$2 = { class: "_sb_col" };
const _hoisted_9$2 = /* @__PURE__ */ _withScopeId$h(() => /* @__PURE__ */ createBaseVNode("div", { class: "_sb_col _sb_arrow" }, "◀", -1));
const _hoisted_10$2 = /* @__PURE__ */ _withScopeId$h(() => /* @__PURE__ */ createBaseVNode("div", { class: "_sb_col middle-column" }, null, -1));
const _hoisted_11$2 = /* @__PURE__ */ _withScopeId$h(() => /* @__PURE__ */ createBaseVNode("div", { class: "_sb_col _sb_arrow" }, "▶", -1));
const _sfc_main$y = /* @__PURE__ */ defineComponent({
  __name: "NodePreview",
  props: {
    nodeDef: {
      type: ComfyNodeDefImpl,
      required: true
    }
  },
  setup(__props) {
    const props = __props;
    const colors = getColorPalette()?.colors?.litegraph_base;
    const litegraphColors = colors ?? defaultColorPalette.colors.litegraph_base;
    const nodeDefStore = useNodeDefStore();
    const nodeDef = props.nodeDef;
    const allInputDefs = nodeDef.input.all;
    const allOutputDefs = nodeDef.output.all;
    const slotInputDefs = allInputDefs.filter(
      (input) => !nodeDefStore.inputIsWidget(input)
    );
    const widgetInputDefs = allInputDefs.filter(
      (input) => nodeDefStore.inputIsWidget(input)
    );
    const truncateDefaultValue = /* @__PURE__ */ __name((value, charLimit = 32) => {
      let stringValue;
      if (typeof value === "object" && value !== null) {
        stringValue = JSON.stringify(value);
      } else if (Array.isArray(value)) {
        stringValue = JSON.stringify(value);
      } else if (typeof value === "string") {
        stringValue = value;
      } else {
        stringValue = String(value);
      }
      return _.truncate(stringValue, { length: charLimit });
    }, "truncateDefaultValue");
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", _hoisted_1$x, [
        createBaseVNode("div", _hoisted_2$p, [
          createBaseVNode("div", {
            class: "node_header",
            style: normalizeStyle({
              backgroundColor: unref(litegraphColors).NODE_DEFAULT_COLOR,
              color: unref(litegraphColors).NODE_TITLE_COLOR
            })
          }, [
            _hoisted_3$j,
            createTextVNode(" " + toDisplayString(unref(nodeDef).display_name), 1)
          ], 4),
          _hoisted_4$c,
          (openBlock(true), createElementBlock(Fragment, null, renderList(unref(_).zip(unref(slotInputDefs), unref(allOutputDefs)), ([slotInput, slotOutput]) => {
            return openBlock(), createElementBlock("div", {
              class: "_sb_row slot_row",
              key: (slotInput?.name || "") + (slotOutput?.index.toString() || "")
            }, [
              createBaseVNode("div", _hoisted_5$8, [
                slotInput ? (openBlock(), createElementBlock("div", {
                  key: 0,
                  class: normalizeClass(["_sb_dot", slotInput.type])
                }, null, 2)) : createCommentVNode("", true)
              ]),
              createBaseVNode("div", _hoisted_6$6, toDisplayString(slotInput ? slotInput.name : ""), 1),
              _hoisted_7$3,
              createBaseVNode("div", {
                class: "_sb_col _sb_inherit",
                style: normalizeStyle({
                  color: unref(litegraphColors).NODE_TEXT_COLOR
                })
              }, toDisplayString(slotOutput ? slotOutput.name : ""), 5),
              createBaseVNode("div", _hoisted_8$2, [
                slotOutput ? (openBlock(), createElementBlock("div", {
                  key: 0,
                  class: normalizeClass(["_sb_dot", slotOutput.type])
                }, null, 2)) : createCommentVNode("", true)
              ])
            ]);
          }), 128)),
          (openBlock(true), createElementBlock(Fragment, null, renderList(unref(widgetInputDefs), (widgetInput) => {
            return openBlock(), createElementBlock("div", {
              class: "_sb_row _long_field",
              key: widgetInput.name
            }, [
              _hoisted_9$2,
              createBaseVNode("div", {
                class: "_sb_col",
                style: normalizeStyle({
                  color: unref(litegraphColors).WIDGET_SECONDARY_TEXT_COLOR
                })
              }, toDisplayString(widgetInput.name), 5),
              _hoisted_10$2,
              createBaseVNode("div", {
                class: "_sb_col _sb_inherit",
                style: normalizeStyle({ color: unref(litegraphColors).WIDGET_TEXT_COLOR })
              }, toDisplayString(truncateDefaultValue(widgetInput.default)), 5),
              _hoisted_11$2
            ]);
          }), 128))
        ]),
        unref(nodeDef).description ? (openBlock(), createElementBlock("div", {
          key: 0,
          class: "_sb_description",
          style: normalizeStyle({
            color: unref(litegraphColors).WIDGET_SECONDARY_TEXT_COLOR,
            backgroundColor: unref(litegraphColors).WIDGET_BGCOLOR
          })
        }, toDisplayString(unref(nodeDef).description), 5)) : createCommentVNode("", true)
      ]);
    };
  }
});
const NodePreview = /* @__PURE__ */ _export_sfc(_sfc_main$y, [["__scopeId", "data-v-ff07c900"]]);
const _withScopeId$g = /* @__PURE__ */ __name((n) => (pushScopeId("data-v-2d409367"), n = n(), popScopeId(), n), "_withScopeId$g");
const _hoisted_1$w = { class: "comfy-vue-node-search-container" };
const _hoisted_2$o = {
  key: 0,
  class: "comfy-vue-node-preview-container"
};
const _hoisted_3$i = /* @__PURE__ */ _withScopeId$g(() => /* @__PURE__ */ createBaseVNode("h3", null, "Add node filter condition", -1));
const _hoisted_4$b = { class: "_dialog-body" };
const _sfc_main$x = /* @__PURE__ */ defineComponent({
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
    const setHoverSuggestion = /* @__PURE__ */ __name((index2) => {
      if (index2 === -1) {
        hoveredSuggestion.value = null;
        return;
      }
      const value = suggestions2.value[index2];
      hoveredSuggestion.value = value;
    }, "setHoverSuggestion");
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", _hoisted_1$w, [
        enableNodePreview.value ? (openBlock(), createElementBlock("div", _hoisted_2$o, [
          hoveredSuggestion.value ? (openBlock(), createBlock(NodePreview, {
            nodeDef: hoveredSuggestion.value,
            key: hoveredSuggestion.value?.name || ""
          }, null, 8, ["nodeDef"])) : createCommentVNode("", true)
        ])) : createCommentVNode("", true),
        createVNode(unref(script$o), {
          icon: "pi pi-filter",
          severity: "secondary",
          class: "_filter-button",
          onClick: _cache[0] || (_cache[0] = ($event) => nodeSearchFilterVisible.value = true)
        }),
        createVNode(unref(script$x), {
          visible: nodeSearchFilterVisible.value,
          "onUpdate:visible": _cache[1] || (_cache[1] = ($event) => nodeSearchFilterVisible.value = $event),
          class: "_dialog"
        }, {
          header: withCtx(() => [
            _hoisted_3$i
          ]),
          default: withCtx(() => [
            createBaseVNode("div", _hoisted_4$b, [
              createVNode(NodeSearchFilter, { onAddFilter })
            ])
          ]),
          _: 1
        }, 8, ["visible"]),
        createVNode(_sfc_main$B, {
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
const NodeSearchBox = /* @__PURE__ */ _export_sfc(_sfc_main$x, [["__scopeId", "data-v-2d409367"]]);
class ConnectingLinkImpl {
  static {
    __name(this, "ConnectingLinkImpl");
  }
  node;
  slot;
  input;
  output;
  pos;
  constructor(node2, slot, input, output, pos) {
    this.node = node2;
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
const _sfc_main$w = /* @__PURE__ */ defineComponent({
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
      const node2 = app.addNodeOnGraph(nodeDef, { pos: getNewNodeLocation() });
      const eventDetail = triggerEvent.value.detail;
      if (eventDetail.subType === "empty-release") {
        eventDetail.linkReleaseContext.links.forEach((link) => {
          ConnectingLinkImpl.createFromPlainObject(link).connectTo(node2);
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
        createVNode(unref(script$x), {
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
const _sfc_main$v = /* @__PURE__ */ defineComponent({
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
      const node2 = app.canvas.node_over;
      if (!node2.widgets) return;
      const graphPos = app.canvas.graph_mouse;
      const x = graphPos[0] - node2.pos[0];
      const y = graphPos[1] - node2.pos[1];
      for (const w of node2.widgets) {
        let widgetWidth, widgetHeight;
        if (w.computeSize) {
          ;
          [widgetWidth, widgetHeight] = w.computeSize(node2.size[0]);
        } else {
          widgetWidth = w.width || node2.size[0];
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
      const node2 = canvas.node_over;
      if (!node2) return;
      const ctor = node2.constructor;
      const nodeDef = nodeDefStore.nodeDefsByName[node2.type];
      if (ctor.title_mode !== LiteGraph.NO_TITLE && canvas.graph_mouse[1] < node2.pos[1]) {
        return showTooltip(nodeDef.description);
      }
      if (node2.flags?.collapsed) return;
      const inputSlot = canvas.isOverNodeInput(
        node2,
        canvas.graph_mouse[0],
        canvas.graph_mouse[1],
        [0, 0]
      );
      if (inputSlot !== -1) {
        const inputName = node2.inputs[inputSlot].name;
        return showTooltip(nodeDef.input.getInput(inputName)?.tooltip);
      }
      const outputSlot = canvas.isOverNodeOutput(
        node2,
        canvas.graph_mouse[0],
        canvas.graph_mouse[1],
        [0, 0]
      );
      if (outputSlot !== -1) {
        return showTooltip(nodeDef.output.all?.[outputSlot]?.tooltip);
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
const NodeTooltip = /* @__PURE__ */ _export_sfc(_sfc_main$v, [["__scopeId", "data-v-0a4402f9"]]);
function _arrayWithHoles(r) {
  if (Array.isArray(r)) return r;
}
__name(_arrayWithHoles, "_arrayWithHoles");
function _iterableToArrayLimit(r, l) {
  var t = null == r ? null : "undefined" != typeof Symbol && r[Symbol.iterator] || r["@@iterator"];
  if (null != t) {
    var e, n, i, u, a = [], f = true, o = false;
    try {
      if (i = (t = t.call(r)).next, 0 === l) {
        if (Object(t) !== t) return;
        f = false;
      } else for (; !(f = (e = i.call(t)).done) && (a.push(e.value), a.length !== l); f = true) ;
    } catch (r2) {
      o = true, n = r2;
    } finally {
      try {
        if (!f && null != t["return"] && (u = t["return"](), Object(u) !== u)) return;
      } finally {
        if (o) throw n;
      }
    }
    return a;
  }
}
__name(_iterableToArrayLimit, "_iterableToArrayLimit");
function _arrayLikeToArray$6(r, a) {
  (null == a || a > r.length) && (a = r.length);
  for (var e = 0, n = Array(a); e < a; e++) n[e] = r[e];
  return n;
}
__name(_arrayLikeToArray$6, "_arrayLikeToArray$6");
function _unsupportedIterableToArray$6(r, a) {
  if (r) {
    if ("string" == typeof r) return _arrayLikeToArray$6(r, a);
    var t = {}.toString.call(r).slice(8, -1);
    return "Object" === t && r.constructor && (t = r.constructor.name), "Map" === t || "Set" === t ? Array.from(r) : "Arguments" === t || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(t) ? _arrayLikeToArray$6(r, a) : void 0;
  }
}
__name(_unsupportedIterableToArray$6, "_unsupportedIterableToArray$6");
function _nonIterableRest() {
  throw new TypeError("Invalid attempt to destructure non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.");
}
__name(_nonIterableRest, "_nonIterableRest");
function _slicedToArray(r, e) {
  return _arrayWithHoles(r) || _iterableToArrayLimit(r, e) || _unsupportedIterableToArray$6(r, e) || _nonIterableRest();
}
__name(_slicedToArray, "_slicedToArray");
var dist = {};
var bind$1 = {};
"use strict";
Object.defineProperty(bind$1, "__esModule", { value: true });
var bind_2 = bind$1.bind = void 0;
function bind(target, _a) {
  var type = _a.type, listener = _a.listener, options = _a.options;
  target.addEventListener(type, listener, options);
  return /* @__PURE__ */ __name(function unbind() {
    target.removeEventListener(type, listener, options);
  }, "unbind");
}
__name(bind, "bind");
bind_2 = bind$1.bind = bind;
var bindAll$1 = {};
"use strict";
var __assign = commonjsGlobal && commonjsGlobal.__assign || function() {
  __assign = Object.assign || function(t) {
    for (var s, i = 1, n = arguments.length; i < n; i++) {
      s = arguments[i];
      for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p))
        t[p] = s[p];
    }
    return t;
  };
  return __assign.apply(this, arguments);
};
Object.defineProperty(bindAll$1, "__esModule", { value: true });
var bindAll_2 = bindAll$1.bindAll = void 0;
var bind_1 = bind$1;
function toOptions(value) {
  if (typeof value === "undefined") {
    return void 0;
  }
  if (typeof value === "boolean") {
    return {
      capture: value
    };
  }
  return value;
}
__name(toOptions, "toOptions");
function getBinding(original, sharedOptions) {
  if (sharedOptions == null) {
    return original;
  }
  var binding = __assign(__assign({}, original), { options: __assign(__assign({}, toOptions(sharedOptions)), toOptions(original.options)) });
  return binding;
}
__name(getBinding, "getBinding");
function bindAll(target, bindings, sharedOptions) {
  var unbinds = bindings.map(function(original) {
    var binding = getBinding(original, sharedOptions);
    return (0, bind_1.bind)(target, binding);
  });
  return /* @__PURE__ */ __name(function unbindAll() {
    unbinds.forEach(function(unbind) {
      return unbind();
    });
  }, "unbindAll");
}
__name(bindAll, "bindAll");
bindAll_2 = bindAll$1.bindAll = bindAll;
(function(exports) {
  "use strict";
  Object.defineProperty(exports, "__esModule", { value: true });
  exports.bindAll = exports.bind = void 0;
  var bind_12 = bind$1;
  Object.defineProperty(exports, "bind", { enumerable: true, get: /* @__PURE__ */ __name(function() {
    return bind_12.bind;
  }, "get") });
  var bind_all_1 = bindAll$1;
  Object.defineProperty(exports, "bindAll", { enumerable: true, get: /* @__PURE__ */ __name(function() {
    return bind_all_1.bindAll;
  }, "get") });
})(dist);
const index = /* @__PURE__ */ getDefaultExportFromCjs(dist);
var honeyPotDataAttribute = "data-pdnd-honey-pot";
function isHoneyPotElement(target) {
  return target instanceof Element && target.hasAttribute(honeyPotDataAttribute);
}
__name(isHoneyPotElement, "isHoneyPotElement");
function getElementFromPointWithoutHoneypot(client) {
  var _document$elementsFro = document.elementsFromPoint(client.x, client.y), _document$elementsFro2 = _slicedToArray(_document$elementsFro, 2), top = _document$elementsFro2[0], second = _document$elementsFro2[1];
  if (!top) {
    return null;
  }
  if (isHoneyPotElement(top)) {
    return second !== null && second !== void 0 ? second : null;
  }
  return top;
}
__name(getElementFromPointWithoutHoneypot, "getElementFromPointWithoutHoneypot");
function _typeof$6(o) {
  "@babel/helpers - typeof";
  return _typeof$6 = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function(o2) {
    return typeof o2;
  } : function(o2) {
    return o2 && "function" == typeof Symbol && o2.constructor === Symbol && o2 !== Symbol.prototype ? "symbol" : typeof o2;
  }, _typeof$6(o);
}
__name(_typeof$6, "_typeof$6");
function toPrimitive(t, r) {
  if ("object" != _typeof$6(t) || !t) return t;
  var e = t[Symbol.toPrimitive];
  if (void 0 !== e) {
    var i = e.call(t, r || "default");
    if ("object" != _typeof$6(i)) return i;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return ("string" === r ? String : Number)(t);
}
__name(toPrimitive, "toPrimitive");
function toPropertyKey(t) {
  var i = toPrimitive(t, "string");
  return "symbol" == _typeof$6(i) ? i : i + "";
}
__name(toPropertyKey, "toPropertyKey");
function _defineProperty$6(e, r, t) {
  return (r = toPropertyKey(r)) in e ? Object.defineProperty(e, r, {
    value: t,
    enumerable: true,
    configurable: true,
    writable: true
  }) : e[r] = t, e;
}
__name(_defineProperty$6, "_defineProperty$6");
var maxZIndex = 2147483647;
function ownKeys$7(e, r) {
  var t = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var o = Object.getOwnPropertySymbols(e);
    r && (o = o.filter(function(r2) {
      return Object.getOwnPropertyDescriptor(e, r2).enumerable;
    })), t.push.apply(t, o);
  }
  return t;
}
__name(ownKeys$7, "ownKeys$7");
function _objectSpread$7(e) {
  for (var r = 1; r < arguments.length; r++) {
    var t = null != arguments[r] ? arguments[r] : {};
    r % 2 ? ownKeys$7(Object(t), true).forEach(function(r2) {
      _defineProperty$6(e, r2, t[r2]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(t)) : ownKeys$7(Object(t)).forEach(function(r2) {
      Object.defineProperty(e, r2, Object.getOwnPropertyDescriptor(t, r2));
    });
  }
  return e;
}
__name(_objectSpread$7, "_objectSpread$7");
var honeyPotSize = 2;
var halfHoneyPotSize = honeyPotSize / 2;
function floorToClosestPixel(point) {
  return {
    x: Math.floor(point.x),
    y: Math.floor(point.y)
  };
}
__name(floorToClosestPixel, "floorToClosestPixel");
function pullBackByHalfHoneyPotSize(point) {
  return {
    x: point.x - halfHoneyPotSize,
    y: point.y - halfHoneyPotSize
  };
}
__name(pullBackByHalfHoneyPotSize, "pullBackByHalfHoneyPotSize");
function preventGoingBackwardsOffScreen(point) {
  return {
    x: Math.max(point.x, 0),
    y: Math.max(point.y, 0)
  };
}
__name(preventGoingBackwardsOffScreen, "preventGoingBackwardsOffScreen");
function preventGoingForwardsOffScreen(point) {
  return {
    x: Math.min(point.x, window.innerWidth - honeyPotSize),
    y: Math.min(point.y, window.innerHeight - honeyPotSize)
  };
}
__name(preventGoingForwardsOffScreen, "preventGoingForwardsOffScreen");
function getHoneyPotRectFor(_ref) {
  var client = _ref.client;
  var point = preventGoingForwardsOffScreen(preventGoingBackwardsOffScreen(pullBackByHalfHoneyPotSize(floorToClosestPixel(client))));
  return DOMRect.fromRect({
    x: point.x,
    y: point.y,
    width: honeyPotSize,
    height: honeyPotSize
  });
}
__name(getHoneyPotRectFor, "getHoneyPotRectFor");
function getRectStyles(_ref2) {
  var clientRect = _ref2.clientRect;
  return {
    left: "".concat(clientRect.left, "px"),
    top: "".concat(clientRect.top, "px"),
    width: "".concat(clientRect.width, "px"),
    height: "".concat(clientRect.height, "px")
  };
}
__name(getRectStyles, "getRectStyles");
function isWithin(_ref3) {
  var client = _ref3.client, clientRect = _ref3.clientRect;
  return (
    // is within horizontal bounds
    client.x >= clientRect.x && client.x <= clientRect.x + clientRect.width && // is within vertical bounds
    client.y >= clientRect.y && client.y <= clientRect.y + clientRect.height
  );
}
__name(isWithin, "isWithin");
function mountHoneyPot(_ref4) {
  var initial = _ref4.initial;
  var element = document.createElement("div");
  element.setAttribute(honeyPotDataAttribute, "true");
  var clientRect = getHoneyPotRectFor({
    client: initial
  });
  Object.assign(element.style, _objectSpread$7(_objectSpread$7({
    // Setting a background color explicitly to avoid any inherited styles.
    // Looks like this could be `opacity: 0`, but worried that _might_
    // cause the element to be ignored on some platforms.
    // When debugging, set backgroundColor to something like "red".
    backgroundColor: "transparent",
    position: "fixed",
    // Being explicit to avoid inheriting styles
    padding: 0,
    margin: 0,
    boxSizing: "border-box"
  }, getRectStyles({
    clientRect
  })), {}, {
    // We want this element to absorb pointer events,
    // it's kind of the whole point 😉
    pointerEvents: "auto",
    // Want to make sure the honey pot is top of everything else.
    // Don't need to worry about native drag previews, as they will
    // have been rendered (and removed) before the honey pot is rendered
    zIndex: maxZIndex
  }));
  document.body.appendChild(element);
  var unbindPointerMove = dist.bind(window, {
    type: "pointermove",
    listener: /* @__PURE__ */ __name(function listener(event) {
      var client = {
        x: event.clientX,
        y: event.clientY
      };
      clientRect = getHoneyPotRectFor({
        client
      });
      Object.assign(element.style, getRectStyles({
        clientRect
      }));
    }, "listener"),
    // using capture so we are less likely to be impacted by event stopping
    options: {
      capture: true
    }
  });
  return /* @__PURE__ */ __name(function finish(_ref5) {
    var current = _ref5.current;
    unbindPointerMove();
    if (isWithin({
      client: current,
      clientRect
    })) {
      element.remove();
      return;
    }
    function cleanup() {
      unbindPostDragEvents();
      element.remove();
    }
    __name(cleanup, "cleanup");
    var unbindPostDragEvents = dist.bindAll(window, [
      {
        type: "pointerdown",
        listener: cleanup
      },
      {
        type: "pointermove",
        listener: cleanup
      },
      {
        type: "focusin",
        listener: cleanup
      },
      {
        type: "focusout",
        listener: cleanup
      },
      // a 'pointerdown' should happen before 'dragstart', but just being super safe
      {
        type: "dragstart",
        listener: cleanup
      },
      // if the user has dragged something out of the window
      // and then is dragging something back into the window
      // the first events we will see are "dragenter" (and then "dragover").
      // So if we see any of these we need to clear the post drag fix.
      {
        type: "dragenter",
        listener: cleanup
      },
      {
        type: "dragover",
        listener: cleanup
      }
      // Not adding a "wheel" event listener, as "wheel" by itself does not
      // resolve the bug.
    ], {
      // Using `capture` so less likely to be impacted by other code stopping events
      capture: true
    });
  }, "finish");
}
__name(mountHoneyPot, "mountHoneyPot");
function makeHoneyPotFix() {
  var latestPointerMove = null;
  function bindEvents() {
    latestPointerMove = null;
    return dist.bind(window, {
      type: "pointermove",
      listener: /* @__PURE__ */ __name(function listener(event) {
        latestPointerMove = {
          x: event.clientX,
          y: event.clientY
        };
      }, "listener"),
      // listening for pointer move in capture phase
      // so we are less likely to be impacted by events being stopped.
      options: {
        capture: true
      }
    });
  }
  __name(bindEvents, "bindEvents");
  function getOnPostDispatch() {
    var finish = null;
    return /* @__PURE__ */ __name(function onPostEvent(_ref6) {
      var eventName = _ref6.eventName, payload = _ref6.payload;
      if (eventName === "onDragStart") {
        var _latestPointerMove;
        var input = payload.location.initial.input;
        var initial = (_latestPointerMove = latestPointerMove) !== null && _latestPointerMove !== void 0 ? _latestPointerMove : {
          x: input.clientX,
          y: input.clientY
        };
        finish = mountHoneyPot({
          initial
        });
      }
      if (eventName === "onDrop") {
        var _finish;
        var _input = payload.location.current.input;
        (_finish = finish) === null || _finish === void 0 || _finish({
          current: {
            x: _input.clientX,
            y: _input.clientY
          }
        });
        finish = null;
        latestPointerMove = null;
      }
    }, "onPostEvent");
  }
  __name(getOnPostDispatch, "getOnPostDispatch");
  return {
    bindEvents,
    getOnPostDispatch
  };
}
__name(makeHoneyPotFix, "makeHoneyPotFix");
function _arrayWithoutHoles$4(r) {
  if (Array.isArray(r)) return _arrayLikeToArray$6(r);
}
__name(_arrayWithoutHoles$4, "_arrayWithoutHoles$4");
function _iterableToArray$4(r) {
  if ("undefined" != typeof Symbol && null != r[Symbol.iterator] || null != r["@@iterator"]) return Array.from(r);
}
__name(_iterableToArray$4, "_iterableToArray$4");
function _nonIterableSpread$4() {
  throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.");
}
__name(_nonIterableSpread$4, "_nonIterableSpread$4");
function _toConsumableArray$4(r) {
  return _arrayWithoutHoles$4(r) || _iterableToArray$4(r) || _unsupportedIterableToArray$6(r) || _nonIterableSpread$4();
}
__name(_toConsumableArray$4, "_toConsumableArray$4");
function once(fn) {
  var cache = null;
  return /* @__PURE__ */ __name(function wrapped() {
    if (!cache) {
      for (var _len = arguments.length, args = new Array(_len), _key = 0; _key < _len; _key++) {
        args[_key] = arguments[_key];
      }
      var result = fn.apply(this, args);
      cache = {
        result
      };
    }
    return cache.result;
  }, "wrapped");
}
__name(once, "once");
var isFirefox = once(/* @__PURE__ */ __name(function isFirefox2() {
  if (false) {
    return false;
  }
  return navigator.userAgent.includes("Firefox");
}, "isFirefox2"));
var isSafari = once(/* @__PURE__ */ __name(function isSafari2() {
  if (false) {
    return false;
  }
  var _navigator = navigator, userAgent = _navigator.userAgent;
  return userAgent.includes("AppleWebKit") && !userAgent.includes("Chrome");
}, "isSafari2"));
var symbols = {
  isLeavingWindow: Symbol("leaving"),
  isEnteringWindow: Symbol("entering")
};
function isEnteringWindowInSafari(_ref) {
  var dragEnter = _ref.dragEnter;
  if (!isSafari()) {
    return false;
  }
  return dragEnter.hasOwnProperty(symbols.isEnteringWindow);
}
__name(isEnteringWindowInSafari, "isEnteringWindowInSafari");
function isLeavingWindowInSafari(_ref2) {
  var dragLeave = _ref2.dragLeave;
  if (!isSafari()) {
    return false;
  }
  return dragLeave.hasOwnProperty(symbols.isLeavingWindow);
}
__name(isLeavingWindowInSafari, "isLeavingWindowInSafari");
(/* @__PURE__ */ __name(function fixSafari() {
  if (typeof window === "undefined") {
    return;
  }
  if (false) {
    return;
  }
  if (!isSafari()) {
    return;
  }
  function getInitialState() {
    return {
      enterCount: 0,
      isOverWindow: false
    };
  }
  __name(getInitialState, "getInitialState");
  var state = getInitialState();
  function resetState() {
    state = getInitialState();
  }
  __name(resetState, "resetState");
  dist.bindAll(
    window,
    [{
      type: "dragstart",
      listener: /* @__PURE__ */ __name(function listener() {
        state.enterCount = 0;
        state.isOverWindow = true;
      }, "listener")
    }, {
      type: "drop",
      listener: resetState
    }, {
      type: "dragend",
      listener: resetState
    }, {
      type: "dragenter",
      listener: /* @__PURE__ */ __name(function listener(event) {
        if (!state.isOverWindow && state.enterCount === 0) {
          event[symbols.isEnteringWindow] = true;
        }
        state.isOverWindow = true;
        state.enterCount++;
      }, "listener")
    }, {
      type: "dragleave",
      listener: /* @__PURE__ */ __name(function listener(event) {
        state.enterCount--;
        if (state.isOverWindow && state.enterCount === 0) {
          event[symbols.isLeavingWindow] = true;
          state.isOverWindow = false;
        }
      }, "listener")
    }],
    // using `capture: true` so that adding event listeners
    // in bubble phase will have the correct symbols
    {
      capture: true
    }
  );
}, "fixSafari"))();
function isNodeLike(target) {
  return "nodeName" in target;
}
__name(isNodeLike, "isNodeLike");
function isFromAnotherWindow(eventTarget) {
  return isNodeLike(eventTarget) && eventTarget.ownerDocument !== document;
}
__name(isFromAnotherWindow, "isFromAnotherWindow");
function isLeavingWindow(_ref) {
  var dragLeave = _ref.dragLeave;
  var type = dragLeave.type, relatedTarget = dragLeave.relatedTarget;
  if (type !== "dragleave") {
    return false;
  }
  if (isSafari()) {
    return isLeavingWindowInSafari({
      dragLeave
    });
  }
  if (relatedTarget == null) {
    return true;
  }
  if (isFirefox()) {
    return isFromAnotherWindow(relatedTarget);
  }
  return relatedTarget instanceof HTMLIFrameElement;
}
__name(isLeavingWindow, "isLeavingWindow");
function getBindingsForBrokenDrags(_ref) {
  var onDragEnd2 = _ref.onDragEnd;
  return [
    // ## Detecting drag ending for removed draggables
    //
    // If a draggable element is removed during a drag and the user drops:
    // 1. if over a valid drop target: we get a "drop" event to know the drag is finished
    // 2. if not over a valid drop target (or cancelled): we get nothing
    // The "dragend" event will not fire on the source draggable if it has been
    // removed from the DOM.
    // So we need to figure out if a drag operation has finished by looking at other events
    // We can do this by looking at other events
    // ### First detection: "pointermove" events
    // 1. "pointermove" events cannot fire during a drag and drop operation
    // according to the spec. So if we get a "pointermove" it means that
    // the drag and drop operations has finished. So if we get a "pointermove"
    // we know that the drag is over
    // 2. 🦊😤 Drag and drop operations are _supposed_ to suppress
    // other pointer events. However, firefox will allow a few
    // pointer event to get through after a drag starts.
    // The most I've seen is 3
    {
      type: "pointermove",
      listener: /* @__PURE__ */ function() {
        var callCount = 0;
        return /* @__PURE__ */ __name(function listener() {
          if (callCount < 20) {
            callCount++;
            return;
          }
          onDragEnd2();
        }, "listener");
      }()
    },
    // ### Second detection: "pointerdown" events
    // If we receive this event then we know that a drag operation has finished
    // and potentially another one is about to start.
    // Note: `pointerdown` fires on all browsers / platforms before "dragstart"
    {
      type: "pointerdown",
      listener: onDragEnd2
    }
  ];
}
__name(getBindingsForBrokenDrags, "getBindingsForBrokenDrags");
function getInput(event) {
  return {
    altKey: event.altKey,
    button: event.button,
    buttons: event.buttons,
    ctrlKey: event.ctrlKey,
    metaKey: event.metaKey,
    shiftKey: event.shiftKey,
    clientX: event.clientX,
    clientY: event.clientY,
    pageX: event.pageX,
    pageY: event.pageY
  };
}
__name(getInput, "getInput");
var rafSchd = /* @__PURE__ */ __name(function rafSchd2(fn) {
  var lastArgs = [];
  var frameId = null;
  var wrapperFn = /* @__PURE__ */ __name(function wrapperFn2() {
    for (var _len = arguments.length, args = new Array(_len), _key = 0; _key < _len; _key++) {
      args[_key] = arguments[_key];
    }
    lastArgs = args;
    if (frameId) {
      return;
    }
    frameId = requestAnimationFrame(function() {
      frameId = null;
      fn.apply(void 0, lastArgs);
    });
  }, "wrapperFn");
  wrapperFn.cancel = function() {
    if (!frameId) {
      return;
    }
    cancelAnimationFrame(frameId);
    frameId = null;
  };
  return wrapperFn;
}, "rafSchd");
var scheduleOnDrag = rafSchd(function(fn) {
  return fn();
});
var dragStart = /* @__PURE__ */ function() {
  var scheduled = null;
  function schedule(fn) {
    var frameId = requestAnimationFrame(function() {
      scheduled = null;
      fn();
    });
    scheduled = {
      frameId,
      fn
    };
  }
  __name(schedule, "schedule");
  function flush() {
    if (scheduled) {
      cancelAnimationFrame(scheduled.frameId);
      scheduled.fn();
      scheduled = null;
    }
  }
  __name(flush, "flush");
  return {
    schedule,
    flush
  };
}();
function makeDispatch(_ref) {
  var source = _ref.source, initial = _ref.initial, dispatchEvent = _ref.dispatchEvent;
  var previous = {
    dropTargets: []
  };
  function safeDispatch(args) {
    dispatchEvent(args);
    previous = {
      dropTargets: args.payload.location.current.dropTargets
    };
  }
  __name(safeDispatch, "safeDispatch");
  var dispatch = {
    start: /* @__PURE__ */ __name(function start2(_ref2) {
      var nativeSetDragImage = _ref2.nativeSetDragImage;
      var location = {
        current: initial,
        previous,
        initial
      };
      safeDispatch({
        eventName: "onGenerateDragPreview",
        payload: {
          source,
          location,
          nativeSetDragImage
        }
      });
      dragStart.schedule(function() {
        safeDispatch({
          eventName: "onDragStart",
          payload: {
            source,
            location
          }
        });
      });
    }, "start"),
    dragUpdate: /* @__PURE__ */ __name(function dragUpdate(_ref3) {
      var current = _ref3.current;
      dragStart.flush();
      scheduleOnDrag.cancel();
      safeDispatch({
        eventName: "onDropTargetChange",
        payload: {
          source,
          location: {
            initial,
            previous,
            current
          }
        }
      });
    }, "dragUpdate"),
    drag: /* @__PURE__ */ __name(function drag(_ref4) {
      var current = _ref4.current;
      scheduleOnDrag(function() {
        dragStart.flush();
        var location = {
          initial,
          previous,
          current
        };
        safeDispatch({
          eventName: "onDrag",
          payload: {
            source,
            location
          }
        });
      });
    }, "drag"),
    drop: /* @__PURE__ */ __name(function drop(_ref5) {
      var current = _ref5.current, updatedSourcePayload = _ref5.updatedSourcePayload;
      dragStart.flush();
      scheduleOnDrag.cancel();
      safeDispatch({
        eventName: "onDrop",
        payload: {
          source: updatedSourcePayload !== null && updatedSourcePayload !== void 0 ? updatedSourcePayload : source,
          location: {
            current,
            previous,
            initial
          }
        }
      });
    }, "drop")
  };
  return dispatch;
}
__name(makeDispatch, "makeDispatch");
var globalState = {
  isActive: false
};
function canStart() {
  return !globalState.isActive;
}
__name(canStart, "canStart");
function getNativeSetDragImage(event) {
  if (event.dataTransfer) {
    return event.dataTransfer.setDragImage.bind(event.dataTransfer);
  }
  return null;
}
__name(getNativeSetDragImage, "getNativeSetDragImage");
function hasHierarchyChanged(_ref) {
  var current = _ref.current, next2 = _ref.next;
  if (current.length !== next2.length) {
    return true;
  }
  for (var i = 0; i < current.length; i++) {
    if (current[i].element !== next2[i].element) {
      return true;
    }
  }
  return false;
}
__name(hasHierarchyChanged, "hasHierarchyChanged");
function start(_ref2) {
  var event = _ref2.event, dragType = _ref2.dragType, getDropTargetsOver = _ref2.getDropTargetsOver, dispatchEvent = _ref2.dispatchEvent;
  if (!canStart()) {
    return;
  }
  var initial = getStartLocation({
    event,
    dragType,
    getDropTargetsOver
  });
  globalState.isActive = true;
  var state = {
    current: initial
  };
  setDropEffectOnEvent({
    event,
    current: initial.dropTargets
  });
  var dispatch = makeDispatch({
    source: dragType.payload,
    dispatchEvent,
    initial
  });
  function updateState(next2) {
    var hasChanged = hasHierarchyChanged({
      current: state.current.dropTargets,
      next: next2.dropTargets
    });
    state.current = next2;
    if (hasChanged) {
      dispatch.dragUpdate({
        current: state.current
      });
    }
  }
  __name(updateState, "updateState");
  function onUpdateEvent(event2) {
    var input = getInput(event2);
    var target = isHoneyPotElement(event2.target) ? getElementFromPointWithoutHoneypot({
      x: input.clientX,
      y: input.clientY
    }) : event2.target;
    var nextDropTargets = getDropTargetsOver({
      target,
      input,
      source: dragType.payload,
      current: state.current.dropTargets
    });
    if (nextDropTargets.length) {
      event2.preventDefault();
      setDropEffectOnEvent({
        event: event2,
        current: nextDropTargets
      });
    }
    updateState({
      dropTargets: nextDropTargets,
      input
    });
  }
  __name(onUpdateEvent, "onUpdateEvent");
  function cancel() {
    if (state.current.dropTargets.length) {
      updateState({
        dropTargets: [],
        input: state.current.input
      });
    }
    dispatch.drop({
      current: state.current,
      updatedSourcePayload: null
    });
    finish();
  }
  __name(cancel, "cancel");
  function finish() {
    globalState.isActive = false;
    unbindEvents();
  }
  __name(finish, "finish");
  var unbindEvents = dist.bindAll(
    window,
    [{
      // 👋 Note: we are repurposing the `dragover` event as our `drag` event
      // this is because firefox does not publish pointer coordinates during
      // a `drag` event, but does for every other type of drag event
      // `dragover` fires on all elements that are being dragged over
      // Because we are binding to `window` - our `dragover` is effectively the same as a `drag`
      // 🦊😤
      type: "dragover",
      listener: /* @__PURE__ */ __name(function listener(event2) {
        onUpdateEvent(event2);
        dispatch.drag({
          current: state.current
        });
      }, "listener")
    }, {
      type: "dragenter",
      listener: onUpdateEvent
    }, {
      type: "dragleave",
      listener: /* @__PURE__ */ __name(function listener(event2) {
        if (!isLeavingWindow({
          dragLeave: event2
        })) {
          return;
        }
        updateState({
          input: state.current.input,
          dropTargets: []
        });
        if (dragType.startedFrom === "external") {
          cancel();
        }
      }, "listener")
    }, {
      // A "drop" can only happen if the browser allowed the drop
      type: "drop",
      listener: /* @__PURE__ */ __name(function listener(event2) {
        state.current = {
          dropTargets: state.current.dropTargets,
          input: getInput(event2)
        };
        if (!state.current.dropTargets.length) {
          cancel();
          return;
        }
        event2.preventDefault();
        setDropEffectOnEvent({
          event: event2,
          current: state.current.dropTargets
        });
        dispatch.drop({
          current: state.current,
          // When dropping something native, we need to extract the latest
          // `.items` from the "drop" event as it is now accessible
          updatedSourcePayload: dragType.type === "external" ? dragType.getDropPayload(event2) : null
        });
        finish();
      }, "listener")
    }, {
      // "dragend" fires when on the drag source (eg a draggable element)
      // when the drag is finished.
      // "dragend" will fire after "drop" (if there was a successful drop)
      // "dragend" does not fire if the draggable source has been removed during the drag
      // or for external drag sources (eg files)
      // This "dragend" listener will not fire if there was a successful drop
      // as we will have already removed the event listener
      type: "dragend",
      listener: /* @__PURE__ */ __name(function listener(event2) {
        state.current = {
          dropTargets: state.current.dropTargets,
          input: getInput(event2)
        };
        cancel();
      }, "listener")
    }].concat(_toConsumableArray$4(getBindingsForBrokenDrags({
      onDragEnd: cancel
    }))),
    // Once we have started a managed drag operation it is important that we see / own all drag events
    // We got one adoption bug pop up where some code was stopping (`event.stopPropagation()`)
    // all "drop" events in the bubble phase on the `document.body`.
    // This meant that we never saw the "drop" event.
    {
      capture: true
    }
  );
  dispatch.start({
    nativeSetDragImage: getNativeSetDragImage(event)
  });
}
__name(start, "start");
function setDropEffectOnEvent(_ref3) {
  var _current$;
  var event = _ref3.event, current = _ref3.current;
  var innerMost = (_current$ = current[0]) === null || _current$ === void 0 ? void 0 : _current$.dropEffect;
  if (innerMost != null && event.dataTransfer) {
    event.dataTransfer.dropEffect = innerMost;
  }
}
__name(setDropEffectOnEvent, "setDropEffectOnEvent");
function getStartLocation(_ref4) {
  var event = _ref4.event, dragType = _ref4.dragType, getDropTargetsOver = _ref4.getDropTargetsOver;
  var input = getInput(event);
  if (dragType.startedFrom === "external") {
    return {
      input,
      dropTargets: []
    };
  }
  var dropTargets = getDropTargetsOver({
    input,
    source: dragType.payload,
    target: event.target,
    current: []
  });
  return {
    input,
    dropTargets
  };
}
__name(getStartLocation, "getStartLocation");
var lifecycle = {
  canStart,
  start
};
var ledger = /* @__PURE__ */ new Map();
function registerUsage(_ref) {
  var typeKey = _ref.typeKey, mount2 = _ref.mount;
  var entry = ledger.get(typeKey);
  if (entry) {
    entry.usageCount++;
    return entry;
  }
  var initial = {
    typeKey,
    unmount: mount2(),
    usageCount: 1
  };
  ledger.set(typeKey, initial);
  return initial;
}
__name(registerUsage, "registerUsage");
function register(args) {
  var entry = registerUsage(args);
  return /* @__PURE__ */ __name(function unregister() {
    entry.usageCount--;
    if (entry.usageCount > 0) {
      return;
    }
    entry.unmount();
    ledger.delete(args.typeKey);
  }, "unregister");
}
__name(register, "register");
function combine() {
  for (var _len = arguments.length, fns = new Array(_len), _key = 0; _key < _len; _key++) {
    fns[_key] = arguments[_key];
  }
  return /* @__PURE__ */ __name(function cleanup() {
    fns.forEach(function(fn) {
      return fn();
    });
  }, "cleanup");
}
__name(combine, "combine");
function addAttribute(element, _ref) {
  var attribute = _ref.attribute, value = _ref.value;
  element.setAttribute(attribute, value);
  return function() {
    return element.removeAttribute(attribute);
  };
}
__name(addAttribute, "addAttribute");
function ownKeys$6(e, r) {
  var t = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var o = Object.getOwnPropertySymbols(e);
    r && (o = o.filter(function(r2) {
      return Object.getOwnPropertyDescriptor(e, r2).enumerable;
    })), t.push.apply(t, o);
  }
  return t;
}
__name(ownKeys$6, "ownKeys$6");
function _objectSpread$6(e) {
  for (var r = 1; r < arguments.length; r++) {
    var t = null != arguments[r] ? arguments[r] : {};
    r % 2 ? ownKeys$6(Object(t), true).forEach(function(r2) {
      _defineProperty$6(e, r2, t[r2]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(t)) : ownKeys$6(Object(t)).forEach(function(r2) {
      Object.defineProperty(e, r2, Object.getOwnPropertyDescriptor(t, r2));
    });
  }
  return e;
}
__name(_objectSpread$6, "_objectSpread$6");
function _createForOfIteratorHelper$3(o, allowArrayLike) {
  var it = typeof Symbol !== "undefined" && o[Symbol.iterator] || o["@@iterator"];
  if (!it) {
    if (Array.isArray(o) || (it = _unsupportedIterableToArray$5(o)) || allowArrayLike && o && typeof o.length === "number") {
      if (it) o = it;
      var i = 0;
      var F = /* @__PURE__ */ __name(function F2() {
      }, "F2");
      return { s: F, n: /* @__PURE__ */ __name(function n() {
        if (i >= o.length) return { done: true };
        return { done: false, value: o[i++] };
      }, "n"), e: /* @__PURE__ */ __name(function e(_e) {
        throw _e;
      }, "e"), f: F };
    }
    throw new TypeError("Invalid attempt to iterate non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.");
  }
  var normalCompletion = true, didErr = false, err;
  return { s: /* @__PURE__ */ __name(function s() {
    it = it.call(o);
  }, "s"), n: /* @__PURE__ */ __name(function n() {
    var step2 = it.next();
    normalCompletion = step2.done;
    return step2;
  }, "n"), e: /* @__PURE__ */ __name(function e(_e2) {
    didErr = true;
    err = _e2;
  }, "e"), f: /* @__PURE__ */ __name(function f() {
    try {
      if (!normalCompletion && it.return != null) it.return();
    } finally {
      if (didErr) throw err;
    }
  }, "f") };
}
__name(_createForOfIteratorHelper$3, "_createForOfIteratorHelper$3");
function _unsupportedIterableToArray$5(o, minLen) {
  if (!o) return;
  if (typeof o === "string") return _arrayLikeToArray$5(o, minLen);
  var n = Object.prototype.toString.call(o).slice(8, -1);
  if (n === "Object" && o.constructor) n = o.constructor.name;
  if (n === "Map" || n === "Set") return Array.from(o);
  if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray$5(o, minLen);
}
__name(_unsupportedIterableToArray$5, "_unsupportedIterableToArray$5");
function _arrayLikeToArray$5(arr, len) {
  if (len == null || len > arr.length) len = arr.length;
  for (var i = 0, arr2 = new Array(len); i < len; i++) arr2[i] = arr[i];
  return arr2;
}
__name(_arrayLikeToArray$5, "_arrayLikeToArray$5");
function copyReverse(array) {
  return array.slice(0).reverse();
}
__name(copyReverse, "copyReverse");
function makeDropTarget(_ref) {
  var typeKey = _ref.typeKey, defaultDropEffect = _ref.defaultDropEffect;
  var registry = /* @__PURE__ */ new WeakMap();
  var dropTargetDataAtt = "data-drop-target-for-".concat(typeKey);
  var dropTargetSelector = "[".concat(dropTargetDataAtt, "]");
  function addToRegistry2(args) {
    registry.set(args.element, args);
    return function() {
      return registry.delete(args.element);
    };
  }
  __name(addToRegistry2, "addToRegistry");
  function dropTargetForConsumers(args) {
    if (false) {
      var existing = registry.get(args.element);
      if (existing) {
        console.warn("You have already registered a [".concat(typeKey, "] dropTarget on the same element"), {
          existing,
          proposed: args
        });
      }
      if (args.element instanceof HTMLIFrameElement) {
        console.warn("\n            We recommend not registering <iframe> elements as drop targets\n            as it can result in some strange browser event ordering.\n          ".replace(/\s{2,}/g, " ").trim());
      }
    }
    return combine(addAttribute(args.element, {
      attribute: dropTargetDataAtt,
      value: "true"
    }), addToRegistry2(args));
  }
  __name(dropTargetForConsumers, "dropTargetForConsumers");
  function getActualDropTargets(_ref2) {
    var _args$getData, _args$getData2, _args$getDropEffect, _args$getDropEffect2;
    var source = _ref2.source, target = _ref2.target, input = _ref2.input, _ref2$result = _ref2.result, result = _ref2$result === void 0 ? [] : _ref2$result;
    if (target == null) {
      return result;
    }
    if (!(target instanceof Element)) {
      if (target instanceof Node) {
        return getActualDropTargets({
          source,
          target: target.parentElement,
          input,
          result
        });
      }
      return result;
    }
    var closest = target.closest(dropTargetSelector);
    if (closest == null) {
      return result;
    }
    var args = registry.get(closest);
    if (args == null) {
      return result;
    }
    var feedback = {
      input,
      source,
      element: args.element
    };
    if (args.canDrop && !args.canDrop(feedback)) {
      return getActualDropTargets({
        source,
        target: args.element.parentElement,
        input,
        result
      });
    }
    var data17 = (_args$getData = (_args$getData2 = args.getData) === null || _args$getData2 === void 0 ? void 0 : _args$getData2.call(args, feedback)) !== null && _args$getData !== void 0 ? _args$getData : {};
    var dropEffect = (_args$getDropEffect = (_args$getDropEffect2 = args.getDropEffect) === null || _args$getDropEffect2 === void 0 ? void 0 : _args$getDropEffect2.call(args, feedback)) !== null && _args$getDropEffect !== void 0 ? _args$getDropEffect : defaultDropEffect;
    var record = {
      data: data17,
      element: args.element,
      dropEffect,
      // we are collecting _actual_ drop targets, so these are
      // being applied _not_ due to stickiness
      isActiveDueToStickiness: false
    };
    return getActualDropTargets({
      source,
      target: args.element.parentElement,
      input,
      // Using bubble ordering. Same ordering as `event.getPath()`
      result: [].concat(_toConsumableArray$4(result), [record])
    });
  }
  __name(getActualDropTargets, "getActualDropTargets");
  function notifyCurrent(_ref3) {
    var eventName = _ref3.eventName, payload = _ref3.payload;
    var _iterator = _createForOfIteratorHelper$3(payload.location.current.dropTargets), _step;
    try {
      for (_iterator.s(); !(_step = _iterator.n()).done; ) {
        var _entry$eventName;
        var record = _step.value;
        var entry = registry.get(record.element);
        var _args = _objectSpread$6(_objectSpread$6({}, payload), {}, {
          self: record
        });
        entry === null || entry === void 0 || (_entry$eventName = entry[eventName]) === null || _entry$eventName === void 0 || _entry$eventName.call(
          entry,
          // I cannot seem to get the types right here.
          // TS doesn't seem to like that one event can need `nativeSetDragImage`
          // @ts-expect-error
          _args
        );
      }
    } catch (err) {
      _iterator.e(err);
    } finally {
      _iterator.f();
    }
  }
  __name(notifyCurrent, "notifyCurrent");
  var actions = {
    onGenerateDragPreview: notifyCurrent,
    onDrag: notifyCurrent,
    onDragStart: notifyCurrent,
    onDrop: notifyCurrent,
    onDropTargetChange: /* @__PURE__ */ __name(function onDropTargetChange(_ref4) {
      var payload = _ref4.payload;
      var isCurrent = new Set(payload.location.current.dropTargets.map(function(record2) {
        return record2.element;
      }));
      var visited = /* @__PURE__ */ new Set();
      var _iterator2 = _createForOfIteratorHelper$3(payload.location.previous.dropTargets), _step2;
      try {
        for (_iterator2.s(); !(_step2 = _iterator2.n()).done; ) {
          var _entry$onDropTargetCh;
          var record = _step2.value;
          visited.add(record.element);
          var entry = registry.get(record.element);
          var isOver = isCurrent.has(record.element);
          var _args2 = _objectSpread$6(_objectSpread$6({}, payload), {}, {
            self: record
          });
          entry === null || entry === void 0 || (_entry$onDropTargetCh = entry.onDropTargetChange) === null || _entry$onDropTargetCh === void 0 || _entry$onDropTargetCh.call(entry, _args2);
          if (!isOver) {
            var _entry$onDragLeave;
            entry === null || entry === void 0 || (_entry$onDragLeave = entry.onDragLeave) === null || _entry$onDragLeave === void 0 || _entry$onDragLeave.call(entry, _args2);
          }
        }
      } catch (err) {
        _iterator2.e(err);
      } finally {
        _iterator2.f();
      }
      var _iterator3 = _createForOfIteratorHelper$3(payload.location.current.dropTargets), _step3;
      try {
        for (_iterator3.s(); !(_step3 = _iterator3.n()).done; ) {
          var _entry$onDropTargetCh2, _entry$onDragEnter;
          var _record = _step3.value;
          if (visited.has(_record.element)) {
            continue;
          }
          var _args3 = _objectSpread$6(_objectSpread$6({}, payload), {}, {
            self: _record
          });
          var _entry = registry.get(_record.element);
          _entry === null || _entry === void 0 || (_entry$onDropTargetCh2 = _entry.onDropTargetChange) === null || _entry$onDropTargetCh2 === void 0 || _entry$onDropTargetCh2.call(_entry, _args3);
          _entry === null || _entry === void 0 || (_entry$onDragEnter = _entry.onDragEnter) === null || _entry$onDragEnter === void 0 || _entry$onDragEnter.call(_entry, _args3);
        }
      } catch (err) {
        _iterator3.e(err);
      } finally {
        _iterator3.f();
      }
    }, "onDropTargetChange")
  };
  function dispatchEvent(args) {
    actions[args.eventName](args);
  }
  __name(dispatchEvent, "dispatchEvent");
  function getIsOver(_ref5) {
    var source = _ref5.source, target = _ref5.target, input = _ref5.input, current = _ref5.current;
    var actual = getActualDropTargets({
      source,
      target,
      input
    });
    if (actual.length >= current.length) {
      return actual;
    }
    var lastCaptureOrdered = copyReverse(current);
    var actualCaptureOrdered = copyReverse(actual);
    var resultCaptureOrdered = [];
    for (var index2 = 0; index2 < lastCaptureOrdered.length; index2++) {
      var _argsForLast$getIsSti;
      var last = lastCaptureOrdered[index2];
      var fresh = actualCaptureOrdered[index2];
      if (fresh != null) {
        resultCaptureOrdered.push(fresh);
        continue;
      }
      var parent = resultCaptureOrdered[index2 - 1];
      var lastParent = lastCaptureOrdered[index2 - 1];
      if ((parent === null || parent === void 0 ? void 0 : parent.element) !== (lastParent === null || lastParent === void 0 ? void 0 : lastParent.element)) {
        break;
      }
      var argsForLast = registry.get(last.element);
      if (!argsForLast) {
        break;
      }
      var feedback = {
        input,
        source,
        element: argsForLast.element
      };
      if (argsForLast.canDrop && !argsForLast.canDrop(feedback)) {
        break;
      }
      if (!((_argsForLast$getIsSti = argsForLast.getIsSticky) !== null && _argsForLast$getIsSti !== void 0 && _argsForLast$getIsSti.call(argsForLast, feedback))) {
        break;
      }
      resultCaptureOrdered.push(_objectSpread$6(_objectSpread$6({}, last), {}, {
        // making it clear to consumers this drop target is active due to stickiness
        isActiveDueToStickiness: true
      }));
    }
    return copyReverse(resultCaptureOrdered);
  }
  __name(getIsOver, "getIsOver");
  return {
    dropTargetForConsumers,
    getIsOver,
    dispatchEvent
  };
}
__name(makeDropTarget, "makeDropTarget");
function _createForOfIteratorHelper$2(o, allowArrayLike) {
  var it = typeof Symbol !== "undefined" && o[Symbol.iterator] || o["@@iterator"];
  if (!it) {
    if (Array.isArray(o) || (it = _unsupportedIterableToArray$4(o)) || allowArrayLike && o && typeof o.length === "number") {
      if (it) o = it;
      var i = 0;
      var F = /* @__PURE__ */ __name(function F2() {
      }, "F");
      return { s: F, n: /* @__PURE__ */ __name(function n() {
        if (i >= o.length) return { done: true };
        return { done: false, value: o[i++] };
      }, "n"), e: /* @__PURE__ */ __name(function e(_e) {
        throw _e;
      }, "e"), f: F };
    }
    throw new TypeError("Invalid attempt to iterate non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.");
  }
  var normalCompletion = true, didErr = false, err;
  return { s: /* @__PURE__ */ __name(function s() {
    it = it.call(o);
  }, "s"), n: /* @__PURE__ */ __name(function n() {
    var step2 = it.next();
    normalCompletion = step2.done;
    return step2;
  }, "n"), e: /* @__PURE__ */ __name(function e(_e2) {
    didErr = true;
    err = _e2;
  }, "e"), f: /* @__PURE__ */ __name(function f() {
    try {
      if (!normalCompletion && it.return != null) it.return();
    } finally {
      if (didErr) throw err;
    }
  }, "f") };
}
__name(_createForOfIteratorHelper$2, "_createForOfIteratorHelper$2");
function _unsupportedIterableToArray$4(o, minLen) {
  if (!o) return;
  if (typeof o === "string") return _arrayLikeToArray$4(o, minLen);
  var n = Object.prototype.toString.call(o).slice(8, -1);
  if (n === "Object" && o.constructor) n = o.constructor.name;
  if (n === "Map" || n === "Set") return Array.from(o);
  if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray$4(o, minLen);
}
__name(_unsupportedIterableToArray$4, "_unsupportedIterableToArray$4");
function _arrayLikeToArray$4(arr, len) {
  if (len == null || len > arr.length) len = arr.length;
  for (var i = 0, arr2 = new Array(len); i < len; i++) arr2[i] = arr[i];
  return arr2;
}
__name(_arrayLikeToArray$4, "_arrayLikeToArray$4");
function ownKeys$5(e, r) {
  var t = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var o = Object.getOwnPropertySymbols(e);
    r && (o = o.filter(function(r2) {
      return Object.getOwnPropertyDescriptor(e, r2).enumerable;
    })), t.push.apply(t, o);
  }
  return t;
}
__name(ownKeys$5, "ownKeys$5");
function _objectSpread$5(e) {
  for (var r = 1; r < arguments.length; r++) {
    var t = null != arguments[r] ? arguments[r] : {};
    r % 2 ? ownKeys$5(Object(t), true).forEach(function(r2) {
      _defineProperty$6(e, r2, t[r2]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(t)) : ownKeys$5(Object(t)).forEach(function(r2) {
      Object.defineProperty(e, r2, Object.getOwnPropertyDescriptor(t, r2));
    });
  }
  return e;
}
__name(_objectSpread$5, "_objectSpread$5");
function makeMonitor() {
  var registry = /* @__PURE__ */ new Set();
  var dragging = null;
  function tryAddToActive(monitor) {
    if (!dragging) {
      return;
    }
    if (!monitor.canMonitor || monitor.canMonitor(dragging.canMonitorArgs)) {
      dragging.active.add(monitor);
    }
  }
  __name(tryAddToActive, "tryAddToActive");
  function monitorForConsumers(args) {
    var entry = _objectSpread$5({}, args);
    registry.add(entry);
    tryAddToActive(entry);
    return /* @__PURE__ */ __name(function cleanup() {
      registry.delete(entry);
      if (dragging) {
        dragging.active.delete(entry);
      }
    }, "cleanup");
  }
  __name(monitorForConsumers, "monitorForConsumers");
  function dispatchEvent(_ref) {
    var eventName = _ref.eventName, payload = _ref.payload;
    if (eventName === "onGenerateDragPreview") {
      dragging = {
        canMonitorArgs: {
          initial: payload.location.initial,
          source: payload.source
        },
        active: /* @__PURE__ */ new Set()
      };
      var _iterator = _createForOfIteratorHelper$2(registry), _step;
      try {
        for (_iterator.s(); !(_step = _iterator.n()).done; ) {
          var monitor = _step.value;
          tryAddToActive(monitor);
        }
      } catch (err) {
        _iterator.e(err);
      } finally {
        _iterator.f();
      }
    }
    if (!dragging) {
      return;
    }
    var active2 = Array.from(dragging.active);
    for (var _i = 0, _active = active2; _i < _active.length; _i++) {
      var _monitor = _active[_i];
      if (dragging.active.has(_monitor)) {
        var _monitor$eventName;
        (_monitor$eventName = _monitor[eventName]) === null || _monitor$eventName === void 0 || _monitor$eventName.call(_monitor, payload);
      }
    }
    if (eventName === "onDrop") {
      dragging.active.clear();
      dragging = null;
    }
  }
  __name(dispatchEvent, "dispatchEvent");
  return {
    dispatchEvent,
    monitorForConsumers
  };
}
__name(makeMonitor, "makeMonitor");
function makeAdapter(_ref) {
  var typeKey = _ref.typeKey, mount2 = _ref.mount, dispatchEventToSource2 = _ref.dispatchEventToSource, onPostDispatch = _ref.onPostDispatch, defaultDropEffect = _ref.defaultDropEffect;
  var monitorAPI = makeMonitor();
  var dropTargetAPI = makeDropTarget({
    typeKey,
    defaultDropEffect
  });
  function dispatchEvent(args) {
    dispatchEventToSource2 === null || dispatchEventToSource2 === void 0 || dispatchEventToSource2(args);
    dropTargetAPI.dispatchEvent(args);
    monitorAPI.dispatchEvent(args);
    onPostDispatch === null || onPostDispatch === void 0 || onPostDispatch(args);
  }
  __name(dispatchEvent, "dispatchEvent");
  function start2(_ref2) {
    var event = _ref2.event, dragType = _ref2.dragType;
    lifecycle.start({
      event,
      dragType,
      getDropTargetsOver: dropTargetAPI.getIsOver,
      dispatchEvent
    });
  }
  __name(start2, "start");
  function registerUsage2() {
    function mountAdapter() {
      var api2 = {
        canStart: lifecycle.canStart,
        start: start2
      };
      return mount2(api2);
    }
    __name(mountAdapter, "mountAdapter");
    return register({
      typeKey,
      mount: mountAdapter
    });
  }
  __name(registerUsage2, "registerUsage");
  return {
    registerUsage: registerUsage2,
    dropTarget: dropTargetAPI.dropTargetForConsumers,
    monitor: monitorAPI.monitorForConsumers
  };
}
__name(makeAdapter, "makeAdapter");
var isAndroid = once(/* @__PURE__ */ __name(function isAndroid2() {
  return navigator.userAgent.toLocaleLowerCase().includes("android");
}, "isAndroid"));
var androidFallbackText = "pdnd:android-fallback";
var textMediaType = "text/plain";
var urlMediaType = "text/uri-list";
var elementAdapterNativeDataKey = "application/vnd.pdnd";
var draggableRegistry = /* @__PURE__ */ new WeakMap();
function addToRegistry(args) {
  draggableRegistry.set(args.element, args);
  return /* @__PURE__ */ __name(function cleanup() {
    draggableRegistry.delete(args.element);
  }, "cleanup");
}
__name(addToRegistry, "addToRegistry");
var honeyPotFix = makeHoneyPotFix();
var adapter = makeAdapter({
  typeKey: "element",
  defaultDropEffect: "move",
  mount: /* @__PURE__ */ __name(function mount(api2) {
    return combine(honeyPotFix.bindEvents(), dist.bind(document, {
      type: "dragstart",
      listener: /* @__PURE__ */ __name(function listener(event) {
        var _entry$dragHandle, _entry$getInitialData, _entry$getInitialData2, _entry$dragHandle2, _entry$getInitialData3, _entry$getInitialData4;
        if (!api2.canStart(event)) {
          return;
        }
        if (event.defaultPrevented) {
          return;
        }
        if (!event.dataTransfer) {
          if (false) {
            console.warn("\n              It appears as though you have are not testing DragEvents correctly.\n\n              - If you are unit testing, ensure you have polyfilled DragEvent.\n              - If you are browser testing, ensure you are dispatching drag events correctly.\n\n              Please see our testing guides for more information:\n              https://atlassian.design/components/pragmatic-drag-and-drop/core-package/testing\n            ".replace(/ {2}/g, ""));
          }
          return;
        }
        var target = event.target;
        if (!(target instanceof HTMLElement)) {
          return null;
        }
        var entry = draggableRegistry.get(target);
        if (!entry) {
          return null;
        }
        var input = getInput(event);
        var feedback = {
          element: entry.element,
          dragHandle: (_entry$dragHandle = entry.dragHandle) !== null && _entry$dragHandle !== void 0 ? _entry$dragHandle : null,
          input
        };
        if (entry.canDrag && !entry.canDrag(feedback)) {
          event.preventDefault();
          return null;
        }
        if (entry.dragHandle) {
          var over = getElementFromPointWithoutHoneypot({
            x: input.clientX,
            y: input.clientY
          });
          if (!entry.dragHandle.contains(over)) {
            event.preventDefault();
            return null;
          }
        }
        var nativeData = (_entry$getInitialData = (_entry$getInitialData2 = entry.getInitialDataForExternal) === null || _entry$getInitialData2 === void 0 ? void 0 : _entry$getInitialData2.call(entry, feedback)) !== null && _entry$getInitialData !== void 0 ? _entry$getInitialData : null;
        if (nativeData) {
          for (var _i = 0, _Object$entries = Object.entries(nativeData); _i < _Object$entries.length; _i++) {
            var _Object$entries$_i = _slicedToArray(_Object$entries[_i], 2), key = _Object$entries$_i[0], data17 = _Object$entries$_i[1];
            event.dataTransfer.setData(key, data17 !== null && data17 !== void 0 ? data17 : "");
          }
        }
        var types = event.dataTransfer.types;
        if (isAndroid() && !types.includes(textMediaType) && !types.includes(urlMediaType)) {
          event.dataTransfer.setData(textMediaType, androidFallbackText);
        }
        event.dataTransfer.setData(elementAdapterNativeDataKey, "");
        var payload = {
          element: entry.element,
          dragHandle: (_entry$dragHandle2 = entry.dragHandle) !== null && _entry$dragHandle2 !== void 0 ? _entry$dragHandle2 : null,
          data: (_entry$getInitialData3 = (_entry$getInitialData4 = entry.getInitialData) === null || _entry$getInitialData4 === void 0 ? void 0 : _entry$getInitialData4.call(entry, feedback)) !== null && _entry$getInitialData3 !== void 0 ? _entry$getInitialData3 : {}
        };
        var dragType = {
          type: "element",
          payload,
          startedFrom: "internal"
        };
        api2.start({
          event,
          dragType
        });
      }, "listener")
    }));
  }, "mount"),
  dispatchEventToSource: /* @__PURE__ */ __name(function dispatchEventToSource(_ref) {
    var _draggableRegistry$ge, _draggableRegistry$ge2;
    var eventName = _ref.eventName, payload = _ref.payload;
    (_draggableRegistry$ge = draggableRegistry.get(payload.source.element)) === null || _draggableRegistry$ge === void 0 || (_draggableRegistry$ge2 = _draggableRegistry$ge[eventName]) === null || _draggableRegistry$ge2 === void 0 || _draggableRegistry$ge2.call(
      _draggableRegistry$ge,
      // I cannot seem to get the types right here.
      // TS doesn't seem to like that one event can need `nativeSetDragImage`
      // @ts-expect-error
      payload
    );
  }, "dispatchEventToSource"),
  onPostDispatch: honeyPotFix.getOnPostDispatch()
});
var dropTargetForElements = adapter.dropTarget;
var monitorForElements = adapter.monitor;
function draggable(args) {
  if (false) {
    if (args.dragHandle && !args.element.contains(args.dragHandle)) {
      console.warn("Drag handle element must be contained in draggable element", {
        element: args.element,
        dragHandle: args.dragHandle
      });
    }
  }
  if (false) {
    var existing = draggableRegistry.get(args.element);
    if (existing) {
      console.warn("You have already registered a `draggable` on the same element", {
        existing,
        proposed: args
      });
    }
  }
  return combine(
    // making the draggable register the adapter rather than drop targets
    // this is because you *must* have a draggable element to start a drag
    // but you _might_ not have any drop targets immediately
    // (You might create drop targets async)
    adapter.registerUsage(),
    addToRegistry(args),
    addAttribute(args.element, {
      attribute: "draggable",
      value: "true"
    })
  );
}
__name(draggable, "draggable");
class ModelNodeProvider {
  static {
    __name(this, "ModelNodeProvider");
  }
  /** The node definition to use for this model. */
  nodeDef;
  /** The node input key for where to inside the model name. */
  key;
  constructor(nodeDef, key) {
    this.nodeDef = nodeDef;
    this.key = key;
  }
}
const useModelToNodeStore = defineStore("modelToNode", {
  state: /* @__PURE__ */ __name(() => ({
    modelToNodeMap: {},
    nodeDefStore: useNodeDefStore(),
    haveDefaultsLoaded: false
  }), "state"),
  actions: {
    /**
     * Get the node provider for the given model type name.
     * @param modelType The name of the model type to get the node provider for.
     * @returns The node provider for the given model type name.
     */
    getNodeProvider(modelType) {
      this.registerDefaults();
      return this.modelToNodeMap[modelType]?.[0];
    },
    /**
     * Get the list of all valid node providers for the given model type name.
     * @param modelType The name of the model type to get the node providers for.
     * @returns The list of all valid node providers for the given model type name.
     */
    getAllNodeProviders(modelType) {
      this.registerDefaults();
      return this.modelToNodeMap[modelType] ?? [];
    },
    /**
     * Register a node provider for the given model type name.
     * @param modelType The name of the model type to register the node provider for.
     * @param nodeProvider The node provider to register.
     */
    registerNodeProvider(modelType, nodeProvider) {
      this.registerDefaults();
      this.modelToNodeMap[modelType] ??= [];
      this.modelToNodeMap[modelType].push(nodeProvider);
    },
    /**
     * Register a node provider for the given simple names.
     * @param modelType The name of the model type to register the node provider for.
     * @param nodeClass The node class name to register.
     * @param key The key to use for the node input.
     */
    quickRegister(modelType, nodeClass, key) {
      this.registerNodeProvider(
        modelType,
        new ModelNodeProvider(this.nodeDefStore.nodeDefsByName[nodeClass], key)
      );
    },
    registerDefaults() {
      if (this.haveDefaultsLoaded) {
        return;
      }
      if (Object.keys(this.nodeDefStore.nodeDefsByName).length === 0) {
        return;
      }
      this.haveDefaultsLoaded = true;
      this.quickRegister("checkpoints", "CheckpointLoaderSimple", "ckpt_name");
      this.quickRegister(
        "checkpoints",
        "ImageOnlyCheckpointLoader",
        "ckpt_name"
      );
      this.quickRegister("loras", "LoraLoader", "lora_name");
      this.quickRegister("loras", "LoraLoaderModelOnly", "lora_name");
      this.quickRegister("vae", "VAELoader", "vae_name");
      this.quickRegister("controlnet", "ControlNetLoader", "control_net_name");
    }
  }
});
const _hoisted_1$v = {
  viewBox: "0 0 1024 1024",
  width: "1.2em",
  height: "1.2em"
};
const _hoisted_2$n = /* @__PURE__ */ createBaseVNode("path", {
  fill: "currentColor",
  d: "M921.088 103.232L584.832 889.024L465.52 544.512L121.328 440.48zM1004.46.769c-6.096 0-13.52 1.728-22.096 5.36L27.708 411.2c-34.383 14.592-36.56 42.704-4.847 62.464l395.296 123.584l129.36 403.264c9.28 15.184 20.496 22.72 31.263 22.72c11.936 0 23.296-9.152 31.04-27.248l408.272-953.728C1029.148 16.368 1022.86.769 1004.46.769"
}, null, -1);
const _hoisted_3$h = [
  _hoisted_2$n
];
function render$g(_ctx, _cache) {
  return openBlock(), createElementBlock("svg", _hoisted_1$v, [..._hoisted_3$h]);
}
__name(render$g, "render$g");
const __unplugin_components_1 = markRaw({ name: "simple-line-icons-cursor", render: render$g });
const _hoisted_1$u = {
  viewBox: "0 0 24 24",
  width: "1.2em",
  height: "1.2em"
};
const _hoisted_2$m = /* @__PURE__ */ createBaseVNode("path", {
  fill: "currentColor",
  d: "M10.05 23q-.75 0-1.4-.337T7.575 21.7L1.2 12.375l.6-.575q.475-.475 1.125-.55t1.175.3L7 13.575V4q0-.425.288-.712T8 3t.713.288T9 4v13.425l-3.7-2.6l3.925 5.725q.125.2.35.325t.475.125H17q.825 0 1.413-.587T19 19V5q0-.425.288-.712T20 4t.713.288T21 5v14q0 1.65-1.175 2.825T17 23zM11 12V2q0-.425.288-.712T12 1t.713.288T13 2v10zm4 0V3q0-.425.288-.712T16 2t.713.288T17 3v9zm-2.85 4.5"
}, null, -1);
const _hoisted_3$g = [
  _hoisted_2$m
];
function render$f(_ctx, _cache) {
  return openBlock(), createElementBlock("svg", _hoisted_1$u, [..._hoisted_3$g]);
}
__name(render$f, "render$f");
const __unplugin_components_0 = markRaw({ name: "material-symbols-pan-tool-outline", render: render$f });
var theme$c = /* @__PURE__ */ __name(function theme6(_ref) {
  _ref.dt;
  return "\n.p-buttongroup .p-button {\n    margin: 0;\n}\n\n.p-buttongroup .p-button:not(:last-child),\n.p-buttongroup .p-button:not(:last-child):hover {\n    border-right: 0 none;\n}\n\n.p-buttongroup .p-button:not(:first-of-type):not(:last-of-type) {\n    border-radius: 0;\n}\n\n.p-buttongroup .p-button:first-of-type:not(:only-of-type) {\n    border-top-right-radius: 0;\n    border-bottom-right-radius: 0;\n}\n\n.p-buttongroup .p-button:last-of-type:not(:only-of-type) {\n    border-top-left-radius: 0;\n    border-bottom-left-radius: 0;\n}\n\n.p-buttongroup .p-button:focus {\n    position: relative;\n    z-index: 1;\n}\n";
}, "theme");
var classes$c = {
  root: "p-buttongroup p-component"
};
var ButtonGroupStyle = BaseStyle.extend({
  name: "buttongroup",
  theme: theme$c,
  classes: classes$c
});
var script$1$c = {
  name: "BaseButtonGroup",
  "extends": script$p,
  style: ButtonGroupStyle,
  provide: /* @__PURE__ */ __name(function provide8() {
    return {
      $pcButtonGroup: this,
      $parentInstance: this
    };
  }, "provide")
};
var script$f = {
  name: "ButtonGroup",
  "extends": script$1$c,
  inheritAttrs: false
};
function render$e(_ctx, _cache, $props, $setup, $data, $options) {
  return openBlock(), createElementBlock("span", mergeProps({
    "class": _ctx.cx("root"),
    role: "group"
  }, _ctx.ptmi("root")), [renderSlot(_ctx.$slots, "default")], 16);
}
__name(render$e, "render$e");
script$f.render = render$e;
const _sfc_main$u = /* @__PURE__ */ defineComponent({
  __name: "GraphCanvasMenu",
  setup(__props) {
    const { t } = useI18n();
    const commandStore = useCommandStore();
    const canvasStore = useCanvasStore();
    const settingStore = useSettingStore();
    const linkHidden = computed(
      () => settingStore.get("Comfy.LinkRenderMode") === LiteGraph.HIDDEN_LINK
    );
    let interval = null;
    const repeat2 = /* @__PURE__ */ __name((command) => {
      if (interval) return;
      const cmd = /* @__PURE__ */ __name(() => commandStore.execute(command), "cmd");
      cmd();
      interval = window.setInterval(cmd, 100);
    }, "repeat");
    const stopRepeat = /* @__PURE__ */ __name(() => {
      if (interval) {
        clearInterval(interval);
        interval = null;
      }
    }, "stopRepeat");
    return (_ctx, _cache) => {
      const _component_i_material_symbols58pan_tool_outline = __unplugin_components_0;
      const _component_i_simple_line_icons58cursor = __unplugin_components_1;
      const _directive_tooltip = resolveDirective("tooltip");
      return openBlock(), createBlock(unref(script$f), { class: "p-buttongroup-vertical absolute bottom-[10px] right-[10px] z-[1000] pointer-events-auto" }, {
        default: withCtx(() => [
          withDirectives(createVNode(unref(script$o), {
            severity: "secondary",
            icon: "pi pi-plus",
            onMousedown: _cache[0] || (_cache[0] = ($event) => repeat2("Comfy.Canvas.ZoomIn")),
            onMouseup: stopRepeat
          }, null, 512), [
            [
              _directive_tooltip,
              unref(t)("graphCanvasMenu.zoomIn"),
              void 0,
              { left: true }
            ]
          ]),
          withDirectives(createVNode(unref(script$o), {
            severity: "secondary",
            icon: "pi pi-minus",
            onMousedown: _cache[1] || (_cache[1] = ($event) => repeat2("Comfy.Canvas.ZoomOut")),
            onMouseup: stopRepeat
          }, null, 512), [
            [
              _directive_tooltip,
              unref(t)("graphCanvasMenu.zoomOut"),
              void 0,
              { left: true }
            ]
          ]),
          withDirectives(createVNode(unref(script$o), {
            severity: "secondary",
            icon: "pi pi-expand",
            onClick: _cache[2] || (_cache[2] = () => unref(commandStore).execute("Comfy.Canvas.ResetView"))
          }, null, 512), [
            [
              _directive_tooltip,
              unref(t)("graphCanvasMenu.resetView"),
              void 0,
              { left: true }
            ]
          ]),
          withDirectives((openBlock(), createBlock(unref(script$o), {
            severity: "secondary",
            onClick: _cache[3] || (_cache[3] = () => unref(commandStore).execute("Comfy.Canvas.ToggleLock"))
          }, {
            icon: withCtx(() => [
              unref(canvasStore).readOnly ? (openBlock(), createBlock(_component_i_material_symbols58pan_tool_outline, { key: 0 })) : (openBlock(), createBlock(_component_i_simple_line_icons58cursor, { key: 1 }))
            ]),
            _: 1
          })), [
            [
              _directive_tooltip,
              unref(t)(
                "graphCanvasMenu." + (unref(canvasStore).readOnly ? "panMode" : "selectMode")
              ) + " (Space)",
              void 0,
              { left: true }
            ]
          ]),
          withDirectives(createVNode(unref(script$o), {
            severity: "secondary",
            icon: linkHidden.value ? "pi pi-eye-slash" : "pi pi-eye",
            onClick: _cache[4] || (_cache[4] = () => unref(commandStore).execute("Comfy.Canvas.ToggleLinkVisibility")),
            "data-testid": "toggle-link-visibility-button"
          }, null, 8, ["icon"]), [
            [
              _directive_tooltip,
              unref(t)("graphCanvasMenu.toggleLinkVisibility"),
              void 0,
              { left: true }
            ]
          ])
        ]),
        _: 1
      });
    };
  }
});
const GraphCanvasMenu = /* @__PURE__ */ _export_sfc(_sfc_main$u, [["__scopeId", "data-v-ce8bd6ac"]]);
const _sfc_main$t = /* @__PURE__ */ defineComponent({
  __name: "GraphCanvas",
  emits: ["ready"],
  setup(__props, { emit: __emit }) {
    const emit = __emit;
    const canvasRef = ref(null);
    const settingStore = useSettingStore();
    const nodeDefStore = useNodeDefStore();
    const workspaceStore = useWorkspaceStore();
    const canvasStore = useCanvasStore();
    const modelToNodeStore = useModelToNodeStore();
    const betaMenuEnabled = computed(
      () => settingStore.get("Comfy.UseNewMenu") !== "Disabled"
    );
    const canvasMenuEnabled = computed(
      () => settingStore.get("Comfy.Graph.CanvasMenu")
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
    watchEffect(() => {
      if (!canvasStore.canvas) return;
      if (canvasStore.draggingCanvas) {
        canvasStore.canvas.canvas.style.cursor = "grabbing";
        return;
      }
      if (canvasStore.readOnly) {
        canvasStore.canvas.canvas.style.cursor = "grab";
        return;
      }
      canvasStore.canvas.canvas.style.cursor = "default";
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
      window["LGraphBadge"] = LGraphBadge;
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
            const node2 = dndData.data;
            if (node2.data instanceof ComfyNodeDefImpl) {
              const nodeDef = node2.data;
              const pos = app.clientPosToCanvasPos([
                loc.clientX - 20,
                loc.clientY
              ]);
              app.addNodeOnGraph(nodeDef, { pos });
            } else if (node2.data instanceof ComfyModelDef) {
              const model = node2.data;
              const pos = app.clientPosToCanvasPos([loc.clientX, loc.clientY]);
              const nodeAtPos = app.graph.getNodeOnPos(pos[0], pos[1]);
              let targetProvider = null;
              let targetGraphNode = null;
              if (nodeAtPos) {
                const providers = modelToNodeStore.getAllNodeProviders(
                  model.directory
                );
                for (const provider of providers) {
                  if (provider.nodeDef.name === nodeAtPos.comfyClass) {
                    targetGraphNode = nodeAtPos;
                    targetProvider = provider;
                  }
                }
              }
              if (!targetGraphNode) {
                const provider = modelToNodeStore.getNodeProvider(model.directory);
                if (provider) {
                  targetGraphNode = app.addNodeOnGraph(provider.nodeDef, {
                    pos
                  });
                  targetProvider = provider;
                }
              }
              if (targetGraphNode) {
                const widget = targetGraphNode.widgets.find(
                  (widget2) => widget2.name === targetProvider.key
                );
                if (widget) {
                  widget.value = model.file_name;
                }
              }
            }
          }
        }, "onDrop")
      });
      useKeybindingStore().loadUserKeybindings();
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
            "graph-canvas-panel": withCtx(() => [
              canvasMenuEnabled.value ? (openBlock(), createBlock(GraphCanvasMenu, { key: 0 })) : createCommentVNode("", true)
            ]),
            _: 1
          })) : createCommentVNode("", true),
          createVNode(TitleEditor),
          !betaMenuEnabled.value && canvasMenuEnabled.value ? (openBlock(), createBlock(GraphCanvasMenu, { key: 1 })) : createCommentVNode("", true),
          createBaseVNode("canvas", {
            ref_key: "canvasRef",
            ref: canvasRef,
            id: "graph-canvas",
            tabindex: "1"
          }, null, 512)
        ])),
        createVNode(_sfc_main$w),
        createVNode(NodeTooltip)
      ], 64);
    };
  }
});
var theme$b = /* @__PURE__ */ __name(function theme7(_ref) {
  var dt = _ref.dt;
  return "\n.p-confirmpopup {\n    position: absolute;\n    margin-top: ".concat(dt("confirmpopup.gutter"), ";\n    top: 0;\n    left: 0;\n    background: ").concat(dt("confirmpopup.background"), ";\n    color: ").concat(dt("confirmpopup.color"), ";\n    border: 1px solid ").concat(dt("confirmpopup.border.color"), ";\n    border-radius: ").concat(dt("confirmpopup.border.radius"), ";\n    box-shadow: ").concat(dt("confirmpopup.shadow"), ";\n}\n\n.p-confirmpopup-content {\n    display: flex;\n    align-items: center;\n    padding: ").concat(dt("confirmpopup.content.padding"), ";\n    gap: ").concat(dt("confirmpopup.content.gap"), ";\n}\n\n.p-confirmpopup-icon {\n    font-size: ").concat(dt("confirmpopup.icon.size"), ";\n    width: ").concat(dt("confirmpopup.icon.size"), ";\n    height: ").concat(dt("confirmpopup.icon.size"), ";\n    color: ").concat(dt("confirmpopup.icon.color"), ";\n}\n\n.p-confirmpopup-footer {\n    display: flex;\n    justify-content: flex-end;\n    gap: ").concat(dt("confirmpopup.footer.gap"), ";\n    padding: ").concat(dt("confirmpopup.footer.padding"), ";\n}\n\n.p-confirmpopup-footer button {\n    width: auto;\n}\n\n.p-confirmpopup-footer button:last-child {\n    margin: 0;\n}\n\n.p-confirmpopup-flipped {\n    margin-top: calc(").concat(dt("confirmpopup.gutter"), " * -1);\n    margin-bottom: ").concat(dt("confirmpopup.gutter"), ";\n}\n\n.p-confirmpopup-enter-from {\n    opacity: 0;\n    transform: scaleY(0.8);\n}\n\n.p-confirmpopup-leave-to {\n    opacity: 0;\n}\n\n.p-confirmpopup-enter-active {\n    transition: transform 0.12s cubic-bezier(0, 0, 0.2, 1), opacity 0.12s cubic-bezier(0, 0, 0.2, 1);\n}\n\n.p-confirmpopup-leave-active {\n    transition: opacity 0.1s linear;\n}\n\n.p-confirmpopup:after,\n.p-confirmpopup:before {\n    bottom: 100%;\n    left: calc(").concat(dt("confirmpopup.arrow.offset"), " + ").concat(dt("confirmpopup.arrow.left"), ');\n    content: " ";\n    height: 0;\n    width: 0;\n    position: absolute;\n    pointer-events: none;\n}\n\n.p-confirmpopup:after {\n    border-width: calc(').concat(dt("confirmpopup.gutter"), " - 2px);\n    margin-left: calc(-1 * (").concat(dt("confirmpopup.gutter"), " - 2px));\n    border-style: solid;\n    border-color: transparent;\n    border-bottom-color: ").concat(dt("confirmpopup.background"), ";\n}\n\n.p-confirmpopup:before {\n    border-width: ").concat(dt("confirmpopup.gutter"), ";\n    margin-left: calc(-1 * ").concat(dt("confirmpopup.gutter"), ");\n    border-style: solid;\n    border-color: transparent;\n    border-bottom-color: ").concat(dt("confirmpopup.border.color"), ";\n}\n\n.p-confirmpopup-flipped:after,\n.p-confirmpopup-flipped:before {\n    bottom: auto;\n    top: 100%;\n}\n\n.p-confirmpopup-flipped:after {\n    border-bottom-color: transparent;\n    border-top-color: ").concat(dt("confirmpopup.background"), ";\n}\n\n.p-confirmpopup-flipped:before {\n    border-bottom-color: transparent;\n    border-top-color: ").concat(dt("confirmpopup.border.color"), ";\n}\n");
}, "theme");
var classes$b = {
  root: "p-confirmpopup p-component",
  content: "p-confirmpopup-content",
  icon: "p-confirmpopup-icon",
  message: "p-confirmpopup-message",
  footer: "p-confirmpopup-footer",
  pcRejectButton: "p-confirmpopup-reject-button",
  pcAcceptButton: "p-confirmpopup-accept-button"
};
var ConfirmPopupStyle = BaseStyle.extend({
  name: "confirmpopup",
  theme: theme$b,
  classes: classes$b
});
var script$1$b = {
  name: "BaseConfirmPopup",
  "extends": script$p,
  props: {
    group: String
  },
  style: ConfirmPopupStyle,
  provide: /* @__PURE__ */ __name(function provide9() {
    return {
      $pcConfirmPopup: this,
      $parentInstance: this
    };
  }, "provide")
};
var script$e = {
  name: "ConfirmPopup",
  "extends": script$1$b,
  inheritAttrs: false,
  data: /* @__PURE__ */ __name(function data4() {
    return {
      visible: false,
      confirmation: null,
      autoFocusAccept: null,
      autoFocusReject: null,
      target: null
    };
  }, "data"),
  target: null,
  outsideClickListener: null,
  scrollHandler: null,
  resizeListener: null,
  container: null,
  confirmListener: null,
  closeListener: null,
  mounted: /* @__PURE__ */ __name(function mounted3() {
    var _this = this;
    this.confirmListener = function(options) {
      if (!options) {
        return;
      }
      if (options.group === _this.group) {
        _this.confirmation = options;
        _this.target = options.target;
        if (_this.confirmation.onShow) {
          _this.confirmation.onShow();
        }
        _this.visible = true;
      }
    };
    this.closeListener = function() {
      _this.visible = false;
      _this.confirmation = null;
    };
    ConfirmationEventBus.on("confirm", this.confirmListener);
    ConfirmationEventBus.on("close", this.closeListener);
  }, "mounted"),
  beforeUnmount: /* @__PURE__ */ __name(function beforeUnmount3() {
    ConfirmationEventBus.off("confirm", this.confirmListener);
    ConfirmationEventBus.off("close", this.closeListener);
    this.unbindOutsideClickListener();
    if (this.scrollHandler) {
      this.scrollHandler.destroy();
      this.scrollHandler = null;
    }
    this.unbindResizeListener();
    if (this.container) {
      ZIndex.clear(this.container);
      this.container = null;
    }
    this.target = null;
    this.confirmation = null;
  }, "beforeUnmount"),
  methods: {
    accept: /* @__PURE__ */ __name(function accept() {
      if (this.confirmation.accept) {
        this.confirmation.accept();
      }
      this.visible = false;
    }, "accept"),
    reject: /* @__PURE__ */ __name(function reject() {
      if (this.confirmation.reject) {
        this.confirmation.reject();
      }
      this.visible = false;
    }, "reject"),
    onHide: /* @__PURE__ */ __name(function onHide() {
      if (this.confirmation.onHide) {
        this.confirmation.onHide();
      }
      this.visible = false;
    }, "onHide"),
    onAcceptKeydown: /* @__PURE__ */ __name(function onAcceptKeydown(event) {
      if (event.code === "Space" || event.code === "Enter" || event.code === "NumpadEnter") {
        this.accept();
        focus(this.target);
        event.preventDefault();
      }
    }, "onAcceptKeydown"),
    onRejectKeydown: /* @__PURE__ */ __name(function onRejectKeydown(event) {
      if (event.code === "Space" || event.code === "Enter" || event.code === "NumpadEnter") {
        this.reject();
        focus(this.target);
        event.preventDefault();
      }
    }, "onRejectKeydown"),
    onEnter: /* @__PURE__ */ __name(function onEnter(el) {
      this.autoFocusAccept = this.confirmation.defaultFocus === void 0 || this.confirmation.defaultFocus === "accept" ? true : false;
      this.autoFocusReject = this.confirmation.defaultFocus === "reject" ? true : false;
      this.target = document.activeElement;
      this.bindOutsideClickListener();
      this.bindScrollListener();
      this.bindResizeListener();
      ZIndex.set("overlay", el, this.$primevue.config.zIndex.overlay);
    }, "onEnter"),
    onAfterEnter: /* @__PURE__ */ __name(function onAfterEnter() {
      this.focus();
    }, "onAfterEnter"),
    onLeave: /* @__PURE__ */ __name(function onLeave() {
      this.autoFocusAccept = null;
      this.autoFocusReject = null;
      focus(this.target);
      this.target = null;
      this.unbindOutsideClickListener();
      this.unbindScrollListener();
      this.unbindResizeListener();
    }, "onLeave"),
    onAfterLeave: /* @__PURE__ */ __name(function onAfterLeave(el) {
      ZIndex.clear(el);
    }, "onAfterLeave"),
    alignOverlay: /* @__PURE__ */ __name(function alignOverlay2() {
      absolutePosition(this.container, this.target, false);
      var containerOffset = getOffset(this.container);
      var targetOffset = getOffset(this.target);
      var arrowLeft = 0;
      if (containerOffset.left < targetOffset.left) {
        arrowLeft = targetOffset.left - containerOffset.left;
      }
      this.container.style.setProperty($dt("confirmpopup.arrow.left").name, "".concat(arrowLeft, "px"));
      if (containerOffset.top < targetOffset.top) {
        this.container.setAttribute("data-p-confirmpopup-flipped", "true");
        !this.isUnstyled && addClass(this.container, "p-confirmpopup-flipped");
      }
    }, "alignOverlay"),
    bindOutsideClickListener: /* @__PURE__ */ __name(function bindOutsideClickListener2() {
      var _this2 = this;
      if (!this.outsideClickListener) {
        this.outsideClickListener = function(event) {
          if (_this2.visible && _this2.container && !_this2.container.contains(event.target) && !_this2.isTargetClicked(event)) {
            if (_this2.confirmation.onHide) {
              _this2.confirmation.onHide();
            }
            _this2.visible = false;
          } else {
            _this2.alignOverlay();
          }
        };
        document.addEventListener("click", this.outsideClickListener);
      }
    }, "bindOutsideClickListener"),
    unbindOutsideClickListener: /* @__PURE__ */ __name(function unbindOutsideClickListener2() {
      if (this.outsideClickListener) {
        document.removeEventListener("click", this.outsideClickListener);
        this.outsideClickListener = null;
      }
    }, "unbindOutsideClickListener"),
    bindScrollListener: /* @__PURE__ */ __name(function bindScrollListener2() {
      var _this3 = this;
      if (!this.scrollHandler) {
        this.scrollHandler = new ConnectedOverlayScrollHandler(this.target, function() {
          if (_this3.visible) {
            _this3.visible = false;
          }
        });
      }
      this.scrollHandler.bindScrollListener();
    }, "bindScrollListener"),
    unbindScrollListener: /* @__PURE__ */ __name(function unbindScrollListener2() {
      if (this.scrollHandler) {
        this.scrollHandler.unbindScrollListener();
      }
    }, "unbindScrollListener"),
    bindResizeListener: /* @__PURE__ */ __name(function bindResizeListener2() {
      var _this4 = this;
      if (!this.resizeListener) {
        this.resizeListener = function() {
          if (_this4.visible && !isTouchDevice()) {
            _this4.visible = false;
          }
        };
        window.addEventListener("resize", this.resizeListener);
      }
    }, "bindResizeListener"),
    unbindResizeListener: /* @__PURE__ */ __name(function unbindResizeListener2() {
      if (this.resizeListener) {
        window.removeEventListener("resize", this.resizeListener);
        this.resizeListener = null;
      }
    }, "unbindResizeListener"),
    focus: /* @__PURE__ */ __name(function focus2() {
      var focusTarget = this.container.querySelector("[autofocus]");
      if (focusTarget) {
        focusTarget.focus({
          preventScroll: true
        });
      }
    }, "focus"),
    isTargetClicked: /* @__PURE__ */ __name(function isTargetClicked(event) {
      return this.target && (this.target === event.target || this.target.contains(event.target));
    }, "isTargetClicked"),
    containerRef: /* @__PURE__ */ __name(function containerRef(el) {
      this.container = el;
    }, "containerRef"),
    onOverlayClick: /* @__PURE__ */ __name(function onOverlayClick2(event) {
      OverlayEventBus.emit("overlay-click", {
        originalEvent: event,
        target: this.target
      });
    }, "onOverlayClick"),
    onOverlayKeydown: /* @__PURE__ */ __name(function onOverlayKeydown(event) {
      if (event.code === "Escape") {
        ConfirmationEventBus.emit("close", this.closeListener);
        focus(this.target);
      }
    }, "onOverlayKeydown")
  },
  computed: {
    message: /* @__PURE__ */ __name(function message() {
      return this.confirmation ? this.confirmation.message : null;
    }, "message"),
    acceptLabel: /* @__PURE__ */ __name(function acceptLabel() {
      if (this.confirmation) {
        var _confirmation$acceptP;
        var confirmation = this.confirmation;
        return confirmation.acceptLabel || ((_confirmation$acceptP = confirmation.acceptProps) === null || _confirmation$acceptP === void 0 ? void 0 : _confirmation$acceptP.label) || this.$primevue.config.locale.accept;
      }
      return this.$primevue.config.locale.accept;
    }, "acceptLabel"),
    rejectLabel: /* @__PURE__ */ __name(function rejectLabel() {
      if (this.confirmation) {
        var _confirmation$rejectP;
        var confirmation = this.confirmation;
        return confirmation.rejectLabel || ((_confirmation$rejectP = confirmation.rejectProps) === null || _confirmation$rejectP === void 0 ? void 0 : _confirmation$rejectP.label) || this.$primevue.config.locale.reject;
      }
      return this.$primevue.config.locale.reject;
    }, "rejectLabel"),
    acceptIcon: /* @__PURE__ */ __name(function acceptIcon() {
      var _this$confirmation;
      return this.confirmation ? this.confirmation.acceptIcon : (_this$confirmation = this.confirmation) !== null && _this$confirmation !== void 0 && _this$confirmation.acceptProps ? this.confirmation.acceptProps.icon : null;
    }, "acceptIcon"),
    rejectIcon: /* @__PURE__ */ __name(function rejectIcon() {
      var _this$confirmation2;
      return this.confirmation ? this.confirmation.rejectIcon : (_this$confirmation2 = this.confirmation) !== null && _this$confirmation2 !== void 0 && _this$confirmation2.rejectProps ? this.confirmation.rejectProps.icon : null;
    }, "rejectIcon")
  },
  components: {
    Button: script$o,
    Portal: script$r
  },
  directives: {
    focustrap: FocusTrap
  }
};
var _hoisted_1$t = ["aria-modal"];
function render$d(_ctx, _cache, $props, $setup, $data, $options) {
  var _component_Button = resolveComponent("Button");
  var _component_Portal = resolveComponent("Portal");
  var _directive_focustrap = resolveDirective("focustrap");
  return openBlock(), createBlock(_component_Portal, null, {
    "default": withCtx(function() {
      return [createVNode(Transition, mergeProps({
        name: "p-confirmpopup",
        onEnter: $options.onEnter,
        onAfterEnter: $options.onAfterEnter,
        onLeave: $options.onLeave,
        onAfterLeave: $options.onAfterLeave
      }, _ctx.ptm("transition")), {
        "default": withCtx(function() {
          var _$data$confirmation$r, _$data$confirmation$r2, _$data$confirmation$a;
          return [$data.visible ? withDirectives((openBlock(), createElementBlock("div", mergeProps({
            key: 0,
            ref: $options.containerRef,
            role: "alertdialog",
            "class": _ctx.cx("root"),
            "aria-modal": $data.visible,
            onClick: _cache[2] || (_cache[2] = function() {
              return $options.onOverlayClick && $options.onOverlayClick.apply($options, arguments);
            }),
            onKeydown: _cache[3] || (_cache[3] = function() {
              return $options.onOverlayKeydown && $options.onOverlayKeydown.apply($options, arguments);
            })
          }, _ctx.ptmi("root")), [_ctx.$slots.container ? renderSlot(_ctx.$slots, "container", {
            key: 0,
            message: $data.confirmation,
            acceptCallback: $options.accept,
            rejectCallback: $options.reject
          }) : (openBlock(), createElementBlock(Fragment, {
            key: 1
          }, [!_ctx.$slots.message ? (openBlock(), createElementBlock("div", mergeProps({
            key: 0,
            "class": _ctx.cx("content")
          }, _ctx.ptm("content")), [renderSlot(_ctx.$slots, "icon", {}, function() {
            return [_ctx.$slots.icon ? (openBlock(), createBlock(resolveDynamicComponent(_ctx.$slots.icon), {
              key: 0,
              "class": normalizeClass(_ctx.cx("icon"))
            }, null, 8, ["class"])) : $data.confirmation.icon ? (openBlock(), createElementBlock("span", mergeProps({
              key: 1,
              "class": [$data.confirmation.icon, _ctx.cx("icon")]
            }, _ctx.ptm("icon")), null, 16)) : createCommentVNode("", true)];
          }), createBaseVNode("span", mergeProps({
            "class": _ctx.cx("message")
          }, _ctx.ptm("message")), toDisplayString($data.confirmation.message), 17)], 16)) : (openBlock(), createBlock(resolveDynamicComponent(_ctx.$slots.message), {
            key: 1,
            message: $data.confirmation
          }, null, 8, ["message"])), createBaseVNode("div", mergeProps({
            "class": _ctx.cx("footer")
          }, _ctx.ptm("footer")), [createVNode(_component_Button, mergeProps({
            "class": [_ctx.cx("pcRejectButton"), $data.confirmation.rejectClass],
            autofocus: $data.autoFocusReject,
            unstyled: _ctx.unstyled,
            size: ((_$data$confirmation$r = $data.confirmation.rejectProps) === null || _$data$confirmation$r === void 0 ? void 0 : _$data$confirmation$r.size) || "small",
            text: ((_$data$confirmation$r2 = $data.confirmation.rejectProps) === null || _$data$confirmation$r2 === void 0 ? void 0 : _$data$confirmation$r2.text) || false,
            onClick: _cache[0] || (_cache[0] = function($event) {
              return $options.reject();
            }),
            onKeydown: $options.onRejectKeydown
          }, $data.confirmation.rejectProps, {
            label: $options.rejectLabel,
            pt: _ctx.ptm("pcRejectButton")
          }), createSlots({
            _: 2
          }, [$options.rejectIcon || _ctx.$slots.rejecticon ? {
            name: "icon",
            fn: withCtx(function(iconProps) {
              return [renderSlot(_ctx.$slots, "rejecticon", {}, function() {
                return [createBaseVNode("span", mergeProps({
                  "class": [$options.rejectIcon, iconProps["class"]]
                }, _ctx.ptm("pcRejectButton")["icon"], {
                  "data-pc-section": "rejectbuttonicon"
                }), null, 16)];
              })];
            }),
            key: "0"
          } : void 0]), 1040, ["class", "autofocus", "unstyled", "size", "text", "onKeydown", "label", "pt"]), createVNode(_component_Button, mergeProps({
            "class": [_ctx.cx("pcAcceptButton"), $data.confirmation.acceptClass],
            autofocus: $data.autoFocusAccept,
            unstyled: _ctx.unstyled,
            size: ((_$data$confirmation$a = $data.confirmation.acceptProps) === null || _$data$confirmation$a === void 0 ? void 0 : _$data$confirmation$a.size) || "small",
            onClick: _cache[1] || (_cache[1] = function($event) {
              return $options.accept();
            }),
            onKeydown: $options.onAcceptKeydown
          }, $data.confirmation.acceptProps, {
            label: $options.acceptLabel,
            pt: _ctx.ptm("pcAcceptButton")
          }), createSlots({
            _: 2
          }, [$options.acceptIcon || _ctx.$slots.accepticon ? {
            name: "icon",
            fn: withCtx(function(iconProps) {
              return [renderSlot(_ctx.$slots, "accepticon", {}, function() {
                return [createBaseVNode("span", mergeProps({
                  "class": [$options.acceptIcon, iconProps["class"]]
                }, _ctx.ptm("pcAcceptButton")["icon"], {
                  "data-pc-section": "acceptbuttonicon"
                }), null, 16)];
              })];
            }),
            key: "0"
          } : void 0]), 1040, ["class", "autofocus", "unstyled", "size", "onKeydown", "label", "pt"])], 16)], 64))], 16, _hoisted_1$t)), [[_directive_focustrap]]) : createCommentVNode("", true)];
        }),
        _: 3
      }, 16, ["onEnter", "onAfterEnter", "onLeave", "onAfterLeave"])];
    }),
    _: 3
  });
}
__name(render$d, "render$d");
script$e.render = render$d;
var theme$a = /* @__PURE__ */ __name(function theme8(_ref) {
  var dt = _ref.dt;
  return "\n.p-contextmenu {\n    background: ".concat(dt("contextmenu.background"), ";\n    color: ").concat(dt("contextmenu.color"), ";\n    border: 1px solid ").concat(dt("contextmenu.border.color"), ";\n    border-radius: ").concat(dt("contextmenu.border.radius"), ";\n    box-shadow: ").concat(dt("contextmenu.shadow"), ";\n    min-width: 12.5rem;\n}\n\n.p-contextmenu-root-list,\n.p-contextmenu-submenu {\n    margin: 0;\n    padding: ").concat(dt("contextmenu.list.padding"), ";\n    list-style: none;\n    outline: 0 none;\n    display: flex;\n    flex-direction: column;\n    gap: ").concat(dt("contextmenu.list.gap"), ";\n}\n\n.p-contextmenu-submenu {\n    position: absolute;\n    display: flex;\n    flex-direction: column;\n    min-width: 100%;\n    z-index: 1;\n    background: ").concat(dt("contextmenu.background"), ";\n    color: ").concat(dt("contextmenu.color"), ";\n    border: 1px solid ").concat(dt("contextmenu.border.color"), ";\n    border-radius: ").concat(dt("contextmenu.border.radius"), ";\n    box-shadow: ").concat(dt("contextmenu.shadow"), ";\n}\n\n.p-contextmenu-item {\n    position: relative;\n}\n\n.p-contextmenu-item-content {\n    transition: background ").concat(dt("contextmenu.transition.duration"), ", color ").concat(dt("contextmenu.transition.duration"), ";\n    border-radius: ").concat(dt("contextmenu.item.border.radius"), ";\n    color: ").concat(dt("contextmenu.item.color"), ";\n}\n\n.p-contextmenu-item-link {\n    cursor: pointer;\n    display: flex;\n    align-items: center;\n    text-decoration: none;\n    overflow: hidden;\n    position: relative;\n    color: inherit;\n    padding: ").concat(dt("contextmenu.item.padding"), ";\n    gap: ").concat(dt("contextmenu.item.gap"), ";\n    user-select: none;\n}\n\n.p-contextmenu-item-label {\n    line-height: 1;\n}\n\n.p-contextmenu-item-icon {\n    color: ").concat(dt("contextmenu.item.icon.color"), ";\n}\n\n.p-contextmenu-submenu-icon {\n    color: ").concat(dt("contextmenu.submenu.icon.color"), ";\n    margin-left: auto;\n    font-size: ").concat(dt("contextmenu.submenu.icon.size"), ";\n    width: ").concat(dt("contextmenu.submenu.icon.size"), ";\n    height: ").concat(dt("contextmenu.submenu.icon.size"), ";\n}\n\n.p-contextmenu-item.p-focus > .p-contextmenu-item-content {\n    color: ").concat(dt("contextmenu.item.focus.color"), ";\n    background: ").concat(dt("contextmenu.item.focus.background"), ";\n}\n\n.p-contextmenu-item.p-focus > .p-contextmenu-item-content .p-contextmenu-item-icon {\n    color: ").concat(dt("contextmenu.item.icon.focus.color"), ";\n}\n\n.p-contextmenu-item.p-focus > .p-contextmenu-item-content .p-contextmenu-submenu-icon {\n    color: ").concat(dt("contextmenu.submenu.icon.focus.color"), ";\n}\n\n.p-contextmenu-item:not(.p-disabled) > .p-contextmenu-item-content:hover {\n    color: ").concat(dt("contextmenu.item.focus.color"), ";\n    background: ").concat(dt("contextmenu.item.focus.background"), ";\n}\n\n.p-contextmenu-item:not(.p-disabled) > .p-contextmenu-item-content:hover .p-contextmenu-item-icon {\n    color: ").concat(dt("contextmenu.item.icon.focus.color"), ";\n}\n\n.p-contextmenu-item:not(.p-disabled) > .p-contextmenu-item-content:hover .p-contextmenu-submenu-icon {\n    color: ").concat(dt("contextmenu.submenu.icon.focus.color"), ";\n}\n\n.p-contextmenu-item-active > .p-contextmenu-item-content {\n    color: ").concat(dt("contextmenu.item.active.color"), ";\n    background: ").concat(dt("contextmenu.item.active.background"), ";\n}\n\n.p-contextmenu-item-active > .p-contextmenu-item-content .p-contextmenu-item-icon {\n    color: ").concat(dt("contextmenu.item.icon.active.color"), ";\n}\n\n.p-contextmenu-item-active > .p-contextmenu-item-content .p-contextmenu-submenu-icon {\n    color: ").concat(dt("contextmenu.submenu.icon.active.color"), ";\n}\n\n.p-contextmenu-separator {\n    border-top: 1px solid  ").concat(dt("contextmenu.separator.border.color"), ";\n}\n\n.p-contextmenu-enter-from,\n.p-contextmenu-leave-active {\n    opacity: 0;\n}\n\n.p-contextmenu-enter-active {\n    transition: opacity 250ms;\n}\n");
}, "theme");
var classes$a = {
  root: "p-contextmenu p-component",
  rootList: "p-contextmenu-root-list",
  item: /* @__PURE__ */ __name(function item(_ref2) {
    var instance = _ref2.instance, processedItem = _ref2.processedItem;
    return ["p-contextmenu-item", {
      "p-contextmenu-item-active": instance.isItemActive(processedItem),
      "p-focus": instance.isItemFocused(processedItem),
      "p-disabled": instance.isItemDisabled(processedItem)
    }];
  }, "item"),
  itemContent: "p-contextmenu-item-content",
  itemLink: "p-contextmenu-item-link",
  itemIcon: "p-contextmenu-item-icon",
  itemLabel: "p-contextmenu-item-label",
  submenuIcon: "p-contextmenu-submenu-icon",
  submenu: "p-contextmenu-submenu",
  separator: "p-contextmenu-separator"
};
var ContextMenuStyle = BaseStyle.extend({
  name: "contextmenu",
  theme: theme$a,
  classes: classes$a
});
var script$2$5 = {
  name: "BaseContextMenu",
  "extends": script$p,
  props: {
    model: {
      type: Array,
      "default": null
    },
    appendTo: {
      type: [String, Object],
      "default": "body"
    },
    autoZIndex: {
      type: Boolean,
      "default": true
    },
    baseZIndex: {
      type: Number,
      "default": 0
    },
    global: {
      type: Boolean,
      "default": false
    },
    tabindex: {
      type: Number,
      "default": 0
    },
    ariaLabelledby: {
      type: String,
      "default": null
    },
    ariaLabel: {
      type: String,
      "default": null
    }
  },
  style: ContextMenuStyle,
  provide: /* @__PURE__ */ __name(function provide10() {
    return {
      $pcContextMenu: this,
      $parentInstance: this
    };
  }, "provide")
};
var script$1$a = {
  name: "ContextMenuSub",
  hostName: "ContextMenu",
  "extends": script$p,
  emits: ["item-click", "item-mouseenter", "item-mousemove"],
  props: {
    items: {
      type: Array,
      "default": null
    },
    menuId: {
      type: String,
      "default": null
    },
    focusedItemId: {
      type: String,
      "default": null
    },
    root: {
      type: Boolean,
      "default": false
    },
    visible: {
      type: Boolean,
      "default": false
    },
    level: {
      type: Number,
      "default": 0
    },
    templates: {
      type: Object,
      "default": null
    },
    activeItemPath: {
      type: Object,
      "default": null
    },
    tabindex: {
      type: Number,
      "default": 0
    }
  },
  methods: {
    getItemId: /* @__PURE__ */ __name(function getItemId(processedItem) {
      return "".concat(this.menuId, "_").concat(processedItem.key);
    }, "getItemId"),
    getItemKey: /* @__PURE__ */ __name(function getItemKey(processedItem) {
      return this.getItemId(processedItem);
    }, "getItemKey"),
    getItemProp: /* @__PURE__ */ __name(function getItemProp(processedItem, name, params) {
      return processedItem && processedItem.item ? resolve(processedItem.item[name], params) : void 0;
    }, "getItemProp"),
    getItemLabel: /* @__PURE__ */ __name(function getItemLabel(processedItem) {
      return this.getItemProp(processedItem, "label");
    }, "getItemLabel"),
    getItemLabelId: /* @__PURE__ */ __name(function getItemLabelId(processedItem) {
      return "".concat(this.menuId, "_").concat(processedItem.key, "_label");
    }, "getItemLabelId"),
    getPTOptions: /* @__PURE__ */ __name(function getPTOptions6(key, processedItem, index2) {
      return this.ptm(key, {
        context: {
          item: processedItem.item,
          active: this.isItemActive(processedItem),
          focused: this.isItemFocused(processedItem),
          disabled: this.isItemDisabled(processedItem),
          index: index2
        }
      });
    }, "getPTOptions"),
    isItemActive: /* @__PURE__ */ __name(function isItemActive(processedItem) {
      return this.activeItemPath.some(function(path) {
        return path.key === processedItem.key;
      });
    }, "isItemActive"),
    isItemVisible: /* @__PURE__ */ __name(function isItemVisible(processedItem) {
      return this.getItemProp(processedItem, "visible") !== false;
    }, "isItemVisible"),
    isItemDisabled: /* @__PURE__ */ __name(function isItemDisabled(processedItem) {
      return this.getItemProp(processedItem, "disabled");
    }, "isItemDisabled"),
    isItemFocused: /* @__PURE__ */ __name(function isItemFocused(processedItem) {
      return this.focusedItemId === this.getItemId(processedItem);
    }, "isItemFocused"),
    isItemGroup: /* @__PURE__ */ __name(function isItemGroup(processedItem) {
      return isNotEmpty(processedItem.items);
    }, "isItemGroup"),
    onItemClick: /* @__PURE__ */ __name(function onItemClick(event, processedItem) {
      this.getItemProp(processedItem, "command", {
        originalEvent: event,
        item: processedItem.item
      });
      this.$emit("item-click", {
        originalEvent: event,
        processedItem,
        isFocus: true
      });
    }, "onItemClick"),
    onItemMouseEnter: /* @__PURE__ */ __name(function onItemMouseEnter(event, processedItem) {
      this.$emit("item-mouseenter", {
        originalEvent: event,
        processedItem
      });
    }, "onItemMouseEnter"),
    onItemMouseMove: /* @__PURE__ */ __name(function onItemMouseMove(event, processedItem) {
      this.$emit("item-mousemove", {
        originalEvent: event,
        processedItem,
        isFocus: true
      });
    }, "onItemMouseMove"),
    getAriaSetSize: /* @__PURE__ */ __name(function getAriaSetSize() {
      var _this = this;
      return this.items.filter(function(processedItem) {
        return _this.isItemVisible(processedItem) && !_this.getItemProp(processedItem, "separator");
      }).length;
    }, "getAriaSetSize"),
    getAriaPosInset: /* @__PURE__ */ __name(function getAriaPosInset2(index2) {
      var _this2 = this;
      return index2 - this.items.slice(0, index2).filter(function(processedItem) {
        return _this2.isItemVisible(processedItem) && _this2.getItemProp(processedItem, "separator");
      }).length + 1;
    }, "getAriaPosInset"),
    onEnter: /* @__PURE__ */ __name(function onEnter2() {
      nestedPosition(this.$refs.container, this.level);
    }, "onEnter"),
    getMenuItemProps: /* @__PURE__ */ __name(function getMenuItemProps(processedItem, index2) {
      return {
        action: mergeProps({
          "class": this.cx("itemLink"),
          tabindex: -1,
          "aria-hidden": true
        }, this.getPTOptions("itemLink", processedItem, index2)),
        icon: mergeProps({
          "class": [this.cx("itemIcon"), this.getItemProp(processedItem, "icon")]
        }, this.getPTOptions("itemIcon", processedItem, index2)),
        label: mergeProps({
          "class": this.cx("itemLabel")
        }, this.getPTOptions("itemLabel", processedItem, index2)),
        submenuicon: mergeProps({
          "class": this.cx("submenuIcon")
        }, this.getPTOptions("submenuicon", processedItem, index2))
      };
    }, "getMenuItemProps")
  },
  components: {
    AngleRightIcon: script$y
  },
  directives: {
    ripple: Ripple
  }
};
var _hoisted_1$s = ["tabindex"];
var _hoisted_2$l = ["id", "aria-label", "aria-disabled", "aria-expanded", "aria-haspopup", "aria-level", "aria-setsize", "aria-posinset", "data-p-active", "data-p-focused", "data-p-disabled"];
var _hoisted_3$f = ["onClick", "onMouseenter", "onMousemove"];
var _hoisted_4$a = ["href", "target"];
var _hoisted_5$7 = ["id"];
var _hoisted_6$5 = ["id"];
function render$1$5(_ctx, _cache, $props, $setup, $data, $options) {
  var _component_AngleRightIcon = resolveComponent("AngleRightIcon");
  var _component_ContextMenuSub = resolveComponent("ContextMenuSub", true);
  var _directive_ripple = resolveDirective("ripple");
  return openBlock(), createBlock(Transition, mergeProps({
    name: "p-contextmenusub",
    onEnter: $options.onEnter
  }, _ctx.ptm("menu.transition")), {
    "default": withCtx(function() {
      return [($props.root ? true : $props.visible) ? (openBlock(), createElementBlock("ul", mergeProps({
        key: 0,
        ref: "container",
        tabindex: $props.tabindex
      }, _ctx.ptm("rootList")), [(openBlock(true), createElementBlock(Fragment, null, renderList($props.items, function(processedItem, index2) {
        return openBlock(), createElementBlock(Fragment, {
          key: $options.getItemKey(processedItem)
        }, [$options.isItemVisible(processedItem) && !$options.getItemProp(processedItem, "separator") ? (openBlock(), createElementBlock("li", mergeProps({
          key: 0,
          id: $options.getItemId(processedItem),
          style: $options.getItemProp(processedItem, "style"),
          "class": [_ctx.cx("item", {
            processedItem
          }), $options.getItemProp(processedItem, "class")],
          role: "menuitem",
          "aria-label": $options.getItemLabel(processedItem),
          "aria-disabled": $options.isItemDisabled(processedItem) || void 0,
          "aria-expanded": $options.isItemGroup(processedItem) ? $options.isItemActive(processedItem) : void 0,
          "aria-haspopup": $options.isItemGroup(processedItem) && !$options.getItemProp(processedItem, "to") ? "menu" : void 0,
          "aria-level": $props.level + 1,
          "aria-setsize": $options.getAriaSetSize(),
          "aria-posinset": $options.getAriaPosInset(index2),
          ref_for: true
        }, $options.getPTOptions("item", processedItem, index2), {
          "data-p-active": $options.isItemActive(processedItem),
          "data-p-focused": $options.isItemFocused(processedItem),
          "data-p-disabled": $options.isItemDisabled(processedItem)
        }), [createBaseVNode("div", mergeProps({
          "class": _ctx.cx("itemContent"),
          onClick: /* @__PURE__ */ __name(function onClick2($event) {
            return $options.onItemClick($event, processedItem);
          }, "onClick"),
          onMouseenter: /* @__PURE__ */ __name(function onMouseenter($event) {
            return $options.onItemMouseEnter($event, processedItem);
          }, "onMouseenter"),
          onMousemove: /* @__PURE__ */ __name(function onMousemove($event) {
            return $options.onItemMouseMove($event, processedItem);
          }, "onMousemove"),
          ref_for: true
        }, $options.getPTOptions("itemContent", processedItem, index2)), [!$props.templates.item ? withDirectives((openBlock(), createElementBlock("a", mergeProps({
          key: 0,
          href: $options.getItemProp(processedItem, "url"),
          "class": _ctx.cx("itemLink"),
          target: $options.getItemProp(processedItem, "target"),
          tabindex: "-1",
          ref_for: true
        }, $options.getPTOptions("itemLink", processedItem, index2)), [$props.templates.itemicon ? (openBlock(), createBlock(resolveDynamicComponent($props.templates.itemicon), {
          key: 0,
          item: processedItem.item,
          "class": normalizeClass(_ctx.cx("itemIcon"))
        }, null, 8, ["item", "class"])) : $options.getItemProp(processedItem, "icon") ? (openBlock(), createElementBlock("span", mergeProps({
          key: 1,
          "class": [_ctx.cx("itemIcon"), $options.getItemProp(processedItem, "icon")],
          ref_for: true
        }, $options.getPTOptions("itemIcon", processedItem, index2)), null, 16)) : createCommentVNode("", true), createBaseVNode("span", mergeProps({
          id: $options.getItemLabelId(processedItem),
          "class": _ctx.cx("itemLabel"),
          ref_for: true
        }, $options.getPTOptions("itemLabel", processedItem, index2)), toDisplayString($options.getItemLabel(processedItem)), 17, _hoisted_5$7), $options.getItemProp(processedItem, "items") ? (openBlock(), createElementBlock(Fragment, {
          key: 2
        }, [$props.templates.submenuicon ? (openBlock(), createBlock(resolveDynamicComponent($props.templates.submenuicon), {
          key: 0,
          active: $options.isItemActive(processedItem),
          "class": normalizeClass(_ctx.cx("submenuIcon"))
        }, null, 8, ["active", "class"])) : (openBlock(), createBlock(_component_AngleRightIcon, mergeProps({
          key: 1,
          "class": _ctx.cx("submenuIcon"),
          ref_for: true
        }, $options.getPTOptions("submenuicon", processedItem, index2)), null, 16, ["class"]))], 64)) : createCommentVNode("", true)], 16, _hoisted_4$a)), [[_directive_ripple]]) : (openBlock(), createBlock(resolveDynamicComponent($props.templates.item), {
          key: 1,
          item: processedItem.item,
          hasSubmenu: $options.getItemProp(processedItem, "items"),
          label: $options.getItemLabel(processedItem),
          props: $options.getMenuItemProps(processedItem, index2)
        }, null, 8, ["item", "hasSubmenu", "label", "props"]))], 16, _hoisted_3$f), $options.isItemVisible(processedItem) && $options.isItemGroup(processedItem) ? (openBlock(), createBlock(_component_ContextMenuSub, mergeProps({
          key: 0,
          id: $options.getItemId(processedItem) + "_list",
          role: "menu",
          "class": _ctx.cx("submenu"),
          menuId: $props.menuId,
          focusedItemId: $props.focusedItemId,
          items: processedItem.items,
          templates: $props.templates,
          activeItemPath: $props.activeItemPath,
          level: $props.level + 1,
          visible: $options.isItemActive(processedItem) && $options.isItemGroup(processedItem),
          pt: _ctx.pt,
          unstyled: _ctx.unstyled,
          onItemClick: _cache[0] || (_cache[0] = function($event) {
            return _ctx.$emit("item-click", $event);
          }),
          onItemMouseenter: _cache[1] || (_cache[1] = function($event) {
            return _ctx.$emit("item-mouseenter", $event);
          }),
          onItemMousemove: _cache[2] || (_cache[2] = function($event) {
            return _ctx.$emit("item-mousemove", $event);
          }),
          "aria-labelledby": $options.getItemLabelId(processedItem),
          ref_for: true
        }, _ctx.ptm("submenu")), null, 16, ["id", "class", "menuId", "focusedItemId", "items", "templates", "activeItemPath", "level", "visible", "pt", "unstyled", "aria-labelledby"])) : createCommentVNode("", true)], 16, _hoisted_2$l)) : createCommentVNode("", true), $options.isItemVisible(processedItem) && $options.getItemProp(processedItem, "separator") ? (openBlock(), createElementBlock("li", mergeProps({
          key: 1,
          id: $options.getItemId(processedItem),
          style: $options.getItemProp(processedItem, "style"),
          "class": [_ctx.cx("separator"), $options.getItemProp(processedItem, "class")],
          role: "separator",
          ref_for: true
        }, _ctx.ptm("separator")), null, 16, _hoisted_6$5)) : createCommentVNode("", true)], 64);
      }), 128))], 16, _hoisted_1$s)) : createCommentVNode("", true)];
    }),
    _: 1
  }, 16, ["onEnter"]);
}
__name(render$1$5, "render$1$5");
script$1$a.render = render$1$5;
var script$d = {
  name: "ContextMenu",
  "extends": script$2$5,
  inheritAttrs: false,
  emits: ["focus", "blur", "show", "hide", "before-show", "before-hide"],
  target: null,
  outsideClickListener: null,
  resizeListener: null,
  documentContextMenuListener: null,
  pageX: null,
  pageY: null,
  container: null,
  list: null,
  data: /* @__PURE__ */ __name(function data5() {
    return {
      id: this.$attrs.id,
      focused: false,
      focusedItemInfo: {
        index: -1,
        level: 0,
        parentKey: ""
      },
      activeItemPath: [],
      visible: false,
      submenuVisible: false
    };
  }, "data"),
  watch: {
    "$attrs.id": /* @__PURE__ */ __name(function $attrsId2(newValue) {
      this.id = newValue || UniqueComponentId();
    }, "$attrsId"),
    activeItemPath: /* @__PURE__ */ __name(function activeItemPath(newPath) {
      if (isNotEmpty(newPath)) {
        this.bindOutsideClickListener();
        this.bindResizeListener();
      } else if (!this.visible) {
        this.unbindOutsideClickListener();
        this.unbindResizeListener();
      }
    }, "activeItemPath")
  },
  mounted: /* @__PURE__ */ __name(function mounted4() {
    this.id = this.id || UniqueComponentId();
    if (this.global) {
      this.bindDocumentContextMenuListener();
    }
  }, "mounted"),
  beforeUnmount: /* @__PURE__ */ __name(function beforeUnmount4() {
    this.unbindResizeListener();
    this.unbindOutsideClickListener();
    this.unbindDocumentContextMenuListener();
    if (this.container && this.autoZIndex) {
      ZIndex.clear(this.container);
    }
    this.target = null;
    this.container = null;
  }, "beforeUnmount"),
  methods: {
    getItemProp: /* @__PURE__ */ __name(function getItemProp2(item4, name) {
      return item4 ? resolve(item4[name]) : void 0;
    }, "getItemProp"),
    getItemLabel: /* @__PURE__ */ __name(function getItemLabel2(item4) {
      return this.getItemProp(item4, "label");
    }, "getItemLabel"),
    isItemDisabled: /* @__PURE__ */ __name(function isItemDisabled2(item4) {
      return this.getItemProp(item4, "disabled");
    }, "isItemDisabled"),
    isItemVisible: /* @__PURE__ */ __name(function isItemVisible2(item4) {
      return this.getItemProp(item4, "visible") !== false;
    }, "isItemVisible"),
    isItemGroup: /* @__PURE__ */ __name(function isItemGroup2(item4) {
      return isNotEmpty(this.getItemProp(item4, "items"));
    }, "isItemGroup"),
    isItemSeparator: /* @__PURE__ */ __name(function isItemSeparator(item4) {
      return this.getItemProp(item4, "separator");
    }, "isItemSeparator"),
    getProccessedItemLabel: /* @__PURE__ */ __name(function getProccessedItemLabel(processedItem) {
      return processedItem ? this.getItemLabel(processedItem.item) : void 0;
    }, "getProccessedItemLabel"),
    isProccessedItemGroup: /* @__PURE__ */ __name(function isProccessedItemGroup(processedItem) {
      return processedItem && isNotEmpty(processedItem.items);
    }, "isProccessedItemGroup"),
    toggle: /* @__PURE__ */ __name(function toggle(event) {
      this.visible ? this.hide() : this.show(event);
    }, "toggle"),
    show: /* @__PURE__ */ __name(function show2(event) {
      this.$emit("before-show");
      this.activeItemPath = [];
      this.focusedItemInfo = {
        index: -1,
        level: 0,
        parentKey: ""
      };
      focus(this.list);
      this.pageX = event.pageX;
      this.pageY = event.pageY;
      this.visible ? this.position() : this.visible = true;
      event.stopPropagation();
      event.preventDefault();
    }, "show"),
    hide: /* @__PURE__ */ __name(function hide2() {
      this.$emit("before-hide");
      this.visible = false;
      this.activeItemPath = [];
      this.focusedItemInfo = {
        index: -1,
        level: 0,
        parentKey: ""
      };
    }, "hide"),
    onFocus: /* @__PURE__ */ __name(function onFocus2(event) {
      this.focused = true;
      this.focusedItemInfo = this.focusedItemInfo.index !== -1 ? this.focusedItemInfo : {
        index: -1,
        level: 0,
        parentKey: ""
      };
      this.$emit("focus", event);
    }, "onFocus"),
    onBlur: /* @__PURE__ */ __name(function onBlur2(event) {
      this.focused = false;
      this.focusedItemInfo = {
        index: -1,
        level: 0,
        parentKey: ""
      };
      this.searchValue = "";
      this.$emit("blur", event);
    }, "onBlur"),
    onKeyDown: /* @__PURE__ */ __name(function onKeyDown2(event) {
      var metaKey = event.metaKey || event.ctrlKey;
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
        case "Space":
          this.onSpaceKey(event);
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
        case "PageDown":
        case "PageUp":
        case "Backspace":
        case "ShiftLeft":
        case "ShiftRight":
          break;
        default:
          if (!metaKey && isPrintableCharacter(event.key)) {
            this.searchItems(event, event.key);
          }
          break;
      }
    }, "onKeyDown"),
    onItemChange: /* @__PURE__ */ __name(function onItemChange(event) {
      var processedItem = event.processedItem, isFocus = event.isFocus;
      if (isEmpty(processedItem)) return;
      var index2 = processedItem.index, key = processedItem.key, level = processedItem.level, parentKey = processedItem.parentKey, items = processedItem.items;
      var grouped = isNotEmpty(items);
      var activeItemPath4 = this.activeItemPath.filter(function(p) {
        return p.parentKey !== parentKey && p.parentKey !== key;
      });
      if (grouped) {
        activeItemPath4.push(processedItem);
        this.submenuVisible = true;
      }
      this.focusedItemInfo = {
        index: index2,
        level,
        parentKey
      };
      this.activeItemPath = activeItemPath4;
      isFocus && focus(this.list);
    }, "onItemChange"),
    onItemClick: /* @__PURE__ */ __name(function onItemClick2(event) {
      var processedItem = event.processedItem;
      var grouped = this.isProccessedItemGroup(processedItem);
      var selected2 = this.isSelected(processedItem);
      if (selected2) {
        var index2 = processedItem.index, key = processedItem.key, level = processedItem.level, parentKey = processedItem.parentKey;
        this.activeItemPath = this.activeItemPath.filter(function(p) {
          return key !== p.key && key.startsWith(p.key);
        });
        this.focusedItemInfo = {
          index: index2,
          level,
          parentKey
        };
        focus(this.list);
      } else {
        grouped ? this.onItemChange(event) : this.hide();
      }
    }, "onItemClick"),
    onItemMouseEnter: /* @__PURE__ */ __name(function onItemMouseEnter2(event) {
      this.onItemChange(event);
    }, "onItemMouseEnter"),
    onItemMouseMove: /* @__PURE__ */ __name(function onItemMouseMove2(event) {
      if (this.focused) {
        this.changeFocusedItemIndex(event, event.processedItem.index);
      }
    }, "onItemMouseMove"),
    onArrowDownKey: /* @__PURE__ */ __name(function onArrowDownKey2(event) {
      var itemIndex = this.focusedItemInfo.index !== -1 ? this.findNextItemIndex(this.focusedItemInfo.index) : this.findFirstFocusedItemIndex();
      this.changeFocusedItemIndex(event, itemIndex);
      event.preventDefault();
    }, "onArrowDownKey"),
    onArrowUpKey: /* @__PURE__ */ __name(function onArrowUpKey2(event) {
      if (event.altKey) {
        if (this.focusedItemInfo.index !== -1) {
          var processedItem = this.visibleItems[this.focusedItemInfo.index];
          var grouped = this.isProccessedItemGroup(processedItem);
          !grouped && this.onItemChange({
            originalEvent: event,
            processedItem
          });
        }
        this.popup && this.hide();
        event.preventDefault();
      } else {
        var itemIndex = this.focusedItemInfo.index !== -1 ? this.findPrevItemIndex(this.focusedItemInfo.index) : this.findLastFocusedItemIndex();
        this.changeFocusedItemIndex(event, itemIndex);
        event.preventDefault();
      }
    }, "onArrowUpKey"),
    onArrowLeftKey: /* @__PURE__ */ __name(function onArrowLeftKey2(event) {
      var _this = this;
      var processedItem = this.visibleItems[this.focusedItemInfo.index];
      var parentItem = this.activeItemPath.find(function(p) {
        return p.key === processedItem.parentKey;
      });
      var root15 = isEmpty(processedItem.parent);
      if (!root15) {
        this.focusedItemInfo = {
          index: -1,
          parentKey: parentItem ? parentItem.parentKey : ""
        };
        this.searchValue = "";
        this.onArrowDownKey(event);
      }
      this.activeItemPath = this.activeItemPath.filter(function(p) {
        return p.parentKey !== _this.focusedItemInfo.parentKey;
      });
      event.preventDefault();
    }, "onArrowLeftKey"),
    onArrowRightKey: /* @__PURE__ */ __name(function onArrowRightKey2(event) {
      var processedItem = this.visibleItems[this.focusedItemInfo.index];
      var grouped = this.isProccessedItemGroup(processedItem);
      if (grouped) {
        this.onItemChange({
          originalEvent: event,
          processedItem
        });
        this.focusedItemInfo = {
          index: -1,
          parentKey: processedItem.key
        };
        this.searchValue = "";
        this.onArrowDownKey(event);
      }
      event.preventDefault();
    }, "onArrowRightKey"),
    onHomeKey: /* @__PURE__ */ __name(function onHomeKey2(event) {
      this.changeFocusedItemIndex(event, this.findFirstItemIndex());
      event.preventDefault();
    }, "onHomeKey"),
    onEndKey: /* @__PURE__ */ __name(function onEndKey2(event) {
      this.changeFocusedItemIndex(event, this.findLastItemIndex());
      event.preventDefault();
    }, "onEndKey"),
    onEnterKey: /* @__PURE__ */ __name(function onEnterKey2(event) {
      if (this.focusedItemInfo.index !== -1) {
        var element = findSingle(this.list, 'li[id="'.concat("".concat(this.focusedItemIdx), '"]'));
        var anchorElement = element && findSingle(element, '[data-pc-section="itemlink"]');
        anchorElement ? anchorElement.click() : element && element.click();
        var processedItem = this.visibleItems[this.focusedItemInfo.index];
        var grouped = this.isProccessedItemGroup(processedItem);
        !grouped && (this.focusedItemInfo.index = this.findFirstFocusedItemIndex());
      }
      event.preventDefault();
    }, "onEnterKey"),
    onSpaceKey: /* @__PURE__ */ __name(function onSpaceKey(event) {
      this.onEnterKey(event);
    }, "onSpaceKey"),
    onEscapeKey: /* @__PURE__ */ __name(function onEscapeKey2(event) {
      this.hide();
      !this.popup && (this.focusedItemInfo.index = this.findFirstFocusedItemIndex());
      event.preventDefault();
    }, "onEscapeKey"),
    onTabKey: /* @__PURE__ */ __name(function onTabKey2(event) {
      if (this.focusedItemInfo.index !== -1) {
        var processedItem = this.visibleItems[this.focusedItemInfo.index];
        var grouped = this.isProccessedItemGroup(processedItem);
        !grouped && this.onItemChange({
          originalEvent: event,
          processedItem
        });
      }
      this.hide();
    }, "onTabKey"),
    onEnter: /* @__PURE__ */ __name(function onEnter3(el) {
      addStyle(el, {
        position: "absolute"
      });
      this.position();
      if (this.autoZIndex) {
        ZIndex.set("menu", el, this.baseZIndex + this.$primevue.config.zIndex.menu);
      }
    }, "onEnter"),
    onAfterEnter: /* @__PURE__ */ __name(function onAfterEnter2() {
      this.bindOutsideClickListener();
      this.bindResizeListener();
      this.$emit("show");
      focus(this.list);
    }, "onAfterEnter"),
    onLeave: /* @__PURE__ */ __name(function onLeave2() {
      this.$emit("hide");
      this.container = null;
    }, "onLeave"),
    onAfterLeave: /* @__PURE__ */ __name(function onAfterLeave2(el) {
      if (this.autoZIndex) {
        ZIndex.clear(el);
      }
      this.unbindOutsideClickListener();
      this.unbindResizeListener();
    }, "onAfterLeave"),
    position: /* @__PURE__ */ __name(function position() {
      var left = this.pageX + 1;
      var top = this.pageY + 1;
      var width = this.container.offsetParent ? this.container.offsetWidth : getHiddenElementOuterWidth(this.container);
      var height = this.container.offsetParent ? this.container.offsetHeight : getHiddenElementOuterHeight(this.container);
      var viewport = getViewport();
      if (left + width - document.body.scrollLeft > viewport.width) {
        left -= width;
      }
      if (top + height - document.body.scrollTop > viewport.height) {
        top -= height;
      }
      if (left < document.body.scrollLeft) {
        left = document.body.scrollLeft;
      }
      if (top < document.body.scrollTop) {
        top = document.body.scrollTop;
      }
      this.container.style.left = left + "px";
      this.container.style.top = top + "px";
    }, "position"),
    bindOutsideClickListener: /* @__PURE__ */ __name(function bindOutsideClickListener3() {
      var _this2 = this;
      if (!this.outsideClickListener) {
        this.outsideClickListener = function(event) {
          var isOutsideContainer = _this2.container && !_this2.container.contains(event.target);
          var isOutsideTarget = _this2.visible ? !(_this2.target && (_this2.target === event.target || _this2.target.contains(event.target))) : true;
          if (isOutsideContainer && isOutsideTarget) {
            _this2.hide();
          }
        };
        document.addEventListener("click", this.outsideClickListener);
      }
    }, "bindOutsideClickListener"),
    unbindOutsideClickListener: /* @__PURE__ */ __name(function unbindOutsideClickListener3() {
      if (this.outsideClickListener) {
        document.removeEventListener("click", this.outsideClickListener);
        this.outsideClickListener = null;
      }
    }, "unbindOutsideClickListener"),
    bindResizeListener: /* @__PURE__ */ __name(function bindResizeListener3() {
      var _this3 = this;
      if (!this.resizeListener) {
        this.resizeListener = function() {
          if (_this3.visible && !isTouchDevice()) {
            _this3.hide();
          }
        };
        window.addEventListener("resize", this.resizeListener);
      }
    }, "bindResizeListener"),
    unbindResizeListener: /* @__PURE__ */ __name(function unbindResizeListener3() {
      if (this.resizeListener) {
        window.removeEventListener("resize", this.resizeListener);
        this.resizeListener = null;
      }
    }, "unbindResizeListener"),
    bindDocumentContextMenuListener: /* @__PURE__ */ __name(function bindDocumentContextMenuListener() {
      var _this4 = this;
      if (!this.documentContextMenuListener) {
        this.documentContextMenuListener = function(event) {
          event.button === 2 && _this4.show(event);
        };
        document.addEventListener("contextmenu", this.documentContextMenuListener);
      }
    }, "bindDocumentContextMenuListener"),
    unbindDocumentContextMenuListener: /* @__PURE__ */ __name(function unbindDocumentContextMenuListener() {
      if (this.documentContextMenuListener) {
        document.removeEventListener("contextmenu", this.documentContextMenuListener);
        this.documentContextMenuListener = null;
      }
    }, "unbindDocumentContextMenuListener"),
    isItemMatched: /* @__PURE__ */ __name(function isItemMatched(processedItem) {
      var _this$getProccessedIt;
      return this.isValidItem(processedItem) && ((_this$getProccessedIt = this.getProccessedItemLabel(processedItem)) === null || _this$getProccessedIt === void 0 ? void 0 : _this$getProccessedIt.toLocaleLowerCase().startsWith(this.searchValue.toLocaleLowerCase()));
    }, "isItemMatched"),
    isValidItem: /* @__PURE__ */ __name(function isValidItem(processedItem) {
      return !!processedItem && !this.isItemDisabled(processedItem.item) && !this.isItemSeparator(processedItem.item) && this.isItemVisible(processedItem.item);
    }, "isValidItem"),
    isValidSelectedItem: /* @__PURE__ */ __name(function isValidSelectedItem(processedItem) {
      return this.isValidItem(processedItem) && this.isSelected(processedItem);
    }, "isValidSelectedItem"),
    isSelected: /* @__PURE__ */ __name(function isSelected3(processedItem) {
      return this.activeItemPath.some(function(p) {
        return p.key === processedItem.key;
      });
    }, "isSelected"),
    findFirstItemIndex: /* @__PURE__ */ __name(function findFirstItemIndex() {
      var _this5 = this;
      return this.visibleItems.findIndex(function(processedItem) {
        return _this5.isValidItem(processedItem);
      });
    }, "findFirstItemIndex"),
    findLastItemIndex: /* @__PURE__ */ __name(function findLastItemIndex() {
      var _this6 = this;
      return findLastIndex(this.visibleItems, function(processedItem) {
        return _this6.isValidItem(processedItem);
      });
    }, "findLastItemIndex"),
    findNextItemIndex: /* @__PURE__ */ __name(function findNextItemIndex(index2) {
      var _this7 = this;
      var matchedItemIndex = index2 < this.visibleItems.length - 1 ? this.visibleItems.slice(index2 + 1).findIndex(function(processedItem) {
        return _this7.isValidItem(processedItem);
      }) : -1;
      return matchedItemIndex > -1 ? matchedItemIndex + index2 + 1 : index2;
    }, "findNextItemIndex"),
    findPrevItemIndex: /* @__PURE__ */ __name(function findPrevItemIndex(index2) {
      var _this8 = this;
      var matchedItemIndex = index2 > 0 ? findLastIndex(this.visibleItems.slice(0, index2), function(processedItem) {
        return _this8.isValidItem(processedItem);
      }) : -1;
      return matchedItemIndex > -1 ? matchedItemIndex : index2;
    }, "findPrevItemIndex"),
    findSelectedItemIndex: /* @__PURE__ */ __name(function findSelectedItemIndex() {
      var _this9 = this;
      return this.visibleItems.findIndex(function(processedItem) {
        return _this9.isValidSelectedItem(processedItem);
      });
    }, "findSelectedItemIndex"),
    findFirstFocusedItemIndex: /* @__PURE__ */ __name(function findFirstFocusedItemIndex() {
      var selectedIndex = this.findSelectedItemIndex();
      return selectedIndex < 0 ? this.findFirstItemIndex() : selectedIndex;
    }, "findFirstFocusedItemIndex"),
    findLastFocusedItemIndex: /* @__PURE__ */ __name(function findLastFocusedItemIndex() {
      var selectedIndex = this.findSelectedItemIndex();
      return selectedIndex < 0 ? this.findLastItemIndex() : selectedIndex;
    }, "findLastFocusedItemIndex"),
    searchItems: /* @__PURE__ */ __name(function searchItems(event, _char) {
      var _this10 = this;
      this.searchValue = (this.searchValue || "") + _char;
      var itemIndex = -1;
      var matched = false;
      if (this.focusedItemInfo.index !== -1) {
        itemIndex = this.visibleItems.slice(this.focusedItemInfo.index).findIndex(function(processedItem) {
          return _this10.isItemMatched(processedItem);
        });
        itemIndex = itemIndex === -1 ? this.visibleItems.slice(0, this.focusedItemInfo.index).findIndex(function(processedItem) {
          return _this10.isItemMatched(processedItem);
        }) : itemIndex + this.focusedItemInfo.index;
      } else {
        itemIndex = this.visibleItems.findIndex(function(processedItem) {
          return _this10.isItemMatched(processedItem);
        });
      }
      if (itemIndex !== -1) {
        matched = true;
      }
      if (itemIndex === -1 && this.focusedItemInfo.index === -1) {
        itemIndex = this.findFirstFocusedItemIndex();
      }
      if (itemIndex !== -1) {
        this.changeFocusedItemIndex(event, itemIndex);
      }
      if (this.searchTimeout) {
        clearTimeout(this.searchTimeout);
      }
      this.searchTimeout = setTimeout(function() {
        _this10.searchValue = "";
        _this10.searchTimeout = null;
      }, 500);
      return matched;
    }, "searchItems"),
    changeFocusedItemIndex: /* @__PURE__ */ __name(function changeFocusedItemIndex(event, index2) {
      if (this.focusedItemInfo.index !== index2) {
        this.focusedItemInfo.index = index2;
        this.scrollInView();
      }
    }, "changeFocusedItemIndex"),
    scrollInView: /* @__PURE__ */ __name(function scrollInView2() {
      var index2 = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : -1;
      var id = index2 !== -1 ? "".concat(this.id, "_").concat(index2) : this.focusedItemIdx;
      var element = findSingle(this.list, 'li[id="'.concat(id, '"]'));
      if (element) {
        element.scrollIntoView && element.scrollIntoView({
          block: "nearest",
          inline: "start"
        });
      }
    }, "scrollInView"),
    createProcessedItems: /* @__PURE__ */ __name(function createProcessedItems(items) {
      var _this11 = this;
      var level = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : 0;
      var parent = arguments.length > 2 && arguments[2] !== void 0 ? arguments[2] : {};
      var parentKey = arguments.length > 3 && arguments[3] !== void 0 ? arguments[3] : "";
      var processedItems4 = [];
      items && items.forEach(function(item4, index2) {
        var key = (parentKey !== "" ? parentKey + "_" : "") + index2;
        var newItem = {
          item: item4,
          index: index2,
          level,
          key,
          parent,
          parentKey
        };
        newItem["items"] = _this11.createProcessedItems(item4.items, level + 1, newItem, key);
        processedItems4.push(newItem);
      });
      return processedItems4;
    }, "createProcessedItems"),
    containerRef: /* @__PURE__ */ __name(function containerRef2(el) {
      this.container = el;
    }, "containerRef"),
    listRef: /* @__PURE__ */ __name(function listRef2(el) {
      this.list = el ? el.$el : void 0;
    }, "listRef")
  },
  computed: {
    processedItems: /* @__PURE__ */ __name(function processedItems() {
      return this.createProcessedItems(this.model || []);
    }, "processedItems"),
    visibleItems: /* @__PURE__ */ __name(function visibleItems() {
      var _this12 = this;
      var processedItem = this.activeItemPath.find(function(p) {
        return p.key === _this12.focusedItemInfo.parentKey;
      });
      return processedItem ? processedItem.items : this.processedItems;
    }, "visibleItems"),
    focusedItemIdx: /* @__PURE__ */ __name(function focusedItemIdx() {
      return this.focusedItemInfo.index !== -1 ? "".concat(this.id).concat(isNotEmpty(this.focusedItemInfo.parentKey) ? "_" + this.focusedItemInfo.parentKey : "", "_").concat(this.focusedItemInfo.index) : null;
    }, "focusedItemIdx")
  },
  components: {
    ContextMenuSub: script$1$a,
    Portal: script$r
  }
};
function render$c(_ctx, _cache, $props, $setup, $data, $options) {
  var _component_ContextMenuSub = resolveComponent("ContextMenuSub");
  var _component_Portal = resolveComponent("Portal");
  return openBlock(), createBlock(_component_Portal, {
    appendTo: _ctx.appendTo
  }, {
    "default": withCtx(function() {
      return [createVNode(Transition, mergeProps({
        name: "p-contextmenu",
        onEnter: $options.onEnter,
        onAfterEnter: $options.onAfterEnter,
        onLeave: $options.onLeave,
        onAfterLeave: $options.onAfterLeave
      }, _ctx.ptm("transition")), {
        "default": withCtx(function() {
          return [$data.visible ? (openBlock(), createElementBlock("div", mergeProps({
            key: 0,
            ref: $options.containerRef,
            "class": _ctx.cx("root")
          }, _ctx.ptmi("root")), [createVNode(_component_ContextMenuSub, {
            ref: $options.listRef,
            id: $data.id + "_list",
            "class": normalizeClass(_ctx.cx("rootList")),
            role: "menubar",
            root: true,
            tabindex: _ctx.tabindex,
            "aria-orientation": "vertical",
            "aria-activedescendant": $data.focused ? $options.focusedItemIdx : void 0,
            menuId: $data.id,
            focusedItemId: $data.focused ? $options.focusedItemIdx : void 0,
            items: $options.processedItems,
            templates: _ctx.$slots,
            activeItemPath: $data.activeItemPath,
            "aria-labelledby": _ctx.ariaLabelledby,
            "aria-label": _ctx.ariaLabel,
            level: 0,
            visible: $data.submenuVisible,
            pt: _ctx.pt,
            unstyled: _ctx.unstyled,
            onFocus: $options.onFocus,
            onBlur: $options.onBlur,
            onKeydown: $options.onKeyDown,
            onItemClick: $options.onItemClick,
            onItemMouseenter: $options.onItemMouseEnter,
            onItemMousemove: $options.onItemMouseMove
          }, null, 8, ["id", "class", "tabindex", "aria-activedescendant", "menuId", "focusedItemId", "items", "templates", "activeItemPath", "aria-labelledby", "aria-label", "visible", "pt", "unstyled", "onFocus", "onBlur", "onKeydown", "onItemClick", "onItemMouseenter", "onItemMousemove"])], 16)) : createCommentVNode("", true)];
        }),
        _: 1
      }, 16, ["onEnter", "onAfterEnter", "onLeave", "onAfterLeave"])];
    }),
    _: 1
  }, 8, ["appendTo"]);
}
__name(render$c, "render$c");
script$d.render = render$c;
const _withScopeId$f = /* @__PURE__ */ __name((n) => (pushScopeId("data-v-9bc23daf"), n = n(), popScopeId(), n), "_withScopeId$f");
const _hoisted_1$r = ["src", "data-test"];
const _hoisted_2$k = ["src"];
const _hoisted_3$e = {
  key: 1,
  class: "broken-image-placeholder"
};
const _hoisted_4$9 = /* @__PURE__ */ _withScopeId$f(() => /* @__PURE__ */ createBaseVNode("i", { class: "pi pi-image" }, null, -1));
const _sfc_main$s = /* @__PURE__ */ defineComponent({
  __name: "ComfyImage",
  props: {
    src: {},
    class: {},
    contain: { type: Boolean, default: false }
  },
  setup(__props) {
    const props = __props;
    const imageBroken = ref(false);
    const handleImageError = /* @__PURE__ */ __name((e) => {
      imageBroken.value = true;
    }, "handleImageError");
    const classArray = computed(() => {
      if (Array.isArray(props.class)) {
        return props.class;
      } else if (typeof props.class === "string") {
        return props.class.split(" ");
      } else if (typeof props.class === "object") {
        return Object.keys(props.class).filter((key) => props.class[key]);
      }
      return [];
    });
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock(Fragment, null, [
        !imageBroken.value ? (openBlock(), createElementBlock("span", {
          key: 0,
          class: normalizeClass(["comfy-image-wrap", [{ contain: _ctx.contain }]])
        }, [
          _ctx.contain ? (openBlock(), createElementBlock("img", {
            key: 0,
            src: _ctx.src,
            onError: handleImageError,
            "data-test": _ctx.src,
            class: "comfy-image-blur",
            style: normalizeStyle({ "background-image": `url(${_ctx.src})` })
          }, null, 44, _hoisted_1$r)) : createCommentVNode("", true),
          createBaseVNode("img", {
            src: _ctx.src,
            onError: handleImageError,
            class: normalizeClass(["comfy-image-main", [...classArray.value]])
          }, null, 42, _hoisted_2$k)
        ], 2)) : createCommentVNode("", true),
        imageBroken.value ? (openBlock(), createElementBlock("div", _hoisted_3$e, [
          _hoisted_4$9,
          createBaseVNode("span", null, toDisplayString(_ctx.$t("imageFailedToLoad")), 1)
        ])) : createCommentVNode("", true)
      ], 64);
    };
  }
});
const ComfyImage = /* @__PURE__ */ _export_sfc(_sfc_main$s, [["__scopeId", "data-v-9bc23daf"]]);
const _withScopeId$e = /* @__PURE__ */ __name((n) => (pushScopeId("data-v-d9c060ae"), n = n(), popScopeId(), n), "_withScopeId$e");
const _hoisted_1$q = { class: "image-preview-mask" };
const _hoisted_2$j = {
  key: 1,
  class: "task-result-preview"
};
const _hoisted_3$d = /* @__PURE__ */ _withScopeId$e(() => /* @__PURE__ */ createBaseVNode("i", { class: "pi pi-file" }, null, -1));
const _sfc_main$r = /* @__PURE__ */ defineComponent({
  __name: "ResultItem",
  props: {
    result: {}
  },
  emits: ["preview"],
  setup(__props, { emit: __emit }) {
    const props = __props;
    const emit = __emit;
    const resultContainer = ref(null);
    const settingStore = useSettingStore();
    const imageFit = computed(
      () => settingStore.get("Comfy.Queue.ImageFit")
    );
    onMounted(() => {
      if (props.result.mediaType === "images") {
        resultContainer.value?.querySelectorAll("img").forEach((img) => {
          img.draggable = true;
        });
      }
    });
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", {
        class: "result-container",
        ref_key: "resultContainer",
        ref: resultContainer
      }, [
        _ctx.result.mediaType === "images" || _ctx.result.mediaType === "gifs" ? (openBlock(), createElementBlock(Fragment, { key: 0 }, [
          createVNode(ComfyImage, {
            src: _ctx.result.url,
            class: "task-output-image",
            contain: imageFit.value === "contain"
          }, null, 8, ["src", "contain"]),
          createBaseVNode("div", _hoisted_1$q, [
            createVNode(unref(script$o), {
              icon: "pi pi-eye",
              severity: "secondary",
              onClick: _cache[0] || (_cache[0] = ($event) => emit("preview", _ctx.result)),
              rounded: ""
            })
          ])
        ], 64)) : (openBlock(), createElementBlock("div", _hoisted_2$j, [
          _hoisted_3$d,
          createBaseVNode("span", null, toDisplayString(_ctx.result.mediaType), 1)
        ]))
      ], 512);
    };
  }
});
const ResultItem = /* @__PURE__ */ _export_sfc(_sfc_main$r, [["__scopeId", "data-v-d9c060ae"]]);
const _withScopeId$d = /* @__PURE__ */ __name((n) => (pushScopeId("data-v-d4c8a1fe"), n = n(), popScopeId(), n), "_withScopeId$d");
const _hoisted_1$p = { class: "task-result-preview" };
const _hoisted_2$i = {
  key: 0,
  class: "pi pi-spin pi-spinner"
};
const _hoisted_3$c = ["src"];
const _hoisted_4$8 = { key: 2 };
const _hoisted_5$6 = {
  key: 3,
  class: "pi pi-exclamation-triangle"
};
const _hoisted_6$4 = {
  key: 4,
  class: "pi pi-exclamation-circle"
};
const _hoisted_7$2 = { class: "task-item-details" };
const _hoisted_8$1 = { class: "tag-wrapper status-tag-group" };
const _hoisted_9$1 = ["innerHTML"];
const _hoisted_10$1 = {
  key: 0,
  class: "task-time"
};
const _hoisted_11$1 = {
  key: 1,
  class: "task-prompt-id"
};
const _hoisted_12$1 = { class: "tag-wrapper" };
const _hoisted_13$1 = { style: { "font-weight": "bold" } };
const _sfc_main$q = /* @__PURE__ */ defineComponent({
  __name: "TaskItem",
  props: {
    task: {},
    isFlatTask: { type: Boolean }
  },
  emits: ["contextmenu", "preview", "task-output-length-clicked"],
  setup(__props, { emit: __emit }) {
    const props = __props;
    const flatOutputs = props.task.flatOutputs;
    const coverResult = flatOutputs.length ? props.task.previewOutput || flatOutputs[0] : null;
    const node2 = flatOutputs.length ? props.task.workflow.nodes.find(
      (n) => n.id == coverResult.nodeId
    ) ?? null : null;
    const progressPreviewBlobUrl = ref("");
    const emit = __emit;
    onMounted(() => {
      api.addEventListener("b_preview", onProgressPreviewReceived);
    });
    onUnmounted(() => {
      api.removeEventListener("b_preview", onProgressPreviewReceived);
    });
    const handleContextMenu = /* @__PURE__ */ __name((e) => {
      emit("contextmenu", { task: props.task, event: e, node: node2 });
    }, "handleContextMenu");
    const handlePreview = /* @__PURE__ */ __name(() => {
      emit("preview", props.task);
    }, "handlePreview");
    const handleOutputLengthClick = /* @__PURE__ */ __name(() => {
      emit("task-output-length-clicked", props.task);
    }, "handleOutputLengthClick");
    const taskTagSeverity = /* @__PURE__ */ __name((status) => {
      switch (status) {
        case TaskItemDisplayStatus.Pending:
          return "secondary";
        case TaskItemDisplayStatus.Running:
          return "info";
        case TaskItemDisplayStatus.Completed:
          return "success";
        case TaskItemDisplayStatus.Failed:
          return "danger";
        case TaskItemDisplayStatus.Cancelled:
          return "warn";
      }
    }, "taskTagSeverity");
    const taskStatusText = /* @__PURE__ */ __name((status) => {
      switch (status) {
        case TaskItemDisplayStatus.Pending:
          return "Pending";
        case TaskItemDisplayStatus.Running:
          return '<i class="pi pi-spin pi-spinner" style="font-weight: bold"></i> Running';
        case TaskItemDisplayStatus.Completed:
          return '<i class="pi pi-check" style="font-weight: bold"></i>';
        case TaskItemDisplayStatus.Failed:
          return "Failed";
        case TaskItemDisplayStatus.Cancelled:
          return "Cancelled";
      }
    }, "taskStatusText");
    const formatTime = /* @__PURE__ */ __name((time) => {
      if (time === void 0) {
        return "";
      }
      return `${time.toFixed(2)}s`;
    }, "formatTime");
    const onProgressPreviewReceived = /* @__PURE__ */ __name(async ({ detail }) => {
      if (props.task.displayStatus === TaskItemDisplayStatus.Running) {
        progressPreviewBlobUrl.value = URL.createObjectURL(detail);
      }
    }, "onProgressPreviewReceived");
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", {
        class: "task-item",
        onContextmenu: handleContextMenu
      }, [
        createBaseVNode("div", _hoisted_1$p, [
          _ctx.task.displayStatus === unref(TaskItemDisplayStatus).Completed ? (openBlock(), createElementBlock(Fragment, { key: 0 }, [
            unref(flatOutputs).length ? (openBlock(), createBlock(ResultItem, {
              key: 0,
              result: unref(coverResult),
              onPreview: handlePreview
            }, null, 8, ["result"])) : createCommentVNode("", true)
          ], 64)) : createCommentVNode("", true),
          _ctx.task.displayStatus === unref(TaskItemDisplayStatus).Running ? (openBlock(), createElementBlock(Fragment, { key: 1 }, [
            !progressPreviewBlobUrl.value ? (openBlock(), createElementBlock("i", _hoisted_2$i)) : (openBlock(), createElementBlock("img", {
              key: 1,
              src: progressPreviewBlobUrl.value,
              class: "progress-preview-img"
            }, null, 8, _hoisted_3$c))
          ], 64)) : _ctx.task.displayStatus === unref(TaskItemDisplayStatus).Pending ? (openBlock(), createElementBlock("span", _hoisted_4$8, "...")) : _ctx.task.displayStatus === unref(TaskItemDisplayStatus).Cancelled ? (openBlock(), createElementBlock("i", _hoisted_5$6)) : _ctx.task.displayStatus === unref(TaskItemDisplayStatus).Failed ? (openBlock(), createElementBlock("i", _hoisted_6$4)) : createCommentVNode("", true)
        ]),
        createBaseVNode("div", _hoisted_7$2, [
          createBaseVNode("div", _hoisted_8$1, [
            _ctx.isFlatTask && _ctx.task.isHistory ? (openBlock(), createBlock(unref(script$w), {
              key: 0,
              class: "node-name-tag"
            }, {
              default: withCtx(() => [
                createVNode(unref(script$o), {
                  class: "task-node-link",
                  label: `${unref(node2)?.type} (#${unref(node2)?.id})`,
                  link: "",
                  size: "small",
                  onClick: _cache[0] || (_cache[0] = ($event) => unref(app).goToNode(unref(node2)?.id))
                }, null, 8, ["label"])
              ]),
              _: 1
            })) : createCommentVNode("", true),
            createVNode(unref(script$w), {
              severity: taskTagSeverity(_ctx.task.displayStatus)
            }, {
              default: withCtx(() => [
                createBaseVNode("span", {
                  innerHTML: taskStatusText(_ctx.task.displayStatus)
                }, null, 8, _hoisted_9$1),
                _ctx.task.isHistory ? (openBlock(), createElementBlock("span", _hoisted_10$1, toDisplayString(formatTime(_ctx.task.executionTimeInSeconds)), 1)) : createCommentVNode("", true),
                _ctx.isFlatTask ? (openBlock(), createElementBlock("span", _hoisted_11$1, toDisplayString(_ctx.task.promptId.split("-")[0]), 1)) : createCommentVNode("", true)
              ]),
              _: 1
            }, 8, ["severity"])
          ]),
          createBaseVNode("div", _hoisted_12$1, [
            _ctx.task.isHistory && unref(flatOutputs).length > 1 ? (openBlock(), createBlock(unref(script$o), {
              key: 0,
              outlined: "",
              onClick: handleOutputLengthClick
            }, {
              default: withCtx(() => [
                createBaseVNode("span", _hoisted_13$1, toDisplayString(unref(flatOutputs).length), 1)
              ]),
              _: 1
            })) : createCommentVNode("", true)
          ])
        ])
      ], 32);
    };
  }
});
const TaskItem = /* @__PURE__ */ _export_sfc(_sfc_main$q, [["__scopeId", "data-v-d4c8a1fe"]]);
var theme$9 = /* @__PURE__ */ __name(function theme9(_ref) {
  var dt = _ref.dt;
  return "\n.p-galleria {\n    overflow: hidden;\n    border-style: solid;\n    border-width: ".concat(dt("galleria.border.width"), ";\n    border-color: ").concat(dt("galleria.border.color"), ";\n    border-radius: ").concat(dt("galleria.border.radius"), ";\n}\n\n.p-galleria-content {\n    display: flex;\n    flex-direction: column;\n}\n\n.p-galleria-items-container {\n    display: flex;\n    flex-direction: column;\n    position: relative;\n}\n\n.p-galleria-items {\n    position: relative;\n    display: flex;\n    height: 100%;\n}\n\n.p-galleria-nav-button {\n    position: absolute;\n    top: 50%;\n    display: inline-flex;\n    justify-content: center;\n    align-items: center;\n    overflow: hidden;\n    background: ").concat(dt("galleria.nav.button.background"), ";\n    color: ").concat(dt("galleria.nav.button.color"), ";\n    width: ").concat(dt("galleria.nav.button.size"), ";\n    height: ").concat(dt("galleria.nav.button.size"), ";\n    transition: background ").concat(dt("galleria.transition.duration"), ", color ").concat(dt("galleria.transition.duration"), ", outline-color ").concat(dt("galleria.transition.duration"), ", box-shadow ").concat(dt("galleria.transition.duration"), ";\n    margin: calc(-1 * calc(").concat(dt("galleria.nav.button.size"), ") / 2) ").concat(dt("galleria.nav.button.gutter"), " 0 ").concat(dt("galleria.nav.button.gutter"), ";\n    padding: 0;\n    user-select: none;\n    border: 0 none;\n    cursor: pointer;\n    outline-color: transparent;\n}\n\n.p-galleria-nav-button:not(.p-disabled):hover {\n    background: ").concat(dt("galleria.nav.button.hover.background"), ";\n    color: ").concat(dt("galleria.nav.button.hover.color"), ";\n}\n\n.p-galleria-nav-button:not(.p-disabled):focus-visible {\n    box-shadow: ").concat(dt("galleria.nav.button.focus.ring.shadow"), ";\n    outline: ").concat(dt("galleria.nav.button.focus.ring.width"), " ").concat(dt("galleria.nav.button.focus.ring.style"), " ").concat(dt("galleria.nav.button.focus.ring.color"), ";\n    outline-offset: ").concat(dt("galleria.nav.button.focus.ring.offset"), ";\n}\n\n.p-galleria-next-icon,\n.p-galleria-prev-icon {\n    font-size: ").concat(dt("galleria.nav.icon.size"), ";\n    width: ").concat(dt("galleria.nav.icon.size"), ";\n    height: ").concat(dt("galleria.nav.icon.size"), ";\n}\n\n.p-galleria-prev-button {\n    border-radius: ").concat(dt("galleria.nav.button.prev.border.radius"), ";\n    left: 0;\n}\n\n.p-galleria-next-button {\n    border-radius: ").concat(dt("galleria.nav.button.next.border.radius"), ";\n    right: 0;\n}\n\n.p-galleria-item {\n    display: flex;\n    justify-content: center;\n    align-items: center;\n    height: 100%;\n    width: 100%;\n}\n\n.p-galleria-hover-navigators .p-galleria-nav-button {\n    pointer-events: none;\n    opacity: 0;\n    transition: opacity ").concat(dt("galleria.transition.duration"), " ease-in-out;\n}\n\n.p-galleria-hover-navigators .p-galleria-items-container:hover .p-galleria-nav-button {\n    pointer-events: all;\n    opacity: 1;\n}\n\n.p-galleria-hover-navigators .p-galleria-items-container:hover .p-galleria-nav-button.p-disabled {\n    pointer-events: none;\n}\n\n.p-galleria-caption {\n    position: absolute;\n    bottom: 0;\n    left: 0;\n    width: 100%;\n    background: ").concat(dt("galleria.caption.background"), ";\n    color: ").concat(dt("galleria.caption.color"), ";\n    padding: ").concat(dt("galleria.caption.padding"), ";\n}\n\n.p-galleria-thumbnails {\n    display: flex;\n    flex-direction: column;\n    overflow: auto;\n    flex-shrink: 0;\n}\n\n.p-galleria-thumbnail-nav-button {\n    align-self: center;\n    flex: 0 0 auto;\n    display: flex;\n    justify-content: center;\n    align-items: center;\n    overflow: hidden;\n    position: relative;\n    margin: 0 ").concat(dt("galleria.thumbnail.nav.button.gutter"), ";\n    padding: 0;\n    border: none;\n    user-select: none;\n    cursor: pointer;\n    background: transparent;\n    color: ").concat(dt("galleria.thumbnail.nav.button.color"), ";\n    width: ").concat(dt("galleria.thumbnail.nav.button.size"), ";\n    height: ").concat(dt("galleria.thumbnail.nav.button.size"), ";\n    transition: background ").concat(dt("galleria.transition.duration"), ", color ").concat(dt("galleria.transition.duration"), ", outline-color ").concat(dt("galleria.transition.duration"), ";\n    outline-color: transparent;\n    border-radius: ").concat(dt("galleria.thumbnail.nav.button.border.radius"), ";\n}\n\n.p-galleria-thumbnail-nav-button:hover {\n    background: ").concat(dt("galleria.thumbnail.nav.button.hover.background"), ";\n    color: ").concat(dt("galleria.thumbnail.nav.button.hover.color"), ";\n}\n\n.p-galleria-thumbnail-nav-button:focus-visible {\n    box-shadow: ").concat(dt("galleria.thumbnail.nav.button.focus.ring.shadow"), ";\n    outline: ").concat(dt("galleria.thumbnail.nav.button.focus.ring.width"), " ").concat(dt("galleria.thumbnail.nav.button.focus.ring.style"), " ").concat(dt("galleria.thumbnail.nav.button.focus.ring.color"), ";\n    outline-offset: ").concat(dt("galleria.thumbnail.nav.button.focus.ring.offset"), ";\n}\n\n.p-galleria-thumbnail-nav-button .p-galleria-thumbnail-next-icon,\n.p-galleria-thumbnail-nav-button .p-galleria-thumbnail-prev-icon {\n    font-size: ").concat(dt("galleria.thumbnail.nav.button.icon.size"), ";\n    width: ").concat(dt("galleria.thumbnail.nav.button.icon.size"), ";\n    height: ").concat(dt("galleria.thumbnail.nav.button.icon.size"), ";\n}\n\n.p-galleria-thumbnails-content {\n    display: flex;\n    flex-direction: row;\n    background: ").concat(dt("galleria.thumbnails.content.background"), ";\n    padding: ").concat(dt("galleria.thumbnails.content.padding"), ";\n}\n\n.p-galleria-thumbnails-viewport {\n    overflow: hidden;\n    width: 100%;\n}\n\n.p-galleria-thumbnail-items {\n    display: flex;\n}\n\n.p-galleria-thumbnail-item {\n    overflow: auto;\n    display: flex;\n    align-items: center;\n    justify-content: center;\n    cursor: pointer;\n    opacity: 0.5;\n}\n\n.p-galleria-thumbnail {\n    outline-color: transparent;\n}\n\n.p-galleria-thumbnail-item:hover {\n    opacity: 1;\n    transition: opacity 0.3s;\n}\n\n.p-galleria-thumbnail-item-current {\n    opacity: 1;\n}\n\n.p-galleria-thumbnails-left .p-galleria-content,\n.p-galleria-thumbnails-right .p-galleria-content {\n    flex-direction: row;\n}\n\n.p-galleria-thumbnails-left .p-galleria-items-container,\n.p-galleria-thumbnails-right .p-galleria-items-container {\n    flex-direction: row;\n}\n\n.p-galleria-thumbnails-left .p-galleria-items-container,\n.p-galleria-thumbnails-top .p-galleria-items-container {\n    order: 2;\n}\n\n.p-galleria-thumbnails-left .p-galleria-thumbnails,\n.p-galleria-thumbnails-top .p-galleria-thumbnails {\n    order: 1;\n}\n\n.p-galleria-thumbnails-left .p-galleria-thumbnails-content,\n.p-galleria-thumbnails-right .p-galleria-thumbnails-content {\n    flex-direction: column;\n    flex-grow: 1;\n}\n\n.p-galleria-thumbnails-left .p-galleria-thumbnail-items,\n.p-galleria-thumbnails-right .p-galleria-thumbnail-items {\n    flex-direction: column;\n    height: 100%;\n}\n\n.p-galleria-indicator-list {\n    display: flex;\n    align-items: center;\n    justify-content: center;\n    padding: ").concat(dt("galleria.indicator.list.padding"), ";\n    gap: ").concat(dt("galleria.indicator.list.gap"), ";\n    margin: 0;\n    list-style: none;\n}\n\n.p-galleria-indicator-button {\n    display: inline-flex;\n    align-items: center;\n    background: ").concat(dt("galleria.indicator.button.background"), ";\n    width: ").concat(dt("galleria.indicator.button.width"), ";\n    height: ").concat(dt("galleria.indicator.button.height"), ";\n    transition: background ").concat(dt("galleria.transition.duration"), ", color ").concat(dt("galleria.transition.duration"), ", outline-color ").concat(dt("galleria.transition.duration"), ", box-shadow ").concat(dt("galleria.transition.duration"), ";\n    outline-color: transparent;\n    border-radius: ").concat(dt("galleria.indicator.button.border.radius"), ";\n    margin: 0;\n    padding: 0;\n    border: none;\n    user-select: none;\n    cursor: pointer;\n}\n\n.p-galleria-indicator-button:hover {\n    background: ").concat(dt("galleria.indicator.button.hover.background"), ";\n}\n\n.p-galleria-indicator-button:focus-visible {\n    box-shadow: ").concat(dt("galleria.indicator.button.focus.ring.shadow"), ";\n    outline: ").concat(dt("galleria.indicator.button.focus.ring.width"), " ").concat(dt("galleria.indicator.button.focus.ring.style"), " ").concat(dt("galleria.indicator.button.focus.ring.color"), ";\n    outline-offset: ").concat(dt("galleria.indicator.button.focus.ring.offset"), ";\n}\n\n.p-galleria-indicator-active .p-galleria-indicator-button {\n    background: ").concat(dt("galleria.indicator.button.active.background"), ";\n}\n\n.p-galleria-indicators-left .p-galleria-items-container,\n.p-galleria-indicators-right .p-galleria-items-container {\n    flex-direction: row;\n    align-items: center;\n}\n\n.p-galleria-indicators-left .p-galleria-items,\n.p-galleria-indicators-top .p-galleria-items {\n    order: 2;\n}\n\n.p-galleria-indicators-left .p-galleria-indicator-list,\n.p-galleria-indicators-top .p-galleria-indicator-list {\n    order: 1;\n}\n\n.p-galleria-indicators-left .p-galleria-indicator-list,\n.p-galleria-indicators-right .p-galleria-indicator-list {\n    flex-direction: column;\n}\n\n.p-galleria-inset-indicators .p-galleria-indicator-list {\n    position: absolute;\n    display: flex;\n    z-index: 1;\n    background: ").concat(dt("galleria.inset.indicator.list.background"), ";\n}\n\n.p-galleria-inset-indicators .p-galleria-indicator-button {\n    background: ").concat(dt("galleria.inset.indicator.button.background"), ";\n}\n\n.p-galleria-inset-indicators .p-galleria-indicator-button:hover {\n    background: ").concat(dt("galleria.inset.indicator.button.hover.background"), ";\n}\n\n.p-galleria-inset-indicators .p-galleria-indicator-active .p-galleria-indicator-button {\n    background: ").concat(dt("galleria.inset.indicator.button.active.background"), ";\n}\n\n.p-galleria-inset-indicators.p-galleria-indicators-top .p-galleria-indicator-list {\n    top: 0;\n    left: 0;\n    width: 100%;\n    align-items: flex-start;\n}\n\n.p-galleria-inset-indicators.p-galleria-indicators-right .p-galleria-indicator-list {\n    right: 0;\n    top: 0;\n    height: 100%;\n    align-items: flex-end;\n}\n\n.p-galleria-inset-indicators.p-galleria-indicators-bottom .p-galleria-indicator-list {\n    bottom: 0;\n    left: 0;\n    width: 100%;\n    align-items: flex-end;\n}\n\n.p-galleria-inset-indicators.p-galleria-indicators-left .p-galleria-indicator-list {\n    left: 0;\n    top: 0;\n    height: 100%;\n    align-items: flex-start;\n}\n\n.p-galleria-mask {\n    position: fixed;\n    top: 0;\n    left: 0;\n    width: 100%;\n    height: 100%;\n    display: flex;\n    align-items: center;\n    justify-content: center;\n}\n\n.p-galleria-close-button {\n    position: absolute;\n    top: 0;\n    right: 0;\n    display: flex;\n    justify-content: center;\n    align-items: center;\n    overflow: hidden;\n    margin: ").concat(dt("galleria.close.button.gutter"), ";\n    background: ").concat(dt("galleria.close.button.background"), ";\n    color: ").concat(dt("galleria.close.button.color"), ";\n    width: ").concat(dt("galleria.close.button.size"), ";\n    height: ").concat(dt("galleria.close.button.size"), ";\n    padding: 0;\n    border: none;\n    user-select: none;\n    cursor: pointer;\n    border-radius: ").concat(dt("galleria.close.button.border.radius"), ";\n    outline-color: transparent;\n    transition: background ").concat(dt("galleria.transition.duration"), ", color ").concat(dt("galleria.transition.duration"), ", outline-color ").concat(dt("galleria.transition.duration"), ";\n}\n\n.p-galleria-close-icon {\n    font-size: ").concat(dt("galleria.close.button.icon.size"), ";\n    width: ").concat(dt("galleria.close.button.icon.size"), ";\n    height: ").concat(dt("galleria.close.button.icon.size"), ";\n}\n\n.p-galleria-close-button:hover {\n    background: ").concat(dt("galleria.close.button.hover.background"), ";\n    color: ").concat(dt("galleria.close.button.hover.color"), ";\n}\n\n.p-galleria-close-button:focus-visible {\n    box-shadow: ").concat(dt("galleria.close.button.focus.ring.shadow"), ";\n    outline: ").concat(dt("galleria.close.button.focus.ring.width"), " ").concat(dt("galleria.close.button.focus.ring.style"), " ").concat(dt("galleria.close.button.focus.ring.color"), ";\n    outline-offset: ").concat(dt("galleria.close.button.focus.ring.offset"), ";\n}\n\n.p-galleria-mask .p-galleria-nav-button {\n    position: fixed;\n    top: 50%;\n}\n\n.p-galleria-enter-active {\n    transition: all 150ms cubic-bezier(0, 0, 0.2, 1);\n}\n\n.p-galleria-leave-active {\n    transition: all 150ms cubic-bezier(0.4, 0, 0.2, 1);\n}\n\n.p-galleria-enter-from,\n.p-galleria-leave-to {\n    opacity: 0;\n    transform: scale(0.7);\n}\n\n.p-galleria-enter-active .p-galleria-nav-button {\n    opacity: 0;\n}\n\n.p-items-hidden .p-galleria-thumbnail-item {\n    visibility: hidden;\n}\n\n.p-items-hidden .p-galleria-thumbnail-item.p-galleria-thumbnail-item-active {\n    visibility: visible;\n}\n");
}, "theme");
var classes$9 = {
  mask: "p-galleria-mask p-overlay-mask p-overlay-mask-enter",
  root: /* @__PURE__ */ __name(function root7(_ref2) {
    var instance = _ref2.instance;
    var thumbnailsPosClass = instance.$attrs.showThumbnails && instance.getPositionClass("p-galleria-thumbnails", instance.$attrs.thumbnailsPosition);
    var indicatorPosClass = instance.$attrs.showIndicators && instance.getPositionClass("p-galleria-indicators", instance.$attrs.indicatorsPosition);
    return ["p-galleria p-component", {
      "p-galleria-fullscreen": instance.$attrs.fullScreen,
      "p-galleria-inset-indicators": instance.$attrs.showIndicatorsOnItem,
      "p-galleria-hover-navigators": instance.$attrs.showItemNavigatorsOnHover && !instance.$attrs.fullScreen
    }, thumbnailsPosClass, indicatorPosClass];
  }, "root"),
  closeButton: "p-galleria-close-button",
  closeIcon: "p-galleria-close-icon",
  header: "p-galleria-header",
  content: "p-galleria-content",
  footer: "p-galleria-footer",
  itemsContainer: "p-galleria-items-container",
  items: "p-galleria-items",
  prevButton: /* @__PURE__ */ __name(function prevButton(_ref3) {
    var instance = _ref3.instance;
    return ["p-galleria-prev-button p-galleria-nav-button", {
      "p-disabled": instance.isNavBackwardDisabled()
    }];
  }, "prevButton"),
  prevIcon: "p-galleria-prev-icon",
  item: "p-galleria-item",
  nextButton: /* @__PURE__ */ __name(function nextButton(_ref4) {
    var instance = _ref4.instance;
    return ["p-galleria-next-button p-galleria-nav-button", {
      "p-disabled": instance.isNavForwardDisabled()
    }];
  }, "nextButton"),
  nextIcon: "p-galleria-next-icon",
  caption: "p-galleria-caption",
  indicatorList: "p-galleria-indicator-list",
  indicator: /* @__PURE__ */ __name(function indicator(_ref5) {
    var instance = _ref5.instance, index2 = _ref5.index;
    return ["p-galleria-indicator", {
      "p-galleria-indicator-active": instance.isIndicatorItemActive(index2)
    }];
  }, "indicator"),
  indicatorButton: "p-galleria-indicator-button",
  thumbnails: "p-galleria-thumbnails",
  thumbnailContent: "p-galleria-thumbnails-content",
  thumbnailPrevButton: /* @__PURE__ */ __name(function thumbnailPrevButton(_ref6) {
    var instance = _ref6.instance;
    return ["p-galleria-thumbnail-prev-button p-galleria-thumbnail-nav-button", {
      "p-disabled": instance.isNavBackwardDisabled()
    }];
  }, "thumbnailPrevButton"),
  thumbnailPrevIcon: "p-galleria-thumbnail-prev-icon",
  thumbnailsViewport: "p-galleria-thumbnails-viewport",
  thumbnailItems: "p-galleria-thumbnail-items",
  thumbnailItem: /* @__PURE__ */ __name(function thumbnailItem(_ref7) {
    var instance = _ref7.instance, index2 = _ref7.index, activeIndex2 = _ref7.activeIndex;
    return ["p-galleria-thumbnail-item", {
      "p-galleria-thumbnail-item-current": activeIndex2 === index2,
      "p-galleria-thumbnail-item-active": instance.isItemActive(index2),
      "p-galleria-thumbnail-item-start": instance.firstItemAciveIndex() === index2,
      "p-galleria-thumbnail-item-end": instance.lastItemActiveIndex() === index2
    }];
  }, "thumbnailItem"),
  thumbnail: "p-galleria-thumbnail",
  thumbnailNextButton: /* @__PURE__ */ __name(function thumbnailNextButton(_ref8) {
    var instance = _ref8.instance;
    return ["p-galleria-thumbnail-next-button  p-galleria-thumbnail-nav-button", {
      "p-disabled": instance.isNavForwardDisabled()
    }];
  }, "thumbnailNextButton"),
  thumbnailNextIcon: "p-galleria-thumbnail-next-icon"
};
var GalleriaStyle = BaseStyle.extend({
  name: "galleria",
  theme: theme$9,
  classes: classes$9
});
var script$c = {
  name: "ChevronUpIcon",
  "extends": script$z
};
var _hoisted_1$o = /* @__PURE__ */ createBaseVNode("path", {
  d: "M12.2097 10.4113C12.1057 10.4118 12.0027 10.3915 11.9067 10.3516C11.8107 10.3118 11.7237 10.2532 11.6506 10.1792L6.93602 5.46461L2.22139 10.1476C2.07272 10.244 1.89599 10.2877 1.71953 10.2717C1.54307 10.2556 1.3771 10.1808 1.24822 10.0593C1.11933 9.93766 1.035 9.77633 1.00874 9.6011C0.982477 9.42587 1.0158 9.2469 1.10338 9.09287L6.37701 3.81923C6.52533 3.6711 6.72639 3.58789 6.93602 3.58789C7.14565 3.58789 7.3467 3.6711 7.49502 3.81923L12.7687 9.09287C12.9168 9.24119 13 9.44225 13 9.65187C13 9.8615 12.9168 10.0626 12.7687 10.2109C12.616 10.3487 12.4151 10.4207 12.2097 10.4113Z",
  fill: "currentColor"
}, null, -1);
var _hoisted_2$h = [_hoisted_1$o];
function render$b(_ctx, _cache, $props, $setup, $data, $options) {
  return openBlock(), createElementBlock("svg", mergeProps({
    width: "14",
    height: "14",
    viewBox: "0 0 14 14",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg"
  }, _ctx.pti()), _hoisted_2$h, 16);
}
__name(render$b, "render$b");
script$c.render = render$b;
var script$4$1 = {
  name: "BaseGalleria",
  "extends": script$p,
  props: {
    id: {
      type: String,
      "default": null
    },
    value: {
      type: Array,
      "default": null
    },
    activeIndex: {
      type: Number,
      "default": 0
    },
    fullScreen: {
      type: Boolean,
      "default": false
    },
    visible: {
      type: Boolean,
      "default": false
    },
    numVisible: {
      type: Number,
      "default": 3
    },
    responsiveOptions: {
      type: Array,
      "default": null
    },
    showItemNavigators: {
      type: Boolean,
      "default": false
    },
    showThumbnailNavigators: {
      type: Boolean,
      "default": true
    },
    showItemNavigatorsOnHover: {
      type: Boolean,
      "default": false
    },
    changeItemOnIndicatorHover: {
      type: Boolean,
      "default": false
    },
    circular: {
      type: Boolean,
      "default": false
    },
    autoPlay: {
      type: Boolean,
      "default": false
    },
    transitionInterval: {
      type: Number,
      "default": 4e3
    },
    showThumbnails: {
      type: Boolean,
      "default": true
    },
    thumbnailsPosition: {
      type: String,
      "default": "bottom"
    },
    verticalThumbnailViewPortHeight: {
      type: String,
      "default": "300px"
    },
    showIndicators: {
      type: Boolean,
      "default": false
    },
    showIndicatorsOnItem: {
      type: Boolean,
      "default": false
    },
    indicatorsPosition: {
      type: String,
      "default": "bottom"
    },
    baseZIndex: {
      type: Number,
      "default": 0
    },
    maskClass: {
      type: String,
      "default": null
    },
    containerStyle: {
      type: null,
      "default": null
    },
    containerClass: {
      type: null,
      "default": null
    },
    containerProps: {
      type: null,
      "default": null
    },
    prevButtonProps: {
      type: null,
      "default": null
    },
    nextButtonProps: {
      type: null,
      "default": null
    },
    ariaLabel: {
      type: String,
      "default": null
    },
    ariaRoledescription: {
      type: String,
      "default": null
    }
  },
  style: GalleriaStyle,
  provide: /* @__PURE__ */ __name(function provide11() {
    return {
      $pcGalleria: this,
      $parentInstance: this
    };
  }, "provide")
};
function _toConsumableArray$1$1(r) {
  return _arrayWithoutHoles$1$1(r) || _iterableToArray$1$1(r) || _unsupportedIterableToArray$1$1(r) || _nonIterableSpread$1$1();
}
__name(_toConsumableArray$1$1, "_toConsumableArray$1$1");
function _nonIterableSpread$1$1() {
  throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.");
}
__name(_nonIterableSpread$1$1, "_nonIterableSpread$1$1");
function _unsupportedIterableToArray$1$1(r, a) {
  if (r) {
    if ("string" == typeof r) return _arrayLikeToArray$1$1(r, a);
    var t = {}.toString.call(r).slice(8, -1);
    return "Object" === t && r.constructor && (t = r.constructor.name), "Map" === t || "Set" === t ? Array.from(r) : "Arguments" === t || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(t) ? _arrayLikeToArray$1$1(r, a) : void 0;
  }
}
__name(_unsupportedIterableToArray$1$1, "_unsupportedIterableToArray$1$1");
function _iterableToArray$1$1(r) {
  if ("undefined" != typeof Symbol && null != r[Symbol.iterator] || null != r["@@iterator"]) return Array.from(r);
}
__name(_iterableToArray$1$1, "_iterableToArray$1$1");
function _arrayWithoutHoles$1$1(r) {
  if (Array.isArray(r)) return _arrayLikeToArray$1$1(r);
}
__name(_arrayWithoutHoles$1$1, "_arrayWithoutHoles$1$1");
function _arrayLikeToArray$1$1(r, a) {
  (null == a || a > r.length) && (a = r.length);
  for (var e = 0, n = Array(a); e < a; e++) n[e] = r[e];
  return n;
}
__name(_arrayLikeToArray$1$1, "_arrayLikeToArray$1$1");
var script$3$1 = {
  name: "GalleriaItem",
  hostName: "Galleria",
  "extends": script$p,
  emits: ["start-slideshow", "stop-slideshow", "update:activeIndex"],
  props: {
    circular: {
      type: Boolean,
      "default": false
    },
    activeIndex: {
      type: Number,
      "default": 0
    },
    value: {
      type: Array,
      "default": null
    },
    showItemNavigators: {
      type: Boolean,
      "default": true
    },
    showIndicators: {
      type: Boolean,
      "default": true
    },
    slideShowActive: {
      type: Boolean,
      "default": true
    },
    changeItemOnIndicatorHover: {
      type: Boolean,
      "default": true
    },
    autoPlay: {
      type: Boolean,
      "default": false
    },
    templates: {
      type: null,
      "default": null
    },
    id: {
      type: String,
      "default": null
    }
  },
  mounted: /* @__PURE__ */ __name(function mounted5() {
    if (this.autoPlay) {
      this.$emit("start-slideshow");
    }
  }, "mounted"),
  methods: {
    getIndicatorPTOptions: /* @__PURE__ */ __name(function getIndicatorPTOptions(index2) {
      return {
        context: {
          highlighted: this.activeIndex === index2
        }
      };
    }, "getIndicatorPTOptions"),
    next: /* @__PURE__ */ __name(function next() {
      var nextItemIndex = this.activeIndex + 1;
      var activeIndex2 = this.circular && this.value.length - 1 === this.activeIndex ? 0 : nextItemIndex;
      this.$emit("update:activeIndex", activeIndex2);
    }, "next"),
    prev: /* @__PURE__ */ __name(function prev() {
      var prevItemIndex = this.activeIndex !== 0 ? this.activeIndex - 1 : 0;
      var activeIndex2 = this.circular && this.activeIndex === 0 ? this.value.length - 1 : prevItemIndex;
      this.$emit("update:activeIndex", activeIndex2);
    }, "prev"),
    stopSlideShow: /* @__PURE__ */ __name(function stopSlideShow() {
      if (this.slideShowActive && this.stopSlideShow) {
        this.$emit("stop-slideshow");
      }
    }, "stopSlideShow"),
    navBackward: /* @__PURE__ */ __name(function navBackward(e) {
      this.stopSlideShow();
      this.prev();
      if (e && e.cancelable) {
        e.preventDefault();
      }
    }, "navBackward"),
    navForward: /* @__PURE__ */ __name(function navForward(e) {
      this.stopSlideShow();
      this.next();
      if (e && e.cancelable) {
        e.preventDefault();
      }
    }, "navForward"),
    onIndicatorClick: /* @__PURE__ */ __name(function onIndicatorClick(index2) {
      this.stopSlideShow();
      this.$emit("update:activeIndex", index2);
    }, "onIndicatorClick"),
    onIndicatorMouseEnter: /* @__PURE__ */ __name(function onIndicatorMouseEnter(index2) {
      if (this.changeItemOnIndicatorHover) {
        this.stopSlideShow();
        this.$emit("update:activeIndex", index2);
      }
    }, "onIndicatorMouseEnter"),
    onIndicatorKeyDown: /* @__PURE__ */ __name(function onIndicatorKeyDown(event, index2) {
      switch (event.code) {
        case "Enter":
        case "NumpadEnter":
        case "Space":
          this.stopSlideShow();
          this.$emit("update:activeIndex", index2);
          event.preventDefault();
          break;
        case "ArrowRight":
          this.onRightKey();
          break;
        case "ArrowLeft":
          this.onLeftKey();
          break;
        case "Home":
          this.onHomeKey();
          event.preventDefault();
          break;
        case "End":
          this.onEndKey();
          event.preventDefault();
          break;
        case "Tab":
          this.onTabKey();
          break;
        case "ArrowDown":
        case "ArrowUp":
        case "PageUp":
        case "PageDown":
          event.preventDefault();
          break;
      }
    }, "onIndicatorKeyDown"),
    onRightKey: /* @__PURE__ */ __name(function onRightKey() {
      var indicators = _toConsumableArray$1$1(find(this.$refs.indicatorContent, '[data-pc-section="indicator"]'));
      var activeIndex2 = this.findFocusedIndicatorIndex();
      this.changedFocusedIndicator(activeIndex2, activeIndex2 + 1 === indicators.length ? indicators.length - 1 : activeIndex2 + 1);
    }, "onRightKey"),
    onLeftKey: /* @__PURE__ */ __name(function onLeftKey() {
      var activeIndex2 = this.findFocusedIndicatorIndex();
      this.changedFocusedIndicator(activeIndex2, activeIndex2 - 1 <= 0 ? 0 : activeIndex2 - 1);
    }, "onLeftKey"),
    onHomeKey: /* @__PURE__ */ __name(function onHomeKey3() {
      var activeIndex2 = this.findFocusedIndicatorIndex();
      this.changedFocusedIndicator(activeIndex2, 0);
    }, "onHomeKey"),
    onEndKey: /* @__PURE__ */ __name(function onEndKey3() {
      var indicators = _toConsumableArray$1$1(find(this.$refs.indicatorContent, '[data-pc-section="indicator"]'));
      var activeIndex2 = this.findFocusedIndicatorIndex();
      this.changedFocusedIndicator(activeIndex2, indicators.length - 1);
    }, "onEndKey"),
    onTabKey: /* @__PURE__ */ __name(function onTabKey3() {
      var indicators = _toConsumableArray$1$1(find(this.$refs.indicatorContent, '[data-pc-section="indicator"]'));
      var highlightedIndex = indicators.findIndex(function(ind) {
        return getAttribute(ind, "data-p-active") === true;
      });
      var activeIndicator = findSingle(this.$refs.indicatorContent, '[data-pc-section="indicator"] > button[tabindex="0"]');
      var activeIndex2 = indicators.findIndex(function(ind) {
        return ind === activeIndicator.parentElement;
      });
      indicators[activeIndex2].children[0].tabIndex = "-1";
      indicators[highlightedIndex].children[0].tabIndex = "0";
    }, "onTabKey"),
    findFocusedIndicatorIndex: /* @__PURE__ */ __name(function findFocusedIndicatorIndex() {
      var indicators = _toConsumableArray$1$1(find(this.$refs.indicatorContent, '[data-pc-section="indicator"]'));
      var activeIndicator = findSingle(this.$refs.indicatorContent, '[data-pc-section="indicator"] > button[tabindex="0"]');
      return indicators.findIndex(function(ind) {
        return ind === activeIndicator.parentElement;
      });
    }, "findFocusedIndicatorIndex"),
    changedFocusedIndicator: /* @__PURE__ */ __name(function changedFocusedIndicator(prevInd, nextInd) {
      var indicators = _toConsumableArray$1$1(find(this.$refs.indicatorContent, '[data-pc-section="indicator"]'));
      indicators[prevInd].children[0].tabIndex = "-1";
      indicators[nextInd].children[0].tabIndex = "0";
      indicators[nextInd].children[0].focus();
    }, "changedFocusedIndicator"),
    isIndicatorItemActive: /* @__PURE__ */ __name(function isIndicatorItemActive(index2) {
      return this.activeIndex === index2;
    }, "isIndicatorItemActive"),
    isNavBackwardDisabled: /* @__PURE__ */ __name(function isNavBackwardDisabled() {
      return !this.circular && this.activeIndex === 0;
    }, "isNavBackwardDisabled"),
    isNavForwardDisabled: /* @__PURE__ */ __name(function isNavForwardDisabled() {
      return !this.circular && this.activeIndex === this.value.length - 1;
    }, "isNavForwardDisabled"),
    ariaSlideNumber: /* @__PURE__ */ __name(function ariaSlideNumber(value) {
      return this.$primevue.config.locale.aria ? this.$primevue.config.locale.aria.slideNumber.replace(/{slideNumber}/g, value) : void 0;
    }, "ariaSlideNumber"),
    ariaPageLabel: /* @__PURE__ */ __name(function ariaPageLabel(value) {
      return this.$primevue.config.locale.aria ? this.$primevue.config.locale.aria.pageLabel.replace(/{page}/g, value) : void 0;
    }, "ariaPageLabel")
  },
  computed: {
    activeItem: /* @__PURE__ */ __name(function activeItem() {
      return this.value[this.activeIndex];
    }, "activeItem"),
    ariaSlideLabel: /* @__PURE__ */ __name(function ariaSlideLabel() {
      return this.$primevue.config.locale.aria ? this.$primevue.config.locale.aria.slide : void 0;
    }, "ariaSlideLabel")
  },
  components: {
    ChevronLeftIcon: script$A,
    ChevronRightIcon: script$B
  },
  directives: {
    ripple: Ripple
  }
};
var _hoisted_1$3$1 = ["disabled"];
var _hoisted_2$2$1 = ["id", "aria-label", "aria-roledescription"];
var _hoisted_3$2$1 = ["disabled"];
var _hoisted_4$1$1 = ["aria-label", "aria-selected", "aria-controls", "onClick", "onMouseenter", "onKeydown", "data-p-active"];
var _hoisted_5$5 = ["tabindex"];
function render$3$1(_ctx, _cache, $props, $setup, $data, $options) {
  var _directive_ripple = resolveDirective("ripple");
  return openBlock(), createElementBlock("div", mergeProps({
    "class": _ctx.cx("itemsContainer")
  }, _ctx.ptm("itemsContainer")), [createBaseVNode("div", mergeProps({
    "class": _ctx.cx("items")
  }, _ctx.ptm("items")), [$props.showItemNavigators ? withDirectives((openBlock(), createElementBlock("button", mergeProps({
    key: 0,
    type: "button",
    "class": _ctx.cx("prevButton"),
    onClick: _cache[0] || (_cache[0] = function($event) {
      return $options.navBackward($event);
    }),
    disabled: $options.isNavBackwardDisabled()
  }, _ctx.ptm("prevButton"), {
    "data-pc-group-section": "itemnavigator"
  }), [(openBlock(), createBlock(resolveDynamicComponent($props.templates.previousitemicon || "ChevronLeftIcon"), mergeProps({
    "class": _ctx.cx("prevIcon")
  }, _ctx.ptm("prevIcon")), null, 16, ["class"]))], 16, _hoisted_1$3$1)), [[_directive_ripple]]) : createCommentVNode("", true), createBaseVNode("div", mergeProps({
    id: $props.id + "_item_" + $props.activeIndex,
    "class": _ctx.cx("item"),
    role: "group",
    "aria-label": $options.ariaSlideNumber($props.activeIndex + 1),
    "aria-roledescription": $options.ariaSlideLabel
  }, _ctx.ptm("item")), [$props.templates.item ? (openBlock(), createBlock(resolveDynamicComponent($props.templates.item), {
    key: 0,
    item: $options.activeItem
  }, null, 8, ["item"])) : createCommentVNode("", true)], 16, _hoisted_2$2$1), $props.showItemNavigators ? withDirectives((openBlock(), createElementBlock("button", mergeProps({
    key: 1,
    type: "button",
    "class": _ctx.cx("nextButton"),
    onClick: _cache[1] || (_cache[1] = function($event) {
      return $options.navForward($event);
    }),
    disabled: $options.isNavForwardDisabled()
  }, _ctx.ptm("nextButton"), {
    "data-pc-group-section": "itemnavigator"
  }), [(openBlock(), createBlock(resolveDynamicComponent($props.templates.nextitemicon || "ChevronRightIcon"), mergeProps({
    "class": _ctx.cx("nextIcon")
  }, _ctx.ptm("nextIcon")), null, 16, ["class"]))], 16, _hoisted_3$2$1)), [[_directive_ripple]]) : createCommentVNode("", true), $props.templates["caption"] ? (openBlock(), createElementBlock("div", mergeProps({
    key: 2,
    "class": _ctx.cx("caption")
  }, _ctx.ptm("caption")), [$props.templates.caption ? (openBlock(), createBlock(resolveDynamicComponent($props.templates.caption), {
    key: 0,
    item: $options.activeItem
  }, null, 8, ["item"])) : createCommentVNode("", true)], 16)) : createCommentVNode("", true)], 16), $props.showIndicators ? (openBlock(), createElementBlock("ul", mergeProps({
    key: 0,
    ref: "indicatorContent",
    "class": _ctx.cx("indicatorList")
  }, _ctx.ptm("indicatorList")), [(openBlock(true), createElementBlock(Fragment, null, renderList($props.value, function(item4, index2) {
    return openBlock(), createElementBlock("li", mergeProps({
      key: "p-galleria-indicator-".concat(index2),
      "class": _ctx.cx("indicator", {
        index: index2
      }),
      "aria-label": $options.ariaPageLabel(index2 + 1),
      "aria-selected": $props.activeIndex === index2,
      "aria-controls": $props.id + "_item_" + index2,
      onClick: /* @__PURE__ */ __name(function onClick2($event) {
        return $options.onIndicatorClick(index2);
      }, "onClick"),
      onMouseenter: /* @__PURE__ */ __name(function onMouseenter($event) {
        return $options.onIndicatorMouseEnter(index2);
      }, "onMouseenter"),
      onKeydown: /* @__PURE__ */ __name(function onKeydown($event) {
        return $options.onIndicatorKeyDown($event, index2);
      }, "onKeydown"),
      ref_for: true
    }, _ctx.ptm("indicator", $options.getIndicatorPTOptions(index2)), {
      "data-p-active": $options.isIndicatorItemActive(index2)
    }), [!$props.templates["indicator"] ? (openBlock(), createElementBlock("button", mergeProps({
      key: 0,
      type: "button",
      tabindex: $props.activeIndex === index2 ? "0" : "-1",
      "class": _ctx.cx("indicatorButton"),
      ref_for: true
    }, _ctx.ptm("indicatorButton", $options.getIndicatorPTOptions(index2))), null, 16, _hoisted_5$5)) : createCommentVNode("", true), $props.templates.indicator ? (openBlock(), createBlock(resolveDynamicComponent($props.templates.indicator), {
      key: 1,
      index: index2
    }, null, 8, ["index"])) : createCommentVNode("", true)], 16, _hoisted_4$1$1);
  }), 128))], 16)) : createCommentVNode("", true)], 16);
}
__name(render$3$1, "render$3$1");
script$3$1.render = render$3$1;
function _toConsumableArray$3(r) {
  return _arrayWithoutHoles$3(r) || _iterableToArray$3(r) || _unsupportedIterableToArray$3(r) || _nonIterableSpread$3();
}
__name(_toConsumableArray$3, "_toConsumableArray$3");
function _nonIterableSpread$3() {
  throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.");
}
__name(_nonIterableSpread$3, "_nonIterableSpread$3");
function _unsupportedIterableToArray$3(r, a) {
  if (r) {
    if ("string" == typeof r) return _arrayLikeToArray$3(r, a);
    var t = {}.toString.call(r).slice(8, -1);
    return "Object" === t && r.constructor && (t = r.constructor.name), "Map" === t || "Set" === t ? Array.from(r) : "Arguments" === t || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(t) ? _arrayLikeToArray$3(r, a) : void 0;
  }
}
__name(_unsupportedIterableToArray$3, "_unsupportedIterableToArray$3");
function _iterableToArray$3(r) {
  if ("undefined" != typeof Symbol && null != r[Symbol.iterator] || null != r["@@iterator"]) return Array.from(r);
}
__name(_iterableToArray$3, "_iterableToArray$3");
function _arrayWithoutHoles$3(r) {
  if (Array.isArray(r)) return _arrayLikeToArray$3(r);
}
__name(_arrayWithoutHoles$3, "_arrayWithoutHoles$3");
function _arrayLikeToArray$3(r, a) {
  (null == a || a > r.length) && (a = r.length);
  for (var e = 0, n = Array(a); e < a; e++) n[e] = r[e];
  return n;
}
__name(_arrayLikeToArray$3, "_arrayLikeToArray$3");
var script$2$4 = {
  name: "GalleriaThumbnails",
  hostName: "Galleria",
  "extends": script$p,
  emits: ["stop-slideshow", "update:activeIndex"],
  props: {
    containerId: {
      type: String,
      "default": null
    },
    value: {
      type: Array,
      "default": null
    },
    numVisible: {
      type: Number,
      "default": 3
    },
    activeIndex: {
      type: Number,
      "default": 0
    },
    isVertical: {
      type: Boolean,
      "default": false
    },
    slideShowActive: {
      type: Boolean,
      "default": false
    },
    circular: {
      type: Boolean,
      "default": false
    },
    responsiveOptions: {
      type: Array,
      "default": null
    },
    contentHeight: {
      type: String,
      "default": "300px"
    },
    showThumbnailNavigators: {
      type: Boolean,
      "default": true
    },
    templates: {
      type: null,
      "default": null
    },
    prevButtonProps: {
      type: null,
      "default": null
    },
    nextButtonProps: {
      type: null,
      "default": null
    }
  },
  startPos: null,
  thumbnailsStyle: null,
  sortedResponsiveOptions: null,
  data: /* @__PURE__ */ __name(function data6() {
    return {
      d_numVisible: this.numVisible,
      d_oldNumVisible: this.numVisible,
      d_activeIndex: this.activeIndex,
      d_oldActiveItemIndex: this.activeIndex,
      totalShiftedItems: 0,
      page: 0
    };
  }, "data"),
  watch: {
    numVisible: /* @__PURE__ */ __name(function numVisible(newValue, oldValue) {
      this.d_numVisible = newValue;
      this.d_oldNumVisible = oldValue;
    }, "numVisible"),
    activeIndex: /* @__PURE__ */ __name(function activeIndex(newValue, oldValue) {
      this.d_activeIndex = newValue;
      this.d_oldActiveItemIndex = oldValue;
    }, "activeIndex")
  },
  mounted: /* @__PURE__ */ __name(function mounted6() {
    this.createStyle();
    this.calculatePosition();
    if (this.responsiveOptions) {
      this.bindDocumentListeners();
    }
  }, "mounted"),
  updated: /* @__PURE__ */ __name(function updated2() {
    var totalShiftedItems = this.totalShiftedItems;
    if (this.d_oldNumVisible !== this.d_numVisible || this.d_oldActiveItemIndex !== this.d_activeIndex) {
      if (this.d_activeIndex <= this.getMedianItemIndex()) {
        totalShiftedItems = 0;
      } else if (this.value.length - this.d_numVisible + this.getMedianItemIndex() < this.d_activeIndex) {
        totalShiftedItems = this.d_numVisible - this.value.length;
      } else if (this.value.length - this.d_numVisible < this.d_activeIndex && this.d_numVisible % 2 === 0) {
        totalShiftedItems = this.d_activeIndex * -1 + this.getMedianItemIndex() + 1;
      } else {
        totalShiftedItems = this.d_activeIndex * -1 + this.getMedianItemIndex();
      }
      if (totalShiftedItems !== this.totalShiftedItems) {
        this.totalShiftedItems = totalShiftedItems;
      }
      this.$refs.itemsContainer.style.transform = this.isVertical ? "translate3d(0, ".concat(totalShiftedItems * (100 / this.d_numVisible), "%, 0)") : "translate3d(".concat(totalShiftedItems * (100 / this.d_numVisible), "%, 0, 0)");
      if (this.d_oldActiveItemIndex !== this.d_activeIndex) {
        document.body.setAttribute("data-p-items-hidden", "false");
        !this.isUnstyled && removeClass(this.$refs.itemsContainer, "p-items-hidden");
        this.$refs.itemsContainer.style.transition = "transform 500ms ease 0s";
      }
      this.d_oldActiveItemIndex = this.d_activeIndex;
      this.d_oldNumVisible = this.d_numVisible;
    }
  }, "updated"),
  beforeUnmount: /* @__PURE__ */ __name(function beforeUnmount5() {
    if (this.responsiveOptions) {
      this.unbindDocumentListeners();
    }
    if (this.thumbnailsStyle) {
      this.thumbnailsStyle.parentNode.removeChild(this.thumbnailsStyle);
    }
  }, "beforeUnmount"),
  methods: {
    step: /* @__PURE__ */ __name(function step(dir) {
      var totalShiftedItems = this.totalShiftedItems + dir;
      if (dir < 0 && -1 * totalShiftedItems + this.d_numVisible > this.value.length - 1) {
        totalShiftedItems = this.d_numVisible - this.value.length;
      } else if (dir > 0 && totalShiftedItems > 0) {
        totalShiftedItems = 0;
      }
      if (this.circular) {
        if (dir < 0 && this.value.length - 1 === this.d_activeIndex) {
          totalShiftedItems = 0;
        } else if (dir > 0 && this.d_activeIndex === 0) {
          totalShiftedItems = this.d_numVisible - this.value.length;
        }
      }
      if (this.$refs.itemsContainer) {
        document.body.setAttribute("data-p-items-hidden", "false");
        !this.isUnstyled && removeClass(this.$refs.itemsContainer, "p-items-hidden");
        this.$refs.itemsContainer.style.transform = this.isVertical ? "translate3d(0, ".concat(totalShiftedItems * (100 / this.d_numVisible), "%, 0)") : "translate3d(".concat(totalShiftedItems * (100 / this.d_numVisible), "%, 0, 0)");
        this.$refs.itemsContainer.style.transition = "transform 500ms ease 0s";
      }
      this.totalShiftedItems = totalShiftedItems;
    }, "step"),
    stopSlideShow: /* @__PURE__ */ __name(function stopSlideShow2() {
      if (this.slideShowActive && this.stopSlideShow) {
        this.$emit("stop-slideshow");
      }
    }, "stopSlideShow"),
    getMedianItemIndex: /* @__PURE__ */ __name(function getMedianItemIndex() {
      var index2 = Math.floor(this.d_numVisible / 2);
      return this.d_numVisible % 2 ? index2 : index2 - 1;
    }, "getMedianItemIndex"),
    navBackward: /* @__PURE__ */ __name(function navBackward2(e) {
      this.stopSlideShow();
      var prevItemIndex = this.d_activeIndex !== 0 ? this.d_activeIndex - 1 : 0;
      var diff = prevItemIndex + this.totalShiftedItems;
      if (this.d_numVisible - diff - 1 > this.getMedianItemIndex() && (-1 * this.totalShiftedItems !== 0 || this.circular)) {
        this.step(1);
      }
      var activeIndex2 = this.circular && this.d_activeIndex === 0 ? this.value.length - 1 : prevItemIndex;
      this.$emit("update:activeIndex", activeIndex2);
      if (e.cancelable) {
        e.preventDefault();
      }
    }, "navBackward"),
    navForward: /* @__PURE__ */ __name(function navForward2(e) {
      this.stopSlideShow();
      var nextItemIndex = this.d_activeIndex === this.value.length - 1 ? this.value.length - 1 : this.d_activeIndex + 1;
      if (nextItemIndex + this.totalShiftedItems > this.getMedianItemIndex() && (-1 * this.totalShiftedItems < this.getTotalPageNumber() - 1 || this.circular)) {
        this.step(-1);
      }
      var activeIndex2 = this.circular && this.value.length - 1 === this.d_activeIndex ? 0 : nextItemIndex;
      this.$emit("update:activeIndex", activeIndex2);
      if (e.cancelable) {
        e.preventDefault();
      }
    }, "navForward"),
    onItemClick: /* @__PURE__ */ __name(function onItemClick3(index2) {
      this.stopSlideShow();
      var selectedItemIndex = index2;
      if (selectedItemIndex !== this.d_activeIndex) {
        var diff = selectedItemIndex + this.totalShiftedItems;
        var dir = 0;
        if (selectedItemIndex < this.d_activeIndex) {
          dir = this.d_numVisible - diff - 1 - this.getMedianItemIndex();
          if (dir > 0 && -1 * this.totalShiftedItems !== 0) {
            this.step(dir);
          }
        } else {
          dir = this.getMedianItemIndex() - diff;
          if (dir < 0 && -1 * this.totalShiftedItems < this.getTotalPageNumber() - 1) {
            this.step(dir);
          }
        }
        this.$emit("update:activeIndex", selectedItemIndex);
      }
    }, "onItemClick"),
    onThumbnailKeydown: /* @__PURE__ */ __name(function onThumbnailKeydown(event, index2) {
      if (event.code === "Enter" || event.code === "NumpadEnter" || event.code === "Space") {
        this.onItemClick(index2);
        event.preventDefault();
      }
      switch (event.code) {
        case "ArrowRight":
          this.onRightKey();
          break;
        case "ArrowLeft":
          this.onLeftKey();
          break;
        case "Home":
          this.onHomeKey();
          event.preventDefault();
          break;
        case "End":
          this.onEndKey();
          event.preventDefault();
          break;
        case "ArrowUp":
        case "ArrowDown":
          event.preventDefault();
          break;
        case "Tab":
          this.onTabKey();
          break;
      }
    }, "onThumbnailKeydown"),
    onRightKey: /* @__PURE__ */ __name(function onRightKey2() {
      var indicators = find(this.$refs.itemsContainer, '[data-pc-section="thumbnailitem"]');
      var activeIndex2 = this.findFocusedIndicatorIndex();
      this.changedFocusedIndicator(activeIndex2, activeIndex2 + 1 === indicators.length ? indicators.length - 1 : activeIndex2 + 1);
    }, "onRightKey"),
    onLeftKey: /* @__PURE__ */ __name(function onLeftKey2() {
      var activeIndex2 = this.findFocusedIndicatorIndex();
      this.changedFocusedIndicator(activeIndex2, activeIndex2 - 1 <= 0 ? 0 : activeIndex2 - 1);
    }, "onLeftKey"),
    onHomeKey: /* @__PURE__ */ __name(function onHomeKey4() {
      var activeIndex2 = this.findFocusedIndicatorIndex();
      this.changedFocusedIndicator(activeIndex2, 0);
    }, "onHomeKey"),
    onEndKey: /* @__PURE__ */ __name(function onEndKey4() {
      var indicators = find(this.$refs.itemsContainer, '[data-pc-section="thumbnailitem"]');
      var activeIndex2 = this.findFocusedIndicatorIndex();
      this.changedFocusedIndicator(activeIndex2, indicators.length - 1);
    }, "onEndKey"),
    onTabKey: /* @__PURE__ */ __name(function onTabKey4() {
      var indicators = _toConsumableArray$3(find(this.$refs.itemsContainer, '[data-pc-section="thumbnailitem"]'));
      var highlightedIndex = indicators.findIndex(function(ind) {
        return getAttribute(ind, "data-p-active") === true;
      });
      var activeIndicator = findSingle(this.$refs.itemsContainer, '[tabindex="0"]');
      var activeIndex2 = indicators.findIndex(function(ind) {
        return ind === activeIndicator.parentElement;
      });
      indicators[activeIndex2].children[0].tabIndex = "-1";
      indicators[highlightedIndex].children[0].tabIndex = "0";
    }, "onTabKey"),
    findFocusedIndicatorIndex: /* @__PURE__ */ __name(function findFocusedIndicatorIndex2() {
      var indicators = _toConsumableArray$3(find(this.$refs.itemsContainer, '[data-pc-section="thumbnailitem"]'));
      var activeIndicator = findSingle(this.$refs.itemsContainer, '[data-pc-section="thumbnailitem"] > [tabindex="0"]');
      return indicators.findIndex(function(ind) {
        return ind === activeIndicator.parentElement;
      });
    }, "findFocusedIndicatorIndex"),
    changedFocusedIndicator: /* @__PURE__ */ __name(function changedFocusedIndicator2(prevInd, nextInd) {
      var indicators = find(this.$refs.itemsContainer, '[data-pc-section="thumbnailitem"]');
      indicators[prevInd].children[0].tabIndex = "-1";
      indicators[nextInd].children[0].tabIndex = "0";
      indicators[nextInd].children[0].focus();
    }, "changedFocusedIndicator"),
    onTransitionEnd: /* @__PURE__ */ __name(function onTransitionEnd(e) {
      if (this.$refs.itemsContainer && e.propertyName === "transform") {
        document.body.setAttribute("data-p-items-hidden", "true");
        !this.isUnstyled && addClass(this.$refs.itemsContainer, "p-items-hidden");
        this.$refs.itemsContainer.style.transition = "";
      }
    }, "onTransitionEnd"),
    onTouchStart: /* @__PURE__ */ __name(function onTouchStart(e) {
      var touchobj = e.changedTouches[0];
      this.startPos = {
        x: touchobj.pageX,
        y: touchobj.pageY
      };
    }, "onTouchStart"),
    onTouchMove: /* @__PURE__ */ __name(function onTouchMove(e) {
      if (e.cancelable) {
        e.preventDefault();
      }
    }, "onTouchMove"),
    onTouchEnd: /* @__PURE__ */ __name(function onTouchEnd(e) {
      var touchobj = e.changedTouches[0];
      if (this.isVertical) {
        this.changePageOnTouch(e, touchobj.pageY - this.startPos.y);
      } else {
        this.changePageOnTouch(e, touchobj.pageX - this.startPos.x);
      }
    }, "onTouchEnd"),
    changePageOnTouch: /* @__PURE__ */ __name(function changePageOnTouch(e, diff) {
      if (diff < 0) {
        this.navForward(e);
      } else {
        this.navBackward(e);
      }
    }, "changePageOnTouch"),
    getTotalPageNumber: /* @__PURE__ */ __name(function getTotalPageNumber() {
      return this.value.length > this.d_numVisible ? this.value.length - this.d_numVisible + 1 : 0;
    }, "getTotalPageNumber"),
    createStyle: /* @__PURE__ */ __name(function createStyle() {
      if (!this.thumbnailsStyle) {
        var _this$$primevue;
        this.thumbnailsStyle = document.createElement("style");
        this.thumbnailsStyle.type = "text/css";
        setAttribute(this.thumbnailsStyle, "nonce", (_this$$primevue = this.$primevue) === null || _this$$primevue === void 0 || (_this$$primevue = _this$$primevue.config) === null || _this$$primevue === void 0 || (_this$$primevue = _this$$primevue.csp) === null || _this$$primevue === void 0 ? void 0 : _this$$primevue.nonce);
        document.body.appendChild(this.thumbnailsStyle);
      }
      var innerHTML = "\n                #".concat(this.containerId, ' [data-pc-section="thumbnailitem"] {\n                    flex: 1 0 ').concat(100 / this.d_numVisible, "%\n                }\n            ");
      if (this.responsiveOptions && !this.isUnstyled) {
        this.sortedResponsiveOptions = _toConsumableArray$3(this.responsiveOptions);
        var comparer = localeComparator();
        this.sortedResponsiveOptions.sort(function(data1, data22) {
          var value1 = data1.breakpoint;
          var value2 = data22.breakpoint;
          return sort(value1, value2, -1, comparer);
        });
        for (var i = 0; i < this.sortedResponsiveOptions.length; i++) {
          var res = this.sortedResponsiveOptions[i];
          innerHTML += "\n                        @media screen and (max-width: ".concat(res.breakpoint, ") {\n                            #").concat(this.containerId, " .p-galleria-thumbnail-item {\n                                flex: 1 0 ").concat(100 / res.numVisible, "%\n                            }\n                        }\n                    ");
        }
      }
      this.thumbnailsStyle.innerHTML = innerHTML;
    }, "createStyle"),
    calculatePosition: /* @__PURE__ */ __name(function calculatePosition() {
      if (this.$refs.itemsContainer && this.sortedResponsiveOptions) {
        var windowWidth = window.innerWidth;
        var matchedResponsiveData = {
          numVisible: this.numVisible
        };
        for (var i = 0; i < this.sortedResponsiveOptions.length; i++) {
          var res = this.sortedResponsiveOptions[i];
          if (parseInt(res.breakpoint, 10) >= windowWidth) {
            matchedResponsiveData = res;
          }
        }
        if (this.d_numVisible !== matchedResponsiveData.numVisible) {
          this.d_numVisible = matchedResponsiveData.numVisible;
        }
      }
    }, "calculatePosition"),
    bindDocumentListeners: /* @__PURE__ */ __name(function bindDocumentListeners() {
      var _this = this;
      if (!this.documentResizeListener) {
        this.documentResizeListener = function() {
          _this.calculatePosition();
        };
        window.addEventListener("resize", this.documentResizeListener);
      }
    }, "bindDocumentListeners"),
    unbindDocumentListeners: /* @__PURE__ */ __name(function unbindDocumentListeners() {
      if (this.documentResizeListener) {
        window.removeEventListener("resize", this.documentResizeListener);
        this.documentResizeListener = null;
      }
    }, "unbindDocumentListeners"),
    isNavBackwardDisabled: /* @__PURE__ */ __name(function isNavBackwardDisabled2() {
      return !this.circular && this.d_activeIndex === 0 || this.value.length <= this.d_numVisible;
    }, "isNavBackwardDisabled"),
    isNavForwardDisabled: /* @__PURE__ */ __name(function isNavForwardDisabled2() {
      return !this.circular && this.d_activeIndex === this.value.length - 1 || this.value.length <= this.d_numVisible;
    }, "isNavForwardDisabled"),
    firstItemAciveIndex: /* @__PURE__ */ __name(function firstItemAciveIndex() {
      return this.totalShiftedItems * -1;
    }, "firstItemAciveIndex"),
    lastItemActiveIndex: /* @__PURE__ */ __name(function lastItemActiveIndex() {
      return this.firstItemAciveIndex() + this.d_numVisible - 1;
    }, "lastItemActiveIndex"),
    isItemActive: /* @__PURE__ */ __name(function isItemActive2(index2) {
      return this.firstItemAciveIndex() <= index2 && this.lastItemActiveIndex() >= index2;
    }, "isItemActive"),
    ariaPageLabel: /* @__PURE__ */ __name(function ariaPageLabel2(value) {
      return this.$primevue.config.locale.aria ? this.$primevue.config.locale.aria.pageLabel.replace(/{page}/g, value) : void 0;
    }, "ariaPageLabel")
  },
  computed: {
    ariaPrevButtonLabel: /* @__PURE__ */ __name(function ariaPrevButtonLabel() {
      return this.$primevue.config.locale.aria ? this.$primevue.config.locale.aria.prevPageLabel : void 0;
    }, "ariaPrevButtonLabel"),
    ariaNextButtonLabel: /* @__PURE__ */ __name(function ariaNextButtonLabel() {
      return this.$primevue.config.locale.aria ? this.$primevue.config.locale.aria.nextPageLabel : void 0;
    }, "ariaNextButtonLabel")
  },
  components: {
    ChevronLeftIcon: script$A,
    ChevronRightIcon: script$B,
    ChevronUpIcon: script$c,
    ChevronDownIcon: script$s
  },
  directives: {
    ripple: Ripple
  }
};
function _typeof$2$1(o) {
  "@babel/helpers - typeof";
  return _typeof$2$1 = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function(o2) {
    return typeof o2;
  } : function(o2) {
    return o2 && "function" == typeof Symbol && o2.constructor === Symbol && o2 !== Symbol.prototype ? "symbol" : typeof o2;
  }, _typeof$2$1(o);
}
__name(_typeof$2$1, "_typeof$2$1");
function ownKeys$2$1(e, r) {
  var t = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var o = Object.getOwnPropertySymbols(e);
    r && (o = o.filter(function(r2) {
      return Object.getOwnPropertyDescriptor(e, r2).enumerable;
    })), t.push.apply(t, o);
  }
  return t;
}
__name(ownKeys$2$1, "ownKeys$2$1");
function _objectSpread$2$1(e) {
  for (var r = 1; r < arguments.length; r++) {
    var t = null != arguments[r] ? arguments[r] : {};
    r % 2 ? ownKeys$2$1(Object(t), true).forEach(function(r2) {
      _defineProperty$2$1(e, r2, t[r2]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(t)) : ownKeys$2$1(Object(t)).forEach(function(r2) {
      Object.defineProperty(e, r2, Object.getOwnPropertyDescriptor(t, r2));
    });
  }
  return e;
}
__name(_objectSpread$2$1, "_objectSpread$2$1");
function _defineProperty$2$1(e, r, t) {
  return (r = _toPropertyKey$2$1(r)) in e ? Object.defineProperty(e, r, { value: t, enumerable: true, configurable: true, writable: true }) : e[r] = t, e;
}
__name(_defineProperty$2$1, "_defineProperty$2$1");
function _toPropertyKey$2$1(t) {
  var i = _toPrimitive$2$1(t, "string");
  return "symbol" == _typeof$2$1(i) ? i : i + "";
}
__name(_toPropertyKey$2$1, "_toPropertyKey$2$1");
function _toPrimitive$2$1(t, r) {
  if ("object" != _typeof$2$1(t) || !t) return t;
  var e = t[Symbol.toPrimitive];
  if (void 0 !== e) {
    var i = e.call(t, r || "default");
    if ("object" != _typeof$2$1(i)) return i;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return ("string" === r ? String : Number)(t);
}
__name(_toPrimitive$2$1, "_toPrimitive$2$1");
var _hoisted_1$2$1 = ["disabled", "aria-label"];
var _hoisted_2$1$1 = ["data-p-active", "aria-selected", "aria-controls", "onKeydown", "data-p-galleria-thumbnail-item-current", "data-p-galleria-thumbnail-item-active", "data-p-galleria-thumbnail-item-start", "data-p-galleria-thumbnail-item-end"];
var _hoisted_3$1$1 = ["tabindex", "aria-label", "aria-current", "onClick"];
var _hoisted_4$7 = ["disabled", "aria-label"];
function render$2$1(_ctx, _cache, $props, $setup, $data, $options) {
  var _directive_ripple = resolveDirective("ripple");
  return openBlock(), createElementBlock("div", mergeProps({
    "class": _ctx.cx("thumbnails")
  }, _ctx.ptm("thumbnails")), [createBaseVNode("div", mergeProps({
    "class": _ctx.cx("thumbnailContent")
  }, _ctx.ptm("thumbnailContent")), [$props.showThumbnailNavigators ? withDirectives((openBlock(), createElementBlock("button", mergeProps({
    key: 0,
    "class": _ctx.cx("thumbnailPrevButton"),
    disabled: $options.isNavBackwardDisabled(),
    type: "button",
    "aria-label": $options.ariaPrevButtonLabel,
    onClick: _cache[0] || (_cache[0] = function($event) {
      return $options.navBackward($event);
    })
  }, _objectSpread$2$1(_objectSpread$2$1({}, $props.prevButtonProps), _ctx.ptm("thumbnailPrevButton")), {
    "data-pc-group-section": "thumbnailnavigator"
  }), [(openBlock(), createBlock(resolveDynamicComponent($props.templates.previousthumbnailicon || ($props.isVertical ? "ChevronUpIcon" : "ChevronLeftIcon")), mergeProps({
    "class": _ctx.cx("thumbnailPrevIcon")
  }, _ctx.ptm("thumbnailPrevIcon")), null, 16, ["class"]))], 16, _hoisted_1$2$1)), [[_directive_ripple]]) : createCommentVNode("", true), createBaseVNode("div", mergeProps({
    "class": _ctx.cx("thumbnailsViewport"),
    style: {
      height: $props.isVertical ? $props.contentHeight : ""
    }
  }, _ctx.ptm("thumbnailsViewport")), [createBaseVNode("div", mergeProps({
    ref: "itemsContainer",
    "class": _ctx.cx("thumbnailItems"),
    role: "tablist",
    onTransitionend: _cache[1] || (_cache[1] = function($event) {
      return $options.onTransitionEnd($event);
    }),
    onTouchstart: _cache[2] || (_cache[2] = function($event) {
      return $options.onTouchStart($event);
    }),
    onTouchmove: _cache[3] || (_cache[3] = function($event) {
      return $options.onTouchMove($event);
    }),
    onTouchend: _cache[4] || (_cache[4] = function($event) {
      return $options.onTouchEnd($event);
    })
  }, _ctx.ptm("thumbnailItems")), [(openBlock(true), createElementBlock(Fragment, null, renderList($props.value, function(item4, index2) {
    return openBlock(), createElementBlock("div", mergeProps({
      key: "p-galleria-thumbnail-item-".concat(index2),
      "class": _ctx.cx("thumbnailItem", {
        index: index2,
        activeIndex: $props.activeIndex
      }),
      role: "tab",
      "data-p-active": $props.activeIndex === index2,
      "aria-selected": $props.activeIndex === index2,
      "aria-controls": $props.containerId + "_item_" + index2,
      onKeydown: /* @__PURE__ */ __name(function onKeydown($event) {
        return $options.onThumbnailKeydown($event, index2);
      }, "onKeydown"),
      ref_for: true
    }, _ctx.ptm("thumbnailItem"), {
      "data-p-galleria-thumbnail-item-current": $props.activeIndex === index2,
      "data-p-galleria-thumbnail-item-active": $options.isItemActive(index2),
      "data-p-galleria-thumbnail-item-start": $options.firstItemAciveIndex() === index2,
      "data-p-galleria-thumbnail-item-end": $options.lastItemActiveIndex() === index2
    }), [createBaseVNode("div", mergeProps({
      "class": _ctx.cx("thumbnail"),
      tabindex: $props.activeIndex === index2 ? "0" : "-1",
      "aria-label": $options.ariaPageLabel(index2 + 1),
      "aria-current": $props.activeIndex === index2 ? "page" : void 0,
      onClick: /* @__PURE__ */ __name(function onClick2($event) {
        return $options.onItemClick(index2);
      }, "onClick"),
      ref_for: true
    }, _ctx.ptm("thumbnail")), [$props.templates.thumbnail ? (openBlock(), createBlock(resolveDynamicComponent($props.templates.thumbnail), {
      key: 0,
      item: item4
    }, null, 8, ["item"])) : createCommentVNode("", true)], 16, _hoisted_3$1$1)], 16, _hoisted_2$1$1);
  }), 128))], 16)], 16), $props.showThumbnailNavigators ? withDirectives((openBlock(), createElementBlock("button", mergeProps({
    key: 1,
    "class": _ctx.cx("thumbnailNextButton"),
    disabled: $options.isNavForwardDisabled(),
    type: "button",
    "aria-label": $options.ariaNextButtonLabel,
    onClick: _cache[5] || (_cache[5] = function($event) {
      return $options.navForward($event);
    })
  }, _objectSpread$2$1(_objectSpread$2$1({}, $props.nextButtonProps), _ctx.ptm("thumbnailNextButton")), {
    "data-pc-group-section": "thumbnailnavigator"
  }), [(openBlock(), createBlock(resolveDynamicComponent($props.templates.nextthumbnailicon || ($props.isVertical ? "ChevronDownIcon" : "ChevronRightIcon")), mergeProps({
    "class": _ctx.cx("thumbnailNextIcon")
  }, _ctx.ptm("thumbnailNextIcon")), null, 16, ["class"]))], 16, _hoisted_4$7)), [[_directive_ripple]]) : createCommentVNode("", true)], 16)], 16);
}
__name(render$2$1, "render$2$1");
script$2$4.render = render$2$1;
function _typeof$1$2(o) {
  "@babel/helpers - typeof";
  return _typeof$1$2 = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function(o2) {
    return typeof o2;
  } : function(o2) {
    return o2 && "function" == typeof Symbol && o2.constructor === Symbol && o2 !== Symbol.prototype ? "symbol" : typeof o2;
  }, _typeof$1$2(o);
}
__name(_typeof$1$2, "_typeof$1$2");
function ownKeys$1$2(e, r) {
  var t = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var o = Object.getOwnPropertySymbols(e);
    r && (o = o.filter(function(r2) {
      return Object.getOwnPropertyDescriptor(e, r2).enumerable;
    })), t.push.apply(t, o);
  }
  return t;
}
__name(ownKeys$1$2, "ownKeys$1$2");
function _objectSpread$1$2(e) {
  for (var r = 1; r < arguments.length; r++) {
    var t = null != arguments[r] ? arguments[r] : {};
    r % 2 ? ownKeys$1$2(Object(t), true).forEach(function(r2) {
      _defineProperty$1$2(e, r2, t[r2]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(t)) : ownKeys$1$2(Object(t)).forEach(function(r2) {
      Object.defineProperty(e, r2, Object.getOwnPropertyDescriptor(t, r2));
    });
  }
  return e;
}
__name(_objectSpread$1$2, "_objectSpread$1$2");
function _defineProperty$1$2(e, r, t) {
  return (r = _toPropertyKey$1$2(r)) in e ? Object.defineProperty(e, r, { value: t, enumerable: true, configurable: true, writable: true }) : e[r] = t, e;
}
__name(_defineProperty$1$2, "_defineProperty$1$2");
function _toPropertyKey$1$2(t) {
  var i = _toPrimitive$1$2(t, "string");
  return "symbol" == _typeof$1$2(i) ? i : i + "";
}
__name(_toPropertyKey$1$2, "_toPropertyKey$1$2");
function _toPrimitive$1$2(t, r) {
  if ("object" != _typeof$1$2(t) || !t) return t;
  var e = t[Symbol.toPrimitive];
  if (void 0 !== e) {
    var i = e.call(t, r || "default");
    if ("object" != _typeof$1$2(i)) return i;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return ("string" === r ? String : Number)(t);
}
__name(_toPrimitive$1$2, "_toPrimitive$1$2");
var script$1$9 = {
  name: "GalleriaContent",
  hostName: "Galleria",
  "extends": script$p,
  inheritAttrs: false,
  interval: null,
  emits: ["activeitem-change", "mask-hide"],
  data: /* @__PURE__ */ __name(function data7() {
    return {
      id: this.$attrs.id || UniqueComponentId(),
      activeIndex: this.$attrs.activeIndex,
      numVisible: this.$attrs.numVisible,
      slideShowActive: false
    };
  }, "data"),
  watch: {
    "$attrs.id": /* @__PURE__ */ __name(function $attrsId3(newValue) {
      this.id = newValue || UniqueComponentId();
    }, "$attrsId"),
    "$attrs.value": /* @__PURE__ */ __name(function $attrsValue(newVal) {
      if (newVal && newVal.length < this.numVisible) {
        this.numVisible = newVal.length;
      }
    }, "$attrsValue"),
    "$attrs.activeIndex": /* @__PURE__ */ __name(function $attrsActiveIndex(newVal) {
      this.activeIndex = newVal;
    }, "$attrsActiveIndex"),
    "$attrs.numVisible": /* @__PURE__ */ __name(function $attrsNumVisible(newVal) {
      this.numVisible = newVal;
    }, "$attrsNumVisible"),
    "$attrs.autoPlay": /* @__PURE__ */ __name(function $attrsAutoPlay(newVal) {
      newVal ? this.startSlideShow() : this.stopSlideShow();
    }, "$attrsAutoPlay")
  },
  mounted: /* @__PURE__ */ __name(function mounted7() {
    this.id = this.id || UniqueComponentId();
  }, "mounted"),
  updated: /* @__PURE__ */ __name(function updated3() {
    this.$emit("activeitem-change", this.activeIndex);
  }, "updated"),
  beforeUnmount: /* @__PURE__ */ __name(function beforeUnmount6() {
    if (this.slideShowActive) {
      this.stopSlideShow();
    }
  }, "beforeUnmount"),
  methods: {
    getPTOptions: /* @__PURE__ */ __name(function getPTOptions7(key) {
      return this.ptm(key, {
        props: _objectSpread$1$2(_objectSpread$1$2({}, this.$attrs), {}, {
          pt: this.pt,
          unstyled: this.unstyled
        })
      });
    }, "getPTOptions"),
    isAutoPlayActive: /* @__PURE__ */ __name(function isAutoPlayActive() {
      return this.slideShowActive;
    }, "isAutoPlayActive"),
    startSlideShow: /* @__PURE__ */ __name(function startSlideShow() {
      var _this = this;
      this.interval = setInterval(function() {
        var activeIndex2 = _this.$attrs.circular && _this.$attrs.value.length - 1 === _this.activeIndex ? 0 : _this.activeIndex + 1;
        _this.activeIndex = activeIndex2;
      }, this.$attrs.transitionInterval);
      this.slideShowActive = true;
    }, "startSlideShow"),
    stopSlideShow: /* @__PURE__ */ __name(function stopSlideShow3() {
      if (this.interval) {
        clearInterval(this.interval);
      }
      this.slideShowActive = false;
    }, "stopSlideShow"),
    getPositionClass: /* @__PURE__ */ __name(function getPositionClass(preClassName, position2) {
      var positions = ["top", "left", "bottom", "right"];
      var pos = positions.find(function(item4) {
        return item4 === position2;
      });
      return pos ? "".concat(preClassName, "-").concat(pos) : "";
    }, "getPositionClass"),
    isVertical: /* @__PURE__ */ __name(function isVertical() {
      return this.$attrs.thumbnailsPosition === "left" || this.$attrs.thumbnailsPosition === "right";
    }, "isVertical")
  },
  computed: {
    closeAriaLabel: /* @__PURE__ */ __name(function closeAriaLabel() {
      return this.$primevue.config.locale.aria ? this.$primevue.config.locale.aria.close : void 0;
    }, "closeAriaLabel")
  },
  components: {
    GalleriaItem: script$3$1,
    GalleriaThumbnails: script$2$4,
    TimesIcon: script$C
  },
  directives: {
    ripple: Ripple
  }
};
function _typeof$5(o) {
  "@babel/helpers - typeof";
  return _typeof$5 = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function(o2) {
    return typeof o2;
  } : function(o2) {
    return o2 && "function" == typeof Symbol && o2.constructor === Symbol && o2 !== Symbol.prototype ? "symbol" : typeof o2;
  }, _typeof$5(o);
}
__name(_typeof$5, "_typeof$5");
function ownKeys$4(e, r) {
  var t = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var o = Object.getOwnPropertySymbols(e);
    r && (o = o.filter(function(r2) {
      return Object.getOwnPropertyDescriptor(e, r2).enumerable;
    })), t.push.apply(t, o);
  }
  return t;
}
__name(ownKeys$4, "ownKeys$4");
function _objectSpread$4(e) {
  for (var r = 1; r < arguments.length; r++) {
    var t = null != arguments[r] ? arguments[r] : {};
    r % 2 ? ownKeys$4(Object(t), true).forEach(function(r2) {
      _defineProperty$5(e, r2, t[r2]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(t)) : ownKeys$4(Object(t)).forEach(function(r2) {
      Object.defineProperty(e, r2, Object.getOwnPropertyDescriptor(t, r2));
    });
  }
  return e;
}
__name(_objectSpread$4, "_objectSpread$4");
function _defineProperty$5(e, r, t) {
  return (r = _toPropertyKey$5(r)) in e ? Object.defineProperty(e, r, { value: t, enumerable: true, configurable: true, writable: true }) : e[r] = t, e;
}
__name(_defineProperty$5, "_defineProperty$5");
function _toPropertyKey$5(t) {
  var i = _toPrimitive$5(t, "string");
  return "symbol" == _typeof$5(i) ? i : i + "";
}
__name(_toPropertyKey$5, "_toPropertyKey$5");
function _toPrimitive$5(t, r) {
  if ("object" != _typeof$5(t) || !t) return t;
  var e = t[Symbol.toPrimitive];
  if (void 0 !== e) {
    var i = e.call(t, r || "default");
    if ("object" != _typeof$5(i)) return i;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return ("string" === r ? String : Number)(t);
}
__name(_toPrimitive$5, "_toPrimitive$5");
var _hoisted_1$1$4 = ["id", "aria-label", "aria-roledescription"];
var _hoisted_2$g = ["aria-label"];
var _hoisted_3$b = ["aria-live"];
function render$1$4(_ctx, _cache, $props, $setup, $data, $options) {
  var _component_GalleriaItem = resolveComponent("GalleriaItem");
  var _component_GalleriaThumbnails = resolveComponent("GalleriaThumbnails");
  var _directive_ripple = resolveDirective("ripple");
  return _ctx.$attrs.value && _ctx.$attrs.value.length > 0 ? (openBlock(), createElementBlock("div", mergeProps({
    key: 0,
    id: $data.id,
    role: "region",
    "class": [_ctx.cx("root"), _ctx.$attrs.containerClass],
    style: _ctx.$attrs.containerStyle,
    "aria-label": _ctx.$attrs.ariaLabel,
    "aria-roledescription": _ctx.$attrs.ariaRoledescription
  }, _objectSpread$4(_objectSpread$4({}, _ctx.$attrs.containerProps), $options.getPTOptions("root"))), [_ctx.$attrs.fullScreen ? withDirectives((openBlock(), createElementBlock("button", mergeProps({
    key: 0,
    autofocus: "",
    type: "button",
    "class": _ctx.cx("closeButton"),
    "aria-label": $options.closeAriaLabel,
    onClick: _cache[0] || (_cache[0] = function($event) {
      return _ctx.$emit("mask-hide");
    })
  }, $options.getPTOptions("closeButton")), [(openBlock(), createBlock(resolveDynamicComponent(_ctx.$attrs.templates["closeicon"] || "TimesIcon"), mergeProps({
    "class": _ctx.cx("closeIcon")
  }, $options.getPTOptions("closeIcon")), null, 16, ["class"]))], 16, _hoisted_2$g)), [[_directive_ripple]]) : createCommentVNode("", true), _ctx.$attrs.templates && _ctx.$attrs.templates["header"] ? (openBlock(), createElementBlock("div", mergeProps({
    key: 1,
    "class": _ctx.cx("header")
  }, $options.getPTOptions("header")), [(openBlock(), createBlock(resolveDynamicComponent(_ctx.$attrs.templates["header"])))], 16)) : createCommentVNode("", true), createBaseVNode("div", mergeProps({
    "class": _ctx.cx("content"),
    "aria-live": _ctx.$attrs.autoPlay ? "polite" : "off"
  }, $options.getPTOptions("content")), [createVNode(_component_GalleriaItem, {
    id: $data.id,
    activeIndex: $data.activeIndex,
    "onUpdate:activeIndex": _cache[1] || (_cache[1] = function($event) {
      return $data.activeIndex = $event;
    }),
    slideShowActive: $data.slideShowActive,
    "onUpdate:slideShowActive": _cache[2] || (_cache[2] = function($event) {
      return $data.slideShowActive = $event;
    }),
    value: _ctx.$attrs.value,
    circular: _ctx.$attrs.circular,
    templates: _ctx.$attrs.templates,
    showIndicators: _ctx.$attrs.showIndicators,
    changeItemOnIndicatorHover: _ctx.$attrs.changeItemOnIndicatorHover,
    showItemNavigators: _ctx.$attrs.showItemNavigators,
    autoPlay: _ctx.$attrs.autoPlay,
    onStartSlideshow: $options.startSlideShow,
    onStopSlideshow: $options.stopSlideShow,
    pt: _ctx.pt,
    unstyled: _ctx.unstyled
  }, null, 8, ["id", "activeIndex", "slideShowActive", "value", "circular", "templates", "showIndicators", "changeItemOnIndicatorHover", "showItemNavigators", "autoPlay", "onStartSlideshow", "onStopSlideshow", "pt", "unstyled"]), _ctx.$attrs.showThumbnails ? (openBlock(), createBlock(_component_GalleriaThumbnails, {
    key: 0,
    activeIndex: $data.activeIndex,
    "onUpdate:activeIndex": _cache[3] || (_cache[3] = function($event) {
      return $data.activeIndex = $event;
    }),
    slideShowActive: $data.slideShowActive,
    "onUpdate:slideShowActive": _cache[4] || (_cache[4] = function($event) {
      return $data.slideShowActive = $event;
    }),
    containerId: $data.id,
    value: _ctx.$attrs.value,
    templates: _ctx.$attrs.templates,
    numVisible: $data.numVisible,
    responsiveOptions: _ctx.$attrs.responsiveOptions,
    circular: _ctx.$attrs.circular,
    isVertical: $options.isVertical(),
    contentHeight: _ctx.$attrs.verticalThumbnailViewPortHeight,
    showThumbnailNavigators: _ctx.$attrs.showThumbnailNavigators,
    prevButtonProps: _ctx.$attrs.prevButtonProps,
    nextButtonProps: _ctx.$attrs.nextButtonProps,
    onStopSlideshow: $options.stopSlideShow,
    pt: _ctx.pt,
    unstyled: _ctx.unstyled
  }, null, 8, ["activeIndex", "slideShowActive", "containerId", "value", "templates", "numVisible", "responsiveOptions", "circular", "isVertical", "contentHeight", "showThumbnailNavigators", "prevButtonProps", "nextButtonProps", "onStopSlideshow", "pt", "unstyled"])) : createCommentVNode("", true)], 16, _hoisted_3$b), _ctx.$attrs.templates && _ctx.$attrs.templates["footer"] ? (openBlock(), createElementBlock("div", mergeProps({
    key: 2,
    "class": _ctx.cx("footer")
  }, $options.getPTOptions("footer")), [(openBlock(), createBlock(resolveDynamicComponent(_ctx.$attrs.templates["footer"])))], 16)) : createCommentVNode("", true)], 16, _hoisted_1$1$4)) : createCommentVNode("", true);
}
__name(render$1$4, "render$1$4");
script$1$9.render = render$1$4;
var script$b = {
  name: "Galleria",
  "extends": script$4$1,
  inheritAttrs: false,
  emits: ["update:activeIndex", "update:visible"],
  container: null,
  mask: null,
  data: /* @__PURE__ */ __name(function data8() {
    return {
      containerVisible: this.visible
    };
  }, "data"),
  updated: /* @__PURE__ */ __name(function updated4() {
    if (this.fullScreen && this.visible) {
      this.containerVisible = this.visible;
    }
  }, "updated"),
  beforeUnmount: /* @__PURE__ */ __name(function beforeUnmount7() {
    if (this.fullScreen) {
      unblockBodyScroll();
    }
    this.mask = null;
    if (this.container) {
      ZIndex.clear(this.container);
      this.container = null;
    }
  }, "beforeUnmount"),
  methods: {
    onBeforeEnter: /* @__PURE__ */ __name(function onBeforeEnter(el) {
      ZIndex.set("modal", el, this.baseZIndex || this.$primevue.config.zIndex.modal);
    }, "onBeforeEnter"),
    onEnter: /* @__PURE__ */ __name(function onEnter4(el) {
      this.mask.style.zIndex = String(parseInt(el.style.zIndex, 10) - 1);
      blockBodyScroll();
      this.focus();
    }, "onEnter"),
    onBeforeLeave: /* @__PURE__ */ __name(function onBeforeLeave() {
      !this.isUnstyled && addClass(this.mask, "p-overlay-mask-leave");
    }, "onBeforeLeave"),
    onAfterLeave: /* @__PURE__ */ __name(function onAfterLeave3(el) {
      ZIndex.clear(el);
      this.containerVisible = false;
      unblockBodyScroll();
    }, "onAfterLeave"),
    onActiveItemChange: /* @__PURE__ */ __name(function onActiveItemChange(index2) {
      if (this.activeIndex !== index2) {
        this.$emit("update:activeIndex", index2);
      }
    }, "onActiveItemChange"),
    maskHide: /* @__PURE__ */ __name(function maskHide() {
      this.$emit("update:visible", false);
    }, "maskHide"),
    containerRef: /* @__PURE__ */ __name(function containerRef3(el) {
      this.container = el;
    }, "containerRef"),
    maskRef: /* @__PURE__ */ __name(function maskRef(el) {
      this.mask = el;
    }, "maskRef"),
    focus: /* @__PURE__ */ __name(function focus3() {
      var focusTarget = this.container.$el.querySelector("[autofocus]");
      if (focusTarget) {
        focusTarget.focus();
      }
    }, "focus")
  },
  components: {
    GalleriaContent: script$1$9,
    Portal: script$r
  },
  directives: {
    focustrap: FocusTrap
  }
};
var _hoisted_1$n = ["aria-modal"];
function render$a(_ctx, _cache, $props, $setup, $data, $options) {
  var _component_GalleriaContent = resolveComponent("GalleriaContent");
  var _component_Portal = resolveComponent("Portal");
  var _directive_focustrap = resolveDirective("focustrap");
  return _ctx.fullScreen ? (openBlock(), createBlock(_component_Portal, {
    key: 0
  }, {
    "default": withCtx(function() {
      return [$data.containerVisible ? (openBlock(), createElementBlock("div", mergeProps({
        key: 0,
        ref: $options.maskRef,
        "class": [_ctx.cx("mask"), _ctx.maskClass],
        role: "dialog",
        "aria-modal": _ctx.fullScreen ? "true" : void 0
      }, _ctx.ptm("mask")), [createVNode(Transition, mergeProps({
        name: "p-galleria",
        onBeforeEnter: $options.onBeforeEnter,
        onEnter: $options.onEnter,
        onBeforeLeave: $options.onBeforeLeave,
        onAfterLeave: $options.onAfterLeave,
        appear: ""
      }, _ctx.ptm("transition")), {
        "default": withCtx(function() {
          return [_ctx.visible ? withDirectives((openBlock(), createBlock(_component_GalleriaContent, mergeProps({
            key: 0,
            ref: $options.containerRef,
            onMaskHide: $options.maskHide,
            templates: _ctx.$slots,
            onActiveitemChange: $options.onActiveItemChange,
            pt: _ctx.pt,
            unstyled: _ctx.unstyled
          }, _ctx.$props), null, 16, ["onMaskHide", "templates", "onActiveitemChange", "pt", "unstyled"])), [[_directive_focustrap]]) : createCommentVNode("", true)];
        }),
        _: 1
      }, 16, ["onBeforeEnter", "onEnter", "onBeforeLeave", "onAfterLeave"])], 16, _hoisted_1$n)) : createCommentVNode("", true)];
    }),
    _: 1
  })) : (openBlock(), createBlock(_component_GalleriaContent, mergeProps({
    key: 1,
    templates: _ctx.$slots,
    onActiveitemChange: $options.onActiveItemChange,
    pt: _ctx.pt,
    unstyled: _ctx.unstyled
  }, _ctx.$props), null, 16, ["templates", "onActiveitemChange", "pt", "unstyled"]));
}
__name(render$a, "render$a");
script$b.render = render$a;
const _sfc_main$p = /* @__PURE__ */ defineComponent({
  __name: "ResultGallery",
  props: {
    allGalleryItems: {},
    activeIndex: {}
  },
  emits: ["update:activeIndex"],
  setup(__props, { emit: __emit }) {
    const galleryVisible = ref(false);
    const emit = __emit;
    const props = __props;
    let maskMouseDownTarget = null;
    const onMaskMouseDown = /* @__PURE__ */ __name((event) => {
      maskMouseDownTarget = event.target;
    }, "onMaskMouseDown");
    const onMaskMouseUp = /* @__PURE__ */ __name((event) => {
      const maskEl = document.querySelector("[data-mask]");
      if (galleryVisible.value && maskMouseDownTarget === event.target && maskMouseDownTarget === maskEl) {
        galleryVisible.value = false;
        handleVisibilityChange(false);
      }
    }, "onMaskMouseUp");
    watch(
      () => props.activeIndex,
      (index2) => {
        if (index2 !== -1) {
          galleryVisible.value = true;
        }
      }
    );
    const handleVisibilityChange = /* @__PURE__ */ __name((visible) => {
      if (!visible) {
        emit("update:activeIndex", -1);
      }
    }, "handleVisibilityChange");
    const handleActiveIndexChange = /* @__PURE__ */ __name((index2) => {
      emit("update:activeIndex", index2);
    }, "handleActiveIndexChange");
    const handleKeyDown = /* @__PURE__ */ __name((event) => {
      if (!galleryVisible.value) return;
      switch (event.key) {
        case "ArrowLeft":
          navigateImage(-1);
          break;
        case "ArrowRight":
          navigateImage(1);
          break;
        case "Escape":
          galleryVisible.value = false;
          handleVisibilityChange(false);
          break;
      }
    }, "handleKeyDown");
    const navigateImage = /* @__PURE__ */ __name((direction) => {
      const newIndex = (props.activeIndex + direction + props.allGalleryItems.length) % props.allGalleryItems.length;
      emit("update:activeIndex", newIndex);
    }, "navigateImage");
    onMounted(() => {
      window.addEventListener("keydown", handleKeyDown);
    });
    onUnmounted(() => {
      window.removeEventListener("keydown", handleKeyDown);
    });
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(script$b), {
        visible: galleryVisible.value,
        "onUpdate:visible": [
          _cache[0] || (_cache[0] = ($event) => galleryVisible.value = $event),
          handleVisibilityChange
        ],
        activeIndex: _ctx.activeIndex,
        "onUpdate:activeIndex": handleActiveIndexChange,
        value: _ctx.allGalleryItems,
        showIndicators: false,
        changeItemOnIndicatorHover: "",
        showItemNavigators: "",
        fullScreen: "",
        circular: "",
        showThumbnails: false,
        pt: {
          mask: {
            onMousedown: onMaskMouseDown,
            onMouseup: onMaskMouseUp,
            "data-mask": true
          }
        }
      }, {
        item: withCtx(({ item: item4 }) => [
          (openBlock(), createBlock(ComfyImage, {
            key: item4.url,
            src: item4.url,
            contain: false,
            class: "galleria-image"
          }, null, 8, ["src"]))
        ]),
        _: 1
      }, 8, ["visible", "activeIndex", "value", "pt"]);
    };
  }
});
var theme$8 = /* @__PURE__ */ __name(function theme10(_ref) {
  var dt = _ref.dt;
  return "\n.p-toolbar {\n    display: flex;\n    align-items: center;\n    justify-content: space-between;\n    flex-wrap: wrap;\n    padding: ".concat(dt("toolbar.padding"), ";\n    background: ").concat(dt("toolbar.background"), ";\n    border: 1px solid ").concat(dt("toolbar.border.color"), ";\n    color: ").concat(dt("toolbar.color"), ";\n    border-radius: ").concat(dt("toolbar.border.radius"), ";\n    gap: ").concat(dt("toolbar.gap"), ";\n}\n\n.p-toolbar-start,\n.p-toolbar-center,\n.p-toolbar-end {\n    display: flex;\n    align-items: center;\n}\n");
}, "theme");
var classes$8 = {
  root: "p-toolbar p-component",
  start: "p-toolbar-start",
  center: "p-toolbar-center",
  end: "p-toolbar-end"
};
var ToolbarStyle = BaseStyle.extend({
  name: "toolbar",
  theme: theme$8,
  classes: classes$8
});
var script$1$8 = {
  name: "BaseToolbar",
  "extends": script$p,
  props: {
    ariaLabelledby: {
      type: String,
      "default": null
    }
  },
  style: ToolbarStyle,
  provide: /* @__PURE__ */ __name(function provide12() {
    return {
      $pcToolbar: this,
      $parentInstance: this
    };
  }, "provide")
};
var script$a = {
  name: "Toolbar",
  "extends": script$1$8,
  inheritAttrs: false
};
var _hoisted_1$m = ["aria-labelledby"];
function render$9(_ctx, _cache, $props, $setup, $data, $options) {
  return openBlock(), createElementBlock("div", mergeProps({
    "class": _ctx.cx("root"),
    role: "toolbar",
    "aria-labelledby": _ctx.ariaLabelledby
  }, _ctx.ptmi("root")), [createBaseVNode("div", mergeProps({
    "class": _ctx.cx("start")
  }, _ctx.ptm("start")), [renderSlot(_ctx.$slots, "start")], 16), createBaseVNode("div", mergeProps({
    "class": _ctx.cx("center")
  }, _ctx.ptm("center")), [renderSlot(_ctx.$slots, "center")], 16), createBaseVNode("div", mergeProps({
    "class": _ctx.cx("end")
  }, _ctx.ptm("end")), [renderSlot(_ctx.$slots, "end")], 16)], 16, _hoisted_1$m);
}
__name(render$9, "render$9");
script$a.render = render$9;
const _withScopeId$c = /* @__PURE__ */ __name((n) => (pushScopeId("data-v-1b0a8fe3"), n = n(), popScopeId(), n), "_withScopeId$c");
const _hoisted_1$l = { class: "comfy-vue-side-bar-container" };
const _hoisted_2$f = { class: "comfy-vue-side-bar-header-span" };
const _hoisted_3$a = { class: "comfy-vue-side-bar-body" };
const _sfc_main$o = /* @__PURE__ */ defineComponent({
  __name: "SidebarTabTemplate",
  props: {
    title: {
      type: String,
      required: true
    }
  },
  setup(__props) {
    const props = __props;
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", _hoisted_1$l, [
        createVNode(unref(script$a), { class: "comfy-vue-side-bar-header" }, {
          start: withCtx(() => [
            createBaseVNode("span", _hoisted_2$f, toDisplayString(props.title.toUpperCase()), 1)
          ]),
          end: withCtx(() => [
            renderSlot(_ctx.$slots, "tool-buttons", {}, void 0, true)
          ]),
          _: 3
        }),
        createBaseVNode("div", _hoisted_3$a, [
          renderSlot(_ctx.$slots, "body", {}, void 0, true)
        ])
      ]);
    };
  }
});
const SidebarTabTemplate = /* @__PURE__ */ _export_sfc(_sfc_main$o, [["__scopeId", "data-v-1b0a8fe3"]]);
const _withScopeId$b = /* @__PURE__ */ __name((n) => (pushScopeId("data-v-08fa89b1"), n = n(), popScopeId(), n), "_withScopeId$b");
const _hoisted_1$k = { class: "queue-grid" };
const _hoisted_2$e = { key: 1 };
const _hoisted_3$9 = { key: 2 };
const IMAGE_FIT = "Comfy.Queue.ImageFit";
const ITEMS_PER_PAGE = 8;
const SCROLL_THRESHOLD = 100;
const _sfc_main$n = /* @__PURE__ */ defineComponent({
  __name: "QueueSidebarTab",
  setup(__props) {
    const confirm = useConfirm();
    const toast = useToast();
    const queueStore = useQueueStore();
    const settingStore = useSettingStore();
    const commandStore = useCommandStore();
    const { t } = useI18n();
    const isExpanded = ref(false);
    const visibleTasks = ref([]);
    const scrollContainer = ref(null);
    const loadMoreTrigger = ref(null);
    const galleryActiveIndex = ref(-1);
    const folderTask = ref(null);
    const isInFolderView = computed(() => folderTask.value !== null);
    const imageFit = computed(() => settingStore.get(IMAGE_FIT));
    const allTasks = computed(
      () => isInFolderView.value ? folderTask.value ? folderTask.value.flatten() : [] : isExpanded.value ? queueStore.flatTasks : queueStore.tasks
    );
    const allGalleryItems = computed(
      () => allTasks.value.flatMap((task) => {
        const previewOutput = task.previewOutput;
        return previewOutput ? [previewOutput] : [];
      })
    );
    const loadMoreItems = /* @__PURE__ */ __name(() => {
      const currentLength = visibleTasks.value.length;
      const newTasks = allTasks.value.slice(
        currentLength,
        currentLength + ITEMS_PER_PAGE
      );
      visibleTasks.value.push(...newTasks);
    }, "loadMoreItems");
    const checkAndLoadMore = /* @__PURE__ */ __name(() => {
      if (!scrollContainer.value) return;
      const { scrollHeight, scrollTop, clientHeight } = scrollContainer.value;
      if (scrollHeight - scrollTop - clientHeight < SCROLL_THRESHOLD) {
        loadMoreItems();
      }
    }, "checkAndLoadMore");
    useInfiniteScroll(
      scrollContainer,
      () => {
        if (visibleTasks.value.length < allTasks.value.length) {
          loadMoreItems();
        }
      },
      { distance: SCROLL_THRESHOLD }
    );
    useResizeObserver(scrollContainer, () => {
      nextTick(() => {
        checkAndLoadMore();
      });
    });
    const updateVisibleTasks = /* @__PURE__ */ __name(() => {
      visibleTasks.value = allTasks.value.slice(0, ITEMS_PER_PAGE);
    }, "updateVisibleTasks");
    const toggleExpanded = /* @__PURE__ */ __name(() => {
      isExpanded.value = !isExpanded.value;
      updateVisibleTasks();
    }, "toggleExpanded");
    const removeTask = /* @__PURE__ */ __name((task) => {
      if (task.isRunning) {
        api.interrupt();
      }
      queueStore.delete(task);
    }, "removeTask");
    const removeAllTasks = /* @__PURE__ */ __name(async () => {
      await queueStore.clear();
    }, "removeAllTasks");
    const confirmRemoveAll = /* @__PURE__ */ __name((event) => {
      confirm.require({
        target: event.currentTarget,
        message: "Do you want to delete all tasks?",
        icon: "pi pi-info-circle",
        rejectProps: {
          label: "Cancel",
          severity: "secondary",
          outlined: true
        },
        acceptProps: {
          label: "Delete",
          severity: "danger"
        },
        accept: /* @__PURE__ */ __name(async () => {
          await removeAllTasks();
          toast.add({
            severity: "info",
            summary: "Confirmed",
            detail: "Tasks deleted",
            life: 3e3
          });
        }, "accept")
      });
    }, "confirmRemoveAll");
    const onStatus = /* @__PURE__ */ __name(async () => {
      await queueStore.update();
      updateVisibleTasks();
    }, "onStatus");
    const menu = ref(null);
    const menuTargetTask = ref(null);
    const menuTargetNode = ref(null);
    const menuItems = computed(() => [
      {
        label: t("delete"),
        icon: "pi pi-trash",
        command: /* @__PURE__ */ __name(() => menuTargetTask.value && removeTask(menuTargetTask.value), "command"),
        disabled: isExpanded.value || isInFolderView.value
      },
      {
        label: t("loadWorkflow"),
        icon: "pi pi-file-export",
        command: /* @__PURE__ */ __name(() => menuTargetTask.value?.loadWorkflow(), "command")
      },
      {
        label: t("goToNode"),
        icon: "pi pi-arrow-circle-right",
        command: /* @__PURE__ */ __name(() => app.goToNode(menuTargetNode.value?.id), "command"),
        visible: !!menuTargetNode.value
      }
    ]);
    const handleContextMenu = /* @__PURE__ */ __name(({
      task,
      event,
      node: node2
    }) => {
      menuTargetTask.value = task;
      menuTargetNode.value = node2;
      menu.value?.show(event);
    }, "handleContextMenu");
    const handlePreview = /* @__PURE__ */ __name((task) => {
      galleryActiveIndex.value = allGalleryItems.value.findIndex(
        (item4) => item4.url === task.previewOutput?.url
      );
    }, "handlePreview");
    const enterFolderView = /* @__PURE__ */ __name((task) => {
      folderTask.value = task;
      updateVisibleTasks();
    }, "enterFolderView");
    const exitFolderView = /* @__PURE__ */ __name(() => {
      folderTask.value = null;
      updateVisibleTasks();
    }, "exitFolderView");
    const toggleImageFit = /* @__PURE__ */ __name(() => {
      settingStore.set(IMAGE_FIT, imageFit.value === "cover" ? "contain" : "cover");
    }, "toggleImageFit");
    onMounted(() => {
      api.addEventListener("status", onStatus);
      queueStore.update();
    });
    onUnmounted(() => {
      api.removeEventListener("status", onStatus);
    });
    watch(
      allTasks,
      (newTasks) => {
        if (visibleTasks.value.length === 0 || visibleTasks.value.length > newTasks.length) {
          updateVisibleTasks();
        }
        nextTick(() => {
          checkAndLoadMore();
        });
      },
      { immediate: true }
    );
    return (_ctx, _cache) => {
      const _directive_tooltip = resolveDirective("tooltip");
      return openBlock(), createElementBlock(Fragment, null, [
        createVNode(SidebarTabTemplate, {
          title: _ctx.$t("sideToolbar.queue")
        }, {
          "tool-buttons": withCtx(() => [
            withDirectives(createVNode(unref(script$o), {
              icon: imageFit.value === "cover" ? "pi pi-arrow-down-left-and-arrow-up-right-to-center" : "pi pi-arrow-up-right-and-arrow-down-left-from-center",
              text: "",
              severity: "secondary",
              onClick: toggleImageFit,
              class: "toggle-expanded-button"
            }, null, 8, ["icon"]), [
              [_directive_tooltip, _ctx.$t(`sideToolbar.queueTab.${imageFit.value}ImagePreview`)]
            ]),
            isInFolderView.value ? withDirectives((openBlock(), createBlock(unref(script$o), {
              key: 0,
              icon: "pi pi-arrow-left",
              text: "",
              severity: "secondary",
              onClick: exitFolderView,
              class: "back-button"
            }, null, 512)), [
              [_directive_tooltip, _ctx.$t("sideToolbar.queueTab.backToAllTasks")]
            ]) : (openBlock(), createElementBlock(Fragment, { key: 1 }, [
              withDirectives(createVNode(unref(script$o), {
                icon: isExpanded.value ? "pi pi-images" : "pi pi-image",
                text: "",
                severity: "secondary",
                onClick: toggleExpanded,
                class: "toggle-expanded-button"
              }, null, 8, ["icon"]), [
                [_directive_tooltip, _ctx.$t("sideToolbar.queueTab.showFlatList")]
              ]),
              unref(queueStore).hasPendingTasks ? withDirectives((openBlock(), createBlock(unref(script$o), {
                key: 0,
                icon: "pi pi-stop",
                severity: "danger",
                text: "",
                onClick: _cache[0] || (_cache[0] = () => unref(commandStore).execute("Comfy.ClearPendingTasks"))
              }, null, 512)), [
                [
                  _directive_tooltip,
                  _ctx.$t("sideToolbar.queueTab.clearPendingTasks"),
                  void 0,
                  { bottom: true }
                ]
              ]) : createCommentVNode("", true),
              createVNode(unref(script$o), {
                icon: "pi pi-trash",
                text: "",
                severity: "primary",
                onClick: _cache[1] || (_cache[1] = ($event) => confirmRemoveAll($event)),
                class: "clear-all-button"
              })
            ], 64))
          ]),
          body: withCtx(() => [
            visibleTasks.value.length > 0 ? (openBlock(), createElementBlock("div", {
              key: 0,
              ref_key: "scrollContainer",
              ref: scrollContainer,
              class: "scroll-container"
            }, [
              createBaseVNode("div", _hoisted_1$k, [
                (openBlock(true), createElementBlock(Fragment, null, renderList(visibleTasks.value, (task) => {
                  return openBlock(), createBlock(TaskItem, {
                    key: task.key,
                    task,
                    isFlatTask: isExpanded.value || isInFolderView.value,
                    onContextmenu: handleContextMenu,
                    onPreview: handlePreview,
                    onTaskOutputLengthClicked: _cache[2] || (_cache[2] = ($event) => enterFolderView($event))
                  }, null, 8, ["task", "isFlatTask"]);
                }), 128))
              ]),
              createBaseVNode("div", {
                ref_key: "loadMoreTrigger",
                ref: loadMoreTrigger,
                style: { "height": "1px" }
              }, null, 512)
            ], 512)) : unref(queueStore).isLoading ? (openBlock(), createElementBlock("div", _hoisted_2$e, [
              createVNode(unref(script$D), { style: { "width": "50px", "left": "50%", "transform": "translateX(-50%)" } })
            ])) : (openBlock(), createElementBlock("div", _hoisted_3$9, [
              createVNode(NoResultsPlaceholder, {
                icon: "pi pi-info-circle",
                title: _ctx.$t("noTasksFound"),
                message: _ctx.$t("noTasksFoundMessage")
              }, null, 8, ["title", "message"])
            ]))
          ]),
          _: 1
        }, 8, ["title"]),
        createVNode(unref(script$e)),
        createVNode(unref(script$d), {
          ref_key: "menu",
          ref: menu,
          model: menuItems.value
        }, null, 8, ["model"]),
        createVNode(_sfc_main$p, {
          activeIndex: galleryActiveIndex.value,
          "onUpdate:activeIndex": _cache[3] || (_cache[3] = ($event) => galleryActiveIndex.value = $event),
          allGalleryItems: allGalleryItems.value
        }, null, 8, ["activeIndex", "allGalleryItems"])
      ], 64);
    };
  }
});
const QueueSidebarTab = /* @__PURE__ */ _export_sfc(_sfc_main$n, [["__scopeId", "data-v-08fa89b1"]]);
var theme$7 = /* @__PURE__ */ __name(function theme11(_ref) {
  var dt = _ref.dt;
  return "\n.p-popover {\n    margin-top: ".concat(dt("popover.gutter"), ";\n    background: ").concat(dt("popover.background"), ";\n    color: ").concat(dt("popover.color"), ";\n    border: 1px solid ").concat(dt("popover.border.color"), ";\n    border-radius: ").concat(dt("popover.border.radius"), ";\n    box-shadow: ").concat(dt("popover.shadow"), ";\n}\n\n.p-popover-content {\n    padding: ").concat(dt("popover.content.padding"), ";\n}\n\n.p-popover-flipped {\n    margin-top: calc(").concat(dt("popover.gutter"), " * -1);\n    margin-bottom: ").concat(dt("popover.gutter"), ";\n}\n\n.p-popover-enter-from {\n    opacity: 0;\n    transform: scaleY(0.8);\n}\n\n.p-popover-leave-to {\n    opacity: 0;\n}\n\n.p-popover-enter-active {\n    transition: transform 0.12s cubic-bezier(0, 0, 0.2, 1), opacity 0.12s cubic-bezier(0, 0, 0.2, 1);\n}\n\n.p-popover-leave-active {\n    transition: opacity 0.1s linear;\n}\n\n.p-popover:after,\n.p-popover:before {\n    bottom: 100%;\n    left: calc(").concat(dt("popover.arrow.offset"), " + ").concat(dt("popover.arrow.left"), ');\n    content: " ";\n    height: 0;\n    width: 0;\n    position: absolute;\n    pointer-events: none;\n}\n\n.p-popover:after {\n    border-width: calc(').concat(dt("popover.gutter"), " - 2px);\n    margin-left: calc(-1 * (").concat(dt("popover.gutter"), " - 2px));\n    border-style: solid;\n    border-color: transparent;\n    border-bottom-color: ").concat(dt("popover.background"), ";\n}\n\n.p-popover:before {\n    border-width: ").concat(dt("popover.gutter"), ";\n    margin-left: calc(-1 * ").concat(dt("popover.gutter"), ");\n    border-style: solid;\n    border-color: transparent;\n    border-bottom-color: ").concat(dt("popover.border.color"), ";\n}\n\n.p-popover-flipped:after,\n.p-popover-flipped:before {\n    bottom: auto;\n    top: 100%;\n}\n\n.p-popover.p-popover-flipped:after {\n    border-bottom-color: transparent;\n    border-top-color: ").concat(dt("popover.background"), ";\n}\n\n.p-popover.p-popover-flipped:before {\n    border-bottom-color: transparent;\n    border-top-color: ").concat(dt("popover.border.color"), ";\n}\n");
}, "theme");
var classes$7 = {
  root: "p-popover p-component",
  content: "p-popover-content"
};
var PopoverStyle = BaseStyle.extend({
  name: "popover",
  theme: theme$7,
  classes: classes$7
});
var script$1$7 = {
  name: "BasePopover",
  "extends": script$p,
  props: {
    dismissable: {
      type: Boolean,
      "default": true
    },
    appendTo: {
      type: [String, Object],
      "default": "body"
    },
    baseZIndex: {
      type: Number,
      "default": 0
    },
    autoZIndex: {
      type: Boolean,
      "default": true
    },
    breakpoints: {
      type: Object,
      "default": null
    },
    closeOnEscape: {
      type: Boolean,
      "default": true
    }
  },
  style: PopoverStyle,
  provide: /* @__PURE__ */ __name(function provide13() {
    return {
      $pcPopover: this,
      $parentInstance: this
    };
  }, "provide")
};
var script$9 = {
  name: "Popover",
  "extends": script$1$7,
  inheritAttrs: false,
  emits: ["show", "hide"],
  data: /* @__PURE__ */ __name(function data9() {
    return {
      visible: false
    };
  }, "data"),
  watch: {
    dismissable: {
      immediate: true,
      handler: /* @__PURE__ */ __name(function handler(newValue) {
        if (newValue) {
          this.bindOutsideClickListener();
        } else {
          this.unbindOutsideClickListener();
        }
      }, "handler")
    }
  },
  selfClick: false,
  target: null,
  eventTarget: null,
  outsideClickListener: null,
  scrollHandler: null,
  resizeListener: null,
  container: null,
  styleElement: null,
  overlayEventListener: null,
  documentKeydownListener: null,
  beforeUnmount: /* @__PURE__ */ __name(function beforeUnmount8() {
    if (this.dismissable) {
      this.unbindOutsideClickListener();
    }
    if (this.scrollHandler) {
      this.scrollHandler.destroy();
      this.scrollHandler = null;
    }
    this.destroyStyle();
    this.unbindResizeListener();
    this.target = null;
    if (this.container && this.autoZIndex) {
      ZIndex.clear(this.container);
    }
    if (this.overlayEventListener) {
      OverlayEventBus.off("overlay-click", this.overlayEventListener);
      this.overlayEventListener = null;
    }
    this.container = null;
  }, "beforeUnmount"),
  mounted: /* @__PURE__ */ __name(function mounted8() {
    if (this.breakpoints) {
      this.createStyle();
    }
  }, "mounted"),
  methods: {
    toggle: /* @__PURE__ */ __name(function toggle2(event, target) {
      if (this.visible) this.hide();
      else this.show(event, target);
    }, "toggle"),
    show: /* @__PURE__ */ __name(function show3(event, target) {
      this.visible = true;
      this.eventTarget = event.currentTarget;
      this.target = target || event.currentTarget;
    }, "show"),
    hide: /* @__PURE__ */ __name(function hide3() {
      this.visible = false;
    }, "hide"),
    onContentClick: /* @__PURE__ */ __name(function onContentClick() {
      this.selfClick = true;
    }, "onContentClick"),
    onEnter: /* @__PURE__ */ __name(function onEnter5(el) {
      var _this = this;
      this.container.setAttribute(this.attributeSelector, "");
      addStyle(el, {
        position: "absolute",
        top: "0",
        left: "0"
      });
      this.alignOverlay();
      if (this.dismissable) {
        this.bindOutsideClickListener();
      }
      this.bindScrollListener();
      this.bindResizeListener();
      if (this.autoZIndex) {
        ZIndex.set("overlay", el, this.baseZIndex + this.$primevue.config.zIndex.overlay);
      }
      this.overlayEventListener = function(e) {
        if (_this.container.contains(e.target)) {
          _this.selfClick = true;
        }
      };
      this.focus();
      OverlayEventBus.on("overlay-click", this.overlayEventListener);
      this.$emit("show");
      if (this.closeOnEscape) {
        this.bindDocumentKeyDownListener();
      }
    }, "onEnter"),
    onLeave: /* @__PURE__ */ __name(function onLeave3() {
      this.unbindOutsideClickListener();
      this.unbindScrollListener();
      this.unbindResizeListener();
      this.unbindDocumentKeyDownListener();
      OverlayEventBus.off("overlay-click", this.overlayEventListener);
      this.overlayEventListener = null;
      this.$emit("hide");
    }, "onLeave"),
    onAfterLeave: /* @__PURE__ */ __name(function onAfterLeave4(el) {
      if (this.autoZIndex) {
        ZIndex.clear(el);
      }
    }, "onAfterLeave"),
    alignOverlay: /* @__PURE__ */ __name(function alignOverlay3() {
      absolutePosition(this.container, this.target, false);
      var containerOffset = getOffset(this.container);
      var targetOffset = getOffset(this.target);
      var arrowLeft = 0;
      if (containerOffset.left < targetOffset.left) {
        arrowLeft = targetOffset.left - containerOffset.left;
      }
      this.container.style.setProperty($dt("popover.arrow.left").name, "".concat(arrowLeft, "px"));
      if (containerOffset.top < targetOffset.top) {
        this.container.setAttribute("data-p-popover-flipped", "true");
        !this.isUnstyled && addClass(this.container, "p-popover-flipped");
      }
    }, "alignOverlay"),
    onContentKeydown: /* @__PURE__ */ __name(function onContentKeydown(event) {
      if (event.code === "Escape" && this.closeOnEscape) {
        this.hide();
        focus(this.target);
      }
    }, "onContentKeydown"),
    onButtonKeydown: /* @__PURE__ */ __name(function onButtonKeydown(event) {
      switch (event.code) {
        case "ArrowDown":
        case "ArrowUp":
        case "ArrowLeft":
        case "ArrowRight":
          event.preventDefault();
      }
    }, "onButtonKeydown"),
    focus: /* @__PURE__ */ __name(function focus4() {
      var focusTarget = this.container.querySelector("[autofocus]");
      if (focusTarget) {
        focusTarget.focus();
      }
    }, "focus"),
    onKeyDown: /* @__PURE__ */ __name(function onKeyDown3(event) {
      if (event.code === "Escape" && this.closeOnEscape) {
        this.visible = false;
      }
    }, "onKeyDown"),
    bindDocumentKeyDownListener: /* @__PURE__ */ __name(function bindDocumentKeyDownListener() {
      if (!this.documentKeydownListener) {
        this.documentKeydownListener = this.onKeyDown.bind(this);
        window.document.addEventListener("keydown", this.documentKeydownListener);
      }
    }, "bindDocumentKeyDownListener"),
    unbindDocumentKeyDownListener: /* @__PURE__ */ __name(function unbindDocumentKeyDownListener() {
      if (this.documentKeydownListener) {
        window.document.removeEventListener("keydown", this.documentKeydownListener);
        this.documentKeydownListener = null;
      }
    }, "unbindDocumentKeyDownListener"),
    bindOutsideClickListener: /* @__PURE__ */ __name(function bindOutsideClickListener4() {
      var _this2 = this;
      if (!this.outsideClickListener && isClient()) {
        this.outsideClickListener = function(event) {
          if (_this2.visible && !_this2.selfClick && !_this2.isTargetClicked(event)) {
            _this2.visible = false;
          }
          _this2.selfClick = false;
        };
        document.addEventListener("click", this.outsideClickListener);
      }
    }, "bindOutsideClickListener"),
    unbindOutsideClickListener: /* @__PURE__ */ __name(function unbindOutsideClickListener4() {
      if (this.outsideClickListener) {
        document.removeEventListener("click", this.outsideClickListener);
        this.outsideClickListener = null;
        this.selfClick = false;
      }
    }, "unbindOutsideClickListener"),
    bindScrollListener: /* @__PURE__ */ __name(function bindScrollListener3() {
      var _this3 = this;
      if (!this.scrollHandler) {
        this.scrollHandler = new ConnectedOverlayScrollHandler(this.target, function() {
          if (_this3.visible) {
            _this3.visible = false;
          }
        });
      }
      this.scrollHandler.bindScrollListener();
    }, "bindScrollListener"),
    unbindScrollListener: /* @__PURE__ */ __name(function unbindScrollListener3() {
      if (this.scrollHandler) {
        this.scrollHandler.unbindScrollListener();
      }
    }, "unbindScrollListener"),
    bindResizeListener: /* @__PURE__ */ __name(function bindResizeListener4() {
      var _this4 = this;
      if (!this.resizeListener) {
        this.resizeListener = function() {
          if (_this4.visible && !isTouchDevice()) {
            _this4.visible = false;
          }
        };
        window.addEventListener("resize", this.resizeListener);
      }
    }, "bindResizeListener"),
    unbindResizeListener: /* @__PURE__ */ __name(function unbindResizeListener4() {
      if (this.resizeListener) {
        window.removeEventListener("resize", this.resizeListener);
        this.resizeListener = null;
      }
    }, "unbindResizeListener"),
    isTargetClicked: /* @__PURE__ */ __name(function isTargetClicked2(event) {
      return this.eventTarget && (this.eventTarget === event.target || this.eventTarget.contains(event.target));
    }, "isTargetClicked"),
    containerRef: /* @__PURE__ */ __name(function containerRef4(el) {
      this.container = el;
    }, "containerRef"),
    createStyle: /* @__PURE__ */ __name(function createStyle2() {
      if (!this.styleElement && !this.isUnstyled) {
        var _this$$primevue;
        this.styleElement = document.createElement("style");
        this.styleElement.type = "text/css";
        setAttribute(this.styleElement, "nonce", (_this$$primevue = this.$primevue) === null || _this$$primevue === void 0 || (_this$$primevue = _this$$primevue.config) === null || _this$$primevue === void 0 || (_this$$primevue = _this$$primevue.csp) === null || _this$$primevue === void 0 ? void 0 : _this$$primevue.nonce);
        document.head.appendChild(this.styleElement);
        var innerHTML = "";
        for (var breakpoint in this.breakpoints) {
          innerHTML += "\n                        @media screen and (max-width: ".concat(breakpoint, ") {\n                            .p-popover[").concat(this.attributeSelector, "] {\n                                width: ").concat(this.breakpoints[breakpoint], " !important;\n                            }\n                        }\n                    ");
        }
        this.styleElement.innerHTML = innerHTML;
      }
    }, "createStyle"),
    destroyStyle: /* @__PURE__ */ __name(function destroyStyle() {
      if (this.styleElement) {
        document.head.removeChild(this.styleElement);
        this.styleElement = null;
      }
    }, "destroyStyle"),
    onOverlayClick: /* @__PURE__ */ __name(function onOverlayClick3(event) {
      OverlayEventBus.emit("overlay-click", {
        originalEvent: event,
        target: this.target
      });
    }, "onOverlayClick")
  },
  computed: {
    attributeSelector: /* @__PURE__ */ __name(function attributeSelector() {
      return UniqueComponentId();
    }, "attributeSelector")
  },
  directives: {
    focustrap: FocusTrap,
    ripple: Ripple
  },
  components: {
    Portal: script$r
  }
};
var _hoisted_1$j = ["aria-modal"];
function render$8(_ctx, _cache, $props, $setup, $data, $options) {
  var _component_Portal = resolveComponent("Portal");
  var _directive_focustrap = resolveDirective("focustrap");
  return openBlock(), createBlock(_component_Portal, {
    appendTo: _ctx.appendTo
  }, {
    "default": withCtx(function() {
      return [createVNode(Transition, mergeProps({
        name: "p-popover",
        onEnter: $options.onEnter,
        onLeave: $options.onLeave,
        onAfterLeave: $options.onAfterLeave
      }, _ctx.ptm("transition")), {
        "default": withCtx(function() {
          return [$data.visible ? withDirectives((openBlock(), createElementBlock("div", mergeProps({
            key: 0,
            ref: $options.containerRef,
            role: "dialog",
            "aria-modal": $data.visible,
            onClick: _cache[3] || (_cache[3] = function() {
              return $options.onOverlayClick && $options.onOverlayClick.apply($options, arguments);
            }),
            "class": _ctx.cx("root")
          }, _ctx.ptmi("root")), [_ctx.$slots.container ? renderSlot(_ctx.$slots, "container", {
            key: 0,
            closeCallback: $options.hide,
            keydownCallback: /* @__PURE__ */ __name(function keydownCallback(event) {
              return $options.onButtonKeydown(event);
            }, "keydownCallback")
          }) : (openBlock(), createElementBlock("div", mergeProps({
            key: 1,
            "class": _ctx.cx("content"),
            onClick: _cache[0] || (_cache[0] = function() {
              return $options.onContentClick && $options.onContentClick.apply($options, arguments);
            }),
            onMousedown: _cache[1] || (_cache[1] = function() {
              return $options.onContentClick && $options.onContentClick.apply($options, arguments);
            }),
            onKeydown: _cache[2] || (_cache[2] = function() {
              return $options.onContentKeydown && $options.onContentKeydown.apply($options, arguments);
            })
          }, _ctx.ptm("content")), [renderSlot(_ctx.$slots, "default")], 16))], 16, _hoisted_1$j)), [[_directive_focustrap]]) : createCommentVNode("", true)];
        }),
        _: 3
      }, 16, ["onEnter", "onLeave", "onAfterLeave"])];
    }),
    _: 3
  }, 8, ["appendTo"]);
}
__name(render$8, "render$8");
script$9.render = render$8;
var theme$6 = /* @__PURE__ */ __name(function theme12(_ref) {
  var dt = _ref.dt;
  return "\n.p-tree {\n    background: ".concat(dt("tree.background"), ";\n    color: ").concat(dt("tree.color"), ";\n    padding: ").concat(dt("tree.padding"), ";\n}\n\n.p-tree-root-children,\n.p-tree-node-children {\n    display: flex;\n    list-style-type: none;\n    flex-direction: column;\n    margin: 0;\n    gap: ").concat(dt("tree.gap"), ";\n}\n\n.p-tree-root-children {\n    padding: ").concat(dt("tree.gap"), " 0 0 0;\n}\n\n.p-tree-node-children {\n    padding: ").concat(dt("tree.gap"), " 0 0 ").concat(dt("tree.indent"), ";\n}\n\n.p-tree-node {\n    padding: 0;\n    outline: 0 none;\n}\n\n.p-tree-node-content {\n    border-radius: ").concat(dt("tree.node.border.radius"), ";\n    padding: ").concat(dt("tree.node.padding"), ";\n    display: flex;\n    align-items: center;\n    outline-color: transparent;\n    color: ").concat(dt("tree.node.color"), ";\n    gap: ").concat(dt("tree.node.gap"), ";\n    transition: background ").concat(dt("tree.transition.duration"), ", color ").concat(dt("tree.transition.duration"), ", outline-color ").concat(dt("tree.transition.duration"), ", box-shadow ").concat(dt("tree.transition.duration"), ";\n}\n\n.p-tree-node:focus-visible > .p-tree-node-content {\n    box-shadow: ").concat(dt("tree.node.focus.ring.shadow"), ";\n    outline: ").concat(dt("tree.node.focus.ring.width"), " ").concat(dt("tree.node.focus.ring.style"), " ").concat(dt("tree.node.focus.ring.color"), ";\n    outline-offset: ").concat(dt("tree.node.focus.ring.offset"), ";\n}\n\n.p-tree-node-content.p-tree-node-selectable:not(.p-tree-node-selected):hover {\n    background: ").concat(dt("tree.node.hover.background"), ";\n    color: ").concat(dt("tree.node.hover.color"), ";\n}\n\n.p-tree-node-content.p-tree-node-selectable:not(.p-tree-node-selected):hover .p-tree-node-icon {\n    color: ").concat(dt("tree.node.icon.hover.color"), ";\n}\n\n.p-tree-node-content.p-tree-node-selected {\n    background: ").concat(dt("tree.node.selected.background"), ";\n    color: ").concat(dt("tree.node.selected.color"), ";\n}\n\n.p-tree-node-content.p-tree-node-selected .p-tree-node-toggle-button {\n    color: inherit;\n}\n\n.p-tree-node-toggle-button {\n    cursor: pointer;\n    user-select: none;\n    display: inline-flex;\n    align-items: center;\n    justify-content: center;\n    overflow: hidden;\n    position: relative;\n    flex-shrink: 0;\n    width: ").concat(dt("tree.node.toggle.button.size"), ";\n    height: ").concat(dt("tree.node.toggle.button.size"), ";\n    color: ").concat(dt("tree.node.toggle.button.color"), ";\n    border: 0 none;\n    background: transparent;\n    border-radius: ").concat(dt("tree.node.toggle.button.border.radius"), ";\n    transition: background ").concat(dt("tree.transition.duration"), ", color ").concat(dt("tree.transition.duration"), ", border-color ").concat(dt("tree.transition.duration"), ", outline-color ").concat(dt("tree.transition.duration"), ", box-shadow ").concat(dt("tree.transition.duration"), ";\n    outline-color: transparent;\n    padding: 0;\n}\n\n.p-tree-node-toggle-button:enabled:hover {\n    background: ").concat(dt("tree.node.toggle.button.hover.background"), ";\n    color: ").concat(dt("tree.node.toggle.button.hover.color"), ";\n}\n\n.p-tree-node-content.p-tree-node-selected .p-tree-node-toggle-button:hover {\n    background: ").concat(dt("tree.node.toggle.button.selected.hover.background"), ";\n    color: ").concat(dt("tree.node.toggle.button.selected.hover.color"), ";\n}\n\n.p-tree-root {\n    overflow: auto;\n}\n\n.p-tree-node-selectable {\n    cursor: pointer;\n    user-select: none;\n}\n\n.p-tree-node-leaf > .p-tree-node-content .p-tree-node-toggle-button {\n    visibility: hidden;\n}\n\n.p-tree-node-icon {\n    color: ").concat(dt("tree.node.icon.color"), ";\n    transition: color ").concat(dt("tree.transition.duration"), ";\n}\n\n.p-tree-node-content.p-tree-node-selected .p-tree-node-icon {\n    color: ").concat(dt("tree.node.icon.selected.color"), ";\n}\n\n.p-tree-filter-input {\n    width: 100%;\n}\n\n.p-tree-loading {\n    position: relative;\n    height: 100%;\n}\n\n.p-tree-loading-icon {\n    font-size: ").concat(dt("tree.loading.icon.size"), ";\n    width: ").concat(dt("tree.loading.icon.size"), ";\n    height: ").concat(dt("tree.loading.icon.size"), ";\n}\n\n.p-tree .p-tree-mask {\n    position: absolute;\n    z-index: 1;\n    display: flex;\n    align-items: center;\n    justify-content: center;\n}\n\n.p-tree-flex-scrollable {\n    display: flex;\n    flex: 1;\n    height: 100%;\n    flex-direction: column;\n}\n\n.p-tree-flex-scrollable .p-tree-root {\n    flex: 1;\n}\n");
}, "theme");
var classes$6 = {
  root: /* @__PURE__ */ __name(function root8(_ref2) {
    var props = _ref2.props;
    return ["p-tree p-component", {
      "p-tree-selectable": props.selectionMode != null,
      "p-tree-loading": props.loading,
      "p-tree-flex-scrollable": props.scrollHeight === "flex"
    }];
  }, "root"),
  mask: "p-tree-mask p-overlay-mask",
  loadingIcon: "p-tree-loading-icon",
  pcFilterInput: "p-tree-filter-input",
  wrapper: "p-tree-root",
  //TODO: discuss
  rootChildren: "p-tree-root-children",
  node: /* @__PURE__ */ __name(function node(_ref3) {
    var instance = _ref3.instance;
    return ["p-tree-node", {
      "p-tree-node-leaf": instance.leaf
    }];
  }, "node"),
  nodeContent: /* @__PURE__ */ __name(function nodeContent(_ref4) {
    var instance = _ref4.instance;
    return ["p-tree-node-content", instance.node.styleClass, {
      "p-tree-node-selectable": instance.selectable,
      "p-tree-node-selected": instance.checkboxMode && instance.$parentInstance.highlightOnSelect ? instance.checked : instance.selected
    }];
  }, "nodeContent"),
  nodeToggleButton: "p-tree-node-toggle-button",
  nodeToggleIcon: "p-tree-node-toggle-icon",
  nodeCheckbox: "p-tree-node-checkbox",
  nodeIcon: "p-tree-node-icon",
  nodeLabel: "p-tree-node-label",
  nodeChildren: "p-tree-node-children"
};
var TreeStyle = BaseStyle.extend({
  name: "tree",
  theme: theme$6,
  classes: classes$6
});
var script$2$3 = {
  name: "BaseTree",
  "extends": script$p,
  props: {
    value: {
      type: null,
      "default": null
    },
    expandedKeys: {
      type: null,
      "default": null
    },
    selectionKeys: {
      type: null,
      "default": null
    },
    selectionMode: {
      type: String,
      "default": null
    },
    metaKeySelection: {
      type: Boolean,
      "default": false
    },
    loading: {
      type: Boolean,
      "default": false
    },
    loadingIcon: {
      type: String,
      "default": void 0
    },
    loadingMode: {
      type: String,
      "default": "mask"
    },
    filter: {
      type: Boolean,
      "default": false
    },
    filterBy: {
      type: String,
      "default": "label"
    },
    filterMode: {
      type: String,
      "default": "lenient"
    },
    filterPlaceholder: {
      type: String,
      "default": null
    },
    filterLocale: {
      type: String,
      "default": void 0
    },
    highlightOnSelect: {
      type: Boolean,
      "default": false
    },
    scrollHeight: {
      type: String,
      "default": null
    },
    level: {
      type: Number,
      "default": 0
    },
    ariaLabelledby: {
      type: String,
      "default": null
    },
    ariaLabel: {
      type: String,
      "default": null
    }
  },
  style: TreeStyle,
  provide: /* @__PURE__ */ __name(function provide14() {
    return {
      $pcTree: this,
      $parentInstance: this
    };
  }, "provide")
};
function _typeof$1$1(o) {
  "@babel/helpers - typeof";
  return _typeof$1$1 = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function(o2) {
    return typeof o2;
  } : function(o2) {
    return o2 && "function" == typeof Symbol && o2.constructor === Symbol && o2 !== Symbol.prototype ? "symbol" : typeof o2;
  }, _typeof$1$1(o);
}
__name(_typeof$1$1, "_typeof$1$1");
function _createForOfIteratorHelper$1(r, e) {
  var t = "undefined" != typeof Symbol && r[Symbol.iterator] || r["@@iterator"];
  if (!t) {
    if (Array.isArray(r) || (t = _unsupportedIterableToArray$1(r)) || e) {
      t && (r = t);
      var _n = 0, F = /* @__PURE__ */ __name(function F2() {
      }, "F");
      return { s: F, n: /* @__PURE__ */ __name(function n() {
        return _n >= r.length ? { done: true } : { done: false, value: r[_n++] };
      }, "n"), e: /* @__PURE__ */ __name(function e2(r2) {
        throw r2;
      }, "e"), f: F };
    }
    throw new TypeError("Invalid attempt to iterate non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.");
  }
  var o, a = true, u = false;
  return { s: /* @__PURE__ */ __name(function s() {
    t = t.call(r);
  }, "s"), n: /* @__PURE__ */ __name(function n() {
    var r2 = t.next();
    return a = r2.done, r2;
  }, "n"), e: /* @__PURE__ */ __name(function e2(r2) {
    u = true, o = r2;
  }, "e"), f: /* @__PURE__ */ __name(function f() {
    try {
      a || null == t["return"] || t["return"]();
    } finally {
      if (u) throw o;
    }
  }, "f") };
}
__name(_createForOfIteratorHelper$1, "_createForOfIteratorHelper$1");
function ownKeys$1$1(e, r) {
  var t = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var o = Object.getOwnPropertySymbols(e);
    r && (o = o.filter(function(r2) {
      return Object.getOwnPropertyDescriptor(e, r2).enumerable;
    })), t.push.apply(t, o);
  }
  return t;
}
__name(ownKeys$1$1, "ownKeys$1$1");
function _objectSpread$1$1(e) {
  for (var r = 1; r < arguments.length; r++) {
    var t = null != arguments[r] ? arguments[r] : {};
    r % 2 ? ownKeys$1$1(Object(t), true).forEach(function(r2) {
      _defineProperty$1$1(e, r2, t[r2]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(t)) : ownKeys$1$1(Object(t)).forEach(function(r2) {
      Object.defineProperty(e, r2, Object.getOwnPropertyDescriptor(t, r2));
    });
  }
  return e;
}
__name(_objectSpread$1$1, "_objectSpread$1$1");
function _defineProperty$1$1(e, r, t) {
  return (r = _toPropertyKey$1$1(r)) in e ? Object.defineProperty(e, r, { value: t, enumerable: true, configurable: true, writable: true }) : e[r] = t, e;
}
__name(_defineProperty$1$1, "_defineProperty$1$1");
function _toPropertyKey$1$1(t) {
  var i = _toPrimitive$1$1(t, "string");
  return "symbol" == _typeof$1$1(i) ? i : i + "";
}
__name(_toPropertyKey$1$1, "_toPropertyKey$1$1");
function _toPrimitive$1$1(t, r) {
  if ("object" != _typeof$1$1(t) || !t) return t;
  var e = t[Symbol.toPrimitive];
  if (void 0 !== e) {
    var i = e.call(t, r || "default");
    if ("object" != _typeof$1$1(i)) return i;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return ("string" === r ? String : Number)(t);
}
__name(_toPrimitive$1$1, "_toPrimitive$1$1");
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
var script$1$6 = {
  name: "TreeNode",
  hostName: "Tree",
  "extends": script$p,
  emits: ["node-toggle", "node-click", "checkbox-change"],
  props: {
    node: {
      type: null,
      "default": null
    },
    expandedKeys: {
      type: null,
      "default": null
    },
    loadingMode: {
      type: String,
      "default": "mask"
    },
    selectionKeys: {
      type: null,
      "default": null
    },
    selectionMode: {
      type: String,
      "default": null
    },
    templates: {
      type: null,
      "default": null
    },
    level: {
      type: Number,
      "default": null
    },
    index: null
  },
  nodeTouched: false,
  toggleClicked: false,
  mounted: /* @__PURE__ */ __name(function mounted9() {
    this.setAllNodesTabIndexes();
  }, "mounted"),
  methods: {
    toggle: /* @__PURE__ */ __name(function toggle3() {
      this.$emit("node-toggle", this.node);
      this.toggleClicked = true;
    }, "toggle"),
    label: /* @__PURE__ */ __name(function label2(node2) {
      return typeof node2.label === "function" ? node2.label() : node2.label;
    }, "label"),
    onChildNodeToggle: /* @__PURE__ */ __name(function onChildNodeToggle(node2) {
      this.$emit("node-toggle", node2);
    }, "onChildNodeToggle"),
    getPTOptions: /* @__PURE__ */ __name(function getPTOptions8(key) {
      return this.ptm(key, {
        context: {
          index: this.index,
          expanded: this.expanded,
          selected: this.selected,
          checked: this.checked,
          leaf: this.leaf
        }
      });
    }, "getPTOptions"),
    onClick: /* @__PURE__ */ __name(function onClick(event) {
      if (this.toggleClicked || getAttribute(event.target, '[data-pc-section="nodetogglebutton"]') || getAttribute(event.target.parentElement, '[data-pc-section="nodetogglebutton"]')) {
        this.toggleClicked = false;
        return;
      }
      if (this.isCheckboxSelectionMode()) {
        if (this.node.selectable != false) {
          this.toggleCheckbox();
        }
      } else {
        this.$emit("node-click", {
          originalEvent: event,
          nodeTouched: this.nodeTouched,
          node: this.node
        });
      }
      this.nodeTouched = false;
    }, "onClick"),
    onChildNodeClick: /* @__PURE__ */ __name(function onChildNodeClick(event) {
      this.$emit("node-click", event);
    }, "onChildNodeClick"),
    onTouchEnd: /* @__PURE__ */ __name(function onTouchEnd2() {
      this.nodeTouched = true;
    }, "onTouchEnd"),
    onKeyDown: /* @__PURE__ */ __name(function onKeyDown4(event) {
      if (!this.isSameNode(event)) return;
      switch (event.code) {
        case "Tab":
          this.onTabKey(event);
          break;
        case "ArrowDown":
          this.onArrowDown(event);
          break;
        case "ArrowUp":
          this.onArrowUp(event);
          break;
        case "ArrowRight":
          this.onArrowRight(event);
          break;
        case "ArrowLeft":
          this.onArrowLeft(event);
          break;
        case "Enter":
        case "NumpadEnter":
        case "Space":
          this.onEnterKey(event);
          break;
      }
    }, "onKeyDown"),
    onArrowDown: /* @__PURE__ */ __name(function onArrowDown(event) {
      var nodeElement = event.target.getAttribute("data-pc-section") === "nodetogglebutton" ? event.target.closest('[role="treeitem"]') : event.target;
      var listElement = nodeElement.children[1];
      if (listElement) {
        this.focusRowChange(nodeElement, listElement.children[0]);
      } else {
        if (nodeElement.nextElementSibling) {
          this.focusRowChange(nodeElement, nodeElement.nextElementSibling);
        } else {
          var nextSiblingAncestor = this.findNextSiblingOfAncestor(nodeElement);
          if (nextSiblingAncestor) {
            this.focusRowChange(nodeElement, nextSiblingAncestor);
          }
        }
      }
      event.preventDefault();
    }, "onArrowDown"),
    onArrowUp: /* @__PURE__ */ __name(function onArrowUp(event) {
      var nodeElement = event.target;
      if (nodeElement.previousElementSibling) {
        this.focusRowChange(nodeElement, nodeElement.previousElementSibling, this.findLastVisibleDescendant(nodeElement.previousElementSibling));
      } else {
        var parentNodeElement = this.getParentNodeElement(nodeElement);
        if (parentNodeElement) {
          this.focusRowChange(nodeElement, parentNodeElement);
        }
      }
      event.preventDefault();
    }, "onArrowUp"),
    onArrowRight: /* @__PURE__ */ __name(function onArrowRight(event) {
      var _this = this;
      if (this.leaf || this.expanded) return;
      event.currentTarget.tabIndex = -1;
      this.$emit("node-toggle", this.node);
      this.$nextTick(function() {
        _this.onArrowDown(event);
      });
    }, "onArrowRight"),
    onArrowLeft: /* @__PURE__ */ __name(function onArrowLeft(event) {
      var togglerElement = findSingle(event.currentTarget, '[data-pc-section="nodetogglebutton"]');
      if (this.level === 0 && !this.expanded) {
        return false;
      }
      if (this.expanded && !this.leaf) {
        togglerElement.click();
        return false;
      }
      var target = this.findBeforeClickableNode(event.currentTarget);
      if (target) {
        this.focusRowChange(event.currentTarget, target);
      }
    }, "onArrowLeft"),
    onEnterKey: /* @__PURE__ */ __name(function onEnterKey3(event) {
      this.setTabIndexForSelectionMode(event, this.nodeTouched);
      this.onClick(event);
      event.preventDefault();
    }, "onEnterKey"),
    onTabKey: /* @__PURE__ */ __name(function onTabKey5() {
      this.setAllNodesTabIndexes();
    }, "onTabKey"),
    setAllNodesTabIndexes: /* @__PURE__ */ __name(function setAllNodesTabIndexes() {
      var nodes = find(this.$refs.currentNode.closest('[data-pc-section="rootchildren"]'), '[role="treeitem"]');
      var hasSelectedNode = _toConsumableArray$1(nodes).some(function(node2) {
        return node2.getAttribute("aria-selected") === "true" || node2.getAttribute("aria-checked") === "true";
      });
      _toConsumableArray$1(nodes).forEach(function(node2) {
        node2.tabIndex = -1;
      });
      if (hasSelectedNode) {
        var selectedNodes = _toConsumableArray$1(nodes).filter(function(node2) {
          return node2.getAttribute("aria-selected") === "true" || node2.getAttribute("aria-checked") === "true";
        });
        selectedNodes[0].tabIndex = 0;
        return;
      }
      _toConsumableArray$1(nodes)[0].tabIndex = 0;
    }, "setAllNodesTabIndexes"),
    setTabIndexForSelectionMode: /* @__PURE__ */ __name(function setTabIndexForSelectionMode(event, nodeTouched) {
      if (this.selectionMode !== null) {
        var elements = _toConsumableArray$1(find(this.$refs.currentNode.parentElement, '[role="treeitem"]'));
        event.currentTarget.tabIndex = nodeTouched === false ? -1 : 0;
        if (elements.every(function(element) {
          return element.tabIndex === -1;
        })) {
          elements[0].tabIndex = 0;
        }
      }
    }, "setTabIndexForSelectionMode"),
    focusRowChange: /* @__PURE__ */ __name(function focusRowChange(firstFocusableRow, currentFocusedRow, lastVisibleDescendant) {
      firstFocusableRow.tabIndex = "-1";
      currentFocusedRow.tabIndex = "0";
      this.focusNode(lastVisibleDescendant || currentFocusedRow);
    }, "focusRowChange"),
    findBeforeClickableNode: /* @__PURE__ */ __name(function findBeforeClickableNode(node2) {
      var parentListElement = node2.closest("ul").closest("li");
      if (parentListElement) {
        var prevNodeButton = findSingle(parentListElement, "button");
        if (prevNodeButton && prevNodeButton.style.visibility !== "hidden") {
          return parentListElement;
        }
        return this.findBeforeClickableNode(node2.previousElementSibling);
      }
      return null;
    }, "findBeforeClickableNode"),
    toggleCheckbox: /* @__PURE__ */ __name(function toggleCheckbox() {
      var _selectionKeys = this.selectionKeys ? _objectSpread$1$1({}, this.selectionKeys) : {};
      var _check = !this.checked;
      this.propagateDown(this.node, _check, _selectionKeys);
      this.$emit("checkbox-change", {
        node: this.node,
        check: _check,
        selectionKeys: _selectionKeys
      });
    }, "toggleCheckbox"),
    propagateDown: /* @__PURE__ */ __name(function propagateDown(node2, check, selectionKeys) {
      if (check && node2.selectable != false) selectionKeys[node2.key] = {
        checked: true,
        partialChecked: false
      };
      else delete selectionKeys[node2.key];
      if (node2.children && node2.children.length) {
        var _iterator = _createForOfIteratorHelper$1(node2.children), _step;
        try {
          for (_iterator.s(); !(_step = _iterator.n()).done; ) {
            var child = _step.value;
            this.propagateDown(child, check, selectionKeys);
          }
        } catch (err) {
          _iterator.e(err);
        } finally {
          _iterator.f();
        }
      }
    }, "propagateDown"),
    propagateUp: /* @__PURE__ */ __name(function propagateUp(event) {
      var check = event.check;
      var _selectionKeys = _objectSpread$1$1({}, event.selectionKeys);
      var checkedChildCount = 0;
      var childPartialSelected = false;
      var _iterator2 = _createForOfIteratorHelper$1(this.node.children), _step2;
      try {
        for (_iterator2.s(); !(_step2 = _iterator2.n()).done; ) {
          var child = _step2.value;
          if (_selectionKeys[child.key] && _selectionKeys[child.key].checked) checkedChildCount++;
          else if (_selectionKeys[child.key] && _selectionKeys[child.key].partialChecked) childPartialSelected = true;
        }
      } catch (err) {
        _iterator2.e(err);
      } finally {
        _iterator2.f();
      }
      if (check && checkedChildCount === this.node.children.length) {
        _selectionKeys[this.node.key] = {
          checked: true,
          partialChecked: false
        };
      } else {
        if (!check) {
          delete _selectionKeys[this.node.key];
        }
        if (childPartialSelected || checkedChildCount > 0 && checkedChildCount !== this.node.children.length) _selectionKeys[this.node.key] = {
          checked: false,
          partialChecked: true
        };
        else delete _selectionKeys[this.node.key];
      }
      this.$emit("checkbox-change", {
        node: event.node,
        check: event.check,
        selectionKeys: _selectionKeys
      });
    }, "propagateUp"),
    onChildCheckboxChange: /* @__PURE__ */ __name(function onChildCheckboxChange(event) {
      this.$emit("checkbox-change", event);
    }, "onChildCheckboxChange"),
    findNextSiblingOfAncestor: /* @__PURE__ */ __name(function findNextSiblingOfAncestor(nodeElement) {
      var parentNodeElement = this.getParentNodeElement(nodeElement);
      if (parentNodeElement) {
        if (parentNodeElement.nextElementSibling) return parentNodeElement.nextElementSibling;
        else return this.findNextSiblingOfAncestor(parentNodeElement);
      } else {
        return null;
      }
    }, "findNextSiblingOfAncestor"),
    findLastVisibleDescendant: /* @__PURE__ */ __name(function findLastVisibleDescendant(nodeElement) {
      var childrenListElement = nodeElement.children[1];
      if (childrenListElement) {
        var lastChildElement = childrenListElement.children[childrenListElement.children.length - 1];
        return this.findLastVisibleDescendant(lastChildElement);
      } else {
        return nodeElement;
      }
    }, "findLastVisibleDescendant"),
    getParentNodeElement: /* @__PURE__ */ __name(function getParentNodeElement(nodeElement) {
      var parentNodeElement = nodeElement.parentElement.parentElement;
      return getAttribute(parentNodeElement, "role") === "treeitem" ? parentNodeElement : null;
    }, "getParentNodeElement"),
    focusNode: /* @__PURE__ */ __name(function focusNode(element) {
      element.focus();
    }, "focusNode"),
    isCheckboxSelectionMode: /* @__PURE__ */ __name(function isCheckboxSelectionMode() {
      return this.selectionMode === "checkbox";
    }, "isCheckboxSelectionMode"),
    isSameNode: /* @__PURE__ */ __name(function isSameNode(event) {
      return event.currentTarget && (event.currentTarget.isSameNode(event.target) || event.currentTarget.isSameNode(event.target.closest('[role="treeitem"]')));
    }, "isSameNode")
  },
  computed: {
    hasChildren: /* @__PURE__ */ __name(function hasChildren() {
      return this.node.children && this.node.children.length > 0;
    }, "hasChildren"),
    expanded: /* @__PURE__ */ __name(function expanded() {
      return this.expandedKeys && this.expandedKeys[this.node.key] === true;
    }, "expanded"),
    leaf: /* @__PURE__ */ __name(function leaf() {
      return this.node.leaf === false ? false : !(this.node.children && this.node.children.length);
    }, "leaf"),
    selectable: /* @__PURE__ */ __name(function selectable() {
      return this.node.selectable === false ? false : this.selectionMode != null;
    }, "selectable"),
    selected: /* @__PURE__ */ __name(function selected() {
      return this.selectionMode && this.selectionKeys ? this.selectionKeys[this.node.key] === true : false;
    }, "selected"),
    checkboxMode: /* @__PURE__ */ __name(function checkboxMode() {
      return this.selectionMode === "checkbox" && this.node.selectable !== false;
    }, "checkboxMode"),
    checked: /* @__PURE__ */ __name(function checked() {
      return this.selectionKeys ? this.selectionKeys[this.node.key] && this.selectionKeys[this.node.key].checked : false;
    }, "checked"),
    partialChecked: /* @__PURE__ */ __name(function partialChecked() {
      return this.selectionKeys ? this.selectionKeys[this.node.key] && this.selectionKeys[this.node.key].partialChecked : false;
    }, "partialChecked"),
    ariaChecked: /* @__PURE__ */ __name(function ariaChecked() {
      return this.selectionMode === "single" || this.selectionMode === "multiple" ? this.selected : void 0;
    }, "ariaChecked"),
    ariaSelected: /* @__PURE__ */ __name(function ariaSelected() {
      return this.checkboxMode ? this.checked : void 0;
    }, "ariaSelected")
  },
  components: {
    Checkbox: script$E,
    ChevronDownIcon: script$s,
    ChevronRightIcon: script$B,
    CheckIcon: script$F,
    MinusIcon: script$G,
    SpinnerIcon: script$t
  },
  directives: {
    ripple: Ripple
  }
};
var _hoisted_1$1$3 = ["aria-label", "aria-selected", "aria-expanded", "aria-setsize", "aria-posinset", "aria-level", "aria-checked", "tabindex"];
var _hoisted_2$d = ["data-p-selected", "data-p-selectable"];
function render$1$3(_ctx, _cache, $props, $setup, $data, $options) {
  var _component_SpinnerIcon = resolveComponent("SpinnerIcon");
  var _component_Checkbox = resolveComponent("Checkbox");
  var _component_TreeNode = resolveComponent("TreeNode", true);
  var _directive_ripple = resolveDirective("ripple");
  return openBlock(), createElementBlock("li", mergeProps({
    ref: "currentNode",
    "class": _ctx.cx("node"),
    role: "treeitem",
    "aria-label": $options.label($props.node),
    "aria-selected": $options.ariaSelected,
    "aria-expanded": $options.expanded,
    "aria-setsize": $props.node.children ? $props.node.children.length : 0,
    "aria-posinset": $props.index + 1,
    "aria-level": $props.level,
    "aria-checked": $options.ariaChecked,
    tabindex: $props.index === 0 ? 0 : -1,
    onKeydown: _cache[4] || (_cache[4] = function() {
      return $options.onKeyDown && $options.onKeyDown.apply($options, arguments);
    })
  }, $props.level === 1 ? $options.getPTOptions("node") : _ctx.ptm("nodeChildren")), [createBaseVNode("div", mergeProps({
    "class": _ctx.cx("nodeContent"),
    onClick: _cache[2] || (_cache[2] = function() {
      return $options.onClick && $options.onClick.apply($options, arguments);
    }),
    onTouchend: _cache[3] || (_cache[3] = function() {
      return $options.onTouchEnd && $options.onTouchEnd.apply($options, arguments);
    }),
    style: $props.node.style
  }, $options.getPTOptions("nodeContent"), {
    "data-p-selected": $options.checkboxMode ? $options.checked : $options.selected,
    "data-p-selectable": $options.selectable
  }), [withDirectives((openBlock(), createElementBlock("button", mergeProps({
    type: "button",
    "class": _ctx.cx("nodeToggleButton"),
    onClick: _cache[0] || (_cache[0] = function() {
      return $options.toggle && $options.toggle.apply($options, arguments);
    }),
    tabindex: "-1"
  }, $options.getPTOptions("nodeToggleButton")), [$props.node.loading && $props.loadingMode === "icon" ? (openBlock(), createElementBlock(Fragment, {
    key: 0
  }, [$props.templates["nodetoggleicon"] || $props.templates["nodetogglericon"] ? (openBlock(), createBlock(resolveDynamicComponent($props.templates["nodetoggleicon"] || $props.templates["nodetogglericon"]), {
    key: 0,
    "class": normalizeClass(_ctx.cx("nodeToggleIcon"))
  }, null, 8, ["class"])) : (openBlock(), createBlock(_component_SpinnerIcon, mergeProps({
    key: 1,
    spin: "",
    "class": _ctx.cx("nodetogglericon")
  }, _ctx.ptm("nodeToggleIcon")), null, 16, ["class"]))], 64)) : (openBlock(), createElementBlock(Fragment, {
    key: 1
  }, [$props.templates["nodetoggleicon"] || $props.templates["togglericon"] ? (openBlock(), createBlock(resolveDynamicComponent($props.templates["nodetoggleicon"] || $props.templates["togglericon"]), {
    key: 0,
    node: $props.node,
    expanded: $options.expanded,
    "class": normalizeClass(_ctx.cx("nodeToggleIcon"))
  }, null, 8, ["node", "expanded", "class"])) : $options.expanded ? (openBlock(), createBlock(resolveDynamicComponent($props.node.expandedIcon ? "span" : "ChevronDownIcon"), mergeProps({
    key: 1,
    "class": _ctx.cx("nodeToggleIcon")
  }, $options.getPTOptions("nodeToggleIcon")), null, 16, ["class"])) : (openBlock(), createBlock(resolveDynamicComponent($props.node.collapsedIcon ? "span" : "ChevronRightIcon"), mergeProps({
    key: 2,
    "class": _ctx.cx("nodeToggleIcon")
  }, $options.getPTOptions("nodeToggleIcon")), null, 16, ["class"]))], 64))], 16)), [[_directive_ripple]]), $options.checkboxMode ? (openBlock(), createBlock(_component_Checkbox, {
    key: 0,
    modelValue: $options.checked,
    binary: true,
    indeterminate: $options.partialChecked,
    "class": normalizeClass(_ctx.cx("nodeCheckbox")),
    tabindex: -1,
    unstyled: _ctx.unstyled,
    pt: $options.getPTOptions("nodeCheckbox"),
    "data-p-partialchecked": $options.partialChecked
  }, {
    icon: withCtx(function(slotProps) {
      return [$props.templates["checkboxicon"] ? (openBlock(), createBlock(resolveDynamicComponent($props.templates["checkboxicon"]), {
        key: 0,
        checked: slotProps.checked,
        partialChecked: $options.partialChecked,
        "class": normalizeClass(slotProps["class"])
      }, null, 8, ["checked", "partialChecked", "class"])) : createCommentVNode("", true)];
    }),
    _: 1
  }, 8, ["modelValue", "indeterminate", "class", "unstyled", "pt", "data-p-partialchecked"])) : createCommentVNode("", true), $props.templates["nodeicon"] ? (openBlock(), createBlock(resolveDynamicComponent($props.templates["nodeicon"]), mergeProps({
    key: 1,
    node: $props.node,
    "class": [_ctx.cx("nodeIcon")]
  }, $options.getPTOptions("nodeIcon")), null, 16, ["node", "class"])) : (openBlock(), createElementBlock("span", mergeProps({
    key: 2,
    "class": [_ctx.cx("nodeIcon"), $props.node.icon]
  }, $options.getPTOptions("nodeIcon")), null, 16)), createBaseVNode("span", mergeProps({
    "class": _ctx.cx("nodeLabel")
  }, $options.getPTOptions("nodeLabel"), {
    onKeydown: _cache[1] || (_cache[1] = withModifiers(function() {
    }, ["stop"]))
  }), [$props.templates[$props.node.type] || $props.templates["default"] ? (openBlock(), createBlock(resolveDynamicComponent($props.templates[$props.node.type] || $props.templates["default"]), {
    key: 0,
    node: $props.node,
    selected: $options.checkboxMode ? $options.checked : $options.selected
  }, null, 8, ["node", "selected"])) : (openBlock(), createElementBlock(Fragment, {
    key: 1
  }, [createTextVNode(toDisplayString($options.label($props.node)), 1)], 64))], 16)], 16, _hoisted_2$d), $options.hasChildren && $options.expanded ? (openBlock(), createElementBlock("ul", mergeProps({
    key: 0,
    "class": _ctx.cx("nodeChildren"),
    role: "group"
  }, _ctx.ptm("nodeChildren")), [(openBlock(true), createElementBlock(Fragment, null, renderList($props.node.children, function(childNode) {
    return openBlock(), createBlock(_component_TreeNode, {
      key: childNode.key,
      node: childNode,
      templates: $props.templates,
      level: $props.level + 1,
      loadingMode: $props.loadingMode,
      expandedKeys: $props.expandedKeys,
      onNodeToggle: $options.onChildNodeToggle,
      onNodeClick: $options.onChildNodeClick,
      selectionMode: $props.selectionMode,
      selectionKeys: $props.selectionKeys,
      onCheckboxChange: $options.propagateUp,
      unstyled: _ctx.unstyled,
      pt: _ctx.pt
    }, null, 8, ["node", "templates", "level", "loadingMode", "expandedKeys", "onNodeToggle", "onNodeClick", "selectionMode", "selectionKeys", "onCheckboxChange", "unstyled", "pt"]);
  }), 128))], 16)) : createCommentVNode("", true)], 16, _hoisted_1$1$3);
}
__name(render$1$3, "render$1$3");
script$1$6.render = render$1$3;
function _typeof$4(o) {
  "@babel/helpers - typeof";
  return _typeof$4 = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function(o2) {
    return typeof o2;
  } : function(o2) {
    return o2 && "function" == typeof Symbol && o2.constructor === Symbol && o2 !== Symbol.prototype ? "symbol" : typeof o2;
  }, _typeof$4(o);
}
__name(_typeof$4, "_typeof$4");
function _createForOfIteratorHelper(r, e) {
  var t = "undefined" != typeof Symbol && r[Symbol.iterator] || r["@@iterator"];
  if (!t) {
    if (Array.isArray(r) || (t = _unsupportedIterableToArray$2(r)) || e) {
      t && (r = t);
      var _n = 0, F = /* @__PURE__ */ __name(function F2() {
      }, "F");
      return { s: F, n: /* @__PURE__ */ __name(function n() {
        return _n >= r.length ? { done: true } : { done: false, value: r[_n++] };
      }, "n"), e: /* @__PURE__ */ __name(function e2(r2) {
        throw r2;
      }, "e"), f: F };
    }
    throw new TypeError("Invalid attempt to iterate non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.");
  }
  var o, a = true, u = false;
  return { s: /* @__PURE__ */ __name(function s() {
    t = t.call(r);
  }, "s"), n: /* @__PURE__ */ __name(function n() {
    var r2 = t.next();
    return a = r2.done, r2;
  }, "n"), e: /* @__PURE__ */ __name(function e2(r2) {
    u = true, o = r2;
  }, "e"), f: /* @__PURE__ */ __name(function f() {
    try {
      a || null == t["return"] || t["return"]();
    } finally {
      if (u) throw o;
    }
  }, "f") };
}
__name(_createForOfIteratorHelper, "_createForOfIteratorHelper");
function _toConsumableArray$2(r) {
  return _arrayWithoutHoles$2(r) || _iterableToArray$2(r) || _unsupportedIterableToArray$2(r) || _nonIterableSpread$2();
}
__name(_toConsumableArray$2, "_toConsumableArray$2");
function _nonIterableSpread$2() {
  throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.");
}
__name(_nonIterableSpread$2, "_nonIterableSpread$2");
function _unsupportedIterableToArray$2(r, a) {
  if (r) {
    if ("string" == typeof r) return _arrayLikeToArray$2(r, a);
    var t = {}.toString.call(r).slice(8, -1);
    return "Object" === t && r.constructor && (t = r.constructor.name), "Map" === t || "Set" === t ? Array.from(r) : "Arguments" === t || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(t) ? _arrayLikeToArray$2(r, a) : void 0;
  }
}
__name(_unsupportedIterableToArray$2, "_unsupportedIterableToArray$2");
function _iterableToArray$2(r) {
  if ("undefined" != typeof Symbol && null != r[Symbol.iterator] || null != r["@@iterator"]) return Array.from(r);
}
__name(_iterableToArray$2, "_iterableToArray$2");
function _arrayWithoutHoles$2(r) {
  if (Array.isArray(r)) return _arrayLikeToArray$2(r);
}
__name(_arrayWithoutHoles$2, "_arrayWithoutHoles$2");
function _arrayLikeToArray$2(r, a) {
  (null == a || a > r.length) && (a = r.length);
  for (var e = 0, n = Array(a); e < a; e++) n[e] = r[e];
  return n;
}
__name(_arrayLikeToArray$2, "_arrayLikeToArray$2");
function ownKeys$3(e, r) {
  var t = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var o = Object.getOwnPropertySymbols(e);
    r && (o = o.filter(function(r2) {
      return Object.getOwnPropertyDescriptor(e, r2).enumerable;
    })), t.push.apply(t, o);
  }
  return t;
}
__name(ownKeys$3, "ownKeys$3");
function _objectSpread$3(e) {
  for (var r = 1; r < arguments.length; r++) {
    var t = null != arguments[r] ? arguments[r] : {};
    r % 2 ? ownKeys$3(Object(t), true).forEach(function(r2) {
      _defineProperty$4(e, r2, t[r2]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(t)) : ownKeys$3(Object(t)).forEach(function(r2) {
      Object.defineProperty(e, r2, Object.getOwnPropertyDescriptor(t, r2));
    });
  }
  return e;
}
__name(_objectSpread$3, "_objectSpread$3");
function _defineProperty$4(e, r, t) {
  return (r = _toPropertyKey$4(r)) in e ? Object.defineProperty(e, r, { value: t, enumerable: true, configurable: true, writable: true }) : e[r] = t, e;
}
__name(_defineProperty$4, "_defineProperty$4");
function _toPropertyKey$4(t) {
  var i = _toPrimitive$4(t, "string");
  return "symbol" == _typeof$4(i) ? i : i + "";
}
__name(_toPropertyKey$4, "_toPropertyKey$4");
function _toPrimitive$4(t, r) {
  if ("object" != _typeof$4(t) || !t) return t;
  var e = t[Symbol.toPrimitive];
  if (void 0 !== e) {
    var i = e.call(t, r || "default");
    if ("object" != _typeof$4(i)) return i;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return ("string" === r ? String : Number)(t);
}
__name(_toPrimitive$4, "_toPrimitive$4");
var script$8 = {
  name: "Tree",
  "extends": script$2$3,
  inheritAttrs: false,
  emits: ["node-expand", "node-collapse", "update:expandedKeys", "update:selectionKeys", "node-select", "node-unselect", "filter"],
  data: /* @__PURE__ */ __name(function data10() {
    return {
      d_expandedKeys: this.expandedKeys || {},
      filterValue: null
    };
  }, "data"),
  watch: {
    expandedKeys: /* @__PURE__ */ __name(function expandedKeys(newValue) {
      this.d_expandedKeys = newValue;
    }, "expandedKeys")
  },
  methods: {
    onNodeToggle: /* @__PURE__ */ __name(function onNodeToggle(node2) {
      var key = node2.key;
      if (this.d_expandedKeys[key]) {
        delete this.d_expandedKeys[key];
        this.$emit("node-collapse", node2);
      } else {
        this.d_expandedKeys[key] = true;
        this.$emit("node-expand", node2);
      }
      this.d_expandedKeys = _objectSpread$3({}, this.d_expandedKeys);
      this.$emit("update:expandedKeys", this.d_expandedKeys);
    }, "onNodeToggle"),
    onNodeClick: /* @__PURE__ */ __name(function onNodeClick(event) {
      if (this.selectionMode != null && event.node.selectable !== false) {
        var metaSelection = event.nodeTouched ? false : this.metaKeySelection;
        var _selectionKeys = metaSelection ? this.handleSelectionWithMetaKey(event) : this.handleSelectionWithoutMetaKey(event);
        this.$emit("update:selectionKeys", _selectionKeys);
      }
    }, "onNodeClick"),
    onCheckboxChange: /* @__PURE__ */ __name(function onCheckboxChange(event) {
      this.$emit("update:selectionKeys", event.selectionKeys);
      if (event.check) this.$emit("node-select", event.node);
      else this.$emit("node-unselect", event.node);
    }, "onCheckboxChange"),
    handleSelectionWithMetaKey: /* @__PURE__ */ __name(function handleSelectionWithMetaKey(event) {
      var originalEvent = event.originalEvent;
      var node2 = event.node;
      var metaKey = originalEvent.metaKey || originalEvent.ctrlKey;
      var selected2 = this.isNodeSelected(node2);
      var _selectionKeys;
      if (selected2 && metaKey) {
        if (this.isSingleSelectionMode()) {
          _selectionKeys = {};
        } else {
          _selectionKeys = _objectSpread$3({}, this.selectionKeys);
          delete _selectionKeys[node2.key];
        }
        this.$emit("node-unselect", node2);
      } else {
        if (this.isSingleSelectionMode()) {
          _selectionKeys = {};
        } else if (this.isMultipleSelectionMode()) {
          _selectionKeys = !metaKey ? {} : this.selectionKeys ? _objectSpread$3({}, this.selectionKeys) : {};
        }
        _selectionKeys[node2.key] = true;
        this.$emit("node-select", node2);
      }
      return _selectionKeys;
    }, "handleSelectionWithMetaKey"),
    handleSelectionWithoutMetaKey: /* @__PURE__ */ __name(function handleSelectionWithoutMetaKey(event) {
      var node2 = event.node;
      var selected2 = this.isNodeSelected(node2);
      var _selectionKeys;
      if (this.isSingleSelectionMode()) {
        if (selected2) {
          _selectionKeys = {};
          this.$emit("node-unselect", node2);
        } else {
          _selectionKeys = {};
          _selectionKeys[node2.key] = true;
          this.$emit("node-select", node2);
        }
      } else {
        if (selected2) {
          _selectionKeys = _objectSpread$3({}, this.selectionKeys);
          delete _selectionKeys[node2.key];
          this.$emit("node-unselect", node2);
        } else {
          _selectionKeys = this.selectionKeys ? _objectSpread$3({}, this.selectionKeys) : {};
          _selectionKeys[node2.key] = true;
          this.$emit("node-select", node2);
        }
      }
      return _selectionKeys;
    }, "handleSelectionWithoutMetaKey"),
    isSingleSelectionMode: /* @__PURE__ */ __name(function isSingleSelectionMode() {
      return this.selectionMode === "single";
    }, "isSingleSelectionMode"),
    isMultipleSelectionMode: /* @__PURE__ */ __name(function isMultipleSelectionMode() {
      return this.selectionMode === "multiple";
    }, "isMultipleSelectionMode"),
    isNodeSelected: /* @__PURE__ */ __name(function isNodeSelected(node2) {
      return this.selectionMode && this.selectionKeys ? this.selectionKeys[node2.key] === true : false;
    }, "isNodeSelected"),
    isChecked: /* @__PURE__ */ __name(function isChecked(node2) {
      return this.selectionKeys ? this.selectionKeys[node2.key] && this.selectionKeys[node2.key].checked : false;
    }, "isChecked"),
    isNodeLeaf: /* @__PURE__ */ __name(function isNodeLeaf(node2) {
      return node2.leaf === false ? false : !(node2.children && node2.children.length);
    }, "isNodeLeaf"),
    onFilterKeydown: /* @__PURE__ */ __name(function onFilterKeydown(event) {
      if (event.code === "Enter" || event.code === "NumpadEnter") {
        event.preventDefault();
      }
      this.$emit("filter", {
        originalEvent: event,
        value: event.target.value
      });
    }, "onFilterKeydown"),
    findFilteredNodes: /* @__PURE__ */ __name(function findFilteredNodes(node2, paramsWithoutNode) {
      if (node2) {
        var matched = false;
        if (node2.children) {
          var childNodes = _toConsumableArray$2(node2.children);
          node2.children = [];
          var _iterator = _createForOfIteratorHelper(childNodes), _step;
          try {
            for (_iterator.s(); !(_step = _iterator.n()).done; ) {
              var childNode = _step.value;
              var copyChildNode = _objectSpread$3({}, childNode);
              if (this.isFilterMatched(copyChildNode, paramsWithoutNode)) {
                matched = true;
                node2.children.push(copyChildNode);
              }
            }
          } catch (err) {
            _iterator.e(err);
          } finally {
            _iterator.f();
          }
        }
        if (matched) {
          return true;
        }
      }
    }, "findFilteredNodes"),
    isFilterMatched: /* @__PURE__ */ __name(function isFilterMatched(node2, _ref) {
      var searchFields = _ref.searchFields, filterText = _ref.filterText, strict = _ref.strict;
      var matched = false;
      var _iterator2 = _createForOfIteratorHelper(searchFields), _step2;
      try {
        for (_iterator2.s(); !(_step2 = _iterator2.n()).done; ) {
          var field = _step2.value;
          var fieldValue = String(resolveFieldData(node2, field)).toLocaleLowerCase(this.filterLocale);
          if (fieldValue.indexOf(filterText) > -1) {
            matched = true;
          }
        }
      } catch (err) {
        _iterator2.e(err);
      } finally {
        _iterator2.f();
      }
      if (!matched || strict && !this.isNodeLeaf(node2)) {
        matched = this.findFilteredNodes(node2, {
          searchFields,
          filterText,
          strict
        }) || matched;
      }
      return matched;
    }, "isFilterMatched")
  },
  computed: {
    filteredValue: /* @__PURE__ */ __name(function filteredValue() {
      var filteredNodes = [];
      var searchFields = this.filterBy.split(",");
      var filterText = this.filterValue.trim().toLocaleLowerCase(this.filterLocale);
      var strict = this.filterMode === "strict";
      var _iterator3 = _createForOfIteratorHelper(this.value), _step3;
      try {
        for (_iterator3.s(); !(_step3 = _iterator3.n()).done; ) {
          var node2 = _step3.value;
          var _node = _objectSpread$3({}, node2);
          var paramsWithoutNode = {
            searchFields,
            filterText,
            strict
          };
          if (strict && (this.findFilteredNodes(_node, paramsWithoutNode) || this.isFilterMatched(_node, paramsWithoutNode)) || !strict && (this.isFilterMatched(_node, paramsWithoutNode) || this.findFilteredNodes(_node, paramsWithoutNode))) {
            filteredNodes.push(_node);
          }
        }
      } catch (err) {
        _iterator3.e(err);
      } finally {
        _iterator3.f();
      }
      return filteredNodes;
    }, "filteredValue"),
    valueToRender: /* @__PURE__ */ __name(function valueToRender() {
      if (this.filterValue && this.filterValue.trim().length > 0) return this.filteredValue;
      else return this.value;
    }, "valueToRender")
  },
  components: {
    TreeNode: script$1$6,
    InputText: script$m,
    InputIcon: script$H,
    IconField: script$I,
    SearchIcon: script$J,
    SpinnerIcon: script$t
  }
};
var _hoisted_1$i = ["aria-labelledby", "aria-label"];
function render$7(_ctx, _cache, $props, $setup, $data, $options) {
  var _component_SpinnerIcon = resolveComponent("SpinnerIcon");
  var _component_InputText = resolveComponent("InputText");
  var _component_SearchIcon = resolveComponent("SearchIcon");
  var _component_InputIcon = resolveComponent("InputIcon");
  var _component_IconField = resolveComponent("IconField");
  var _component_TreeNode = resolveComponent("TreeNode");
  return openBlock(), createElementBlock("div", mergeProps({
    "class": _ctx.cx("root")
  }, _ctx.ptmi("root")), [_ctx.loading && _ctx.loadingMode === "mask" ? (openBlock(), createElementBlock("div", mergeProps({
    key: 0,
    "class": _ctx.cx("mask")
  }, _ctx.ptm("mask")), [renderSlot(_ctx.$slots, "loadingicon", {
    "class": normalizeClass(_ctx.cx("loadingIcon"))
  }, function() {
    return [_ctx.loadingIcon ? (openBlock(), createElementBlock("i", mergeProps({
      key: 0,
      "class": [_ctx.cx("loadingIcon"), "pi-spin", _ctx.loadingIcon]
    }, _ctx.ptm("loadingIcon")), null, 16)) : (openBlock(), createBlock(_component_SpinnerIcon, mergeProps({
      key: 1,
      spin: "",
      "class": _ctx.cx("loadingIcon")
    }, _ctx.ptm("loadingIcon")), null, 16, ["class"]))];
  })], 16)) : createCommentVNode("", true), _ctx.filter ? (openBlock(), createBlock(_component_IconField, {
    key: 1,
    unstyled: _ctx.unstyled,
    pt: _ctx.ptm("pcFilterContainer")
  }, {
    "default": withCtx(function() {
      return [createVNode(_component_InputText, {
        modelValue: $data.filterValue,
        "onUpdate:modelValue": _cache[0] || (_cache[0] = function($event) {
          return $data.filterValue = $event;
        }),
        autocomplete: "off",
        "class": normalizeClass(_ctx.cx("pcFilter")),
        placeholder: _ctx.filterPlaceholder,
        unstyled: _ctx.unstyled,
        onKeydown: $options.onFilterKeydown,
        pt: _ctx.ptm("pcFilter")
      }, null, 8, ["modelValue", "class", "placeholder", "unstyled", "onKeydown", "pt"]), createVNode(_component_InputIcon, {
        unstyled: _ctx.unstyled,
        pt: _ctx.ptm("pcFilterIconContainer")
      }, {
        "default": withCtx(function() {
          return [renderSlot(_ctx.$slots, _ctx.$slots.filtericon ? "filtericon" : "searchicon", {
            "class": normalizeClass(_ctx.cx("filterIcon"))
          }, function() {
            return [createVNode(_component_SearchIcon, mergeProps({
              "class": _ctx.cx("filterIcon")
            }, _ctx.ptm("filterIcon")), null, 16, ["class"])];
          })];
        }),
        _: 3
      }, 8, ["unstyled", "pt"])];
    }),
    _: 3
  }, 8, ["unstyled", "pt"])) : createCommentVNode("", true), createBaseVNode("div", mergeProps({
    "class": _ctx.cx("wrapper"),
    style: {
      maxHeight: _ctx.scrollHeight
    }
  }, _ctx.ptm("wrapper")), [createBaseVNode("ul", mergeProps({
    "class": _ctx.cx("rootChildren"),
    role: "tree",
    "aria-labelledby": _ctx.ariaLabelledby,
    "aria-label": _ctx.ariaLabel
  }, _ctx.ptm("rootChildren")), [(openBlock(true), createElementBlock(Fragment, null, renderList($options.valueToRender, function(node2, index2) {
    return openBlock(), createBlock(_component_TreeNode, {
      key: node2.key,
      node: node2,
      templates: _ctx.$slots,
      level: _ctx.level + 1,
      index: index2,
      expandedKeys: $data.d_expandedKeys,
      onNodeToggle: $options.onNodeToggle,
      onNodeClick: $options.onNodeClick,
      selectionMode: _ctx.selectionMode,
      selectionKeys: _ctx.selectionKeys,
      onCheckboxChange: $options.onCheckboxChange,
      loadingMode: _ctx.loadingMode,
      unstyled: _ctx.unstyled,
      pt: _ctx.pt
    }, null, 8, ["node", "templates", "level", "index", "expandedKeys", "onNodeToggle", "onNodeClick", "selectionMode", "selectionKeys", "onCheckboxChange", "loadingMode", "unstyled", "pt"]);
  }), 128))], 16, _hoisted_1$i)], 16)], 16);
}
__name(render$7, "render$7");
script$8.render = render$7;
const _withScopeId$a = /* @__PURE__ */ __name((n) => (pushScopeId("data-v-633e27ab"), n = n(), popScopeId(), n), "_withScopeId$a");
const _hoisted_1$h = { class: "node-content" };
const _hoisted_2$c = { class: "node-label" };
const _hoisted_3$8 = { class: "node-actions" };
const _sfc_main$m = /* @__PURE__ */ defineComponent({
  __name: "TreeExplorerTreeNode",
  props: {
    node: {}
  },
  emits: ["itemDropped", "dragStart", "dragEnd"],
  setup(__props, { emit: __emit }) {
    const props = __props;
    const emit = __emit;
    const labelEditable = computed(() => !!props.node.handleRename);
    const renameEditingNode = inject("renameEditingNode");
    const isEditing = computed(
      () => labelEditable.value && renameEditingNode.value?.key === props.node.key
    );
    const errorHandling = useErrorHandling();
    const handleRename = errorHandling.wrapWithErrorHandlingAsync(
      async (newName) => {
        await props.node.handleRename(props.node, newName);
        renameEditingNode.value = null;
      },
      props.node.handleError
    );
    const container = ref(null);
    const canDrop = ref(false);
    const treeNodeElement = ref(null);
    let dropTargetCleanup = /* @__PURE__ */ __name(() => {
    }, "dropTargetCleanup");
    let draggableCleanup = /* @__PURE__ */ __name(() => {
    }, "draggableCleanup");
    onMounted(() => {
      treeNodeElement.value = container.value?.closest(
        ".p-tree-node-content"
      );
      if (props.node.droppable) {
        dropTargetCleanup = dropTargetForElements({
          element: treeNodeElement.value,
          onDrop: /* @__PURE__ */ __name(async (event) => {
            const dndData = event.source.data;
            if (dndData.type === "tree-explorer-node") {
              await props.node.handleDrop?.(props.node, dndData);
              canDrop.value = false;
              emit("itemDropped", props.node, dndData.data);
            }
          }, "onDrop"),
          onDragEnter: /* @__PURE__ */ __name((event) => {
            const dndData = event.source.data;
            if (dndData.type === "tree-explorer-node") {
              canDrop.value = true;
            }
          }, "onDragEnter"),
          onDragLeave: /* @__PURE__ */ __name(() => {
            canDrop.value = false;
          }, "onDragLeave")
        });
      }
      if (props.node.draggable) {
        draggableCleanup = draggable({
          element: treeNodeElement.value,
          getInitialData() {
            return {
              type: "tree-explorer-node",
              data: props.node
            };
          },
          onDragStart: /* @__PURE__ */ __name(() => emit("dragStart", props.node), "onDragStart"),
          onDrop: /* @__PURE__ */ __name(() => emit("dragEnd", props.node), "onDrop")
        });
      }
    });
    onUnmounted(() => {
      dropTargetCleanup();
      draggableCleanup();
    });
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", {
        class: normalizeClass([
          "tree-node",
          {
            "can-drop": canDrop.value,
            "tree-folder": !props.node.leaf,
            "tree-leaf": props.node.leaf
          }
        ]),
        ref_key: "container",
        ref: container
      }, [
        createBaseVNode("div", _hoisted_1$h, [
          createBaseVNode("span", _hoisted_2$c, [
            renderSlot(_ctx.$slots, "before-label", {
              node: props.node
            }, void 0, true),
            createVNode(EditableText, {
              modelValue: _ctx.node.label,
              isEditing: isEditing.value,
              onEdit: unref(handleRename)
            }, null, 8, ["modelValue", "isEditing", "onEdit"]),
            renderSlot(_ctx.$slots, "after-label", {
              node: props.node
            }, void 0, true)
          ]),
          !props.node.leaf ? (openBlock(), createBlock(unref(script$n), {
            key: 0,
            value: props.node.badgeText ?? props.node.totalLeaves,
            severity: "secondary",
            class: "leaf-count-badge"
          }, null, 8, ["value"])) : createCommentVNode("", true)
        ]),
        createBaseVNode("div", _hoisted_3$8, [
          renderSlot(_ctx.$slots, "actions", {
            node: props.node
          }, void 0, true)
        ])
      ], 2);
    };
  }
});
const TreeExplorerTreeNode = /* @__PURE__ */ _export_sfc(_sfc_main$m, [["__scopeId", "data-v-633e27ab"]]);
const _sfc_main$l = /* @__PURE__ */ defineComponent({
  __name: "TreeExplorer",
  props: /* @__PURE__ */ mergeModels({
    roots: {},
    class: {}
  }, {
    "expandedKeys": {},
    "expandedKeysModifiers": {},
    "selectionKeys": {},
    "selectionKeysModifiers": {}
  }),
  emits: /* @__PURE__ */ mergeModels(["nodeClick", "nodeDelete", "contextMenu"], ["update:expandedKeys", "update:selectionKeys"]),
  setup(__props, { expose: __expose, emit: __emit }) {
    const expandedKeys2 = useModel(__props, "expandedKeys");
    provide("expandedKeys", expandedKeys2);
    const selectionKeys = useModel(__props, "selectionKeys");
    provide("selectionKeys", selectionKeys);
    const storeSelectionKeys = selectionKeys.value !== void 0;
    const props = __props;
    const emit = __emit;
    const renderedRoots = computed(() => {
      return props.roots.map(fillNodeInfo);
    });
    const getTreeNodeIcon = /* @__PURE__ */ __name((node2) => {
      if (node2.getIcon) {
        const icon = node2.getIcon(node2);
        if (icon) {
          return icon;
        }
      } else if (node2.icon) {
        return node2.icon;
      }
      if (node2.leaf) {
        return "pi pi-file";
      }
      const isExpanded = expandedKeys2.value[node2.key];
      return isExpanded ? "pi pi-folder-open" : "pi pi-folder";
    }, "getTreeNodeIcon");
    const fillNodeInfo = /* @__PURE__ */ __name((node2) => {
      const children = node2.children?.map(fillNodeInfo);
      const totalLeaves = node2.leaf ? 1 : children.reduce((acc, child) => acc + child.totalLeaves, 0);
      return {
        ...node2,
        icon: getTreeNodeIcon(node2),
        children,
        type: node2.leaf ? "node" : "folder",
        totalLeaves,
        badgeText: node2.getBadgeText ? node2.getBadgeText(node2) : null
      };
    }, "fillNodeInfo");
    const onNodeContentClick = /* @__PURE__ */ __name(async (e, node2) => {
      if (!storeSelectionKeys) {
        selectionKeys.value = {};
      }
      if (node2.handleClick) {
        await node2.handleClick(node2, e);
      }
      emit("nodeClick", node2, e);
    }, "onNodeContentClick");
    const menu = ref(null);
    const menuTargetNode = ref(null);
    provide("menuTargetNode", menuTargetNode);
    const extraMenuItems = computed(() => {
      return menuTargetNode.value?.contextMenuItems ? typeof menuTargetNode.value.contextMenuItems === "function" ? menuTargetNode.value.contextMenuItems(menuTargetNode.value) : menuTargetNode.value.contextMenuItems : [];
    });
    const renameEditingNode = ref(null);
    provide("renameEditingNode", renameEditingNode);
    const { t } = useI18n();
    const renameCommand = /* @__PURE__ */ __name((node2) => {
      renameEditingNode.value = node2;
    }, "renameCommand");
    const deleteCommand = /* @__PURE__ */ __name(async (node2) => {
      await node2.handleDelete?.(node2);
      emit("nodeDelete", node2);
    }, "deleteCommand");
    const menuItems = computed(
      () => [
        {
          label: t("rename"),
          icon: "pi pi-file-edit",
          command: /* @__PURE__ */ __name(() => renameCommand(menuTargetNode.value), "command"),
          visible: menuTargetNode.value?.handleRename !== void 0
        },
        {
          label: t("delete"),
          icon: "pi pi-trash",
          command: /* @__PURE__ */ __name(() => deleteCommand(menuTargetNode.value), "command"),
          visible: menuTargetNode.value?.handleDelete !== void 0,
          isAsync: true
          // The delete command can be async
        },
        ...extraMenuItems.value
      ].map((menuItem) => ({
        ...menuItem,
        command: wrapCommandWithErrorHandler(menuItem.command, {
          isAsync: menuItem.isAsync ?? false
        })
      }))
    );
    const handleContextMenu = /* @__PURE__ */ __name((node2, e) => {
      menuTargetNode.value = node2;
      emit("contextMenu", node2, e);
      if (menuItems.value.filter((item4) => item4.visible).length > 0) {
        menu.value?.show(e);
      }
    }, "handleContextMenu");
    const errorHandling = useErrorHandling();
    const wrapCommandWithErrorHandler = /* @__PURE__ */ __name((command, { isAsync = false }) => {
      return isAsync ? errorHandling.wrapWithErrorHandlingAsync(
        command,
        menuTargetNode.value?.handleError
      ) : errorHandling.wrapWithErrorHandling(
        command,
        menuTargetNode.value?.handleError
      );
    }, "wrapCommandWithErrorHandler");
    __expose({
      renameCommand,
      deleteCommand
    });
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock(Fragment, null, [
        createVNode(unref(script$8), {
          class: normalizeClass(["tree-explorer", props.class]),
          expandedKeys: expandedKeys2.value,
          "onUpdate:expandedKeys": _cache[0] || (_cache[0] = ($event) => expandedKeys2.value = $event),
          selectionKeys: selectionKeys.value,
          "onUpdate:selectionKeys": _cache[1] || (_cache[1] = ($event) => selectionKeys.value = $event),
          value: renderedRoots.value,
          selectionMode: "single",
          pt: {
            nodeLabel: "tree-explorer-node-label",
            nodeContent: /* @__PURE__ */ __name(({ props: props2 }) => ({
              onClick: /* @__PURE__ */ __name((e) => onNodeContentClick(e, props2.node), "onClick"),
              onContextmenu: /* @__PURE__ */ __name((e) => handleContextMenu(props2.node, e), "onContextmenu")
            }), "nodeContent"),
            nodeToggleButton: /* @__PURE__ */ __name(() => ({
              onClick: /* @__PURE__ */ __name((e) => {
                e.stopImmediatePropagation();
              }, "onClick")
            }), "nodeToggleButton")
          }
        }, {
          folder: withCtx(({ node: node2 }) => [
            renderSlot(_ctx.$slots, "folder", { node: node2 }, () => [
              createVNode(TreeExplorerTreeNode, { node: node2 }, null, 8, ["node"])
            ], true)
          ]),
          node: withCtx(({ node: node2 }) => [
            renderSlot(_ctx.$slots, "node", { node: node2 }, () => [
              createVNode(TreeExplorerTreeNode, { node: node2 }, null, 8, ["node"])
            ], true)
          ]),
          _: 3
        }, 8, ["class", "expandedKeys", "selectionKeys", "value", "pt"]),
        createVNode(unref(script$d), {
          ref_key: "menu",
          ref: menu,
          model: menuItems.value
        }, null, 8, ["model"])
      ], 64);
    };
  }
});
const TreeExplorer = /* @__PURE__ */ _export_sfc(_sfc_main$l, [["__scopeId", "data-v-bd7bae90"]]);
const _sfc_main$k = /* @__PURE__ */ defineComponent({
  __name: "NodeTreeLeaf",
  props: {
    node: {}
  },
  emits: ["toggle-bookmark"],
  setup(__props, { emit: __emit }) {
    const props = __props;
    const nodeDef = computed(() => props.node.data);
    const nodeBookmarkStore = useNodeBookmarkStore();
    const isBookmarked = computed(
      () => nodeBookmarkStore.isBookmarked(nodeDef.value)
    );
    const settingStore = useSettingStore();
    const sidebarLocation = computed(
      () => settingStore.get("Comfy.Sidebar.Location")
    );
    const emit = __emit;
    const toggleBookmark = /* @__PURE__ */ __name(() => {
      nodeBookmarkStore.toggleBookmark(nodeDef.value);
    }, "toggleBookmark");
    const previewRef = ref(null);
    const nodePreviewStyle = ref({
      position: "absolute",
      top: "0px",
      left: "0px"
    });
    const handleNodeHover = /* @__PURE__ */ __name(async () => {
      const hoverTarget = nodeContentElement.value;
      const targetRect = hoverTarget.getBoundingClientRect();
      const previewHeight = previewRef.value?.$el.offsetHeight || 0;
      const availableSpaceBelow = window.innerHeight - targetRect.bottom;
      nodePreviewStyle.value.top = previewHeight > availableSpaceBelow ? `${Math.max(0, targetRect.top - (previewHeight - availableSpaceBelow) - 20)}px` : `${targetRect.top - 40}px`;
      if (sidebarLocation.value === "left") {
        nodePreviewStyle.value.left = `${targetRect.right}px`;
      } else {
        nodePreviewStyle.value.left = `${targetRect.left - 400}px`;
      }
    }, "handleNodeHover");
    const container = ref(null);
    const nodeContentElement = ref(null);
    const isHovered = ref(false);
    const handleMouseEnter = /* @__PURE__ */ __name(async () => {
      isHovered.value = true;
      await nextTick();
      handleNodeHover();
    }, "handleMouseEnter");
    const handleMouseLeave = /* @__PURE__ */ __name(() => {
      isHovered.value = false;
    }, "handleMouseLeave");
    onMounted(() => {
      nodeContentElement.value = container.value?.closest(".p-tree-node-content");
      nodeContentElement.value?.addEventListener("mouseenter", handleMouseEnter);
      nodeContentElement.value?.addEventListener("mouseleave", handleMouseLeave);
    });
    onUnmounted(() => {
      nodeContentElement.value?.removeEventListener("mouseenter", handleMouseEnter);
      nodeContentElement.value?.removeEventListener("mouseleave", handleMouseLeave);
    });
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", {
        ref_key: "container",
        ref: container,
        class: "node-lib-node-container"
      }, [
        createVNode(TreeExplorerTreeNode, { node: _ctx.node }, {
          "before-label": withCtx(() => [
            nodeDef.value.experimental ? (openBlock(), createBlock(unref(script$w), {
              key: 0,
              value: _ctx.$t("experimental"),
              severity: "primary"
            }, null, 8, ["value"])) : createCommentVNode("", true),
            nodeDef.value.deprecated ? (openBlock(), createBlock(unref(script$w), {
              key: 1,
              value: _ctx.$t("deprecated"),
              severity: "danger"
            }, null, 8, ["value"])) : createCommentVNode("", true)
          ]),
          actions: withCtx(() => [
            createVNode(unref(script$o), {
              class: "bookmark-button",
              size: "small",
              icon: isBookmarked.value ? "pi pi-bookmark-fill" : "pi pi-bookmark",
              text: "",
              severity: "secondary",
              onClick: withModifiers(toggleBookmark, ["stop"])
            }, null, 8, ["icon"])
          ]),
          _: 1
        }, 8, ["node"]),
        isHovered.value ? (openBlock(), createBlock(Teleport, {
          key: 0,
          to: "#node-library-node-preview-container"
        }, [
          createBaseVNode("div", {
            class: "node-lib-node-preview",
            style: normalizeStyle(nodePreviewStyle.value)
          }, [
            createVNode(NodePreview, {
              ref_key: "previewRef",
              ref: previewRef,
              nodeDef: nodeDef.value
            }, null, 8, ["nodeDef"])
          ], 4)
        ])) : createCommentVNode("", true)
      ], 512);
    };
  }
});
const NodeTreeLeaf = /* @__PURE__ */ _export_sfc(_sfc_main$k, [["__scopeId", "data-v-90dfee08"]]);
const _sfc_main$j = /* @__PURE__ */ defineComponent({
  __name: "NodeTreeFolder",
  props: {
    node: {}
  },
  setup(__props) {
    const props = __props;
    const nodeBookmarkStore = useNodeBookmarkStore();
    const customization = computed(() => {
      return nodeBookmarkStore.bookmarksCustomization[props.node.data.nodePath];
    });
    const treeNodeElement = ref(null);
    const iconElement = ref(null);
    let stopWatchCustomization = null;
    const container = ref(null);
    onMounted(() => {
      treeNodeElement.value = container.value?.closest(
        ".p-tree-node-content"
      );
      iconElement.value = treeNodeElement.value.querySelector(
        ":scope > .p-tree-node-icon"
      );
      updateIconColor();
      stopWatchCustomization = watch(customization, updateIconColor, { deep: true });
    });
    const updateIconColor = /* @__PURE__ */ __name(() => {
      if (iconElement.value && customization.value) {
        iconElement.value.style.color = customization.value.color;
      }
    }, "updateIconColor");
    onUnmounted(() => {
      if (stopWatchCustomization) {
        stopWatchCustomization();
      }
    });
    const expandedKeys2 = inject("expandedKeys");
    const handleItemDrop = /* @__PURE__ */ __name((node2) => {
      expandedKeys2.value[node2.key] = true;
    }, "handleItemDrop");
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", {
        ref_key: "container",
        ref: container,
        class: "node-lib-node-container"
      }, [
        createVNode(TreeExplorerTreeNode, {
          node: _ctx.node,
          onItemDropped: handleItemDrop
        }, null, 8, ["node"])
      ], 512);
    };
  }
});
var theme$5 = /* @__PURE__ */ __name(function theme13(_ref) {
  var dt = _ref.dt;
  return "\n.p-colorpicker {\n    display: inline-block;\n    position: relative;\n}\n\n.p-colorpicker-dragging {\n    cursor: pointer;\n}\n\n.p-colorpicker-preview {\n    width: ".concat(dt("colorpicker.preview.width"), ";\n    height: ").concat(dt("colorpicker.preview.height"), ";\n    padding: 0;\n    border: 0 none;\n    border-radius: ").concat(dt("colorpicker.preview.border.radius"), ";\n    transition: background ").concat(dt("colorpicker.transition.duration"), ", color ").concat(dt("colorpicker.transition.duration"), ", border-color ").concat(dt("colorpicker.transition.duration"), ", outline-color ").concat(dt("colorpicker.transition.duration"), ", box-shadow ").concat(dt("colorpicker.transition.duration"), ";\n    outline-color: transparent;\n    cursor: pointer;\n}\n\n.p-colorpicker-preview:enabled:focus-visible {\n    border-color: ").concat(dt("colorpicker.preview.focus.border.color"), ";\n    box-shadow: ").concat(dt("colorpicker.preview.focus.ring.shadow"), ";\n    outline: ").concat(dt("colorpicker.preview.focus.ring.width"), " ").concat(dt("colorpicker.preview.focus.ring.style"), " ").concat(dt("colorpicker.preview.focus.ring.color"), ";\n    outline-offset: ").concat(dt("colorpicker.preview.focus.ring.offset"), ";\n}\n\n.p-colorpicker-panel {\n    background: ").concat(dt("colorpicker.panel.background"), ";\n    border: 1px solid ").concat(dt("colorpicker.panel.border.color"), ";\n    border-radius: ").concat(dt("colorpicker.panel.border.radius"), ";\n    box-shadow: ").concat(dt("colorpicker.panel.shadow"), ";\n    width: 193px;\n    height: 166px;\n    position: absolute;\n    top: 0;\n    left: 0;\n}\n\n.p-colorpicker-panel-inline {\n    box-shadow: none;\n    position: static;\n}\n\n.p-colorpicker-content {\n    position: relative;\n}\n\n.p-colorpicker-color-selector {\n    width: 150px;\n    height: 150px;\n    top: 8px;\n    left: 8px;\n    position: absolute;\n}\n\n.p-colorpicker-color-background {\n    width: 100%;\n    height: 100%;\n    background: linear-gradient(to top, #000 0%, rgba(0, 0, 0, 0) 100%), linear-gradient(to right, #fff 0%, rgba(255, 255, 255, 0) 100%);\n}\n\n.p-colorpicker-color-handle {\n    position: absolute;\n    top: 0px;\n    left: 150px;\n    border-radius: 100%;\n    width: 10px;\n    height: 10px;\n    border-width: 1px;\n    border-style: solid;\n    margin: -5px 0 0 -5px;\n    cursor: pointer;\n    opacity: 0.85;\n    border-color: ").concat(dt("colorpicker.handle.color"), ";\n}\n\n.p-colorpicker-hue {\n    width: 17px;\n    height: 150px;\n    top: 8px;\n    left: 167px;\n    position: absolute;\n    opacity: 0.85;\n    background: linear-gradient(0deg,\n        red 0,\n        #ff0 17%,\n        #0f0 33%,\n        #0ff 50%,\n        #00f 67%,\n        #f0f 83%,\n        red);\n}\n\n.p-colorpicker-hue-handle {\n    position: absolute;\n    top: 150px;\n    left: 0px;\n    width: 21px;\n    margin-left: -2px;\n    margin-top: -5px;\n    height: 10px;\n    border-width: 2px;\n    border-style: solid;\n    opacity: 0.85;\n    cursor: pointer;\n    border-color: ").concat(dt("colorpicker.handle.color"), ";\n}\n");
}, "theme");
var classes$5 = {
  root: "p-colorpicker p-component",
  preview: /* @__PURE__ */ __name(function preview(_ref2) {
    var props = _ref2.props;
    return ["p-colorpicker-preview", {
      "p-disabled": props.disabled
    }];
  }, "preview"),
  panel: /* @__PURE__ */ __name(function panel(_ref3) {
    var props = _ref3.props;
    return ["p-colorpicker-panel", {
      "p-colorpicker-panel-inline": props.inline,
      "p-disabled": props.disabled
    }];
  }, "panel"),
  colorSelector: "p-colorpicker-color-selector",
  colorBackground: "p-colorpicker-color-background",
  colorHandle: "p-colorpicker-color-handle",
  hue: "p-colorpicker-hue",
  hueHandle: "p-colorpicker-hue-handle"
};
var ColorPickerStyle = BaseStyle.extend({
  name: "colorpicker",
  theme: theme$5,
  classes: classes$5
});
var script$1$5 = {
  name: "BaseColorPicker",
  "extends": script$p,
  props: {
    modelValue: {
      type: null,
      "default": null
    },
    defaultColor: {
      type: null,
      "default": "ff0000"
    },
    inline: {
      type: Boolean,
      "default": false
    },
    format: {
      type: String,
      "default": "hex"
    },
    disabled: {
      type: Boolean,
      "default": false
    },
    tabindex: {
      type: String,
      "default": null
    },
    autoZIndex: {
      type: Boolean,
      "default": true
    },
    baseZIndex: {
      type: Number,
      "default": 0
    },
    appendTo: {
      type: [String, Object],
      "default": "body"
    },
    inputId: {
      type: String,
      "default": null
    },
    panelClass: null
  },
  style: ColorPickerStyle,
  provide: /* @__PURE__ */ __name(function provide15() {
    return {
      $pcColorPicker: this,
      $parentInstance: this
    };
  }, "provide")
};
var script$7 = {
  name: "ColorPicker",
  "extends": script$1$5,
  inheritAttrs: false,
  emits: ["update:modelValue", "change", "show", "hide"],
  data: /* @__PURE__ */ __name(function data11() {
    return {
      overlayVisible: false
    };
  }, "data"),
  hsbValue: null,
  outsideClickListener: null,
  documentMouseMoveListener: null,
  documentMouseUpListener: null,
  scrollHandler: null,
  resizeListener: null,
  hueDragging: null,
  colorDragging: null,
  selfUpdate: null,
  picker: null,
  colorSelector: null,
  colorHandle: null,
  hueView: null,
  hueHandle: null,
  watch: {
    modelValue: {
      immediate: true,
      handler: /* @__PURE__ */ __name(function handler2(newValue) {
        this.hsbValue = this.toHSB(newValue);
        if (this.selfUpdate) this.selfUpdate = false;
        else this.updateUI();
      }, "handler")
    }
  },
  beforeUnmount: /* @__PURE__ */ __name(function beforeUnmount9() {
    this.unbindOutsideClickListener();
    this.unbindDragListeners();
    this.unbindResizeListener();
    if (this.scrollHandler) {
      this.scrollHandler.destroy();
      this.scrollHandler = null;
    }
    if (this.picker && this.autoZIndex) {
      ZIndex.clear(this.picker);
    }
    this.clearRefs();
  }, "beforeUnmount"),
  mounted: /* @__PURE__ */ __name(function mounted10() {
    this.updateUI();
  }, "mounted"),
  methods: {
    pickColor: /* @__PURE__ */ __name(function pickColor(event) {
      var rect = this.colorSelector.getBoundingClientRect();
      var top = rect.top + (window.pageYOffset || document.documentElement.scrollTop || document.body.scrollTop || 0);
      var left = rect.left + document.body.scrollLeft;
      var saturation = Math.floor(100 * Math.max(0, Math.min(150, (event.pageX || event.changedTouches[0].pageX) - left)) / 150);
      var brightness = Math.floor(100 * (150 - Math.max(0, Math.min(150, (event.pageY || event.changedTouches[0].pageY) - top))) / 150);
      this.hsbValue = this.validateHSB({
        h: this.hsbValue.h,
        s: saturation,
        b: brightness
      });
      this.selfUpdate = true;
      this.updateColorHandle();
      this.updateInput();
      this.updateModel(event);
    }, "pickColor"),
    pickHue: /* @__PURE__ */ __name(function pickHue(event) {
      var top = this.hueView.getBoundingClientRect().top + (window.pageYOffset || document.documentElement.scrollTop || document.body.scrollTop || 0);
      this.hsbValue = this.validateHSB({
        h: Math.floor(360 * (150 - Math.max(0, Math.min(150, (event.pageY || event.changedTouches[0].pageY) - top))) / 150),
        s: 100,
        b: 100
      });
      this.selfUpdate = true;
      this.updateColorSelector();
      this.updateHue();
      this.updateModel(event);
      this.updateInput();
    }, "pickHue"),
    updateModel: /* @__PURE__ */ __name(function updateModel2(event) {
      var value = this.modelValue;
      switch (this.format) {
        case "hex":
          value = this.HSBtoHEX(this.hsbValue);
          break;
        case "rgb":
          value = this.HSBtoRGB(this.hsbValue);
          break;
        case "hsb":
          value = this.hsbValue;
          break;
      }
      this.$emit("update:modelValue", value);
      this.$emit("change", {
        event,
        value
      });
    }, "updateModel"),
    updateColorSelector: /* @__PURE__ */ __name(function updateColorSelector() {
      if (this.colorSelector) {
        var hsbValue = this.validateHSB({
          h: this.hsbValue.h,
          s: 100,
          b: 100
        });
        this.colorSelector.style.backgroundColor = "#" + this.HSBtoHEX(hsbValue);
      }
    }, "updateColorSelector"),
    updateColorHandle: /* @__PURE__ */ __name(function updateColorHandle() {
      if (this.colorHandle) {
        this.colorHandle.style.left = Math.floor(150 * this.hsbValue.s / 100) + "px";
        this.colorHandle.style.top = Math.floor(150 * (100 - this.hsbValue.b) / 100) + "px";
      }
    }, "updateColorHandle"),
    updateHue: /* @__PURE__ */ __name(function updateHue() {
      if (this.hueHandle) {
        this.hueHandle.style.top = Math.floor(150 - 150 * this.hsbValue.h / 360) + "px";
      }
    }, "updateHue"),
    updateInput: /* @__PURE__ */ __name(function updateInput() {
      if (this.$refs.input) {
        this.$refs.input.style.backgroundColor = "#" + this.HSBtoHEX(this.hsbValue);
      }
    }, "updateInput"),
    updateUI: /* @__PURE__ */ __name(function updateUI() {
      this.updateHue();
      this.updateColorHandle();
      this.updateInput();
      this.updateColorSelector();
    }, "updateUI"),
    validateHSB: /* @__PURE__ */ __name(function validateHSB(hsb) {
      return {
        h: Math.min(360, Math.max(0, hsb.h)),
        s: Math.min(100, Math.max(0, hsb.s)),
        b: Math.min(100, Math.max(0, hsb.b))
      };
    }, "validateHSB"),
    validateRGB: /* @__PURE__ */ __name(function validateRGB(rgb) {
      return {
        r: Math.min(255, Math.max(0, rgb.r)),
        g: Math.min(255, Math.max(0, rgb.g)),
        b: Math.min(255, Math.max(0, rgb.b))
      };
    }, "validateRGB"),
    validateHEX: /* @__PURE__ */ __name(function validateHEX(hex) {
      var len = 6 - hex.length;
      if (len > 0) {
        var o = [];
        for (var i = 0; i < len; i++) {
          o.push("0");
        }
        o.push(hex);
        hex = o.join("");
      }
      return hex;
    }, "validateHEX"),
    HEXtoRGB: /* @__PURE__ */ __name(function HEXtoRGB(hex) {
      var hexValue = parseInt(hex.indexOf("#") > -1 ? hex.substring(1) : hex, 16);
      return {
        r: hexValue >> 16,
        g: (hexValue & 65280) >> 8,
        b: hexValue & 255
      };
    }, "HEXtoRGB"),
    HEXtoHSB: /* @__PURE__ */ __name(function HEXtoHSB(hex) {
      return this.RGBtoHSB(this.HEXtoRGB(hex));
    }, "HEXtoHSB"),
    RGBtoHSB: /* @__PURE__ */ __name(function RGBtoHSB(rgb) {
      var hsb = {
        h: 0,
        s: 0,
        b: 0
      };
      var min = Math.min(rgb.r, rgb.g, rgb.b);
      var max = Math.max(rgb.r, rgb.g, rgb.b);
      var delta = max - min;
      hsb.b = max;
      hsb.s = max !== 0 ? 255 * delta / max : 0;
      if (hsb.s !== 0) {
        if (rgb.r === max) {
          hsb.h = (rgb.g - rgb.b) / delta;
        } else if (rgb.g === max) {
          hsb.h = 2 + (rgb.b - rgb.r) / delta;
        } else {
          hsb.h = 4 + (rgb.r - rgb.g) / delta;
        }
      } else {
        hsb.h = -1;
      }
      hsb.h *= 60;
      if (hsb.h < 0) {
        hsb.h += 360;
      }
      hsb.s *= 100 / 255;
      hsb.b *= 100 / 255;
      return hsb;
    }, "RGBtoHSB"),
    HSBtoRGB: /* @__PURE__ */ __name(function HSBtoRGB(hsb) {
      var rgb = {
        r: null,
        g: null,
        b: null
      };
      var h = Math.round(hsb.h);
      var s = Math.round(hsb.s * 255 / 100);
      var v = Math.round(hsb.b * 255 / 100);
      if (s === 0) {
        rgb = {
          r: v,
          g: v,
          b: v
        };
      } else {
        var t1 = v;
        var t2 = (255 - s) * v / 255;
        var t3 = (t1 - t2) * (h % 60) / 60;
        if (h === 360) h = 0;
        if (h < 60) {
          rgb.r = t1;
          rgb.b = t2;
          rgb.g = t2 + t3;
        } else if (h < 120) {
          rgb.g = t1;
          rgb.b = t2;
          rgb.r = t1 - t3;
        } else if (h < 180) {
          rgb.g = t1;
          rgb.r = t2;
          rgb.b = t2 + t3;
        } else if (h < 240) {
          rgb.b = t1;
          rgb.r = t2;
          rgb.g = t1 - t3;
        } else if (h < 300) {
          rgb.b = t1;
          rgb.g = t2;
          rgb.r = t2 + t3;
        } else if (h < 360) {
          rgb.r = t1;
          rgb.g = t2;
          rgb.b = t1 - t3;
        } else {
          rgb.r = 0;
          rgb.g = 0;
          rgb.b = 0;
        }
      }
      return {
        r: Math.round(rgb.r),
        g: Math.round(rgb.g),
        b: Math.round(rgb.b)
      };
    }, "HSBtoRGB"),
    RGBtoHEX: /* @__PURE__ */ __name(function RGBtoHEX(rgb) {
      var hex = [rgb.r.toString(16), rgb.g.toString(16), rgb.b.toString(16)];
      for (var key in hex) {
        if (hex[key].length === 1) {
          hex[key] = "0" + hex[key];
        }
      }
      return hex.join("");
    }, "RGBtoHEX"),
    HSBtoHEX: /* @__PURE__ */ __name(function HSBtoHEX(hsb) {
      return this.RGBtoHEX(this.HSBtoRGB(hsb));
    }, "HSBtoHEX"),
    toHSB: /* @__PURE__ */ __name(function toHSB(value) {
      var hsb;
      if (value) {
        switch (this.format) {
          case "hex":
            hsb = this.HEXtoHSB(value);
            break;
          case "rgb":
            hsb = this.RGBtoHSB(value);
            break;
          case "hsb":
            hsb = value;
            break;
        }
      } else {
        hsb = this.HEXtoHSB(this.defaultColor);
      }
      return hsb;
    }, "toHSB"),
    onOverlayEnter: /* @__PURE__ */ __name(function onOverlayEnter2(el) {
      this.updateUI();
      this.alignOverlay();
      this.bindOutsideClickListener();
      this.bindScrollListener();
      this.bindResizeListener();
      if (this.autoZIndex) {
        ZIndex.set("overlay", el, this.baseZIndex, this.$primevue.config.zIndex.overlay);
      }
      this.$emit("show");
    }, "onOverlayEnter"),
    onOverlayLeave: /* @__PURE__ */ __name(function onOverlayLeave2() {
      this.unbindOutsideClickListener();
      this.unbindScrollListener();
      this.unbindResizeListener();
      this.clearRefs();
      this.$emit("hide");
    }, "onOverlayLeave"),
    onOverlayAfterLeave: /* @__PURE__ */ __name(function onOverlayAfterLeave2(el) {
      if (this.autoZIndex) {
        ZIndex.clear(el);
      }
    }, "onOverlayAfterLeave"),
    alignOverlay: /* @__PURE__ */ __name(function alignOverlay4() {
      if (this.appendTo === "self") relativePosition(this.picker, this.$refs.input);
      else absolutePosition(this.picker, this.$refs.input);
    }, "alignOverlay"),
    onInputClick: /* @__PURE__ */ __name(function onInputClick() {
      if (this.disabled) {
        return;
      }
      this.overlayVisible = !this.overlayVisible;
    }, "onInputClick"),
    onInputKeydown: /* @__PURE__ */ __name(function onInputKeydown(event) {
      switch (event.code) {
        case "Space":
          this.overlayVisible = !this.overlayVisible;
          event.preventDefault();
          break;
        case "Escape":
        case "Tab":
          this.overlayVisible = false;
          break;
      }
    }, "onInputKeydown"),
    onColorMousedown: /* @__PURE__ */ __name(function onColorMousedown(event) {
      if (this.disabled) {
        return;
      }
      this.bindDragListeners();
      this.onColorDragStart(event);
    }, "onColorMousedown"),
    onColorDragStart: /* @__PURE__ */ __name(function onColorDragStart(event) {
      if (this.disabled) {
        return;
      }
      this.colorDragging = true;
      this.pickColor(event);
      this.$el.setAttribute("p-colorpicker-dragging", "true");
      !this.isUnstyled && addClass(this.$el, "p-colorpicker-dragging");
      event.preventDefault();
    }, "onColorDragStart"),
    onDrag: /* @__PURE__ */ __name(function onDrag(event) {
      if (this.colorDragging) {
        this.pickColor(event);
        event.preventDefault();
      }
      if (this.hueDragging) {
        this.pickHue(event);
        event.preventDefault();
      }
    }, "onDrag"),
    onDragEnd: /* @__PURE__ */ __name(function onDragEnd() {
      this.colorDragging = false;
      this.hueDragging = false;
      this.$el.setAttribute("p-colorpicker-dragging", "false");
      !this.isUnstyled && removeClass(this.$el, "p-colorpicker-dragging");
      this.unbindDragListeners();
    }, "onDragEnd"),
    onHueMousedown: /* @__PURE__ */ __name(function onHueMousedown(event) {
      if (this.disabled) {
        return;
      }
      this.bindDragListeners();
      this.onHueDragStart(event);
    }, "onHueMousedown"),
    onHueDragStart: /* @__PURE__ */ __name(function onHueDragStart(event) {
      if (this.disabled) {
        return;
      }
      this.hueDragging = true;
      this.pickHue(event);
      !this.isUnstyled && addClass(this.$el, "p-colorpicker-dragging");
    }, "onHueDragStart"),
    isInputClicked: /* @__PURE__ */ __name(function isInputClicked2(event) {
      return this.$refs.input && this.$refs.input.isSameNode(event.target);
    }, "isInputClicked"),
    bindDragListeners: /* @__PURE__ */ __name(function bindDragListeners() {
      this.bindDocumentMouseMoveListener();
      this.bindDocumentMouseUpListener();
    }, "bindDragListeners"),
    unbindDragListeners: /* @__PURE__ */ __name(function unbindDragListeners() {
      this.unbindDocumentMouseMoveListener();
      this.unbindDocumentMouseUpListener();
    }, "unbindDragListeners"),
    bindOutsideClickListener: /* @__PURE__ */ __name(function bindOutsideClickListener5() {
      var _this = this;
      if (!this.outsideClickListener) {
        this.outsideClickListener = function(event) {
          if (_this.overlayVisible && _this.picker && !_this.picker.contains(event.target) && !_this.isInputClicked(event)) {
            _this.overlayVisible = false;
          }
        };
        document.addEventListener("click", this.outsideClickListener);
      }
    }, "bindOutsideClickListener"),
    unbindOutsideClickListener: /* @__PURE__ */ __name(function unbindOutsideClickListener5() {
      if (this.outsideClickListener) {
        document.removeEventListener("click", this.outsideClickListener);
        this.outsideClickListener = null;
      }
    }, "unbindOutsideClickListener"),
    bindScrollListener: /* @__PURE__ */ __name(function bindScrollListener4() {
      var _this2 = this;
      if (!this.scrollHandler) {
        this.scrollHandler = new ConnectedOverlayScrollHandler(this.$refs.container, function() {
          if (_this2.overlayVisible) {
            _this2.overlayVisible = false;
          }
        });
      }
      this.scrollHandler.bindScrollListener();
    }, "bindScrollListener"),
    unbindScrollListener: /* @__PURE__ */ __name(function unbindScrollListener4() {
      if (this.scrollHandler) {
        this.scrollHandler.unbindScrollListener();
      }
    }, "unbindScrollListener"),
    bindResizeListener: /* @__PURE__ */ __name(function bindResizeListener5() {
      var _this3 = this;
      if (!this.resizeListener) {
        this.resizeListener = function() {
          if (_this3.overlayVisible && !isTouchDevice()) {
            _this3.overlayVisible = false;
          }
        };
        window.addEventListener("resize", this.resizeListener);
      }
    }, "bindResizeListener"),
    unbindResizeListener: /* @__PURE__ */ __name(function unbindResizeListener5() {
      if (this.resizeListener) {
        window.removeEventListener("resize", this.resizeListener);
        this.resizeListener = null;
      }
    }, "unbindResizeListener"),
    bindDocumentMouseMoveListener: /* @__PURE__ */ __name(function bindDocumentMouseMoveListener() {
      if (!this.documentMouseMoveListener) {
        this.documentMouseMoveListener = this.onDrag.bind(this);
        document.addEventListener("mousemove", this.documentMouseMoveListener);
      }
    }, "bindDocumentMouseMoveListener"),
    unbindDocumentMouseMoveListener: /* @__PURE__ */ __name(function unbindDocumentMouseMoveListener() {
      if (this.documentMouseMoveListener) {
        document.removeEventListener("mousemove", this.documentMouseMoveListener);
        this.documentMouseMoveListener = null;
      }
    }, "unbindDocumentMouseMoveListener"),
    bindDocumentMouseUpListener: /* @__PURE__ */ __name(function bindDocumentMouseUpListener() {
      if (!this.documentMouseUpListener) {
        this.documentMouseUpListener = this.onDragEnd.bind(this);
        document.addEventListener("mouseup", this.documentMouseUpListener);
      }
    }, "bindDocumentMouseUpListener"),
    unbindDocumentMouseUpListener: /* @__PURE__ */ __name(function unbindDocumentMouseUpListener() {
      if (this.documentMouseUpListener) {
        document.removeEventListener("mouseup", this.documentMouseUpListener);
        this.documentMouseUpListener = null;
      }
    }, "unbindDocumentMouseUpListener"),
    pickerRef: /* @__PURE__ */ __name(function pickerRef(el) {
      this.picker = el;
    }, "pickerRef"),
    colorSelectorRef: /* @__PURE__ */ __name(function colorSelectorRef(el) {
      this.colorSelector = el;
    }, "colorSelectorRef"),
    colorHandleRef: /* @__PURE__ */ __name(function colorHandleRef(el) {
      this.colorHandle = el;
    }, "colorHandleRef"),
    hueViewRef: /* @__PURE__ */ __name(function hueViewRef(el) {
      this.hueView = el;
    }, "hueViewRef"),
    hueHandleRef: /* @__PURE__ */ __name(function hueHandleRef(el) {
      this.hueHandle = el;
    }, "hueHandleRef"),
    clearRefs: /* @__PURE__ */ __name(function clearRefs() {
      this.picker = null;
      this.colorSelector = null;
      this.colorHandle = null;
      this.hueView = null;
      this.hueHandle = null;
    }, "clearRefs"),
    onOverlayClick: /* @__PURE__ */ __name(function onOverlayClick4(event) {
      OverlayEventBus.emit("overlay-click", {
        originalEvent: event,
        target: this.$el
      });
    }, "onOverlayClick")
  },
  components: {
    Portal: script$r
  }
};
var _hoisted_1$g = ["id", "tabindex", "disabled"];
function render$6(_ctx, _cache, $props, $setup, $data, $options) {
  var _component_Portal = resolveComponent("Portal");
  return openBlock(), createElementBlock("div", mergeProps({
    ref: "container",
    "class": _ctx.cx("root")
  }, _ctx.ptmi("root")), [!_ctx.inline ? (openBlock(), createElementBlock("input", mergeProps({
    key: 0,
    ref: "input",
    id: _ctx.inputId,
    type: "text",
    "class": _ctx.cx("preview"),
    readonly: "readonly",
    tabindex: _ctx.tabindex,
    disabled: _ctx.disabled,
    onClick: _cache[0] || (_cache[0] = function() {
      return $options.onInputClick && $options.onInputClick.apply($options, arguments);
    }),
    onKeydown: _cache[1] || (_cache[1] = function() {
      return $options.onInputKeydown && $options.onInputKeydown.apply($options, arguments);
    })
  }, _ctx.ptm("preview")), null, 16, _hoisted_1$g)) : createCommentVNode("", true), createVNode(_component_Portal, {
    appendTo: _ctx.appendTo,
    disabled: _ctx.inline
  }, {
    "default": withCtx(function() {
      return [createVNode(Transition, mergeProps({
        name: "p-connected-overlay",
        onEnter: $options.onOverlayEnter,
        onLeave: $options.onOverlayLeave,
        onAfterLeave: $options.onOverlayAfterLeave
      }, _ctx.ptm("transition")), {
        "default": withCtx(function() {
          return [(_ctx.inline ? true : $data.overlayVisible) ? (openBlock(), createElementBlock("div", mergeProps({
            key: 0,
            ref: $options.pickerRef,
            "class": [_ctx.cx("panel"), _ctx.panelClass],
            onClick: _cache[10] || (_cache[10] = function() {
              return $options.onOverlayClick && $options.onOverlayClick.apply($options, arguments);
            })
          }, _ctx.ptm("panel")), [createBaseVNode("div", mergeProps({
            "class": _ctx.cx("content")
          }, _ctx.ptm("content")), [createBaseVNode("div", mergeProps({
            ref: $options.colorSelectorRef,
            "class": _ctx.cx("colorSelector"),
            onMousedown: _cache[2] || (_cache[2] = function($event) {
              return $options.onColorMousedown($event);
            }),
            onTouchstart: _cache[3] || (_cache[3] = function($event) {
              return $options.onColorDragStart($event);
            }),
            onTouchmove: _cache[4] || (_cache[4] = function($event) {
              return $options.onDrag($event);
            }),
            onTouchend: _cache[5] || (_cache[5] = function($event) {
              return $options.onDragEnd();
            })
          }, _ctx.ptm("colorSelector")), [createBaseVNode("div", mergeProps({
            "class": _ctx.cx("colorBackground")
          }, _ctx.ptm("colorBackground")), [createBaseVNode("div", mergeProps({
            ref: $options.colorHandleRef,
            "class": _ctx.cx("colorHandle")
          }, _ctx.ptm("colorHandle")), null, 16)], 16)], 16), createBaseVNode("div", mergeProps({
            ref: $options.hueViewRef,
            "class": _ctx.cx("hue"),
            onMousedown: _cache[6] || (_cache[6] = function($event) {
              return $options.onHueMousedown($event);
            }),
            onTouchstart: _cache[7] || (_cache[7] = function($event) {
              return $options.onHueDragStart($event);
            }),
            onTouchmove: _cache[8] || (_cache[8] = function($event) {
              return $options.onDrag($event);
            }),
            onTouchend: _cache[9] || (_cache[9] = function($event) {
              return $options.onDragEnd();
            })
          }, _ctx.ptm("hue")), [createBaseVNode("div", mergeProps({
            ref: $options.hueHandleRef,
            "class": _ctx.cx("hueHandle")
          }, _ctx.ptm("hueHandle")), null, 16)], 16)], 16)], 16)) : createCommentVNode("", true)];
        }),
        _: 1
      }, 16, ["onEnter", "onLeave", "onAfterLeave"])];
    }),
    _: 1
  }, 8, ["appendTo", "disabled"])], 16);
}
__name(render$6, "render$6");
script$7.render = render$6;
const _withScopeId$9 = /* @__PURE__ */ __name((n) => (pushScopeId("data-v-91077f2a"), n = n(), popScopeId(), n), "_withScopeId$9");
const _hoisted_1$f = { class: "p-fluid" };
const _hoisted_2$b = { class: "field icon-field" };
const _hoisted_3$7 = { for: "icon" };
const _hoisted_4$6 = { class: "field color-field" };
const _hoisted_5$4 = { for: "color" };
const _hoisted_6$3 = { class: "color-picker-container" };
const _hoisted_7$1 = {
  key: 1,
  class: "pi pi-palette",
  style: { fontSize: "1.2rem" }
};
const _sfc_main$i = /* @__PURE__ */ defineComponent({
  __name: "CustomizationDialog",
  props: {
    modelValue: { type: Boolean },
    initialIcon: {},
    initialColor: {}
  },
  emits: ["update:modelValue", "confirm"],
  setup(__props, { emit: __emit }) {
    const { t } = useI18n();
    const props = __props;
    const emit = __emit;
    const visible = computed({
      get: /* @__PURE__ */ __name(() => props.modelValue, "get"),
      set: /* @__PURE__ */ __name((value) => emit("update:modelValue", value), "set")
    });
    const nodeBookmarkStore = useNodeBookmarkStore();
    const iconOptions = [
      { name: t("bookmark"), value: nodeBookmarkStore.defaultBookmarkIcon },
      { name: t("folder"), value: "pi-folder" },
      { name: t("star"), value: "pi-star" },
      { name: t("heart"), value: "pi-heart" },
      { name: t("file"), value: "pi-file" },
      { name: t("inbox"), value: "pi-inbox" },
      { name: t("box"), value: "pi-box" },
      { name: t("briefcase"), value: "pi-briefcase" }
    ];
    const colorOptions = [
      { name: t("default"), value: nodeBookmarkStore.defaultBookmarkColor },
      { name: t("blue"), value: "#007bff" },
      { name: t("green"), value: "#28a745" },
      { name: t("red"), value: "#dc3545" },
      { name: t("pink"), value: "#e83e8c" },
      { name: t("yellow"), value: "#ffc107" },
      { name: t("custom"), value: "custom" }
    ];
    const defaultIcon = iconOptions.find(
      (option2) => option2.value === nodeBookmarkStore.defaultBookmarkIcon
    );
    const defaultColor = colorOptions.find(
      (option2) => option2.value === nodeBookmarkStore.defaultBookmarkColor
    );
    const selectedIcon = ref(defaultIcon);
    const selectedColor = ref(defaultColor);
    const finalColor = computed(
      () => selectedColor.value.value === "custom" ? `#${customColor.value}` : selectedColor.value.value
    );
    const customColor = ref("000000");
    const closeDialog = /* @__PURE__ */ __name(() => {
      visible.value = false;
    }, "closeDialog");
    const confirmCustomization = /* @__PURE__ */ __name(() => {
      emit("confirm", selectedIcon.value.value, finalColor.value);
      closeDialog();
    }, "confirmCustomization");
    const resetCustomization = /* @__PURE__ */ __name(() => {
      selectedIcon.value = iconOptions.find((option2) => option2.value === props.initialIcon) || defaultIcon;
      const colorOption = colorOptions.find(
        (option2) => option2.value === props.initialColor
      );
      if (!props.initialColor) {
        selectedColor.value = defaultColor;
      } else if (!colorOption) {
        customColor.value = props.initialColor.replace("#", "");
        selectedColor.value = { name: "Custom", value: "custom" };
      } else {
        selectedColor.value = colorOption;
      }
    }, "resetCustomization");
    watch(
      () => props.modelValue,
      (newValue) => {
        if (newValue) {
          resetCustomization();
        }
      },
      { immediate: true }
    );
    return (_ctx, _cache) => {
      const _directive_tooltip = resolveDirective("tooltip");
      return openBlock(), createBlock(unref(script$x), {
        visible: visible.value,
        "onUpdate:visible": _cache[3] || (_cache[3] = ($event) => visible.value = $event),
        header: _ctx.$t("customizeFolder")
      }, {
        footer: withCtx(() => [
          createVNode(unref(script$o), {
            label: _ctx.$t("reset"),
            icon: "pi pi-refresh",
            onClick: resetCustomization,
            class: "p-button-text"
          }, null, 8, ["label"]),
          createVNode(unref(script$o), {
            label: _ctx.$t("confirm"),
            icon: "pi pi-check",
            onClick: confirmCustomization,
            autofocus: ""
          }, null, 8, ["label"])
        ]),
        default: withCtx(() => [
          createBaseVNode("div", _hoisted_1$f, [
            createBaseVNode("div", _hoisted_2$b, [
              createBaseVNode("label", _hoisted_3$7, toDisplayString(_ctx.$t("icon")), 1),
              createVNode(unref(script$g), {
                modelValue: selectedIcon.value,
                "onUpdate:modelValue": _cache[0] || (_cache[0] = ($event) => selectedIcon.value = $event),
                options: iconOptions,
                optionLabel: "name",
                dataKey: "value"
              }, {
                option: withCtx((slotProps) => [
                  createBaseVNode("i", {
                    class: normalizeClass(["pi", slotProps.option.value, "mr-2"]),
                    style: normalizeStyle({ color: finalColor.value })
                  }, null, 6)
                ]),
                _: 1
              }, 8, ["modelValue"])
            ]),
            createVNode(unref(script$K)),
            createBaseVNode("div", _hoisted_4$6, [
              createBaseVNode("label", _hoisted_5$4, toDisplayString(_ctx.$t("color")), 1),
              createBaseVNode("div", _hoisted_6$3, [
                createVNode(unref(script$g), {
                  modelValue: selectedColor.value,
                  "onUpdate:modelValue": _cache[1] || (_cache[1] = ($event) => selectedColor.value = $event),
                  options: colorOptions,
                  optionLabel: "name",
                  dataKey: "value"
                }, {
                  option: withCtx((slotProps) => [
                    slotProps.option.value !== "custom" ? (openBlock(), createElementBlock("div", {
                      key: 0,
                      style: normalizeStyle({
                        width: "20px",
                        height: "20px",
                        backgroundColor: slotProps.option.value,
                        borderRadius: "50%"
                      })
                    }, null, 4)) : withDirectives((openBlock(), createElementBlock("i", _hoisted_7$1, null, 512)), [
                      [_directive_tooltip, _ctx.$t("customColor")]
                    ])
                  ]),
                  _: 1
                }, 8, ["modelValue"]),
                selectedColor.value.value === "custom" ? (openBlock(), createBlock(unref(script$7), {
                  key: 0,
                  modelValue: customColor.value,
                  "onUpdate:modelValue": _cache[2] || (_cache[2] = ($event) => customColor.value = $event)
                }, null, 8, ["modelValue"])) : createCommentVNode("", true)
              ])
            ])
          ])
        ]),
        _: 1
      }, 8, ["visible", "header"]);
    };
  }
});
const FolderCustomizationDialog = /* @__PURE__ */ _export_sfc(_sfc_main$i, [["__scopeId", "data-v-91077f2a"]]);
function useTreeExpansion(expandedKeys2) {
  const toggleNode = /* @__PURE__ */ __name((node2) => {
    if (node2.key && typeof node2.key === "string") {
      if (node2.key in expandedKeys2.value) {
        delete expandedKeys2.value[node2.key];
      } else {
        expandedKeys2.value[node2.key] = true;
      }
    }
  }, "toggleNode");
  const toggleNodeRecursive = /* @__PURE__ */ __name((node2) => {
    if (node2.key && typeof node2.key === "string") {
      if (node2.key in expandedKeys2.value) {
        collapseNode(node2);
      } else {
        expandNode(node2);
      }
    }
  }, "toggleNodeRecursive");
  const expandNode = /* @__PURE__ */ __name((node2) => {
    if (node2.key && typeof node2.key === "string" && node2.children && node2.children.length) {
      expandedKeys2.value[node2.key] = true;
      for (const child of node2.children) {
        expandNode(child);
      }
    }
  }, "expandNode");
  const collapseNode = /* @__PURE__ */ __name((node2) => {
    if (node2.key && typeof node2.key === "string" && node2.children && node2.children.length) {
      delete expandedKeys2.value[node2.key];
      for (const child of node2.children) {
        collapseNode(child);
      }
    }
  }, "collapseNode");
  const toggleNodeOnEvent = /* @__PURE__ */ __name((e, node2) => {
    if (e.ctrlKey) {
      toggleNodeRecursive(node2);
    } else {
      toggleNode(node2);
    }
  }, "toggleNodeOnEvent");
  return {
    toggleNode,
    toggleNodeRecursive,
    expandNode,
    collapseNode,
    toggleNodeOnEvent
  };
}
__name(useTreeExpansion, "useTreeExpansion");
const _sfc_main$h = /* @__PURE__ */ defineComponent({
  __name: "NodeBookmarkTreeExplorer",
  props: {
    filteredNodeDefs: {}
  },
  setup(__props, { expose: __expose }) {
    const props = __props;
    const expandedKeys2 = ref({});
    const { expandNode, toggleNodeOnEvent } = useTreeExpansion(expandedKeys2);
    const nodeBookmarkStore = useNodeBookmarkStore();
    const bookmarkedRoot = computed(() => {
      const filterTree = /* @__PURE__ */ __name((node2) => {
        if (node2.leaf) {
          return props.filteredNodeDefs.some((def) => def.name === node2.data.name) ? node2 : null;
        }
        const filteredChildren = node2.children?.map(filterTree).filter((child) => child !== null);
        if (filteredChildren && filteredChildren.length > 0) {
          return {
            ...node2,
            children: filteredChildren
          };
        }
        return null;
      }, "filterTree");
      return props.filteredNodeDefs.length ? filterTree(nodeBookmarkStore.bookmarkedRoot) || {
        key: "root",
        label: "Root",
        children: []
      } : nodeBookmarkStore.bookmarkedRoot;
    });
    watch(
      () => props.filteredNodeDefs,
      (newValue) => {
        if (newValue.length) {
          nextTick(() => expandNode(bookmarkedRoot.value));
        }
      }
    );
    const { t } = useI18n();
    const extraMenuItems = /* @__PURE__ */ __name((menuTargetNode) => [
      {
        label: t("newFolder"),
        icon: "pi pi-folder-plus",
        command: /* @__PURE__ */ __name(() => {
          addNewBookmarkFolder(menuTargetNode);
        }, "command"),
        visible: !menuTargetNode?.leaf
      },
      {
        label: t("customize"),
        icon: "pi pi-palette",
        command: /* @__PURE__ */ __name(() => {
          const customization = nodeBookmarkStore.bookmarksCustomization[menuTargetNode.data.nodePath];
          initialIcon.value = customization?.icon || nodeBookmarkStore.defaultBookmarkIcon;
          initialColor.value = customization?.color || nodeBookmarkStore.defaultBookmarkColor;
          showCustomizationDialog.value = true;
          customizationTargetNodePath.value = menuTargetNode.data.nodePath;
        }, "command"),
        visible: !menuTargetNode?.leaf
      }
    ], "extraMenuItems");
    const renderedBookmarkedRoot = computed(
      () => {
        const fillNodeInfo = /* @__PURE__ */ __name((node2) => {
          const children = node2.children?.map(fillNodeInfo);
          const sortedChildren = children?.sort((a, b) => {
            if (a.leaf === b.leaf) {
              return a.label.localeCompare(b.label);
            }
            return a.leaf ? 1 : -1;
          });
          return {
            key: node2.key,
            label: node2.leaf ? node2.data.display_name : node2.label,
            leaf: node2.leaf,
            data: node2.data,
            getIcon: /* @__PURE__ */ __name((node22) => {
              if (node22.leaf) {
                return "pi pi-circle-fill";
              }
              const customization = nodeBookmarkStore.bookmarksCustomization[node22.data.nodePath];
              return customization?.icon ? "pi " + customization.icon : "pi pi-bookmark-fill";
            }, "getIcon"),
            children: sortedChildren,
            draggable: node2.leaf,
            droppable: !node2.leaf,
            handleDrop: /* @__PURE__ */ __name((node22, data17) => {
              const nodeDefToAdd = data17.data.data;
              if (nodeBookmarkStore.isBookmarked(nodeDefToAdd)) {
                nodeBookmarkStore.toggleBookmark(nodeDefToAdd);
              }
              const folderNodeDef = node22.data;
              const nodePath = folderNodeDef.category + "/" + nodeDefToAdd.name;
              nodeBookmarkStore.addBookmark(nodePath);
            }, "handleDrop"),
            handleClick: /* @__PURE__ */ __name((node22, e) => {
              if (node22.leaf) {
                app.addNodeOnGraph(node22.data, { pos: app.getCanvasCenter() });
              } else {
                toggleNodeOnEvent(e, node22);
              }
            }, "handleClick"),
            contextMenuItems: extraMenuItems,
            ...node2.leaf ? {} : {
              handleRename,
              handleDelete: /* @__PURE__ */ __name((node22) => {
                nodeBookmarkStore.deleteBookmarkFolder(node22.data);
              }, "handleDelete")
            }
          };
        }, "fillNodeInfo");
        return fillNodeInfo(bookmarkedRoot.value);
      }
    );
    const treeExplorerRef = ref(null);
    const addNewBookmarkFolder = /* @__PURE__ */ __name((parent) => {
      const newFolderKey = "root/" + nodeBookmarkStore.addNewBookmarkFolder(parent?.data).slice(0, -1);
      nextTick(() => {
        treeExplorerRef.value?.renameCommand(
          findNodeByKey(
            renderedBookmarkedRoot.value,
            newFolderKey
          )
        );
        if (parent) {
          expandedKeys2.value[parent.key] = true;
        }
      });
    }, "addNewBookmarkFolder");
    __expose({
      addNewBookmarkFolder
    });
    const handleRename = useErrorHandling().wrapWithErrorHandling(
      (node2, newName) => {
        if (node2.data && node2.data.isDummyFolder) {
          nodeBookmarkStore.renameBookmarkFolder(node2.data, newName);
        }
      }
    );
    const showCustomizationDialog = ref(false);
    const initialIcon = ref(nodeBookmarkStore.defaultBookmarkIcon);
    const initialColor = ref(nodeBookmarkStore.defaultBookmarkColor);
    const customizationTargetNodePath = ref("");
    const updateCustomization = /* @__PURE__ */ __name((icon, color) => {
      if (customizationTargetNodePath.value) {
        nodeBookmarkStore.updateBookmarkCustomization(
          customizationTargetNodePath.value,
          { icon, color }
        );
      }
    }, "updateCustomization");
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock(Fragment, null, [
        createVNode(TreeExplorer, {
          class: "node-lib-bookmark-tree-explorer",
          ref_key: "treeExplorerRef",
          ref: treeExplorerRef,
          roots: renderedBookmarkedRoot.value.children,
          expandedKeys: expandedKeys2.value
        }, {
          folder: withCtx(({ node: node2 }) => [
            createVNode(_sfc_main$j, { node: node2 }, null, 8, ["node"])
          ]),
          node: withCtx(({ node: node2 }) => [
            createVNode(NodeTreeLeaf, { node: node2 }, null, 8, ["node"])
          ]),
          _: 1
        }, 8, ["roots", "expandedKeys"]),
        createVNode(FolderCustomizationDialog, {
          modelValue: showCustomizationDialog.value,
          "onUpdate:modelValue": _cache[0] || (_cache[0] = ($event) => showCustomizationDialog.value = $event),
          onConfirm: updateCustomization,
          initialIcon: initialIcon.value,
          initialColor: initialColor.value
        }, null, 8, ["modelValue", "initialIcon", "initialColor"])
      ], 64);
    };
  }
});
const _withScopeId$8 = /* @__PURE__ */ __name((n) => (pushScopeId("data-v-f6a7371a"), n = n(), popScopeId(), n), "_withScopeId$8");
const _hoisted_1$e = { class: "flex flex-col h-full" };
const _hoisted_2$a = { class: "flex-shrink-0" };
const _hoisted_3$6 = { class: "flex-grow overflow-y-auto" };
const _hoisted_4$5 = /* @__PURE__ */ _withScopeId$8(() => /* @__PURE__ */ createBaseVNode("div", { id: "node-library-node-preview-container" }, null, -1));
const _sfc_main$g = /* @__PURE__ */ defineComponent({
  __name: "NodeLibrarySidebarTab",
  setup(__props) {
    const nodeDefStore = useNodeDefStore();
    const nodeBookmarkStore = useNodeBookmarkStore();
    const expandedKeys2 = ref({});
    const { expandNode, toggleNodeOnEvent } = useTreeExpansion(expandedKeys2);
    const nodeBookmarkTreeExplorerRef = ref(null);
    const searchFilter = ref(null);
    const alphabeticalSort = ref(false);
    const searchQuery = ref("");
    const root15 = computed(() => {
      const root22 = filteredRoot.value || nodeDefStore.nodeTree;
      return alphabeticalSort.value ? sortedTree(root22) : root22;
    });
    const renderedRoot = computed(() => {
      const fillNodeInfo = /* @__PURE__ */ __name((node2) => {
        const children = node2.children?.map(fillNodeInfo);
        return {
          key: node2.key,
          label: node2.leaf ? node2.data.display_name : node2.label,
          leaf: node2.leaf,
          data: node2.data,
          getIcon: /* @__PURE__ */ __name((node22) => {
            if (node22.leaf) {
              return "pi pi-circle-fill";
            }
          }, "getIcon"),
          children,
          draggable: node2.leaf,
          handleClick: /* @__PURE__ */ __name((node22, e) => {
            if (node22.leaf) {
              app.addNodeOnGraph(node22.data, { pos: app.getCanvasCenter() });
            } else {
              toggleNodeOnEvent(e, node22);
            }
          }, "handleClick")
        };
      }, "fillNodeInfo");
      return fillNodeInfo(root15.value);
    });
    const filteredNodeDefs = ref([]);
    const filteredRoot = computed(() => {
      if (!filteredNodeDefs.value.length) {
        return null;
      }
      return buildNodeDefTree(filteredNodeDefs.value);
    });
    const filters = ref([]);
    const handleSearch = /* @__PURE__ */ __name((query) => {
      if (query.length === 0 && !filters.value.length) {
        filteredNodeDefs.value = [];
        expandedKeys2.value = {};
        return;
      }
      const f = filters.value.map((f2) => f2.filter);
      filteredNodeDefs.value = nodeDefStore.nodeSearchService.searchNode(
        query,
        f,
        {
          limit: 64
        },
        {
          matchWildcards: false
        }
      );
      nextTick(() => {
        expandNode(filteredRoot.value);
      });
    }, "handleSearch");
    const onAddFilter = /* @__PURE__ */ __name((filterAndValue) => {
      filters.value.push({
        filter: filterAndValue,
        badge: filterAndValue[0].invokeSequence.toUpperCase(),
        badgeClass: filterAndValue[0].invokeSequence + "-badge",
        text: filterAndValue[1],
        id: +/* @__PURE__ */ new Date()
      });
      handleSearch(searchQuery.value);
    }, "onAddFilter");
    const onRemoveFilter = /* @__PURE__ */ __name((filterAndValue) => {
      const index2 = filters.value.findIndex((f) => f === filterAndValue);
      if (index2 !== -1) {
        filters.value.splice(index2, 1);
      }
      handleSearch(searchQuery.value);
    }, "onRemoveFilter");
    return (_ctx, _cache) => {
      const _directive_tooltip = resolveDirective("tooltip");
      return openBlock(), createElementBlock(Fragment, null, [
        createVNode(SidebarTabTemplate, {
          title: _ctx.$t("sideToolbar.nodeLibrary")
        }, {
          "tool-buttons": withCtx(() => [
            withDirectives(createVNode(unref(script$o), {
              class: "new-folder-button",
              icon: "pi pi-folder-plus",
              text: "",
              severity: "secondary",
              onClick: _cache[0] || (_cache[0] = ($event) => nodeBookmarkTreeExplorerRef.value?.addNewBookmarkFolder())
            }, null, 512), [
              [_directive_tooltip, _ctx.$t("newFolder")]
            ]),
            withDirectives(createVNode(unref(script$o), {
              class: "sort-button",
              icon: alphabeticalSort.value ? "pi pi-sort-alpha-down" : "pi pi-sort-alt",
              text: "",
              severity: "secondary",
              onClick: _cache[1] || (_cache[1] = ($event) => alphabeticalSort.value = !alphabeticalSort.value)
            }, null, 8, ["icon"]), [
              [_directive_tooltip, _ctx.$t("sideToolbar.nodeLibraryTab.sortOrder")]
            ])
          ]),
          body: withCtx(() => [
            createBaseVNode("div", _hoisted_1$e, [
              createBaseVNode("div", _hoisted_2$a, [
                createVNode(SearchBox, {
                  class: "node-lib-search-box mx-4 mt-4",
                  modelValue: searchQuery.value,
                  "onUpdate:modelValue": _cache[2] || (_cache[2] = ($event) => searchQuery.value = $event),
                  onSearch: handleSearch,
                  onShowFilter: _cache[3] || (_cache[3] = ($event) => searchFilter.value.toggle($event)),
                  onRemoveFilter,
                  placeholder: _ctx.$t("searchNodes") + "...",
                  "filter-icon": "pi pi-filter",
                  filters: filters.value
                }, null, 8, ["modelValue", "placeholder", "filters"]),
                createVNode(unref(script$9), {
                  ref_key: "searchFilter",
                  ref: searchFilter,
                  class: "node-lib-filter-popup"
                }, {
                  default: withCtx(() => [
                    createVNode(NodeSearchFilter, { onAddFilter })
                  ]),
                  _: 1
                }, 512)
              ]),
              createBaseVNode("div", _hoisted_3$6, [
                createVNode(_sfc_main$h, {
                  ref_key: "nodeBookmarkTreeExplorerRef",
                  ref: nodeBookmarkTreeExplorerRef,
                  "filtered-node-defs": filteredNodeDefs.value
                }, null, 8, ["filtered-node-defs"]),
                unref(nodeBookmarkStore).bookmarks.length > 0 ? (openBlock(), createBlock(unref(script$K), {
                  key: 0,
                  type: "dashed"
                })) : createCommentVNode("", true),
                createVNode(TreeExplorer, {
                  class: "node-lib-tree-explorer mt-1",
                  roots: renderedRoot.value.children,
                  expandedKeys: expandedKeys2.value,
                  "onUpdate:expandedKeys": _cache[4] || (_cache[4] = ($event) => expandedKeys2.value = $event)
                }, {
                  node: withCtx(({ node: node2 }) => [
                    createVNode(NodeTreeLeaf, { node: node2 }, null, 8, ["node"])
                  ]),
                  _: 1
                }, 8, ["roots", "expandedKeys"])
              ])
            ])
          ]),
          _: 1
        }, 8, ["title"]),
        _hoisted_4$5
      ], 64);
    };
  }
});
const NodeLibrarySidebarTab = /* @__PURE__ */ _export_sfc(_sfc_main$g, [["__scopeId", "data-v-f6a7371a"]]);
const _withScopeId$7 = /* @__PURE__ */ __name((n) => (pushScopeId("data-v-32e6c4d9"), n = n(), popScopeId(), n), "_withScopeId$7");
const _hoisted_1$d = { class: "model_preview" };
const _hoisted_2$9 = { class: "model_preview_title" };
const _hoisted_3$5 = { class: "model_preview_top_container" };
const _hoisted_4$4 = { class: "model_preview_filename" };
const _hoisted_5$3 = {
  key: 0,
  class: "model_preview_architecture"
};
const _hoisted_6$2 = /* @__PURE__ */ _withScopeId$7(() => /* @__PURE__ */ createBaseVNode("span", { class: "model_preview_prefix" }, "Architecture: ", -1));
const _hoisted_7 = {
  key: 1,
  class: "model_preview_author"
};
const _hoisted_8 = /* @__PURE__ */ _withScopeId$7(() => /* @__PURE__ */ createBaseVNode("span", { class: "model_preview_prefix" }, "Author: ", -1));
const _hoisted_9 = {
  key: 0,
  class: "model_preview_image"
};
const _hoisted_10 = ["src"];
const _hoisted_11 = {
  key: 1,
  class: "model_preview_usage_hint"
};
const _hoisted_12 = /* @__PURE__ */ _withScopeId$7(() => /* @__PURE__ */ createBaseVNode("span", { class: "model_preview_prefix" }, "Usage hint: ", -1));
const _hoisted_13 = {
  key: 2,
  class: "model_preview_trigger_phrase"
};
const _hoisted_14 = /* @__PURE__ */ _withScopeId$7(() => /* @__PURE__ */ createBaseVNode("span", { class: "model_preview_prefix" }, "Trigger phrase: ", -1));
const _hoisted_15 = {
  key: 3,
  class: "model_preview_description"
};
const _hoisted_16 = /* @__PURE__ */ _withScopeId$7(() => /* @__PURE__ */ createBaseVNode("span", { class: "model_preview_prefix" }, "Description: ", -1));
const _sfc_main$f = /* @__PURE__ */ defineComponent({
  __name: "ModelPreview",
  props: {
    modelDef: {
      type: ComfyModelDef,
      required: true
    }
  },
  setup(__props) {
    const props = __props;
    const modelDef = props.modelDef;
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", _hoisted_1$d, [
        createBaseVNode("div", _hoisted_2$9, toDisplayString(unref(modelDef).title), 1),
        createBaseVNode("div", _hoisted_3$5, [
          createBaseVNode("div", _hoisted_4$4, toDisplayString(unref(modelDef).file_name), 1),
          unref(modelDef).architecture_id ? (openBlock(), createElementBlock("div", _hoisted_5$3, [
            _hoisted_6$2,
            createTextVNode(" " + toDisplayString(unref(modelDef).architecture_id), 1)
          ])) : createCommentVNode("", true),
          unref(modelDef).author ? (openBlock(), createElementBlock("div", _hoisted_7, [
            _hoisted_8,
            createTextVNode(" " + toDisplayString(unref(modelDef).author), 1)
          ])) : createCommentVNode("", true)
        ]),
        unref(modelDef).image ? (openBlock(), createElementBlock("div", _hoisted_9, [
          createBaseVNode("img", {
            src: unref(modelDef).image
          }, null, 8, _hoisted_10)
        ])) : createCommentVNode("", true),
        unref(modelDef).usage_hint ? (openBlock(), createElementBlock("div", _hoisted_11, [
          _hoisted_12,
          createTextVNode(" " + toDisplayString(unref(modelDef).usage_hint), 1)
        ])) : createCommentVNode("", true),
        unref(modelDef).trigger_phrase ? (openBlock(), createElementBlock("div", _hoisted_13, [
          _hoisted_14,
          createTextVNode(" " + toDisplayString(unref(modelDef).trigger_phrase), 1)
        ])) : createCommentVNode("", true),
        unref(modelDef).description ? (openBlock(), createElementBlock("div", _hoisted_15, [
          _hoisted_16,
          createTextVNode(" " + toDisplayString(unref(modelDef).description), 1)
        ])) : createCommentVNode("", true)
      ]);
    };
  }
});
const ModelPreview = /* @__PURE__ */ _export_sfc(_sfc_main$f, [["__scopeId", "data-v-32e6c4d9"]]);
const _withScopeId$6 = /* @__PURE__ */ __name((n) => (pushScopeId("data-v-70b69131"), n = n(), popScopeId(), n), "_withScopeId$6");
const _hoisted_1$c = {
  key: 0,
  class: "model-lib-model-icon-container"
};
const _sfc_main$e = /* @__PURE__ */ defineComponent({
  __name: "ModelTreeLeaf",
  props: {
    node: {}
  },
  setup(__props) {
    const props = __props;
    const modelDef = computed(() => props.node.data);
    const previewRef = ref(null);
    const modelPreviewStyle = ref({
      position: "absolute",
      top: "0px",
      left: "0px"
    });
    const settingStore = useSettingStore();
    const sidebarLocation = computed(
      () => settingStore.get("Comfy.Sidebar.Location")
    );
    const handleModelHover = /* @__PURE__ */ __name(async () => {
      if (modelDef.value.is_fake_object) {
        return;
      }
      const hoverTarget = modelContentElement.value;
      const targetRect = hoverTarget.getBoundingClientRect();
      const previewHeight = previewRef.value?.$el.offsetHeight || 0;
      const availableSpaceBelow = window.innerHeight - targetRect.bottom;
      modelPreviewStyle.value.top = previewHeight > availableSpaceBelow ? `${Math.max(0, targetRect.top - (previewHeight - availableSpaceBelow) - 20)}px` : `${targetRect.top - 40}px`;
      if (sidebarLocation.value === "left") {
        modelPreviewStyle.value.left = `${targetRect.right}px`;
      } else {
        modelPreviewStyle.value.left = `${targetRect.left - 400}px`;
      }
      modelDef.value.load();
    }, "handleModelHover");
    const container = ref(null);
    const modelContentElement = ref(null);
    const isHovered = ref(false);
    const showPreview = computed(() => {
      return isHovered.value && modelDef.value && !modelDef.value.is_fake_object && modelDef.value.has_loaded_metadata && (modelDef.value.author || modelDef.value.simplified_file_name != modelDef.value.title || modelDef.value.description || modelDef.value.usage_hint || modelDef.value.trigger_phrase || modelDef.value.image);
    });
    const handleMouseEnter = /* @__PURE__ */ __name(async () => {
      if (modelDef.value.is_fake_object) {
        return;
      }
      isHovered.value = true;
      await nextTick();
      handleModelHover();
    }, "handleMouseEnter");
    const handleMouseLeave = /* @__PURE__ */ __name(() => {
      isHovered.value = false;
    }, "handleMouseLeave");
    onMounted(() => {
      modelContentElement.value = container.value?.closest(".p-tree-node-content");
      modelContentElement.value?.addEventListener("mouseenter", handleMouseEnter);
      modelContentElement.value?.addEventListener("mouseleave", handleMouseLeave);
      if (!modelDef.value.is_fake_object) {
        modelDef.value.load();
      }
    });
    onUnmounted(() => {
      modelContentElement.value?.removeEventListener("mouseenter", handleMouseEnter);
      modelContentElement.value?.removeEventListener("mouseleave", handleMouseLeave);
    });
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", {
        ref_key: "container",
        ref: container,
        class: "model-lib-node-container h-full w-full"
      }, [
        createVNode(TreeExplorerTreeNode, { node: _ctx.node }, {
          "before-label": withCtx(() => [
            modelDef.value && modelDef.value.image ? (openBlock(), createElementBlock("span", _hoisted_1$c, [
              createBaseVNode("span", {
                class: "model-lib-model-icon",
                style: normalizeStyle({ backgroundImage: `url(${modelDef.value.image})` })
              }, null, 4)
            ])) : createCommentVNode("", true)
          ]),
          _: 1
        }, 8, ["node"]),
        showPreview.value ? (openBlock(), createBlock(Teleport, {
          key: 0,
          to: "#model-library-model-preview-container"
        }, [
          createBaseVNode("div", {
            class: "model-lib-model-preview",
            style: normalizeStyle(modelPreviewStyle.value)
          }, [
            createVNode(ModelPreview, {
              ref_key: "previewRef",
              ref: previewRef,
              modelDef: modelDef.value
            }, null, 8, ["modelDef"])
          ], 4)
        ])) : createCommentVNode("", true)
      ], 512);
    };
  }
});
const ModelTreeLeaf = /* @__PURE__ */ _export_sfc(_sfc_main$e, [["__scopeId", "data-v-70b69131"]]);
const _withScopeId$5 = /* @__PURE__ */ __name((n) => (pushScopeId("data-v-74b01bce"), n = n(), popScopeId(), n), "_withScopeId$5");
const _hoisted_1$b = { class: "flex flex-col h-full" };
const _hoisted_2$8 = { class: "flex-shrink-0" };
const _hoisted_3$4 = { class: "flex-grow overflow-y-auto" };
const _hoisted_4$3 = /* @__PURE__ */ _withScopeId$5(() => /* @__PURE__ */ createBaseVNode("div", { id: "model-library-model-preview-container" }, null, -1));
const _sfc_main$d = /* @__PURE__ */ defineComponent({
  __name: "ModelLibrarySidebarTab",
  setup(__props) {
    const { t } = useI18n();
    const modelStore = useModelStore();
    const modelToNodeStore = useModelToNodeStore();
    const settingStore = useSettingStore();
    const searchQuery = ref("");
    const expandedKeys2 = ref({});
    const { toggleNodeOnEvent } = useTreeExpansion(expandedKeys2);
    const root15 = computed(() => {
      let modelList = [];
      if (!modelStore.modelFolders.length) {
        modelStore.getModelFolders();
      }
      if (settingStore.get("Comfy.ModelLibrary.AutoLoadAll")) {
        for (let folder of modelStore.modelFolders) {
          modelStore.getModelsInFolderCached(folder);
        }
      }
      for (let folder of modelStore.modelFolders) {
        const models = modelStore.modelStoreMap[folder];
        if (models) {
          if (Object.values(models.models).length) {
            modelList.push(...Object.values(models.models));
          } else {
            const fakeModel = new ComfyModelDef("(No Content)", folder);
            fakeModel.is_fake_object = true;
            modelList.push(fakeModel);
          }
        } else {
          const fakeModel = new ComfyModelDef("Loading", folder);
          fakeModel.is_fake_object = true;
          modelList.push(fakeModel);
        }
      }
      if (searchQuery.value) {
        const search2 = searchQuery.value.toLocaleLowerCase();
        modelList = modelList.filter((model) => {
          return model.searchable.includes(search2);
        });
      }
      const tree = buildTree(modelList, (model) => {
        return [
          model.directory,
          ...model.file_name.replaceAll("\\", "/").split("/")
        ];
      });
      return tree;
    });
    const renderedRoot = computed(() => {
      const nameFormat = settingStore.get("Comfy.ModelLibrary.NameFormat");
      const fillNodeInfo = /* @__PURE__ */ __name((node2) => {
        const children = node2.children?.map(fillNodeInfo);
        const model = node2.leaf && node2.data ? node2.data : null;
        if (model?.is_fake_object) {
          if (model.file_name === "(No Content)") {
            return {
              key: node2.key,
              label: t("noContent"),
              leaf: true,
              data: node2.data,
              getIcon: /* @__PURE__ */ __name((node22) => {
                return "pi pi-file";
              }, "getIcon"),
              children: []
            };
          } else {
            return {
              key: node2.key,
              label: t("loading") + "...",
              leaf: true,
              data: node2.data,
              getIcon: /* @__PURE__ */ __name((node22) => {
                return "pi pi-spin pi-spinner";
              }, "getIcon"),
              children: []
            };
          }
        }
        return {
          key: node2.key,
          label: model ? nameFormat === "title" ? model.title : model.simplified_file_name : node2.label,
          leaf: node2.leaf,
          data: node2.data,
          getIcon: /* @__PURE__ */ __name((node22) => {
            if (node22.leaf) {
              if (node22.data && node22.data.image) {
                return "pi pi-fake-spacer";
              }
              return "pi pi-file";
            }
          }, "getIcon"),
          getBadgeText: /* @__PURE__ */ __name((node22) => {
            if (node22.leaf) {
              return null;
            }
            if (node22.children?.length === 1) {
              const onlyChild = node22.children[0];
              if (onlyChild.data?.is_fake_object) {
                if (onlyChild.data.file_name === "(No Content)") {
                  return "0";
                } else if (onlyChild.data.file_name === "Loading") {
                  return "?";
                }
              }
            }
            return null;
          }, "getBadgeText"),
          children,
          draggable: node2.leaf,
          handleClick: /* @__PURE__ */ __name((node22, e) => {
            if (node22.leaf) {
              const provider = modelToNodeStore.getNodeProvider(model.directory);
              if (provider) {
                const node3 = app.addNodeOnGraph(provider.nodeDef, {
                  pos: app.getCanvasCenter()
                });
                const widget = node3.widgets.find(
                  (widget2) => widget2.name === provider.key
                );
                if (widget) {
                  widget.value = model.file_name;
                }
              }
            }
          }, "handleClick")
        };
      }, "fillNodeInfo");
      return fillNodeInfo(root15.value);
    });
    const handleNodeClick = /* @__PURE__ */ __name((node2, e) => {
      if (node2.leaf) {
      } else {
        toggleNodeOnEvent(e, node2);
      }
    }, "handleNodeClick");
    watch(
      toRef(expandedKeys2, "value"),
      (newExpandedKeys) => {
        Object.entries(newExpandedKeys).forEach(([key, isExpanded]) => {
          if (isExpanded) {
            const folderPath = key.split("/").slice(1).join("/");
            if (folderPath && !folderPath.includes("/")) {
              modelStore.getModelsInFolderCached(folderPath);
            }
          }
        });
      },
      { deep: true }
    );
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock(Fragment, null, [
        createVNode(SidebarTabTemplate, {
          title: _ctx.$t("sideToolbar.modelLibrary")
        }, {
          "tool-buttons": withCtx(() => []),
          body: withCtx(() => [
            createBaseVNode("div", _hoisted_1$b, [
              createBaseVNode("div", _hoisted_2$8, [
                createVNode(SearchBox, {
                  class: "model-lib-search-box mx-4 mt-4",
                  modelValue: searchQuery.value,
                  "onUpdate:modelValue": _cache[0] || (_cache[0] = ($event) => searchQuery.value = $event),
                  placeholder: _ctx.$t("searchModels") + "..."
                }, null, 8, ["modelValue", "placeholder"])
              ]),
              createBaseVNode("div", _hoisted_3$4, [
                createVNode(TreeExplorer, {
                  class: "model-lib-tree-explorer mt-1",
                  roots: renderedRoot.value.children,
                  expandedKeys: expandedKeys2.value,
                  "onUpdate:expandedKeys": _cache[1] || (_cache[1] = ($event) => expandedKeys2.value = $event),
                  onNodeClick: handleNodeClick
                }, {
                  node: withCtx(({ node: node2 }) => [
                    createVNode(ModelTreeLeaf, { node: node2 }, null, 8, ["node"])
                  ]),
                  _: 1
                }, 8, ["roots", "expandedKeys"])
              ])
            ])
          ]),
          _: 1
        }, 8, ["title"]),
        _hoisted_4$3
      ], 64);
    };
  }
});
const ModelLibrarySidebarTab = /* @__PURE__ */ _export_sfc(_sfc_main$d, [["__scopeId", "data-v-74b01bce"]]);
function _typeof$3(o) {
  "@babel/helpers - typeof";
  return _typeof$3 = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function(o2) {
    return typeof o2;
  } : function(o2) {
    return o2 && "function" == typeof Symbol && o2.constructor === Symbol && o2 !== Symbol.prototype ? "symbol" : typeof o2;
  }, _typeof$3(o);
}
__name(_typeof$3, "_typeof$3");
function _defineProperty$3(e, r, t) {
  return (r = _toPropertyKey$3(r)) in e ? Object.defineProperty(e, r, { value: t, enumerable: true, configurable: true, writable: true }) : e[r] = t, e;
}
__name(_defineProperty$3, "_defineProperty$3");
function _toPropertyKey$3(t) {
  var i = _toPrimitive$3(t, "string");
  return "symbol" == _typeof$3(i) ? i : i + "";
}
__name(_toPropertyKey$3, "_toPropertyKey$3");
function _toPrimitive$3(t, r) {
  if ("object" != _typeof$3(t) || !t) return t;
  var e = t[Symbol.toPrimitive];
  if (void 0 !== e) {
    var i = e.call(t, r || "default");
    if ("object" != _typeof$3(i)) return i;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return ("string" === r ? String : Number)(t);
}
__name(_toPrimitive$3, "_toPrimitive$3");
var theme$4 = /* @__PURE__ */ __name(function theme14(_ref) {
  var dt = _ref.dt;
  return "\n.p-toast {\n    width: ".concat(dt("toast.width"), ";\n    white-space: pre-line;\n    word-break: break-word;\n}\n\n.p-toast-message {\n    margin: 0 0 1rem 0;\n}\n\n.p-toast-message-icon {\n    flex-shrink: 0;\n    font-size: ").concat(dt("toast.icon.size"), ";\n    width: ").concat(dt("toast.icon.size"), ";\n    height: ").concat(dt("toast.icon.size"), ";\n}\n\n.p-toast-message-content {\n    display: flex;\n    align-items: flex-start;\n    padding: ").concat(dt("toast.content.padding"), ";\n    gap: ").concat(dt("toast.content.gap"), ";\n}\n\n.p-toast-message-text {\n    flex: 1 1 auto;\n    display: flex;\n    flex-direction: column;\n    gap: ").concat(dt("toast.text.gap"), ";\n}\n\n.p-toast-summary {\n    font-weight: ").concat(dt("toast.summary.font.weight"), ";\n    font-size: ").concat(dt("toast.summary.font.size"), ";\n}\n\n.p-toast-detail {\n    font-weight: ").concat(dt("toast.detail.font.weight"), ";\n    font-size: ").concat(dt("toast.detail.font.size"), ";\n}\n\n.p-toast-close-button {\n    display: flex;\n    align-items: center;\n    justify-content: center;\n    overflow: hidden;\n    position: relative;\n    cursor: pointer;\n    background: transparent;\n    transition: background ").concat(dt("toast.transition.duration"), ", color ").concat(dt("toast.transition.duration"), ", outline-color ").concat(dt("toast.transition.duration"), ", box-shadow ").concat(dt("toast.transition.duration"), ";\n    outline-color: transparent;\n    color: inherit;\n    width: ").concat(dt("toast.close.button.width"), ";\n    height: ").concat(dt("toast.close.button.height"), ";\n    border-radius: ").concat(dt("toast.close.button.border.radius"), ";\n    margin: -25% 0 0 0;\n    right: -25%;\n    padding: 0;\n    border: none;\n    user-select: none;\n}\n\n.p-toast-message-info,\n.p-toast-message-success,\n.p-toast-message-warn,\n.p-toast-message-error,\n.p-toast-message-secondary,\n.p-toast-message-contrast {\n    border-width: ").concat(dt("toast.border.width"), ";\n    border-style: solid;\n    backdrop-filter: blur(").concat(dt("toast.blur"), ");\n    border-radius: ").concat(dt("toast.border.radius"), ";\n}\n\n.p-toast-close-icon {\n    font-size: ").concat(dt("toast.close.icon.size"), ";\n    width: ").concat(dt("toast.close.icon.size"), ";\n    height: ").concat(dt("toast.close.icon.size"), ";\n}\n\n.p-toast-close-button:focus-visible {\n    outline-width: ").concat(dt("focus.ring.width"), ";\n    outline-style: ").concat(dt("focus.ring.style"), ";\n    outline-offset: ").concat(dt("focus.ring.offset"), ";\n}\n\n.p-toast-message-info {\n    background: ").concat(dt("toast.info.background"), ";\n    border-color: ").concat(dt("toast.info.border.color"), ";\n    color: ").concat(dt("toast.info.color"), ";\n    box-shadow: ").concat(dt("toast.info.shadow"), ";\n}\n\n.p-toast-message-info .p-toast-detail {\n    color: ").concat(dt("toast.info.detail.color"), ";\n}\n\n.p-toast-message-info .p-toast-close-button:focus-visible {\n    outline-color: ").concat(dt("toast.info.close.button.focus.ring.color"), ";\n    box-shadow: ").concat(dt("toast.info.close.button.focus.ring.shadow"), ";\n}\n\n.p-toast-message-info .p-toast-close-button:hover {\n    background: ").concat(dt("toast.info.close.button.hover.background"), ";\n}\n\n.p-toast-message-success {\n    background: ").concat(dt("toast.success.background"), ";\n    border-color: ").concat(dt("toast.success.border.color"), ";\n    color: ").concat(dt("toast.success.color"), ";\n    box-shadow: ").concat(dt("toast.success.shadow"), ";\n}\n\n.p-toast-message-success .p-toast-detail {\n    color: ").concat(dt("toast.success.detail.color"), ";\n}\n\n.p-toast-message-success .p-toast-close-button:focus-visible {\n    outline-color: ").concat(dt("toast.success.close.button.focus.ring.color"), ";\n    box-shadow: ").concat(dt("toast.success.close.button.focus.ring.shadow"), ";\n}\n\n.p-toast-message-success .p-toast-close-button:hover {\n    background: ").concat(dt("toast.success.close.button.hover.background"), ";\n}\n\n.p-toast-message-warn {\n    background: ").concat(dt("toast.warn.background"), ";\n    border-color: ").concat(dt("toast.warn.border.color"), ";\n    color: ").concat(dt("toast.warn.color"), ";\n    box-shadow: ").concat(dt("toast.warn.shadow"), ";\n}\n\n.p-toast-message-warn .p-toast-detail {\n    color: ").concat(dt("toast.warn.detail.color"), ";\n}\n\n.p-toast-message-warn .p-toast-close-button:focus-visible {\n    outline-color: ").concat(dt("toast.warn.close.button.focus.ring.color"), ";\n    box-shadow: ").concat(dt("toast.warn.close.button.focus.ring.shadow"), ";\n}\n\n.p-toast-message-warn .p-toast-close-button:hover {\n    background: ").concat(dt("toast.warn.close.button.hover.background"), ";\n}\n\n.p-toast-message-error {\n    background: ").concat(dt("toast.error.background"), ";\n    border-color: ").concat(dt("toast.error.border.color"), ";\n    color: ").concat(dt("toast.error.color"), ";\n    box-shadow: ").concat(dt("toast.error.shadow"), ";\n}\n\n.p-toast-message-error .p-toast-detail {\n    color: ").concat(dt("toast.error.detail.color"), ";\n}\n\n.p-toast-message-error .p-toast-close-button:focus-visible {\n    outline-color: ").concat(dt("toast.error.close.button.focus.ring.color"), ";\n    box-shadow: ").concat(dt("toast.error.close.button.focus.ring.shadow"), ";\n}\n\n.p-toast-message-error .p-toast-close-button:hover {\n    background: ").concat(dt("toast.error.close.button.hover.background"), ";\n}\n\n.p-toast-message-secondary {\n    background: ").concat(dt("toast.secondary.background"), ";\n    border-color: ").concat(dt("toast.secondary.border.color"), ";\n    color: ").concat(dt("toast.secondary.color"), ";\n    box-shadow: ").concat(dt("toast.secondary.shadow"), ";\n}\n\n.p-toast-message-secondary .p-toast-detail {\n    color: ").concat(dt("toast.secondary.detail.color"), ";\n}\n\n.p-toast-message-secondary .p-toast-close-button:focus-visible {\n    outline-color: ").concat(dt("toast.secondary.close.button.focus.ring.color"), ";\n    box-shadow: ").concat(dt("toast.secondary.close.button.focus.ring.shadow"), ";\n}\n\n.p-toast-message-secondary .p-toast-close-button:hover {\n    background: ").concat(dt("toast.secondary.close.button.hover.background"), ";\n}\n\n.p-toast-message-contrast {\n    background: ").concat(dt("toast.contrast.background"), ";\n    border-color: ").concat(dt("toast.contrast.border.color"), ";\n    color: ").concat(dt("toast.contrast.color"), ";\n    box-shadow: ").concat(dt("toast.contrast.shadow"), ";\n}\n\n.p-toast-message-contrast .p-toast-detail {\n    color: ").concat(dt("toast.contrast.detail.color"), ";\n}\n\n.p-toast-message-contrast .p-toast-close-button:focus-visible {\n    outline-color: ").concat(dt("toast.contrast.close.button.focus.ring.color"), ";\n    box-shadow: ").concat(dt("toast.contrast.close.button.focus.ring.shadow"), ";\n}\n\n.p-toast-message-contrast .p-toast-close-button:hover {\n    background: ").concat(dt("toast.contrast.close.button.hover.background"), ";\n}\n\n.p-toast-top-center {\n    transform: translateX(-50%);\n}\n\n.p-toast-bottom-center {\n    transform: translateX(-50%);\n}\n\n.p-toast-center {\n    min-width: 20vw;\n    transform: translate(-50%, -50%);\n}\n\n.p-toast-message-enter-from {\n    opacity: 0;\n    transform: translateY(50%);\n}\n\n.p-toast-message-leave-from {\n    max-height: 1000px;\n}\n\n.p-toast .p-toast-message.p-toast-message-leave-to {\n    max-height: 0;\n    opacity: 0;\n    margin-bottom: 0;\n    overflow: hidden;\n}\n\n.p-toast-message-enter-active {\n    transition: transform 0.3s, opacity 0.3s;\n}\n\n.p-toast-message-leave-active {\n    transition: max-height 0.45s cubic-bezier(0, 1, 0, 1), opacity 0.3s, margin-bottom 0.3s;\n}\n");
}, "theme");
var inlineStyles$2 = {
  root: /* @__PURE__ */ __name(function root9(_ref2) {
    var position2 = _ref2.position;
    return {
      position: "fixed",
      top: position2 === "top-right" || position2 === "top-left" || position2 === "top-center" ? "20px" : position2 === "center" ? "50%" : null,
      right: (position2 === "top-right" || position2 === "bottom-right") && "20px",
      bottom: (position2 === "bottom-left" || position2 === "bottom-right" || position2 === "bottom-center") && "20px",
      left: position2 === "top-left" || position2 === "bottom-left" ? "20px" : position2 === "center" || position2 === "top-center" || position2 === "bottom-center" ? "50%" : null
    };
  }, "root")
};
var classes$4 = {
  root: /* @__PURE__ */ __name(function root10(_ref3) {
    var props = _ref3.props;
    return ["p-toast p-component p-toast-" + props.position];
  }, "root"),
  message: /* @__PURE__ */ __name(function message2(_ref4) {
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
    return ["p-toast-message-icon", _defineProperty$3(_defineProperty$3(_defineProperty$3(_defineProperty$3({}, props.infoIcon, props.message.severity === "info"), props.warnIcon, props.message.severity === "warn"), props.errorIcon, props.message.severity === "error"), props.successIcon, props.message.severity === "success")];
  }, "messageIcon"),
  messageText: "p-toast-message-text",
  summary: "p-toast-summary",
  detail: "p-toast-detail",
  closeButton: "p-toast-close-button",
  closeIcon: "p-toast-close-icon"
};
var ToastStyle = BaseStyle.extend({
  name: "toast",
  theme: theme$4,
  classes: classes$4,
  inlineStyles: inlineStyles$2
});
var script$2$2 = {
  name: "BaseToast",
  "extends": script$p,
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
  provide: /* @__PURE__ */ __name(function provide16() {
    return {
      $pcToast: this,
      $parentInstance: this
    };
  }, "provide")
};
var script$1$4 = {
  name: "ToastMessage",
  hostName: "Toast",
  "extends": script$p,
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
  mounted: /* @__PURE__ */ __name(function mounted11() {
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
  beforeUnmount: /* @__PURE__ */ __name(function beforeUnmount10() {
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
        info: !this.infoIcon && script$L,
        success: !this.successIcon && script$F,
        warn: !this.warnIcon && script$M,
        error: !this.errorIcon && script$N
      }[this.message.severity];
    }, "iconComponent"),
    closeAriaLabel: /* @__PURE__ */ __name(function closeAriaLabel2() {
      return this.$primevue.config.locale.aria ? this.$primevue.config.locale.aria.close : void 0;
    }, "closeAriaLabel")
  },
  components: {
    TimesIcon: script$C,
    InfoCircleIcon: script$L,
    CheckIcon: script$F,
    ExclamationTriangleIcon: script$M,
    TimesCircleIcon: script$N
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
var _hoisted_1$a = ["aria-label"];
function render$1$2(_ctx, _cache, $props, $setup, $data, $options) {
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
  }, _ctx.ptm("closeIcon")), null, 16, ["class"]))], 16, _hoisted_1$a)), [[_directive_ripple]])], 16)) : createCommentVNode("", true)], 16))], 16);
}
__name(render$1$2, "render$1$2");
script$1$4.render = render$1$2;
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
var script$6 = {
  name: "Toast",
  "extends": script$2$2,
  inheritAttrs: false,
  emits: ["close", "life-end"],
  data: /* @__PURE__ */ __name(function data12() {
    return {
      messages: []
    };
  }, "data"),
  styleElement: null,
  mounted: /* @__PURE__ */ __name(function mounted12() {
    ToastEventBus.on("add", this.onAdd);
    ToastEventBus.on("remove", this.onRemove);
    ToastEventBus.on("remove-group", this.onRemoveGroup);
    ToastEventBus.on("remove-all-groups", this.onRemoveAllGroups);
    if (this.breakpoints) {
      this.createStyle();
    }
  }, "mounted"),
  beforeUnmount: /* @__PURE__ */ __name(function beforeUnmount11() {
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
    add: /* @__PURE__ */ __name(function add(message3) {
      if (message3.id == null) {
        message3.id = messageIdx++;
      }
      this.messages = [].concat(_toConsumableArray(this.messages), [message3]);
    }, "add"),
    remove: /* @__PURE__ */ __name(function remove(params) {
      var index2 = this.messages.findIndex(function(m) {
        return m.id === params.message.id;
      });
      if (index2 !== -1) {
        this.messages.splice(index2, 1);
        this.$emit(params.type, {
          message: params.message
        });
      }
    }, "remove"),
    onAdd: /* @__PURE__ */ __name(function onAdd(message3) {
      if (this.group == message3.group) {
        this.add(message3);
      }
    }, "onAdd"),
    onRemove: /* @__PURE__ */ __name(function onRemove(message3) {
      this.remove({
        message: message3,
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
    onEnter: /* @__PURE__ */ __name(function onEnter6() {
      this.$refs.container.setAttribute(this.attributeSelector, "");
      if (this.autoZIndex) {
        ZIndex.set("modal", this.$refs.container, this.baseZIndex || this.$primevue.config.zIndex.modal);
      }
    }, "onEnter"),
    onLeave: /* @__PURE__ */ __name(function onLeave4() {
      var _this = this;
      if (this.$refs.container && this.autoZIndex && isEmpty(this.messages)) {
        setTimeout(function() {
          ZIndex.clear(_this.$refs.container);
        }, 200);
      }
    }, "onLeave"),
    createStyle: /* @__PURE__ */ __name(function createStyle3() {
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
          innerHTML += "\n                        @media screen and (max-width: ".concat(breakpoint, ") {\n                            .p-toast[").concat(this.attributeSelector, "] {\n                                ").concat(breakpointStyle, "\n                            }\n                        }\n                    ");
        }
        this.styleElement.innerHTML = innerHTML;
      }
    }, "createStyle"),
    destroyStyle: /* @__PURE__ */ __name(function destroyStyle2() {
      if (this.styleElement) {
        document.head.removeChild(this.styleElement);
        this.styleElement = null;
      }
    }, "destroyStyle")
  },
  computed: {
    attributeSelector: /* @__PURE__ */ __name(function attributeSelector2() {
      return UniqueComponentId();
    }, "attributeSelector")
  },
  components: {
    ToastMessage: script$1$4,
    Portal: script$r
  }
};
function _typeof$2(o) {
  "@babel/helpers - typeof";
  return _typeof$2 = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function(o2) {
    return typeof o2;
  } : function(o2) {
    return o2 && "function" == typeof Symbol && o2.constructor === Symbol && o2 !== Symbol.prototype ? "symbol" : typeof o2;
  }, _typeof$2(o);
}
__name(_typeof$2, "_typeof$2");
function ownKeys$2(e, r) {
  var t = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var o = Object.getOwnPropertySymbols(e);
    r && (o = o.filter(function(r2) {
      return Object.getOwnPropertyDescriptor(e, r2).enumerable;
    })), t.push.apply(t, o);
  }
  return t;
}
__name(ownKeys$2, "ownKeys$2");
function _objectSpread$2(e) {
  for (var r = 1; r < arguments.length; r++) {
    var t = null != arguments[r] ? arguments[r] : {};
    r % 2 ? ownKeys$2(Object(t), true).forEach(function(r2) {
      _defineProperty$2(e, r2, t[r2]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(t)) : ownKeys$2(Object(t)).forEach(function(r2) {
      Object.defineProperty(e, r2, Object.getOwnPropertyDescriptor(t, r2));
    });
  }
  return e;
}
__name(_objectSpread$2, "_objectSpread$2");
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
function render$5(_ctx, _cache, $props, $setup, $data, $options) {
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
      }, _objectSpread$2({}, _ctx.ptm("transition"))), {
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
__name(render$5, "render$5");
script$6.render = render$5;
const _sfc_main$c = /* @__PURE__ */ defineComponent({
  __name: "GlobalToast",
  setup(__props) {
    const toast = useToast();
    const toastStore = useToastStore();
    const settingStore = useSettingStore();
    watch(
      () => toastStore.messagesToAdd,
      (newMessages) => {
        if (newMessages.length === 0) {
          return;
        }
        newMessages.forEach((message3) => {
          toast.add(message3);
        });
        toastStore.messagesToAdd = [];
      },
      { deep: true }
    );
    watch(
      () => toastStore.messagesToRemove,
      (messagesToRemove) => {
        if (messagesToRemove.length === 0) {
          return;
        }
        messagesToRemove.forEach((message3) => {
          toast.remove(message3);
        });
        toastStore.messagesToRemove = [];
      },
      { deep: true }
    );
    watch(
      () => toastStore.removeAllRequested,
      (requested) => {
        if (requested) {
          toast.removeAllGroups();
          toastStore.removeAllRequested = false;
        }
      }
    );
    function updateToastPosition() {
      const styleElement = document.getElementById("dynamic-toast-style") || createStyleElement();
      const rect = document.querySelector(".graph-canvas-container").getBoundingClientRect();
      styleElement.textContent = `
    .p-toast.p-component.p-toast-top-right {
      top: ${rect.top + 20}px !important;
      right: ${window.innerWidth - (rect.left + rect.width) + 20}px !important;
    }
  `;
    }
    __name(updateToastPosition, "updateToastPosition");
    function createStyleElement() {
      const style = document.createElement("style");
      style.id = "dynamic-toast-style";
      document.head.appendChild(style);
      return style;
    }
    __name(createStyleElement, "createStyleElement");
    watch(
      () => settingStore.get("Comfy.UseNewMenu"),
      () => nextTick(updateToastPosition),
      { immediate: true }
    );
    watch(
      () => settingStore.get("Comfy.Sidebar.Location"),
      () => nextTick(updateToastPosition),
      { immediate: true }
    );
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(script$6));
    };
  }
});
const _sfc_main$b = /* @__PURE__ */ defineComponent({
  __name: "UnloadWindowConfirmDialog",
  setup(__props) {
    const settingStore = useSettingStore();
    const handleBeforeUnload = /* @__PURE__ */ __name((event) => {
      if (settingStore.get("Comfy.Window.UnloadConfirmation")) {
        event.preventDefault();
        return true;
      }
      return void 0;
    }, "handleBeforeUnload");
    onMounted(() => {
      window.addEventListener("beforeunload", handleBeforeUnload);
    });
    onBeforeUnmount(() => {
      window.removeEventListener("beforeunload", handleBeforeUnload);
    });
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div");
    };
  }
});
const DEFAULT_TITLE = "ComfyUI";
const TITLE_SUFFIX = " - ComfyUI";
const _sfc_main$a = /* @__PURE__ */ defineComponent({
  __name: "BrowserTabTitle",
  setup(__props) {
    const executionStore = useExecutionStore();
    const executionText = computed(
      () => executionStore.isIdle ? "" : `[${executionStore.executionProgress}%]`
    );
    const settingStore = useSettingStore();
    const betaMenuEnabled = computed(
      () => settingStore.get("Comfy.UseNewMenu") !== "Disabled"
    );
    const workflowStore = useWorkflowStore();
    const isUnsavedText = computed(
      () => workflowStore.activeWorkflow?.unsaved ? " *" : ""
    );
    const workflowNameText = computed(() => {
      const workflowName = workflowStore.activeWorkflow?.name;
      return workflowName ? isUnsavedText.value + workflowName + TITLE_SUFFIX : DEFAULT_TITLE;
    });
    const nodeExecutionTitle = computed(
      () => executionStore.executingNode && executionStore.executingNodeProgress ? `${executionText.value}[${executionStore.executingNodeProgress}%] ${executionStore.executingNode.type}` : ""
    );
    const workflowTitle = computed(
      () => executionText.value + (betaMenuEnabled.value ? workflowNameText.value : DEFAULT_TITLE)
    );
    const title = computed(() => nodeExecutionTitle.value || workflowTitle.value);
    useTitle(title);
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div");
    };
  }
});
const _sfc_main$9 = /* @__PURE__ */ defineComponent({
  __name: "WorkflowTreeLeaf",
  props: {
    node: {}
  },
  setup(__props) {
    const props = __props;
    const workflowBookmarkStore = useWorkflowBookmarkStore();
    const isBookmarked = computed(
      () => workflowBookmarkStore.isBookmarked(props.node.data.path)
    );
    return (_ctx, _cache) => {
      return openBlock(), createBlock(TreeExplorerTreeNode, { node: _ctx.node }, {
        actions: withCtx(({ node: node2 }) => [
          createVNode(unref(script$o), {
            icon: isBookmarked.value ? "pi pi-bookmark-fill" : "pi pi-bookmark",
            text: "",
            severity: "secondary",
            size: "small",
            onClick: withModifiers(($event) => unref(workflowBookmarkStore).toggleBookmarked(node2.data.path), ["stop"])
          }, null, 8, ["icon", "onClick"])
        ]),
        _: 1
      }, 8, ["node"]);
    };
  }
});
const _hoisted_1$9 = {
  key: 0,
  class: "mr-2 shrink-0"
};
const _hoisted_2$7 = {
  key: 1,
  class: "ml-2 shrink-0"
};
const _sfc_main$8 = /* @__PURE__ */ defineComponent({
  __name: "TextDivider",
  props: {
    text: {},
    class: {},
    position: { default: "left" },
    align: { default: "center" },
    type: { default: "solid" },
    layout: { default: "horizontal" }
  },
  setup(__props) {
    const props = __props;
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", {
        class: normalizeClass(["flex items-center", props.class])
      }, [
        _ctx.position === "left" ? (openBlock(), createElementBlock("span", _hoisted_1$9, toDisplayString(_ctx.text), 1)) : createCommentVNode("", true),
        createVNode(unref(script$K), {
          align: _ctx.align,
          type: _ctx.type,
          layout: _ctx.layout,
          class: "flex-grow"
        }, null, 8, ["align", "type", "layout"]),
        _ctx.position === "right" ? (openBlock(), createElementBlock("span", _hoisted_2$7, toDisplayString(_ctx.text), 1)) : createCommentVNode("", true)
      ], 2);
    };
  }
});
const _withScopeId$4 = /* @__PURE__ */ __name((n) => (pushScopeId("data-v-d2d58252"), n = n(), popScopeId(), n), "_withScopeId$4");
const _hoisted_1$8 = {
  key: 0,
  class: "comfyui-workflows-panel"
};
const _hoisted_2$6 = {
  key: 0,
  class: "comfyui-workflows-open"
};
const _hoisted_3$3 = { key: 0 };
const _hoisted_4$2 = { class: "comfyui-workflows-bookmarks" };
const _hoisted_5$2 = { class: "comfyui-workflows-browse" };
const _hoisted_6$1 = {
  key: 1,
  class: "comfyui-workflows-search-panel"
};
const _sfc_main$7 = /* @__PURE__ */ defineComponent({
  __name: "WorkflowsSidebarTab",
  setup(__props) {
    const settingStore = useSettingStore();
    const workflowTabsPosition = computed(
      () => settingStore.get("Comfy.Workflow.WorkflowTabsPosition")
    );
    const searchQuery = ref("");
    const isSearching = computed(() => searchQuery.value.length > 0);
    const filteredWorkflows = ref([]);
    const filteredRoot = computed(() => {
      return workflowStore.buildWorkflowTree(
        filteredWorkflows.value
      );
    });
    const handleSearch = /* @__PURE__ */ __name((query) => {
      if (query.length === 0) {
        filteredWorkflows.value = [];
        expandedKeys2.value = {};
        return;
      }
      const lowerQuery = query.toLocaleLowerCase();
      filteredWorkflows.value = workflowStore.workflows.filter((workflow) => {
        return workflow.name.toLocaleLowerCase().includes(lowerQuery);
      });
      nextTick(() => {
        expandNode(filteredRoot.value);
      });
    }, "handleSearch");
    const commandStore = useCommandStore();
    const workflowStore = useWorkflowStore();
    const { t } = useI18n();
    const expandedKeys2 = ref({});
    const { expandNode, toggleNodeOnEvent } = useTreeExpansion(expandedKeys2);
    const renderTreeNode = /* @__PURE__ */ __name((node2) => {
      const children = node2.children?.map(renderTreeNode);
      const workflow = node2.data;
      const handleClick = /* @__PURE__ */ __name((node22, e) => {
        if (node22.leaf) {
          const workflow2 = node22.data;
          workflow2.load();
        } else {
          toggleNodeOnEvent(e, node22);
        }
      }, "handleClick");
      const actions = node2.leaf ? {
        handleClick,
        handleRename: /* @__PURE__ */ __name((node22, newName) => {
          const workflow2 = node22.data;
          workflow2.rename(newName);
        }, "handleRename"),
        handleDelete: workflow.isTemporary ? void 0 : (node22) => {
          const workflow2 = node22.data;
          workflow2.delete();
        },
        contextMenuItems: /* @__PURE__ */ __name((node22) => {
          return [
            {
              label: t("insert"),
              icon: "pi pi-file-export",
              command: /* @__PURE__ */ __name(() => {
                const workflow2 = node22.data;
                workflow2.insert();
              }, "command")
            }
          ];
        }, "contextMenuItems")
      } : { handleClick };
      return {
        key: node2.key,
        label: node2.label,
        leaf: node2.leaf,
        data: node2.data,
        children,
        ...actions
      };
    }, "renderTreeNode");
    const selectionKeys = computed(() => ({
      [`root/${workflowStore.activeWorkflow?.name}.json`]: true
    }));
    return (_ctx, _cache) => {
      const _directive_tooltip = resolveDirective("tooltip");
      return openBlock(), createBlock(SidebarTabTemplate, {
        title: _ctx.$t("sideToolbar.workflows")
      }, {
        "tool-buttons": withCtx(() => [
          withDirectives(createVNode(unref(script$o), {
            class: "browse-templates-button",
            icon: "pi pi-th-large",
            text: "",
            onClick: _cache[0] || (_cache[0] = () => unref(commandStore).execute("Comfy.BrowseTemplates"))
          }, null, 512), [
            [_directive_tooltip, _ctx.$t("sideToolbar.browseTemplates")]
          ]),
          withDirectives(createVNode(unref(script$o), {
            class: "open-workflow-button",
            icon: "pi pi-folder-open",
            text: "",
            onClick: _cache[1] || (_cache[1] = () => unref(commandStore).execute("Comfy.OpenWorkflow"))
          }, null, 512), [
            [_directive_tooltip, _ctx.$t("sideToolbar.openWorkflow")]
          ]),
          withDirectives(createVNode(unref(script$o), {
            class: "new-blank-workflow-button",
            icon: "pi pi-plus",
            onClick: _cache[2] || (_cache[2] = () => unref(commandStore).execute("Comfy.NewBlankWorkflow")),
            text: ""
          }, null, 512), [
            [_directive_tooltip, _ctx.$t("sideToolbar.newBlankWorkflow")]
          ])
        ]),
        body: withCtx(() => [
          createVNode(SearchBox, {
            class: "workflows-search-box mx-4 my-4",
            modelValue: searchQuery.value,
            "onUpdate:modelValue": _cache[3] || (_cache[3] = ($event) => searchQuery.value = $event),
            onSearch: handleSearch,
            placeholder: _ctx.$t("searchWorkflows") + "..."
          }, null, 8, ["modelValue", "placeholder"]),
          !isSearching.value ? (openBlock(), createElementBlock("div", _hoisted_1$8, [
            workflowTabsPosition.value === "Sidebar" ? (openBlock(), createElementBlock("div", _hoisted_2$6, [
              createVNode(_sfc_main$8, {
                text: "Open",
                type: "dashed",
                class: "ml-2"
              }),
              createVNode(TreeExplorer, {
                roots: renderTreeNode(unref(workflowStore).openWorkflowsTree).children,
                selectionKeys: selectionKeys.value,
                "onUpdate:selectionKeys": _cache[4] || (_cache[4] = ($event) => selectionKeys.value = $event)
              }, {
                node: withCtx(({ node: node2 }) => [
                  createVNode(TreeExplorerTreeNode, { node: node2 }, {
                    "before-label": withCtx(({ node: node22 }) => [
                      node22.data.unsaved ? (openBlock(), createElementBlock("span", _hoisted_3$3, "*")) : createCommentVNode("", true)
                    ]),
                    actions: withCtx(({ node: node22 }) => [
                      createVNode(unref(script$o), {
                        icon: "pi pi-times",
                        text: "",
                        severity: "secondary",
                        size: "small",
                        onClick: withModifiers(($event) => unref(app).workflowManager.closeWorkflow(node22.data), ["stop"])
                      }, null, 8, ["onClick"])
                    ]),
                    _: 2
                  }, 1032, ["node"])
                ]),
                _: 1
              }, 8, ["roots", "selectionKeys"])
            ])) : createCommentVNode("", true),
            withDirectives(createBaseVNode("div", _hoisted_4$2, [
              createVNode(_sfc_main$8, {
                text: "Bookmarks",
                type: "dashed",
                class: "ml-2"
              }),
              createVNode(TreeExplorer, {
                roots: renderTreeNode(unref(workflowStore).bookmarkedWorkflowsTree).children
              }, {
                node: withCtx(({ node: node2 }) => [
                  createVNode(_sfc_main$9, { node: node2 }, null, 8, ["node"])
                ]),
                _: 1
              }, 8, ["roots"])
            ], 512), [
              [vShow, unref(workflowStore).bookmarkedWorkflows.length > 0]
            ]),
            createBaseVNode("div", _hoisted_5$2, [
              createVNode(_sfc_main$8, {
                text: "Browse",
                type: "dashed",
                class: "ml-2"
              }),
              createVNode(TreeExplorer, {
                roots: renderTreeNode(unref(workflowStore).workflowsTree).children,
                expandedKeys: expandedKeys2.value,
                "onUpdate:expandedKeys": _cache[5] || (_cache[5] = ($event) => expandedKeys2.value = $event)
              }, {
                node: withCtx(({ node: node2 }) => [
                  createVNode(_sfc_main$9, { node: node2 }, null, 8, ["node"])
                ]),
                _: 1
              }, 8, ["roots", "expandedKeys"])
            ])
          ])) : (openBlock(), createElementBlock("div", _hoisted_6$1, [
            createVNode(TreeExplorer, {
              roots: renderTreeNode(filteredRoot.value).children,
              expandedKeys: expandedKeys2.value,
              "onUpdate:expandedKeys": _cache[6] || (_cache[6] = ($event) => expandedKeys2.value = $event)
            }, {
              node: withCtx(({ node: node2 }) => [
                createVNode(_sfc_main$9, { node: node2 }, null, 8, ["node"])
              ]),
              _: 1
            }, 8, ["roots", "expandedKeys"])
          ]))
        ]),
        _: 1
      }, 8, ["title"]);
    };
  }
});
const WorkflowsSidebarTab = /* @__PURE__ */ _export_sfc(_sfc_main$7, [["__scopeId", "data-v-d2d58252"]]);
const _withScopeId$3 = /* @__PURE__ */ __name((n) => (pushScopeId("data-v-84e785b8"), n = n(), popScopeId(), n), "_withScopeId$3");
const _hoisted_1$7 = { class: "workflow-label text-sm max-w-[150px] truncate inline-block" };
const _hoisted_2$5 = { class: "relative" };
const _hoisted_3$2 = {
  key: 0,
  class: "status-indicator"
};
const _sfc_main$6 = /* @__PURE__ */ defineComponent({
  __name: "WorkflowTabs",
  props: {
    class: {}
  },
  setup(__props) {
    const props = __props;
    const workflowStore = useWorkflowStore();
    const workflowToOption = /* @__PURE__ */ __name((workflow) => ({
      label: workflow.name,
      tooltip: workflow.path,
      value: workflow.key,
      unsaved: workflow.unsaved
    }), "workflowToOption");
    const optionToWorkflow = /* @__PURE__ */ __name((option2) => workflowStore.workflowLookup[option2.value], "optionToWorkflow");
    const options = computed(
      () => workflowStore.openWorkflows.map(workflowToOption)
    );
    const selectedWorkflow = computed(
      () => workflowStore.activeWorkflow ? workflowToOption(workflowStore.activeWorkflow) : null
    );
    const onWorkflowChange = /* @__PURE__ */ __name((option2) => {
      if (!option2) {
        return;
      }
      if (selectedWorkflow.value?.value === option2.value) {
        return;
      }
      const workflow = optionToWorkflow(option2);
      workflow.load();
    }, "onWorkflowChange");
    const onCloseWorkflow = /* @__PURE__ */ __name((option2) => {
      const workflow = optionToWorkflow(option2);
      app.workflowManager.closeWorkflow(workflow);
    }, "onCloseWorkflow");
    return (_ctx, _cache) => {
      const _directive_tooltip = resolveDirective("tooltip");
      return openBlock(), createBlock(unref(script$g), {
        class: normalizeClass(["workflow-tabs bg-transparent flex flex-wrap", props.class]),
        modelValue: selectedWorkflow.value,
        "onUpdate:modelValue": onWorkflowChange,
        options: options.value,
        optionLabel: "label",
        dataKey: "value"
      }, {
        option: withCtx(({ option: option2 }) => [
          withDirectives((openBlock(), createElementBlock("span", _hoisted_1$7, [
            createTextVNode(toDisplayString(option2.label), 1)
          ])), [
            [_directive_tooltip, option2.tooltip]
          ]),
          createBaseVNode("div", _hoisted_2$5, [
            option2.unsaved ? (openBlock(), createElementBlock("span", _hoisted_3$2, "•")) : createCommentVNode("", true),
            createVNode(unref(script$o), {
              class: "close-button p-0 w-auto",
              icon: "pi pi-times",
              text: "",
              severity: "secondary",
              size: "small",
              onClick: withModifiers(($event) => onCloseWorkflow(option2), ["stop"])
            }, null, 8, ["onClick"])
          ])
        ]),
        _: 1
      }, 8, ["class", "modelValue", "options"]);
    };
  }
});
const WorkflowTabs = /* @__PURE__ */ _export_sfc(_sfc_main$6, [["__scopeId", "data-v-84e785b8"]]);
var theme$3 = /* @__PURE__ */ __name(function theme15(_ref) {
  var dt = _ref.dt;
  return "\n.p-menubar {\n    display: flex;\n    align-items: center;\n    background: ".concat(dt("menubar.background"), ";\n    border: 1px solid ").concat(dt("menubar.border.color"), ";\n    border-radius: ").concat(dt("menubar.border.radius"), ";\n    color: ").concat(dt("menubar.color"), ";\n    padding: ").concat(dt("menubar.padding"), ";\n    gap: ").concat(dt("menubar.gap"), ";\n}\n\n.p-menubar-start,\n.p-megamenu-end {\n    display: flex;\n    align-items: center;\n}\n\n.p-menubar-root-list,\n.p-menubar-submenu {\n    display: flex;\n    margin: 0;\n    padding: 0;\n    list-style: none;\n    outline: 0 none;\n}\n\n.p-menubar-root-list {\n    align-items: center;\n    flex-wrap: wrap;\n    gap: ").concat(dt("menubar.gap"), ";\n}\n\n.p-menubar-root-list > .p-menubar-item > .p-menubar-item-content {\n    border-radius: ").concat(dt("menubar.base.item.border.radius"), ";\n}\n\n.p-menubar-root-list > .p-menubar-item > .p-menubar-item-content > .p-menubar-item-link {\n    padding: ").concat(dt("menubar.base.item.padding"), ";\n}\n\n.p-menubar-item-content {\n    transition: background ").concat(dt("menubar.transition.duration"), ", color ").concat(dt("menubar.transition.duration"), ";\n    border-radius: ").concat(dt("menubar.item.border.radius"), ";\n    color: ").concat(dt("menubar.item.color"), ";\n}\n\n.p-menubar-item-link {\n    cursor: pointer;\n    display: flex;\n    align-items: center;\n    text-decoration: none;\n    overflow: hidden;\n    position: relative;\n    color: inherit;\n    padding: ").concat(dt("menubar.item.padding"), ";\n    gap: ").concat(dt("menubar.item.gap"), ";\n    user-select: none;\n    outline: 0 none;\n}\n\n.p-menubar-item-label {\n    line-height: 1;\n}\n\n.p-menubar-item-icon {\n    color: ").concat(dt("menubar.item.icon.color"), ";\n}\n\n.p-menubar-submenu-icon {\n    color: ").concat(dt("menubar.submenu.icon.color"), ";\n    margin-left: auto;\n    font-size: ").concat(dt("menubar.submenu.icon.size"), ";\n    width: ").concat(dt("menubar.submenu.icon.size"), ";\n    height: ").concat(dt("menubar.submenu.icon.size"), ";\n}\n\n.p-menubar-item.p-focus > .p-menubar-item-content {\n    color: ").concat(dt("menubar.item.focus.color"), ";\n    background: ").concat(dt("menubar.item.focus.background"), ";\n}\n\n.p-menubar-item.p-focus > .p-menubar-item-content .p-menubar-item-icon {\n    color: ").concat(dt("menubar.item.icon.focus.color"), ";\n}\n\n.p-menubar-item.p-focus > .p-menubar-item-content .p-menubar-submenu-icon {\n    color: ").concat(dt("menubar.submenu.icon.focus.color"), ";\n}\n\n.p-menubar-item:not(.p-disabled) > .p-menubar-item-content:hover {\n    color: ").concat(dt("menubar.item.focus.color"), ";\n    background: ").concat(dt("menubar.item.focus.background"), ";\n}\n\n.p-menubar-item:not(.p-disabled) > .p-menubar-item-content:hover .p-menubar-item-icon {\n    color: ").concat(dt("menubar.item.icon.focus.color"), ";\n}\n\n.p-menubar-item:not(.p-disabled) > .p-menubar-item-content:hover .p-menubar-submenu-icon {\n    color: ").concat(dt("menubar.submenu.icon.focus.color"), ";\n}\n\n.p-menubar-item-active > .p-menubar-item-content {\n    color: ").concat(dt("menubar.item.active.color"), ";\n    background: ").concat(dt("menubar.item.active.background"), ";\n}\n\n.p-menubar-item-active > .p-menubar-item-content .p-menubar-item-icon {\n    color: ").concat(dt("menubar.item.icon.active.color"), ";\n}\n\n.p-menubar-item-active > .p-menubar-item-content .p-menubar-submenu-icon {\n    color: ").concat(dt("menubar.submenu.icon.active.color"), ";\n}\n\n.p-menubar-submenu {\n    display: none;\n    position: absolute;\n    min-width: 12.5rem;\n    z-index: 1;\n    background: ").concat(dt("menubar.submenu.background"), ";\n    border: 1px solid ").concat(dt("menubar.submenu.border.color"), ";\n    border-radius: ").concat(dt("menubar.border.radius"), ";\n    box-shadow: ").concat(dt("menubar.submenu.shadow"), ";\n    color: ").concat(dt("menubar.submenu.color"), ";\n    flex-direction: column;\n    padding: ").concat(dt("menubar.submenu.padding"), ";\n    gap: ").concat(dt("menubar.submenu.gap"), ";\n}\n\n.p-menubar-submenu .p-menubar-separator {\n    border-top: 1px solid ").concat(dt("menubar.separator.border.color"), ";\n}\n\n.p-menubar-submenu .p-menubar-item {\n    position: relative;\n}\n\n .p-menubar-submenu > .p-menubar-item-active > .p-menubar-submenu {\n    display: block;\n    left: 100%;\n    top: 0;\n}\n\n.p-menubar-end {\n    margin-left: auto;\n    align-self: center;\n}\n\n.p-menubar-button {\n    display: none;\n    justify-content: center;\n    align-items: center;\n    cursor: pointer;\n    width: ").concat(dt("menubar.mobile.button.size"), ";\n    height: ").concat(dt("menubar.mobile.button.size"), ";\n    position: relative;\n    color: ").concat(dt("menubar.mobile.button.color"), ";\n    border: 0 none;\n    background: transparent;\n    border-radius: ").concat(dt("menubar.mobile.button.border.radius"), ";\n    transition: background ").concat(dt("menubar.transition.duration"), ", color ").concat(dt("menubar.transition.duration"), ", outline-color ").concat(dt("menubar.transition.duration"), ";\n    outline-color: transparent;\n}\n\n.p-menubar-button:hover {\n    color: ").concat(dt("menubar.mobile.button.hover.color"), ";\n    background: ").concat(dt("menubar.mobile.button.hover.background"), ";\n}\n\n.p-menubar-button:focus-visible {\n    box-shadow: ").concat(dt("menubar.mobile.button.focus.ring.shadow"), ";\n    outline: ").concat(dt("menubar.mobile.button.focus.ring.width"), " ").concat(dt("menubar.mobile.button.focus.ring.style"), " ").concat(dt("menubar.mobile.button.focus.ring.color"), ";\n    outline-offset: ").concat(dt("menubar.mobile.button.focus.ring.offset"), ";\n}\n\n.p-menubar-mobile {\n    position: relative;\n}\n\n.p-menubar-mobile .p-menubar-button {\n    display: flex;\n}\n\n.p-menubar-mobile .p-menubar-root-list {\n    position: absolute;\n    display: none;\n    width: 100%;\n    padding: ").concat(dt("menubar.submenu.padding"), ";\n    background: ").concat(dt("menubar.submenu.background"), ";\n    border: 1px solid ").concat(dt("menubar.submenu.border.color"), ";\n    box-shadow: ").concat(dt("menubar.submenu.shadow"), ";\n}\n\n.p-menubar-mobile .p-menubar-root-list > .p-menubar-item > .p-menubar-item-content {\n    border-radius: ").concat(dt("menubar.item.border.radius"), ";\n}\n\n.p-menubar-mobile .p-menubar-root-list > .p-menubar-item > .p-menubar-item-content > .p-menubar-item-link {\n    padding: ").concat(dt("menubar.item.padding"), ";\n}\n\n.p-menubar-mobile-active .p-menubar-root-list {\n    display: flex;\n    flex-direction: column;\n    top: 100%;\n    left: 0;\n    z-index: 1;\n}\n\n.p-menubar-mobile .p-menubar-root-list .p-menubar-item {\n    width: 100%;\n    position: static;\n}\n\n.p-menubar-mobile .p-menubar-root-list .p-menubar-separator {\n    border-top: 1px solid ").concat(dt("menubar.separator.border.color"), ";\n}\n\n.p-menubar-mobile .p-menubar-root-list > .p-menubar-item > .p-menubar-item-content .p-menubar-submenu-icon {\n    margin-left: auto;\n    transition: transform 0.2s;\n}\n\n.p-menubar-mobile .p-menubar-root-list > .p-menubar-item-active > .p-menubar-item-content .p-menubar-submenu-icon {\n    transform: rotate(-180deg);\n}\n\n.p-menubar-mobile .p-menubar-submenu .p-menubar-submenu-icon {\n    transition: transform 0.2s;\n    transform: rotate(90deg);\n}\n\n.p-menubar-mobile  .p-menubar-item-active > .p-menubar-item-content .p-menubar-submenu-icon {\n    transform: rotate(-90deg);\n}\n\n.p-menubar-mobile .p-menubar-submenu {\n    width: 100%;\n    position: static;\n    box-shadow: none;\n    border: 0 none;\n    padding-left: ").concat(dt("menubar.submenu.mobile.indent"), ";\n}\n");
}, "theme");
var inlineStyles$1 = {
  submenu: /* @__PURE__ */ __name(function submenu(_ref2) {
    var instance = _ref2.instance, processedItem = _ref2.processedItem;
    return {
      display: instance.isItemActive(processedItem) ? "flex" : "none"
    };
  }, "submenu")
};
var classes$3 = {
  root: /* @__PURE__ */ __name(function root11(_ref3) {
    var instance = _ref3.instance;
    return ["p-menubar p-component", {
      "p-menubar-mobile": instance.queryMatches,
      "p-menubar-mobile-active": instance.mobileActive
    }];
  }, "root"),
  start: "p-menubar-start",
  button: "p-menubar-button",
  rootList: "p-menubar-root-list",
  item: /* @__PURE__ */ __name(function item2(_ref4) {
    var instance = _ref4.instance, processedItem = _ref4.processedItem;
    return ["p-menubar-item", {
      "p-menubar-item-active": instance.isItemActive(processedItem),
      "p-focus": instance.isItemFocused(processedItem),
      "p-disabled": instance.isItemDisabled(processedItem)
    }];
  }, "item"),
  itemContent: "p-menubar-item-content",
  itemLink: "p-menubar-item-link",
  itemIcon: "p-menubar-item-icon",
  itemLabel: "p-menubar-item-label",
  submenuIcon: "p-menubar-submenu-icon",
  submenu: "p-menubar-submenu",
  separator: "p-menubar-separator",
  end: "p-menubar-end"
};
var MenubarStyle = BaseStyle.extend({
  name: "menubar",
  theme: theme$3,
  classes: classes$3,
  inlineStyles: inlineStyles$1
});
var script$2$1 = {
  name: "BaseMenubar",
  "extends": script$p,
  props: {
    model: {
      type: Array,
      "default": null
    },
    buttonProps: {
      type: null,
      "default": null
    },
    breakpoint: {
      type: String,
      "default": "960px"
    },
    ariaLabelledby: {
      type: String,
      "default": null
    },
    ariaLabel: {
      type: String,
      "default": null
    }
  },
  style: MenubarStyle,
  provide: /* @__PURE__ */ __name(function provide17() {
    return {
      $pcMenubar: this,
      $parentInstance: this
    };
  }, "provide")
};
var script$1$3 = {
  name: "MenubarSub",
  hostName: "Menubar",
  "extends": script$p,
  emits: ["item-mouseenter", "item-click", "item-mousemove"],
  props: {
    items: {
      type: Array,
      "default": null
    },
    root: {
      type: Boolean,
      "default": false
    },
    popup: {
      type: Boolean,
      "default": false
    },
    mobileActive: {
      type: Boolean,
      "default": false
    },
    templates: {
      type: Object,
      "default": null
    },
    level: {
      type: Number,
      "default": 0
    },
    menuId: {
      type: String,
      "default": null
    },
    focusedItemId: {
      type: String,
      "default": null
    },
    activeItemPath: {
      type: Object,
      "default": null
    }
  },
  list: null,
  methods: {
    getItemId: /* @__PURE__ */ __name(function getItemId2(processedItem) {
      return "".concat(this.menuId, "_").concat(processedItem.key);
    }, "getItemId"),
    getItemKey: /* @__PURE__ */ __name(function getItemKey2(processedItem) {
      return this.getItemId(processedItem);
    }, "getItemKey"),
    getItemProp: /* @__PURE__ */ __name(function getItemProp3(processedItem, name, params) {
      return processedItem && processedItem.item ? resolve(processedItem.item[name], params) : void 0;
    }, "getItemProp"),
    getItemLabel: /* @__PURE__ */ __name(function getItemLabel3(processedItem) {
      return this.getItemProp(processedItem, "label");
    }, "getItemLabel"),
    getItemLabelId: /* @__PURE__ */ __name(function getItemLabelId2(processedItem) {
      return "".concat(this.menuId, "_").concat(processedItem.key, "_label");
    }, "getItemLabelId"),
    getPTOptions: /* @__PURE__ */ __name(function getPTOptions9(processedItem, index2, key) {
      return this.ptm(key, {
        context: {
          item: processedItem.item,
          index: index2,
          active: this.isItemActive(processedItem),
          focused: this.isItemFocused(processedItem),
          disabled: this.isItemDisabled(processedItem),
          level: this.level
        }
      });
    }, "getPTOptions"),
    isItemActive: /* @__PURE__ */ __name(function isItemActive3(processedItem) {
      return this.activeItemPath.some(function(path) {
        return path.key === processedItem.key;
      });
    }, "isItemActive"),
    isItemVisible: /* @__PURE__ */ __name(function isItemVisible3(processedItem) {
      return this.getItemProp(processedItem, "visible") !== false;
    }, "isItemVisible"),
    isItemDisabled: /* @__PURE__ */ __name(function isItemDisabled3(processedItem) {
      return this.getItemProp(processedItem, "disabled");
    }, "isItemDisabled"),
    isItemFocused: /* @__PURE__ */ __name(function isItemFocused2(processedItem) {
      return this.focusedItemId === this.getItemId(processedItem);
    }, "isItemFocused"),
    isItemGroup: /* @__PURE__ */ __name(function isItemGroup3(processedItem) {
      return isNotEmpty(processedItem.items);
    }, "isItemGroup"),
    onItemClick: /* @__PURE__ */ __name(function onItemClick4(event, processedItem) {
      this.getItemProp(processedItem, "command", {
        originalEvent: event,
        item: processedItem.item
      });
      this.$emit("item-click", {
        originalEvent: event,
        processedItem,
        isFocus: true
      });
    }, "onItemClick"),
    onItemMouseEnter: /* @__PURE__ */ __name(function onItemMouseEnter3(event, processedItem) {
      this.$emit("item-mouseenter", {
        originalEvent: event,
        processedItem
      });
    }, "onItemMouseEnter"),
    onItemMouseMove: /* @__PURE__ */ __name(function onItemMouseMove3(event, processedItem) {
      this.$emit("item-mousemove", {
        originalEvent: event,
        processedItem
      });
    }, "onItemMouseMove"),
    getAriaPosInset: /* @__PURE__ */ __name(function getAriaPosInset3(index2) {
      return index2 - this.calculateAriaSetSize.slice(0, index2).length + 1;
    }, "getAriaPosInset"),
    getMenuItemProps: /* @__PURE__ */ __name(function getMenuItemProps2(processedItem, index2) {
      return {
        action: mergeProps({
          "class": this.cx("itemLink"),
          tabindex: -1,
          "aria-hidden": true
        }, this.getPTOptions(processedItem, index2, "itemLink")),
        icon: mergeProps({
          "class": [this.cx("itemIcon"), this.getItemProp(processedItem, "icon")]
        }, this.getPTOptions(processedItem, index2, "itemIcon")),
        label: mergeProps({
          "class": this.cx("itemLabel")
        }, this.getPTOptions(processedItem, index2, "itemLabel")),
        submenuicon: mergeProps({
          "class": this.cx("submenuIcon")
        }, this.getPTOptions(processedItem, index2, "submenuIcon"))
      };
    }, "getMenuItemProps")
  },
  computed: {
    calculateAriaSetSize: /* @__PURE__ */ __name(function calculateAriaSetSize() {
      var _this = this;
      return this.items.filter(function(processedItem) {
        return _this.isItemVisible(processedItem) && _this.getItemProp(processedItem, "separator");
      });
    }, "calculateAriaSetSize"),
    getAriaSetSize: /* @__PURE__ */ __name(function getAriaSetSize2() {
      var _this2 = this;
      return this.items.filter(function(processedItem) {
        return _this2.isItemVisible(processedItem) && !_this2.getItemProp(processedItem, "separator");
      }).length;
    }, "getAriaSetSize")
  },
  components: {
    AngleRightIcon: script$y,
    AngleDownIcon: script$O
  },
  directives: {
    ripple: Ripple
  }
};
var _hoisted_1$1$2 = ["id", "aria-label", "aria-disabled", "aria-expanded", "aria-haspopup", "aria-level", "aria-setsize", "aria-posinset", "data-p-active", "data-p-focused", "data-p-disabled"];
var _hoisted_2$4 = ["onClick", "onMouseenter", "onMousemove"];
var _hoisted_3$1 = ["href", "target"];
var _hoisted_4$1 = ["id"];
var _hoisted_5$1 = ["id"];
function render$1$1(_ctx, _cache, $props, $setup, $data, $options) {
  var _component_MenubarSub = resolveComponent("MenubarSub", true);
  var _directive_ripple = resolveDirective("ripple");
  return openBlock(), createElementBlock("ul", mergeProps({
    "class": $props.level === 0 ? _ctx.cx("rootList") : _ctx.cx("submenu")
  }, $props.level === 0 ? _ctx.ptm("rootList") : _ctx.ptm("submenu")), [(openBlock(true), createElementBlock(Fragment, null, renderList($props.items, function(processedItem, index2) {
    return openBlock(), createElementBlock(Fragment, {
      key: $options.getItemKey(processedItem)
    }, [$options.isItemVisible(processedItem) && !$options.getItemProp(processedItem, "separator") ? (openBlock(), createElementBlock("li", mergeProps({
      key: 0,
      id: $options.getItemId(processedItem),
      style: $options.getItemProp(processedItem, "style"),
      "class": [_ctx.cx("item", {
        processedItem
      }), $options.getItemProp(processedItem, "class")],
      role: "menuitem",
      "aria-label": $options.getItemLabel(processedItem),
      "aria-disabled": $options.isItemDisabled(processedItem) || void 0,
      "aria-expanded": $options.isItemGroup(processedItem) ? $options.isItemActive(processedItem) : void 0,
      "aria-haspopup": $options.isItemGroup(processedItem) && !$options.getItemProp(processedItem, "to") ? "menu" : void 0,
      "aria-level": $props.level + 1,
      "aria-setsize": $options.getAriaSetSize,
      "aria-posinset": $options.getAriaPosInset(index2),
      ref_for: true
    }, $options.getPTOptions(processedItem, index2, "item"), {
      "data-p-active": $options.isItemActive(processedItem),
      "data-p-focused": $options.isItemFocused(processedItem),
      "data-p-disabled": $options.isItemDisabled(processedItem)
    }), [createBaseVNode("div", mergeProps({
      "class": _ctx.cx("itemContent"),
      onClick: /* @__PURE__ */ __name(function onClick2($event) {
        return $options.onItemClick($event, processedItem);
      }, "onClick"),
      onMouseenter: /* @__PURE__ */ __name(function onMouseenter($event) {
        return $options.onItemMouseEnter($event, processedItem);
      }, "onMouseenter"),
      onMousemove: /* @__PURE__ */ __name(function onMousemove($event) {
        return $options.onItemMouseMove($event, processedItem);
      }, "onMousemove"),
      ref_for: true
    }, $options.getPTOptions(processedItem, index2, "itemContent")), [!$props.templates.item ? withDirectives((openBlock(), createElementBlock("a", mergeProps({
      key: 0,
      href: $options.getItemProp(processedItem, "url"),
      "class": _ctx.cx("itemLink"),
      target: $options.getItemProp(processedItem, "target"),
      tabindex: "-1",
      ref_for: true
    }, $options.getPTOptions(processedItem, index2, "itemLink")), [$props.templates.itemicon ? (openBlock(), createBlock(resolveDynamicComponent($props.templates.itemicon), {
      key: 0,
      item: processedItem.item,
      "class": normalizeClass(_ctx.cx("itemIcon"))
    }, null, 8, ["item", "class"])) : $options.getItemProp(processedItem, "icon") ? (openBlock(), createElementBlock("span", mergeProps({
      key: 1,
      "class": [_ctx.cx("itemIcon"), $options.getItemProp(processedItem, "icon")],
      ref_for: true
    }, $options.getPTOptions(processedItem, index2, "itemIcon")), null, 16)) : createCommentVNode("", true), createBaseVNode("span", mergeProps({
      id: $options.getItemLabelId(processedItem),
      "class": _ctx.cx("itemLabel"),
      ref_for: true
    }, $options.getPTOptions(processedItem, index2, "itemLabel")), toDisplayString($options.getItemLabel(processedItem)), 17, _hoisted_4$1), $options.getItemProp(processedItem, "items") ? (openBlock(), createElementBlock(Fragment, {
      key: 2
    }, [$props.templates.submenuicon ? (openBlock(), createBlock(resolveDynamicComponent($props.templates.submenuicon), {
      key: 0,
      root: $props.root,
      active: $options.isItemActive(processedItem),
      "class": normalizeClass(_ctx.cx("submenuIcon"))
    }, null, 8, ["root", "active", "class"])) : (openBlock(), createBlock(resolveDynamicComponent($props.root ? "AngleDownIcon" : "AngleRightIcon"), mergeProps({
      key: 1,
      "class": _ctx.cx("submenuIcon"),
      ref_for: true
    }, $options.getPTOptions(processedItem, index2, "submenuIcon")), null, 16, ["class"]))], 64)) : createCommentVNode("", true)], 16, _hoisted_3$1)), [[_directive_ripple]]) : (openBlock(), createBlock(resolveDynamicComponent($props.templates.item), {
      key: 1,
      item: processedItem.item,
      root: $props.root,
      hasSubmenu: $options.getItemProp(processedItem, "items"),
      label: $options.getItemLabel(processedItem),
      props: $options.getMenuItemProps(processedItem, index2)
    }, null, 8, ["item", "root", "hasSubmenu", "label", "props"]))], 16, _hoisted_2$4), $options.isItemVisible(processedItem) && $options.isItemGroup(processedItem) ? (openBlock(), createBlock(_component_MenubarSub, {
      key: 0,
      id: $options.getItemId(processedItem) + "_list",
      menuId: $props.menuId,
      role: "menu",
      style: normalizeStyle(_ctx.sx("submenu", true, {
        processedItem
      })),
      focusedItemId: $props.focusedItemId,
      items: processedItem.items,
      mobileActive: $props.mobileActive,
      activeItemPath: $props.activeItemPath,
      templates: $props.templates,
      level: $props.level + 1,
      "aria-labelledby": $options.getItemLabelId(processedItem),
      pt: _ctx.pt,
      unstyled: _ctx.unstyled,
      onItemClick: _cache[0] || (_cache[0] = function($event) {
        return _ctx.$emit("item-click", $event);
      }),
      onItemMouseenter: _cache[1] || (_cache[1] = function($event) {
        return _ctx.$emit("item-mouseenter", $event);
      }),
      onItemMousemove: _cache[2] || (_cache[2] = function($event) {
        return _ctx.$emit("item-mousemove", $event);
      })
    }, null, 8, ["id", "menuId", "style", "focusedItemId", "items", "mobileActive", "activeItemPath", "templates", "level", "aria-labelledby", "pt", "unstyled"])) : createCommentVNode("", true)], 16, _hoisted_1$1$2)) : createCommentVNode("", true), $options.isItemVisible(processedItem) && $options.getItemProp(processedItem, "separator") ? (openBlock(), createElementBlock("li", mergeProps({
      key: 1,
      id: $options.getItemId(processedItem),
      "class": [_ctx.cx("separator"), $options.getItemProp(processedItem, "class")],
      style: $options.getItemProp(processedItem, "style"),
      role: "separator",
      ref_for: true
    }, _ctx.ptm("separator")), null, 16, _hoisted_5$1)) : createCommentVNode("", true)], 64);
  }), 128))], 16);
}
__name(render$1$1, "render$1$1");
script$1$3.render = render$1$1;
var script$5 = {
  name: "Menubar",
  "extends": script$2$1,
  inheritAttrs: false,
  emits: ["focus", "blur"],
  matchMediaListener: null,
  data: /* @__PURE__ */ __name(function data13() {
    return {
      id: this.$attrs.id,
      mobileActive: false,
      focused: false,
      focusedItemInfo: {
        index: -1,
        level: 0,
        parentKey: ""
      },
      activeItemPath: [],
      dirty: false,
      query: null,
      queryMatches: false
    };
  }, "data"),
  watch: {
    "$attrs.id": /* @__PURE__ */ __name(function $attrsId4(newValue) {
      this.id = newValue || UniqueComponentId();
    }, "$attrsId"),
    activeItemPath: /* @__PURE__ */ __name(function activeItemPath2(newPath) {
      if (isNotEmpty(newPath)) {
        this.bindOutsideClickListener();
        this.bindResizeListener();
      } else {
        this.unbindOutsideClickListener();
        this.unbindResizeListener();
      }
    }, "activeItemPath")
  },
  outsideClickListener: null,
  container: null,
  menubar: null,
  mounted: /* @__PURE__ */ __name(function mounted13() {
    this.id = this.id || UniqueComponentId();
    this.bindMatchMediaListener();
  }, "mounted"),
  beforeUnmount: /* @__PURE__ */ __name(function beforeUnmount12() {
    this.mobileActive = false;
    this.unbindOutsideClickListener();
    this.unbindResizeListener();
    this.unbindMatchMediaListener();
    if (this.container) {
      ZIndex.clear(this.container);
    }
    this.container = null;
  }, "beforeUnmount"),
  methods: {
    getItemProp: /* @__PURE__ */ __name(function getItemProp4(item4, name) {
      return item4 ? resolve(item4[name]) : void 0;
    }, "getItemProp"),
    getItemLabel: /* @__PURE__ */ __name(function getItemLabel4(item4) {
      return this.getItemProp(item4, "label");
    }, "getItemLabel"),
    isItemDisabled: /* @__PURE__ */ __name(function isItemDisabled4(item4) {
      return this.getItemProp(item4, "disabled");
    }, "isItemDisabled"),
    isItemVisible: /* @__PURE__ */ __name(function isItemVisible4(item4) {
      return this.getItemProp(item4, "visible") !== false;
    }, "isItemVisible"),
    isItemGroup: /* @__PURE__ */ __name(function isItemGroup4(item4) {
      return isNotEmpty(this.getItemProp(item4, "items"));
    }, "isItemGroup"),
    isItemSeparator: /* @__PURE__ */ __name(function isItemSeparator2(item4) {
      return this.getItemProp(item4, "separator");
    }, "isItemSeparator"),
    getProccessedItemLabel: /* @__PURE__ */ __name(function getProccessedItemLabel2(processedItem) {
      return processedItem ? this.getItemLabel(processedItem.item) : void 0;
    }, "getProccessedItemLabel"),
    isProccessedItemGroup: /* @__PURE__ */ __name(function isProccessedItemGroup2(processedItem) {
      return processedItem && isNotEmpty(processedItem.items);
    }, "isProccessedItemGroup"),
    toggle: /* @__PURE__ */ __name(function toggle4(event) {
      var _this = this;
      if (this.mobileActive) {
        this.mobileActive = false;
        ZIndex.clear(this.menubar);
        this.hide();
      } else {
        this.mobileActive = true;
        ZIndex.set("menu", this.menubar, this.$primevue.config.zIndex.menu);
        setTimeout(function() {
          _this.show();
        }, 1);
      }
      this.bindOutsideClickListener();
      event.preventDefault();
    }, "toggle"),
    show: /* @__PURE__ */ __name(function show4() {
      focus(this.menubar);
    }, "show"),
    hide: /* @__PURE__ */ __name(function hide4(event, isFocus) {
      var _this2 = this;
      if (this.mobileActive) {
        this.mobileActive = false;
        setTimeout(function() {
          focus(_this2.$refs.menubutton);
        }, 0);
      }
      this.activeItemPath = [];
      this.focusedItemInfo = {
        index: -1,
        level: 0,
        parentKey: ""
      };
      isFocus && focus(this.menubar);
      this.dirty = false;
    }, "hide"),
    onFocus: /* @__PURE__ */ __name(function onFocus3(event) {
      this.focused = true;
      this.focusedItemInfo = this.focusedItemInfo.index !== -1 ? this.focusedItemInfo : {
        index: this.findFirstFocusedItemIndex(),
        level: 0,
        parentKey: ""
      };
      this.$emit("focus", event);
    }, "onFocus"),
    onBlur: /* @__PURE__ */ __name(function onBlur3(event) {
      this.focused = false;
      this.focusedItemInfo = {
        index: -1,
        level: 0,
        parentKey: ""
      };
      this.searchValue = "";
      this.dirty = false;
      this.$emit("blur", event);
    }, "onBlur"),
    onKeyDown: /* @__PURE__ */ __name(function onKeyDown5(event) {
      var metaKey = event.metaKey || event.ctrlKey;
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
        case "Space":
          this.onSpaceKey(event);
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
        case "PageDown":
        case "PageUp":
        case "Backspace":
        case "ShiftLeft":
        case "ShiftRight":
          break;
        default:
          if (!metaKey && isPrintableCharacter(event.key)) {
            this.searchItems(event, event.key);
          }
          break;
      }
    }, "onKeyDown"),
    onItemChange: /* @__PURE__ */ __name(function onItemChange2(event) {
      var processedItem = event.processedItem, isFocus = event.isFocus;
      if (isEmpty(processedItem)) return;
      var index2 = processedItem.index, key = processedItem.key, level = processedItem.level, parentKey = processedItem.parentKey, items = processedItem.items;
      var grouped = isNotEmpty(items);
      var activeItemPath4 = this.activeItemPath.filter(function(p) {
        return p.parentKey !== parentKey && p.parentKey !== key;
      });
      grouped && activeItemPath4.push(processedItem);
      this.focusedItemInfo = {
        index: index2,
        level,
        parentKey
      };
      this.activeItemPath = activeItemPath4;
      grouped && (this.dirty = true);
      isFocus && focus(this.menubar);
    }, "onItemChange"),
    onItemClick: /* @__PURE__ */ __name(function onItemClick5(event) {
      var originalEvent = event.originalEvent, processedItem = event.processedItem;
      var grouped = this.isProccessedItemGroup(processedItem);
      var root15 = isEmpty(processedItem.parent);
      var selected2 = this.isSelected(processedItem);
      if (selected2) {
        var index2 = processedItem.index, key = processedItem.key, level = processedItem.level, parentKey = processedItem.parentKey;
        this.activeItemPath = this.activeItemPath.filter(function(p) {
          return key !== p.key && key.startsWith(p.key);
        });
        this.focusedItemInfo = {
          index: index2,
          level,
          parentKey
        };
        this.dirty = !root15;
        focus(this.menubar);
      } else {
        if (grouped) {
          this.onItemChange(event);
        } else {
          var rootProcessedItem = root15 ? processedItem : this.activeItemPath.find(function(p) {
            return p.parentKey === "";
          });
          this.hide(originalEvent);
          this.changeFocusedItemIndex(originalEvent, rootProcessedItem ? rootProcessedItem.index : -1);
          this.mobileActive = false;
          focus(this.menubar);
        }
      }
    }, "onItemClick"),
    onItemMouseEnter: /* @__PURE__ */ __name(function onItemMouseEnter4(event) {
      if (this.dirty) {
        this.onItemChange(event);
      }
    }, "onItemMouseEnter"),
    onItemMouseMove: /* @__PURE__ */ __name(function onItemMouseMove4(event) {
      if (this.focused) {
        this.changeFocusedItemIndex(event, event.processedItem.index);
      }
    }, "onItemMouseMove"),
    menuButtonClick: /* @__PURE__ */ __name(function menuButtonClick(event) {
      this.toggle(event);
    }, "menuButtonClick"),
    menuButtonKeydown: /* @__PURE__ */ __name(function menuButtonKeydown(event) {
      (event.code === "Enter" || event.code === "NumpadEnter" || event.code === "Space") && this.menuButtonClick(event);
    }, "menuButtonKeydown"),
    onArrowDownKey: /* @__PURE__ */ __name(function onArrowDownKey3(event) {
      var processedItem = this.visibleItems[this.focusedItemInfo.index];
      var root15 = processedItem ? isEmpty(processedItem.parent) : null;
      if (root15) {
        var grouped = this.isProccessedItemGroup(processedItem);
        if (grouped) {
          this.onItemChange({
            originalEvent: event,
            processedItem
          });
          this.focusedItemInfo = {
            index: -1,
            parentKey: processedItem.key
          };
          this.onArrowRightKey(event);
        }
      } else {
        var itemIndex = this.focusedItemInfo.index !== -1 ? this.findNextItemIndex(this.focusedItemInfo.index) : this.findFirstFocusedItemIndex();
        this.changeFocusedItemIndex(event, itemIndex);
      }
      event.preventDefault();
    }, "onArrowDownKey"),
    onArrowUpKey: /* @__PURE__ */ __name(function onArrowUpKey3(event) {
      var _this3 = this;
      var processedItem = this.visibleItems[this.focusedItemInfo.index];
      var root15 = isEmpty(processedItem.parent);
      if (root15) {
        var grouped = this.isProccessedItemGroup(processedItem);
        if (grouped) {
          this.onItemChange({
            originalEvent: event,
            processedItem
          });
          this.focusedItemInfo = {
            index: -1,
            parentKey: processedItem.key
          };
          var itemIndex = this.findLastItemIndex();
          this.changeFocusedItemIndex(event, itemIndex);
        }
      } else {
        var parentItem = this.activeItemPath.find(function(p) {
          return p.key === processedItem.parentKey;
        });
        if (this.focusedItemInfo.index === 0) {
          this.focusedItemInfo = {
            index: -1,
            parentKey: parentItem ? parentItem.parentKey : ""
          };
          this.searchValue = "";
          this.onArrowLeftKey(event);
          this.activeItemPath = this.activeItemPath.filter(function(p) {
            return p.parentKey !== _this3.focusedItemInfo.parentKey;
          });
        } else {
          var _itemIndex = this.focusedItemInfo.index !== -1 ? this.findPrevItemIndex(this.focusedItemInfo.index) : this.findLastFocusedItemIndex();
          this.changeFocusedItemIndex(event, _itemIndex);
        }
      }
      event.preventDefault();
    }, "onArrowUpKey"),
    onArrowLeftKey: /* @__PURE__ */ __name(function onArrowLeftKey3(event) {
      var _this4 = this;
      var processedItem = this.visibleItems[this.focusedItemInfo.index];
      var parentItem = processedItem ? this.activeItemPath.find(function(p) {
        return p.key === processedItem.parentKey;
      }) : null;
      if (parentItem) {
        this.onItemChange({
          originalEvent: event,
          processedItem: parentItem
        });
        this.activeItemPath = this.activeItemPath.filter(function(p) {
          return p.parentKey !== _this4.focusedItemInfo.parentKey;
        });
        event.preventDefault();
      } else {
        var itemIndex = this.focusedItemInfo.index !== -1 ? this.findPrevItemIndex(this.focusedItemInfo.index) : this.findLastFocusedItemIndex();
        this.changeFocusedItemIndex(event, itemIndex);
        event.preventDefault();
      }
    }, "onArrowLeftKey"),
    onArrowRightKey: /* @__PURE__ */ __name(function onArrowRightKey3(event) {
      var processedItem = this.visibleItems[this.focusedItemInfo.index];
      var parentItem = processedItem ? this.activeItemPath.find(function(p) {
        return p.key === processedItem.parentKey;
      }) : null;
      if (parentItem) {
        var grouped = this.isProccessedItemGroup(processedItem);
        if (grouped) {
          this.onItemChange({
            originalEvent: event,
            processedItem
          });
          this.focusedItemInfo = {
            index: -1,
            parentKey: processedItem.key
          };
          this.onArrowDownKey(event);
        }
      } else {
        var itemIndex = this.focusedItemInfo.index !== -1 ? this.findNextItemIndex(this.focusedItemInfo.index) : this.findFirstFocusedItemIndex();
        this.changeFocusedItemIndex(event, itemIndex);
        event.preventDefault();
      }
    }, "onArrowRightKey"),
    onHomeKey: /* @__PURE__ */ __name(function onHomeKey5(event) {
      this.changeFocusedItemIndex(event, this.findFirstItemIndex());
      event.preventDefault();
    }, "onHomeKey"),
    onEndKey: /* @__PURE__ */ __name(function onEndKey5(event) {
      this.changeFocusedItemIndex(event, this.findLastItemIndex());
      event.preventDefault();
    }, "onEndKey"),
    onEnterKey: /* @__PURE__ */ __name(function onEnterKey4(event) {
      if (this.focusedItemInfo.index !== -1) {
        var element = findSingle(this.menubar, 'li[id="'.concat("".concat(this.focusedItemId), '"]'));
        var anchorElement = element && findSingle(element, 'a[data-pc-section="itemlink"]');
        anchorElement ? anchorElement.click() : element && element.click();
        var processedItem = this.visibleItems[this.focusedItemInfo.index];
        var grouped = this.isProccessedItemGroup(processedItem);
        !grouped && (this.focusedItemInfo.index = this.findFirstFocusedItemIndex());
      }
      event.preventDefault();
    }, "onEnterKey"),
    onSpaceKey: /* @__PURE__ */ __name(function onSpaceKey2(event) {
      this.onEnterKey(event);
    }, "onSpaceKey"),
    onEscapeKey: /* @__PURE__ */ __name(function onEscapeKey3(event) {
      if (this.focusedItemInfo.level !== 0) {
        var _focusedItemInfo = this.focusedItemInfo;
        this.hide(event, false);
        this.focusedItemInfo = {
          index: Number(_focusedItemInfo.parentKey.split("_")[0]),
          level: 0,
          parentKey: ""
        };
      }
      event.preventDefault();
    }, "onEscapeKey"),
    onTabKey: /* @__PURE__ */ __name(function onTabKey6(event) {
      if (this.focusedItemInfo.index !== -1) {
        var processedItem = this.visibleItems[this.focusedItemInfo.index];
        var grouped = this.isProccessedItemGroup(processedItem);
        !grouped && this.onItemChange({
          originalEvent: event,
          processedItem
        });
      }
      this.hide();
    }, "onTabKey"),
    bindOutsideClickListener: /* @__PURE__ */ __name(function bindOutsideClickListener6() {
      var _this5 = this;
      if (!this.outsideClickListener) {
        this.outsideClickListener = function(event) {
          var isOutsideContainer = _this5.container && !_this5.container.contains(event.target);
          var isOutsideTarget = !(_this5.target && (_this5.target === event.target || _this5.target.contains(event.target)));
          if (isOutsideContainer && isOutsideTarget) {
            _this5.hide();
          }
        };
        document.addEventListener("click", this.outsideClickListener);
      }
    }, "bindOutsideClickListener"),
    unbindOutsideClickListener: /* @__PURE__ */ __name(function unbindOutsideClickListener6() {
      if (this.outsideClickListener) {
        document.removeEventListener("click", this.outsideClickListener);
        this.outsideClickListener = null;
      }
    }, "unbindOutsideClickListener"),
    bindResizeListener: /* @__PURE__ */ __name(function bindResizeListener6() {
      var _this6 = this;
      if (!this.resizeListener) {
        this.resizeListener = function(event) {
          if (!isTouchDevice()) {
            _this6.hide(event, true);
          }
          _this6.mobileActive = false;
        };
        window.addEventListener("resize", this.resizeListener);
      }
    }, "bindResizeListener"),
    unbindResizeListener: /* @__PURE__ */ __name(function unbindResizeListener6() {
      if (this.resizeListener) {
        window.removeEventListener("resize", this.resizeListener);
        this.resizeListener = null;
      }
    }, "unbindResizeListener"),
    bindMatchMediaListener: /* @__PURE__ */ __name(function bindMatchMediaListener() {
      var _this7 = this;
      if (!this.matchMediaListener) {
        var query = matchMedia("(max-width: ".concat(this.breakpoint, ")"));
        this.query = query;
        this.queryMatches = query.matches;
        this.matchMediaListener = function() {
          _this7.queryMatches = query.matches;
          _this7.mobileActive = false;
        };
        this.query.addEventListener("change", this.matchMediaListener);
      }
    }, "bindMatchMediaListener"),
    unbindMatchMediaListener: /* @__PURE__ */ __name(function unbindMatchMediaListener() {
      if (this.matchMediaListener) {
        this.query.removeEventListener("change", this.matchMediaListener);
        this.matchMediaListener = null;
      }
    }, "unbindMatchMediaListener"),
    isItemMatched: /* @__PURE__ */ __name(function isItemMatched2(processedItem) {
      var _this$getProccessedIt;
      return this.isValidItem(processedItem) && ((_this$getProccessedIt = this.getProccessedItemLabel(processedItem)) === null || _this$getProccessedIt === void 0 ? void 0 : _this$getProccessedIt.toLocaleLowerCase().startsWith(this.searchValue.toLocaleLowerCase()));
    }, "isItemMatched"),
    isValidItem: /* @__PURE__ */ __name(function isValidItem2(processedItem) {
      return !!processedItem && !this.isItemDisabled(processedItem.item) && !this.isItemSeparator(processedItem.item) && this.isItemVisible(processedItem.item);
    }, "isValidItem"),
    isValidSelectedItem: /* @__PURE__ */ __name(function isValidSelectedItem2(processedItem) {
      return this.isValidItem(processedItem) && this.isSelected(processedItem);
    }, "isValidSelectedItem"),
    isSelected: /* @__PURE__ */ __name(function isSelected4(processedItem) {
      return this.activeItemPath.some(function(p) {
        return p.key === processedItem.key;
      });
    }, "isSelected"),
    findFirstItemIndex: /* @__PURE__ */ __name(function findFirstItemIndex2() {
      var _this8 = this;
      return this.visibleItems.findIndex(function(processedItem) {
        return _this8.isValidItem(processedItem);
      });
    }, "findFirstItemIndex"),
    findLastItemIndex: /* @__PURE__ */ __name(function findLastItemIndex2() {
      var _this9 = this;
      return findLastIndex(this.visibleItems, function(processedItem) {
        return _this9.isValidItem(processedItem);
      });
    }, "findLastItemIndex"),
    findNextItemIndex: /* @__PURE__ */ __name(function findNextItemIndex2(index2) {
      var _this10 = this;
      var matchedItemIndex = index2 < this.visibleItems.length - 1 ? this.visibleItems.slice(index2 + 1).findIndex(function(processedItem) {
        return _this10.isValidItem(processedItem);
      }) : -1;
      return matchedItemIndex > -1 ? matchedItemIndex + index2 + 1 : index2;
    }, "findNextItemIndex"),
    findPrevItemIndex: /* @__PURE__ */ __name(function findPrevItemIndex2(index2) {
      var _this11 = this;
      var matchedItemIndex = index2 > 0 ? findLastIndex(this.visibleItems.slice(0, index2), function(processedItem) {
        return _this11.isValidItem(processedItem);
      }) : -1;
      return matchedItemIndex > -1 ? matchedItemIndex : index2;
    }, "findPrevItemIndex"),
    findSelectedItemIndex: /* @__PURE__ */ __name(function findSelectedItemIndex2() {
      var _this12 = this;
      return this.visibleItems.findIndex(function(processedItem) {
        return _this12.isValidSelectedItem(processedItem);
      });
    }, "findSelectedItemIndex"),
    findFirstFocusedItemIndex: /* @__PURE__ */ __name(function findFirstFocusedItemIndex2() {
      var selectedIndex = this.findSelectedItemIndex();
      return selectedIndex < 0 ? this.findFirstItemIndex() : selectedIndex;
    }, "findFirstFocusedItemIndex"),
    findLastFocusedItemIndex: /* @__PURE__ */ __name(function findLastFocusedItemIndex2() {
      var selectedIndex = this.findSelectedItemIndex();
      return selectedIndex < 0 ? this.findLastItemIndex() : selectedIndex;
    }, "findLastFocusedItemIndex"),
    searchItems: /* @__PURE__ */ __name(function searchItems2(event, _char) {
      var _this13 = this;
      this.searchValue = (this.searchValue || "") + _char;
      var itemIndex = -1;
      var matched = false;
      if (this.focusedItemInfo.index !== -1) {
        itemIndex = this.visibleItems.slice(this.focusedItemInfo.index).findIndex(function(processedItem) {
          return _this13.isItemMatched(processedItem);
        });
        itemIndex = itemIndex === -1 ? this.visibleItems.slice(0, this.focusedItemInfo.index).findIndex(function(processedItem) {
          return _this13.isItemMatched(processedItem);
        }) : itemIndex + this.focusedItemInfo.index;
      } else {
        itemIndex = this.visibleItems.findIndex(function(processedItem) {
          return _this13.isItemMatched(processedItem);
        });
      }
      if (itemIndex !== -1) {
        matched = true;
      }
      if (itemIndex === -1 && this.focusedItemInfo.index === -1) {
        itemIndex = this.findFirstFocusedItemIndex();
      }
      if (itemIndex !== -1) {
        this.changeFocusedItemIndex(event, itemIndex);
      }
      if (this.searchTimeout) {
        clearTimeout(this.searchTimeout);
      }
      this.searchTimeout = setTimeout(function() {
        _this13.searchValue = "";
        _this13.searchTimeout = null;
      }, 500);
      return matched;
    }, "searchItems"),
    changeFocusedItemIndex: /* @__PURE__ */ __name(function changeFocusedItemIndex2(event, index2) {
      if (this.focusedItemInfo.index !== index2) {
        this.focusedItemInfo.index = index2;
        this.scrollInView();
      }
    }, "changeFocusedItemIndex"),
    scrollInView: /* @__PURE__ */ __name(function scrollInView3() {
      var index2 = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : -1;
      var id = index2 !== -1 ? "".concat(this.id, "_").concat(index2) : this.focusedItemId;
      var element = findSingle(this.menubar, 'li[id="'.concat(id, '"]'));
      if (element) {
        element.scrollIntoView && element.scrollIntoView({
          block: "nearest",
          inline: "start"
        });
      }
    }, "scrollInView"),
    createProcessedItems: /* @__PURE__ */ __name(function createProcessedItems2(items) {
      var _this14 = this;
      var level = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : 0;
      var parent = arguments.length > 2 && arguments[2] !== void 0 ? arguments[2] : {};
      var parentKey = arguments.length > 3 && arguments[3] !== void 0 ? arguments[3] : "";
      var processedItems4 = [];
      items && items.forEach(function(item4, index2) {
        var key = (parentKey !== "" ? parentKey + "_" : "") + index2;
        var newItem = {
          item: item4,
          index: index2,
          level,
          key,
          parent,
          parentKey
        };
        newItem["items"] = _this14.createProcessedItems(item4.items, level + 1, newItem, key);
        processedItems4.push(newItem);
      });
      return processedItems4;
    }, "createProcessedItems"),
    containerRef: /* @__PURE__ */ __name(function containerRef5(el) {
      this.container = el;
    }, "containerRef"),
    menubarRef: /* @__PURE__ */ __name(function menubarRef(el) {
      this.menubar = el ? el.$el : void 0;
    }, "menubarRef")
  },
  computed: {
    processedItems: /* @__PURE__ */ __name(function processedItems2() {
      return this.createProcessedItems(this.model || []);
    }, "processedItems"),
    visibleItems: /* @__PURE__ */ __name(function visibleItems2() {
      var _this15 = this;
      var processedItem = this.activeItemPath.find(function(p) {
        return p.key === _this15.focusedItemInfo.parentKey;
      });
      return processedItem ? processedItem.items : this.processedItems;
    }, "visibleItems"),
    focusedItemId: /* @__PURE__ */ __name(function focusedItemId() {
      return this.focusedItemInfo.index !== -1 ? "".concat(this.id).concat(isNotEmpty(this.focusedItemInfo.parentKey) ? "_" + this.focusedItemInfo.parentKey : "", "_").concat(this.focusedItemInfo.index) : null;
    }, "focusedItemId")
  },
  components: {
    MenubarSub: script$1$3,
    BarsIcon: script$P
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
var _hoisted_1$6 = ["aria-haspopup", "aria-expanded", "aria-controls", "aria-label"];
function render$4(_ctx, _cache, $props, $setup, $data, $options) {
  var _component_BarsIcon = resolveComponent("BarsIcon");
  var _component_MenubarSub = resolveComponent("MenubarSub");
  return openBlock(), createElementBlock("div", mergeProps({
    ref: $options.containerRef,
    "class": _ctx.cx("root")
  }, _ctx.ptmi("root")), [_ctx.$slots.start ? (openBlock(), createElementBlock("div", mergeProps({
    key: 0,
    "class": _ctx.cx("start")
  }, _ctx.ptm("start")), [renderSlot(_ctx.$slots, "start")], 16)) : createCommentVNode("", true), renderSlot(_ctx.$slots, _ctx.$slots.button ? "button" : "menubutton", {
    id: $data.id,
    "class": normalizeClass(_ctx.cx("button")),
    toggleCallback: /* @__PURE__ */ __name(function toggleCallback(event) {
      return $options.menuButtonClick(event);
    }, "toggleCallback")
  }, function() {
    var _ctx$$primevue$config;
    return [_ctx.model && _ctx.model.length > 0 ? (openBlock(), createElementBlock("a", mergeProps({
      key: 0,
      ref: "menubutton",
      role: "button",
      tabindex: "0",
      "class": _ctx.cx("button"),
      "aria-haspopup": _ctx.model.length && _ctx.model.length > 0 ? true : false,
      "aria-expanded": $data.mobileActive,
      "aria-controls": $data.id,
      "aria-label": (_ctx$$primevue$config = _ctx.$primevue.config.locale.aria) === null || _ctx$$primevue$config === void 0 ? void 0 : _ctx$$primevue$config.navigation,
      onClick: _cache[0] || (_cache[0] = function($event) {
        return $options.menuButtonClick($event);
      }),
      onKeydown: _cache[1] || (_cache[1] = function($event) {
        return $options.menuButtonKeydown($event);
      })
    }, _objectSpread(_objectSpread({}, _ctx.buttonProps), _ctx.ptm("button"))), [renderSlot(_ctx.$slots, _ctx.$slots.buttonicon ? "buttonicon" : "menubuttonicon", {}, function() {
      return [createVNode(_component_BarsIcon, normalizeProps(guardReactiveProps(_ctx.ptm("buttonicon"))), null, 16)];
    })], 16, _hoisted_1$6)) : createCommentVNode("", true)];
  }), createVNode(_component_MenubarSub, {
    ref: $options.menubarRef,
    id: $data.id + "_list",
    role: "menubar",
    items: $options.processedItems,
    templates: _ctx.$slots,
    root: true,
    mobileActive: $data.mobileActive,
    tabindex: "0",
    "aria-activedescendant": $data.focused ? $options.focusedItemId : void 0,
    menuId: $data.id,
    focusedItemId: $data.focused ? $options.focusedItemId : void 0,
    activeItemPath: $data.activeItemPath,
    level: 0,
    "aria-labelledby": _ctx.ariaLabelledby,
    "aria-label": _ctx.ariaLabel,
    pt: _ctx.pt,
    unstyled: _ctx.unstyled,
    onFocus: $options.onFocus,
    onBlur: $options.onBlur,
    onKeydown: $options.onKeyDown,
    onItemClick: $options.onItemClick,
    onItemMouseenter: $options.onItemMouseEnter,
    onItemMousemove: $options.onItemMouseMove
  }, null, 8, ["id", "items", "templates", "mobileActive", "aria-activedescendant", "menuId", "focusedItemId", "activeItemPath", "aria-labelledby", "aria-label", "pt", "unstyled", "onFocus", "onBlur", "onKeydown", "onItemClick", "onItemMouseenter", "onItemMousemove"]), _ctx.$slots.end ? (openBlock(), createElementBlock("div", mergeProps({
    key: 1,
    "class": _ctx.cx("end")
  }, _ctx.ptm("end")), [renderSlot(_ctx.$slots, "end")], 16)) : createCommentVNode("", true)], 16);
}
__name(render$4, "render$4");
script$5.render = render$4;
const _withScopeId$2 = /* @__PURE__ */ __name((n) => (pushScopeId("data-v-2ec1b620"), n = n(), popScopeId(), n), "_withScopeId$2");
const _hoisted_1$5 = { class: "p-menubar-item-label" };
const _hoisted_2$3 = {
  key: 1,
  class: "ml-auto border border-surface rounded text-muted text-xs p-1 keybinding-tag"
};
const _sfc_main$5 = /* @__PURE__ */ defineComponent({
  __name: "CommandMenubar",
  setup(__props) {
    const settingStore = useSettingStore();
    const dropdownDirection = computed(
      () => settingStore.get("Comfy.UseNewMenu") === "Top" ? "down" : "up"
    );
    const menuItemsStore = useMenuItemStore();
    const items = menuItemsStore.menuItems;
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(script$5), {
        model: unref(items),
        class: "top-menubar border-none p-0 bg-transparent",
        pt: {
          rootList: "gap-0 flex-nowrap w-auto",
          submenu: `dropdown-direction-${dropdownDirection.value}`,
          item: "relative"
        }
      }, {
        item: withCtx(({ item: item4, props }) => [
          createBaseVNode("a", mergeProps({ class: "p-menubar-item-link" }, props.action), [
            item4.icon ? (openBlock(), createElementBlock("span", {
              key: 0,
              class: normalizeClass(["p-menubar-item-icon", item4.icon])
            }, null, 2)) : createCommentVNode("", true),
            createBaseVNode("span", _hoisted_1$5, toDisplayString(item4.label), 1),
            item4?.comfyCommand?.keybinding ? (openBlock(), createElementBlock("span", _hoisted_2$3, toDisplayString(item4.comfyCommand.keybinding.combo.toString()), 1)) : createCommentVNode("", true)
          ], 16)
        ]),
        _: 1
      }, 8, ["model", "pt"]);
    };
  }
});
const CommandMenubar = /* @__PURE__ */ _export_sfc(_sfc_main$5, [["__scopeId", "data-v-2ec1b620"]]);
var theme$2 = /* @__PURE__ */ __name(function theme16(_ref) {
  var dt = _ref.dt;
  return "\n.p-panel {\n    border: 1px solid ".concat(dt("panel.border.color"), ";\n    border-radius: ").concat(dt("panel.border.radius"), ";\n    background: ").concat(dt("panel.background"), ";\n    color: ").concat(dt("panel.color"), ";\n}\n\n.p-panel-header {\n    display: flex;\n    justify-content: space-between;\n    align-items: center;\n    padding: ").concat(dt("panel.header.padding"), ";\n    background: ").concat(dt("panel.header.background"), ";\n    color: ").concat(dt("panel.header.color"), ";\n    border-style: solid;\n    border-width: ").concat(dt("panel.header.border.width"), ";\n    border-color: ").concat(dt("panel.header.border.color"), ";\n    border-radius: ").concat(dt("panel.header.border.radius"), ";\n}\n\n.p-panel-toggleable .p-panel-header {\n    padding: ").concat(dt("panel.toggleable.header.padding"), ";\n}\n\n.p-panel-title {\n    line-height: 1;\n    font-weight: ").concat(dt("panel.title.font.weight"), ";\n}\n\n.p-panel-content {\n    padding: ").concat(dt("panel.content.padding"), ";\n}\n\n.p-panel-footer {\n    padding: ").concat(dt("panel.footer.padding"), ";\n}\n");
}, "theme");
var classes$2 = {
  root: /* @__PURE__ */ __name(function root12(_ref2) {
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
  theme: theme$2,
  classes: classes$2
});
var script$1$2 = {
  name: "BasePanel",
  "extends": script$p,
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
  provide: /* @__PURE__ */ __name(function provide18() {
    return {
      $pcPanel: this,
      $parentInstance: this
    };
  }, "provide")
};
var script$4 = {
  name: "Panel",
  "extends": script$1$2,
  inheritAttrs: false,
  emits: ["update:collapsed", "toggle"],
  data: /* @__PURE__ */ __name(function data14() {
    return {
      id: this.$attrs.id,
      d_collapsed: this.collapsed
    };
  }, "data"),
  watch: {
    "$attrs.id": /* @__PURE__ */ __name(function $attrsId5(newValue) {
      this.id = newValue || UniqueComponentId();
    }, "$attrsId"),
    collapsed: /* @__PURE__ */ __name(function collapsed(newValue) {
      this.d_collapsed = newValue;
    }, "collapsed")
  },
  mounted: /* @__PURE__ */ __name(function mounted14() {
    this.id = this.id || UniqueComponentId();
  }, "mounted"),
  methods: {
    toggle: /* @__PURE__ */ __name(function toggle5(event) {
      this.d_collapsed = !this.d_collapsed;
      this.$emit("update:collapsed", this.d_collapsed);
      this.$emit("toggle", {
        originalEvent: event,
        value: this.d_collapsed
      });
    }, "toggle"),
    onKeyDown: /* @__PURE__ */ __name(function onKeyDown6(event) {
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
    PlusIcon: script$Q,
    MinusIcon: script$G,
    Button: script$o
  },
  directives: {
    ripple: Ripple
  }
};
var _hoisted_1$4 = ["id"];
var _hoisted_2$2 = ["id", "aria-labelledby"];
function render$3(_ctx, _cache, $props, $setup, $data, $options) {
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
    }, _ctx.ptm("title")), toDisplayString(_ctx.header), 17, _hoisted_1$4)) : createCommentVNode("", true)];
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
      }, _ctx.ptm("footer")), [renderSlot(_ctx.$slots, "footer")], 16)) : createCommentVNode("", true)], 16, _hoisted_2$2), [[vShow, !$data.d_collapsed]])];
    }),
    _: 3
  }, 16)], 16);
}
__name(render$3, "render$3");
script$4.render = render$3;
var theme$1 = /* @__PURE__ */ __name(function theme17(_ref) {
  var dt = _ref.dt;
  return "\n.p-tieredmenu {\n    background: ".concat(dt("tieredmenu.background"), ";\n    color: ").concat(dt("tieredmenu.color"), ";\n    border: 1px solid ").concat(dt("tieredmenu.border.color"), ";\n    border-radius: ").concat(dt("tieredmenu.border.radius"), ";\n    min-width: 12.5rem;\n}\n\n.p-tieredmenu-root-list,\n.p-tieredmenu-submenu {\n    margin: 0;\n    padding: ").concat(dt("tieredmenu.list.padding"), ";\n    list-style: none;\n    outline: 0 none;\n    display: flex;\n    flex-direction: column;\n    gap: ").concat(dt("tieredmenu.list.gap"), ";\n}\n\n.p-tieredmenu-submenu {\n    position: absolute;\n    min-width: 100%;\n    z-index: 1;\n    background: ").concat(dt("tieredmenu.background"), ";\n    color: ").concat(dt("tieredmenu.color"), ";\n    border: 1px solid ").concat(dt("tieredmenu.border.color"), ";\n    border-radius: ").concat(dt("tieredmenu.border.radius"), ";\n    box-shadow: ").concat(dt("tieredmenu.shadow"), ";\n}\n\n.p-tieredmenu-item {\n    position: relative;\n}\n\n.p-tieredmenu-item-content {\n    transition: background ").concat(dt("tieredmenu.transition.duration"), ", color ").concat(dt("tieredmenu.transition.duration"), ";\n    border-radius: ").concat(dt("tieredmenu.item.border.radius"), ";\n    color: ").concat(dt("tieredmenu.item.color"), ";\n}\n\n.p-tieredmenu-item-link {\n    cursor: pointer;\n    display: flex;\n    align-items: center;\n    text-decoration: none;\n    overflow: hidden;\n    position: relative;\n    color: inherit;\n    padding: ").concat(dt("tieredmenu.item.padding"), ";\n    gap: ").concat(dt("tieredmenu.item.gap"), ";\n    user-select: none;\n    outline: 0 none;\n}\n\n.p-tieredmenu-item-label {\n    line-height: 1;\n}\n\n.p-tieredmenu-item-icon {\n    color: ").concat(dt("tieredmenu.item.icon.color"), ";\n}\n\n.p-tieredmenu-submenu-icon {\n    color: ").concat(dt("tieredmenu.submenu.icon.color"), ";\n    margin-left: auto;\n    font-size: ").concat(dt("tieredmenu.submenu.icon.size"), ";\n    width: ").concat(dt("tieredmenu.submenu.icon.size"), ";\n    height: ").concat(dt("tieredmenu.submenu.icon.size"), ";\n}\n\n.p-tieredmenu-item.p-focus > .p-tieredmenu-item-content {\n    color: ").concat(dt("tieredmenu.item.focus.color"), ";\n    background: ").concat(dt("tieredmenu.item.focus.background"), ";\n}\n\n.p-tieredmenu-item.p-focus > .p-tieredmenu-item-content .p-tieredmenu-item-icon {\n    color: ").concat(dt("tieredmenu.item.icon.focus.color"), ";\n}\n\n.p-tieredmenu-item.p-focus > .p-tieredmenu-item-content .p-tieredmenu-submenu-icon {\n    color: ").concat(dt("tieredmenu.submenu.icon.focus.color"), ";\n}\n\n.p-tieredmenu-item:not(.p-disabled) > .p-tieredmenu-item-content:hover {\n    color: ").concat(dt("tieredmenu.item.focus.color"), ";\n    background: ").concat(dt("tieredmenu.item.focus.background"), ";\n}\n\n.p-tieredmenu-item:not(.p-disabled) > .p-tieredmenu-item-content:hover .p-tieredmenu-item-icon {\n    color: ").concat(dt("tieredmenu.item.icon.focus.color"), ";\n}\n\n.p-tieredmenu-item:not(.p-disabled) > .p-tieredmenu-item-content:hover .p-tieredmenu-submenu-icon {\n    color: ").concat(dt("tieredmenu.submenu.icon.focus.color"), ";\n}\n\n.p-tieredmenu-item-active > .p-tieredmenu-item-content {\n    color: ").concat(dt("tieredmenu.item.active.color"), ";\n    background: ").concat(dt("tieredmenu.item.active.background"), ";\n}\n\n.p-tieredmenu-item-active > .p-tieredmenu-item-content .p-tieredmenu-item-icon {\n    color: ").concat(dt("tieredmenu.item.icon.active.color"), ";\n}\n\n.p-tieredmenu-item-active > .p-tieredmenu-item-content .p-tieredmenu-submenu-icon {\n    color: ").concat(dt("tieredmenu.submenu.icon.active.color"), ";\n}\n\n.p-tieredmenu-separator {\n    border-top: 1px solid ").concat(dt("tieredmenu.separator.border.color"), ";\n}\n\n.p-tieredmenu-overlay {\n    box-shadow: ").concat(dt("tieredmenu.shadow"), ";\n}\n\n.p-tieredmenu-enter-from,\n.p-tieredmenu-leave-active {\n    opacity: 0;\n}\n\n.p-tieredmenu-enter-active {\n    transition: opacity 250ms;\n}\n");
}, "theme");
var inlineStyles = {
  submenu: /* @__PURE__ */ __name(function submenu2(_ref2) {
    var instance = _ref2.instance, processedItem = _ref2.processedItem;
    return {
      display: instance.isItemActive(processedItem) ? "flex" : "none"
    };
  }, "submenu")
};
var classes$1 = {
  root: /* @__PURE__ */ __name(function root13(_ref3) {
    _ref3.instance;
    var props = _ref3.props;
    return ["p-tieredmenu p-component", {
      "p-tieredmenu-overlay": props.popup
    }];
  }, "root"),
  start: "p-tieredmenu-start",
  rootList: "p-tieredmenu-root-list",
  item: /* @__PURE__ */ __name(function item3(_ref4) {
    var instance = _ref4.instance, processedItem = _ref4.processedItem;
    return ["p-tieredmenu-item", {
      "p-tieredmenu-item-active": instance.isItemActive(processedItem),
      "p-focus": instance.isItemFocused(processedItem),
      "p-disabled": instance.isItemDisabled(processedItem)
    }];
  }, "item"),
  itemContent: "p-tieredmenu-item-content",
  itemLink: "p-tieredmenu-item-link",
  itemIcon: "p-tieredmenu-item-icon",
  itemLabel: "p-tieredmenu-item-label",
  submenuIcon: "p-tieredmenu-submenu-icon",
  submenu: "p-tieredmenu-submenu",
  separator: "p-tieredmenu-separator",
  end: "p-tieredmenu-end"
};
var TieredMenuStyle = BaseStyle.extend({
  name: "tieredmenu",
  theme: theme$1,
  classes: classes$1,
  inlineStyles
});
var script$2 = {
  name: "BaseTieredMenu",
  "extends": script$p,
  props: {
    popup: {
      type: Boolean,
      "default": false
    },
    model: {
      type: Array,
      "default": null
    },
    appendTo: {
      type: [String, Object],
      "default": "body"
    },
    autoZIndex: {
      type: Boolean,
      "default": true
    },
    baseZIndex: {
      type: Number,
      "default": 0
    },
    disabled: {
      type: Boolean,
      "default": false
    },
    tabindex: {
      type: Number,
      "default": 0
    },
    ariaLabelledby: {
      type: String,
      "default": null
    },
    ariaLabel: {
      type: String,
      "default": null
    }
  },
  style: TieredMenuStyle,
  provide: /* @__PURE__ */ __name(function provide19() {
    return {
      $pcTieredMenu: this,
      $parentInstance: this
    };
  }, "provide")
};
var script$1$1 = {
  name: "TieredMenuSub",
  hostName: "TieredMenu",
  "extends": script$p,
  emits: ["item-click", "item-mouseenter", "item-mousemove"],
  container: null,
  props: {
    menuId: {
      type: String,
      "default": null
    },
    focusedItemId: {
      type: String,
      "default": null
    },
    items: {
      type: Array,
      "default": null
    },
    visible: {
      type: Boolean,
      "default": false
    },
    level: {
      type: Number,
      "default": 0
    },
    templates: {
      type: Object,
      "default": null
    },
    activeItemPath: {
      type: Object,
      "default": null
    },
    tabindex: {
      type: Number,
      "default": 0
    }
  },
  methods: {
    getItemId: /* @__PURE__ */ __name(function getItemId3(processedItem) {
      return "".concat(this.menuId, "_").concat(processedItem.key);
    }, "getItemId"),
    getItemKey: /* @__PURE__ */ __name(function getItemKey3(processedItem) {
      return this.getItemId(processedItem);
    }, "getItemKey"),
    getItemProp: /* @__PURE__ */ __name(function getItemProp5(processedItem, name, params) {
      return processedItem && processedItem.item ? resolve(processedItem.item[name], params) : void 0;
    }, "getItemProp"),
    getItemLabel: /* @__PURE__ */ __name(function getItemLabel5(processedItem) {
      return this.getItemProp(processedItem, "label");
    }, "getItemLabel"),
    getItemLabelId: /* @__PURE__ */ __name(function getItemLabelId3(processedItem) {
      return "".concat(this.menuId, "_").concat(processedItem.key, "_label");
    }, "getItemLabelId"),
    getPTOptions: /* @__PURE__ */ __name(function getPTOptions10(processedItem, index2, key) {
      return this.ptm(key, {
        context: {
          item: processedItem.item,
          index: index2,
          active: this.isItemActive(processedItem),
          focused: this.isItemFocused(processedItem),
          disabled: this.isItemDisabled(processedItem)
        }
      });
    }, "getPTOptions"),
    isItemActive: /* @__PURE__ */ __name(function isItemActive4(processedItem) {
      return this.activeItemPath.some(function(path) {
        return path.key === processedItem.key;
      });
    }, "isItemActive"),
    isItemVisible: /* @__PURE__ */ __name(function isItemVisible5(processedItem) {
      return this.getItemProp(processedItem, "visible") !== false;
    }, "isItemVisible"),
    isItemDisabled: /* @__PURE__ */ __name(function isItemDisabled5(processedItem) {
      return this.getItemProp(processedItem, "disabled");
    }, "isItemDisabled"),
    isItemFocused: /* @__PURE__ */ __name(function isItemFocused3(processedItem) {
      return this.focusedItemId === this.getItemId(processedItem);
    }, "isItemFocused"),
    isItemGroup: /* @__PURE__ */ __name(function isItemGroup5(processedItem) {
      return isNotEmpty(processedItem.items);
    }, "isItemGroup"),
    onEnter: /* @__PURE__ */ __name(function onEnter7() {
      nestedPosition(this.container, this.level);
    }, "onEnter"),
    onItemClick: /* @__PURE__ */ __name(function onItemClick6(event, processedItem) {
      this.getItemProp(processedItem, "command", {
        originalEvent: event,
        item: processedItem.item
      });
      this.$emit("item-click", {
        originalEvent: event,
        processedItem,
        isFocus: true
      });
    }, "onItemClick"),
    onItemMouseEnter: /* @__PURE__ */ __name(function onItemMouseEnter5(event, processedItem) {
      this.$emit("item-mouseenter", {
        originalEvent: event,
        processedItem
      });
    }, "onItemMouseEnter"),
    onItemMouseMove: /* @__PURE__ */ __name(function onItemMouseMove5(event, processedItem) {
      this.$emit("item-mousemove", {
        originalEvent: event,
        processedItem
      });
    }, "onItemMouseMove"),
    getAriaSetSize: /* @__PURE__ */ __name(function getAriaSetSize3() {
      var _this = this;
      return this.items.filter(function(processedItem) {
        return _this.isItemVisible(processedItem) && !_this.getItemProp(processedItem, "separator");
      }).length;
    }, "getAriaSetSize"),
    getAriaPosInset: /* @__PURE__ */ __name(function getAriaPosInset4(index2) {
      var _this2 = this;
      return index2 - this.items.slice(0, index2).filter(function(processedItem) {
        return _this2.isItemVisible(processedItem) && _this2.getItemProp(processedItem, "separator");
      }).length + 1;
    }, "getAriaPosInset"),
    getMenuItemProps: /* @__PURE__ */ __name(function getMenuItemProps3(processedItem, index2) {
      return {
        action: mergeProps({
          "class": this.cx("itemLink"),
          tabindex: -1,
          "aria-hidden": true
        }, this.getPTOptions(processedItem, index2, "itemLink")),
        icon: mergeProps({
          "class": [this.cx("itemIcon"), this.getItemProp(processedItem, "icon")]
        }, this.getPTOptions(processedItem, index2, "itemIcon")),
        label: mergeProps({
          "class": this.cx("itemLabel")
        }, this.getPTOptions(processedItem, index2, "itemLabel")),
        submenuicon: mergeProps({
          "class": this.cx("submenuIcon")
        }, this.getPTOptions(processedItem, index2, "submenuIcon"))
      };
    }, "getMenuItemProps"),
    containerRef: /* @__PURE__ */ __name(function containerRef6(el) {
      this.container = el;
    }, "containerRef")
  },
  components: {
    AngleRightIcon: script$y
  },
  directives: {
    ripple: Ripple
  }
};
var _hoisted_1$1$1 = ["tabindex"];
var _hoisted_2$1 = ["id", "aria-label", "aria-disabled", "aria-expanded", "aria-haspopup", "aria-level", "aria-setsize", "aria-posinset", "data-p-active", "data-p-focused", "data-p-disabled"];
var _hoisted_3 = ["onClick", "onMouseenter", "onMousemove"];
var _hoisted_4 = ["href", "target"];
var _hoisted_5 = ["id"];
var _hoisted_6 = ["id"];
function render$1(_ctx, _cache, $props, $setup, $data, $options) {
  var _component_AngleRightIcon = resolveComponent("AngleRightIcon");
  var _component_TieredMenuSub = resolveComponent("TieredMenuSub", true);
  var _directive_ripple = resolveDirective("ripple");
  return openBlock(), createBlock(Transition, mergeProps({
    name: "p-tieredmenu",
    onEnter: $options.onEnter
  }, _ctx.ptm("menu.transition")), {
    "default": withCtx(function() {
      return [($props.level === 0 ? true : $props.visible) ? (openBlock(), createElementBlock("ul", mergeProps({
        key: 0,
        ref: $options.containerRef,
        "class": $props.level === 0 ? _ctx.cx("rootList") : _ctx.cx("submenu"),
        tabindex: $props.tabindex
      }, $props.level === 0 ? _ctx.ptm("rootList") : _ctx.ptm("submenu")), [(openBlock(true), createElementBlock(Fragment, null, renderList($props.items, function(processedItem, index2) {
        return openBlock(), createElementBlock(Fragment, {
          key: $options.getItemKey(processedItem)
        }, [$options.isItemVisible(processedItem) && !$options.getItemProp(processedItem, "separator") ? (openBlock(), createElementBlock("li", mergeProps({
          key: 0,
          id: $options.getItemId(processedItem),
          style: $options.getItemProp(processedItem, "style"),
          "class": [_ctx.cx("item", {
            processedItem
          }), $options.getItemProp(processedItem, "class")],
          role: "menuitem",
          "aria-label": $options.getItemLabel(processedItem),
          "aria-disabled": $options.isItemDisabled(processedItem) || void 0,
          "aria-expanded": $options.isItemGroup(processedItem) ? $options.isItemActive(processedItem) : void 0,
          "aria-haspopup": $options.isItemGroup(processedItem) && !$options.getItemProp(processedItem, "to") ? "menu" : void 0,
          "aria-level": $props.level + 1,
          "aria-setsize": $options.getAriaSetSize(),
          "aria-posinset": $options.getAriaPosInset(index2),
          ref_for: true
        }, $options.getPTOptions(processedItem, index2, "item"), {
          "data-p-active": $options.isItemActive(processedItem),
          "data-p-focused": $options.isItemFocused(processedItem),
          "data-p-disabled": $options.isItemDisabled(processedItem)
        }), [createBaseVNode("div", mergeProps({
          "class": _ctx.cx("itemContent"),
          onClick: /* @__PURE__ */ __name(function onClick2($event) {
            return $options.onItemClick($event, processedItem);
          }, "onClick"),
          onMouseenter: /* @__PURE__ */ __name(function onMouseenter($event) {
            return $options.onItemMouseEnter($event, processedItem);
          }, "onMouseenter"),
          onMousemove: /* @__PURE__ */ __name(function onMousemove($event) {
            return $options.onItemMouseMove($event, processedItem);
          }, "onMousemove"),
          ref_for: true
        }, $options.getPTOptions(processedItem, index2, "itemContent")), [!$props.templates.item ? withDirectives((openBlock(), createElementBlock("a", mergeProps({
          key: 0,
          href: $options.getItemProp(processedItem, "url"),
          "class": _ctx.cx("itemLink"),
          target: $options.getItemProp(processedItem, "target"),
          tabindex: "-1",
          ref_for: true
        }, $options.getPTOptions(processedItem, index2, "itemLink")), [$props.templates.itemicon ? (openBlock(), createBlock(resolveDynamicComponent($props.templates.itemicon), {
          key: 0,
          item: processedItem.item,
          "class": normalizeClass(_ctx.cx("itemIcon"))
        }, null, 8, ["item", "class"])) : $options.getItemProp(processedItem, "icon") ? (openBlock(), createElementBlock("span", mergeProps({
          key: 1,
          "class": [_ctx.cx("itemIcon"), $options.getItemProp(processedItem, "icon")],
          ref_for: true
        }, $options.getPTOptions(processedItem, index2, "itemIcon")), null, 16)) : createCommentVNode("", true), createBaseVNode("span", mergeProps({
          id: $options.getItemLabelId(processedItem),
          "class": _ctx.cx("itemLabel"),
          ref_for: true
        }, $options.getPTOptions(processedItem, index2, "itemLabel")), toDisplayString($options.getItemLabel(processedItem)), 17, _hoisted_5), $options.getItemProp(processedItem, "items") ? (openBlock(), createElementBlock(Fragment, {
          key: 2
        }, [$props.templates.submenuicon ? (openBlock(), createBlock(resolveDynamicComponent($props.templates.submenuicon), mergeProps({
          key: 0,
          "class": _ctx.cx("submenuIcon"),
          active: $options.isItemActive(processedItem),
          ref_for: true
        }, $options.getPTOptions(processedItem, index2, "submenuIcon")), null, 16, ["class", "active"])) : (openBlock(), createBlock(_component_AngleRightIcon, mergeProps({
          key: 1,
          "class": _ctx.cx("submenuIcon"),
          ref_for: true
        }, $options.getPTOptions(processedItem, index2, "submenuIcon")), null, 16, ["class"]))], 64)) : createCommentVNode("", true)], 16, _hoisted_4)), [[_directive_ripple]]) : (openBlock(), createBlock(resolveDynamicComponent($props.templates.item), {
          key: 1,
          item: processedItem.item,
          hasSubmenu: $options.getItemProp(processedItem, "items"),
          label: $options.getItemLabel(processedItem),
          props: $options.getMenuItemProps(processedItem, index2)
        }, null, 8, ["item", "hasSubmenu", "label", "props"]))], 16, _hoisted_3), $options.isItemVisible(processedItem) && $options.isItemGroup(processedItem) ? (openBlock(), createBlock(_component_TieredMenuSub, {
          key: 0,
          id: $options.getItemId(processedItem) + "_list",
          style: normalizeStyle(_ctx.sx("submenu", true, {
            processedItem
          })),
          "aria-labelledby": $options.getItemLabelId(processedItem),
          role: "menu",
          menuId: $props.menuId,
          focusedItemId: $props.focusedItemId,
          items: processedItem.items,
          templates: $props.templates,
          activeItemPath: $props.activeItemPath,
          level: $props.level + 1,
          visible: $options.isItemActive(processedItem) && $options.isItemGroup(processedItem),
          pt: _ctx.pt,
          unstyled: _ctx.unstyled,
          onItemClick: _cache[0] || (_cache[0] = function($event) {
            return _ctx.$emit("item-click", $event);
          }),
          onItemMouseenter: _cache[1] || (_cache[1] = function($event) {
            return _ctx.$emit("item-mouseenter", $event);
          }),
          onItemMousemove: _cache[2] || (_cache[2] = function($event) {
            return _ctx.$emit("item-mousemove", $event);
          })
        }, null, 8, ["id", "style", "aria-labelledby", "menuId", "focusedItemId", "items", "templates", "activeItemPath", "level", "visible", "pt", "unstyled"])) : createCommentVNode("", true)], 16, _hoisted_2$1)) : createCommentVNode("", true), $options.isItemVisible(processedItem) && $options.getItemProp(processedItem, "separator") ? (openBlock(), createElementBlock("li", mergeProps({
          key: 1,
          id: $options.getItemId(processedItem),
          style: $options.getItemProp(processedItem, "style"),
          "class": [_ctx.cx("separator"), $options.getItemProp(processedItem, "class")],
          role: "separator",
          ref_for: true
        }, _ctx.ptm("separator")), null, 16, _hoisted_6)) : createCommentVNode("", true)], 64);
      }), 128))], 16, _hoisted_1$1$1)) : createCommentVNode("", true)];
    }),
    _: 1
  }, 16, ["onEnter"]);
}
__name(render$1, "render$1");
script$1$1.render = render$1;
var script$3 = {
  name: "TieredMenu",
  "extends": script$2,
  inheritAttrs: false,
  emits: ["focus", "blur", "before-show", "before-hide", "hide", "show"],
  outsideClickListener: null,
  scrollHandler: null,
  resizeListener: null,
  target: null,
  container: null,
  menubar: null,
  searchTimeout: null,
  searchValue: null,
  data: /* @__PURE__ */ __name(function data15() {
    return {
      id: this.$attrs.id,
      focused: false,
      focusedItemInfo: {
        index: -1,
        level: 0,
        parentKey: ""
      },
      activeItemPath: [],
      visible: !this.popup,
      submenuVisible: false,
      dirty: false
    };
  }, "data"),
  watch: {
    "$attrs.id": /* @__PURE__ */ __name(function $attrsId6(newValue) {
      this.id = newValue || UniqueComponentId();
    }, "$attrsId"),
    activeItemPath: /* @__PURE__ */ __name(function activeItemPath3(newPath) {
      if (!this.popup) {
        if (isNotEmpty(newPath)) {
          this.bindOutsideClickListener();
          this.bindResizeListener();
        } else {
          this.unbindOutsideClickListener();
          this.unbindResizeListener();
        }
      }
    }, "activeItemPath")
  },
  mounted: /* @__PURE__ */ __name(function mounted15() {
    this.id = this.id || UniqueComponentId();
  }, "mounted"),
  beforeUnmount: /* @__PURE__ */ __name(function beforeUnmount13() {
    this.unbindOutsideClickListener();
    this.unbindResizeListener();
    if (this.scrollHandler) {
      this.scrollHandler.destroy();
      this.scrollHandler = null;
    }
    if (this.container && this.autoZIndex) {
      ZIndex.clear(this.container);
    }
    this.target = null;
    this.container = null;
  }, "beforeUnmount"),
  methods: {
    getItemProp: /* @__PURE__ */ __name(function getItemProp6(item4, name) {
      return item4 ? resolve(item4[name]) : void 0;
    }, "getItemProp"),
    getItemLabel: /* @__PURE__ */ __name(function getItemLabel6(item4) {
      return this.getItemProp(item4, "label");
    }, "getItemLabel"),
    isItemDisabled: /* @__PURE__ */ __name(function isItemDisabled6(item4) {
      return this.getItemProp(item4, "disabled");
    }, "isItemDisabled"),
    isItemVisible: /* @__PURE__ */ __name(function isItemVisible6(item4) {
      return this.getItemProp(item4, "visible") !== false;
    }, "isItemVisible"),
    isItemGroup: /* @__PURE__ */ __name(function isItemGroup6(item4) {
      return isNotEmpty(this.getItemProp(item4, "items"));
    }, "isItemGroup"),
    isItemSeparator: /* @__PURE__ */ __name(function isItemSeparator3(item4) {
      return this.getItemProp(item4, "separator");
    }, "isItemSeparator"),
    getProccessedItemLabel: /* @__PURE__ */ __name(function getProccessedItemLabel3(processedItem) {
      return processedItem ? this.getItemLabel(processedItem.item) : void 0;
    }, "getProccessedItemLabel"),
    isProccessedItemGroup: /* @__PURE__ */ __name(function isProccessedItemGroup3(processedItem) {
      return processedItem && isNotEmpty(processedItem.items);
    }, "isProccessedItemGroup"),
    toggle: /* @__PURE__ */ __name(function toggle6(event) {
      this.visible ? this.hide(event, true) : this.show(event);
    }, "toggle"),
    show: /* @__PURE__ */ __name(function show5(event, isFocus) {
      if (this.popup) {
        this.$emit("before-show");
        this.visible = true;
        this.target = this.target || event.currentTarget;
        this.relatedTarget = event.relatedTarget || null;
      }
      isFocus && focus(this.menubar);
    }, "show"),
    hide: /* @__PURE__ */ __name(function hide5(event, isFocus) {
      if (this.popup) {
        this.$emit("before-hide");
        this.visible = false;
      }
      this.activeItemPath = [];
      this.focusedItemInfo = {
        index: -1,
        level: 0,
        parentKey: ""
      };
      isFocus && focus(this.relatedTarget || this.target || this.menubar);
      this.dirty = false;
    }, "hide"),
    onFocus: /* @__PURE__ */ __name(function onFocus4(event) {
      this.focused = true;
      if (!this.popup) {
        this.focusedItemInfo = this.focusedItemInfo.index !== -1 ? this.focusedItemInfo : {
          index: this.findFirstFocusedItemIndex(),
          level: 0,
          parentKey: ""
        };
      }
      this.$emit("focus", event);
    }, "onFocus"),
    onBlur: /* @__PURE__ */ __name(function onBlur4(event) {
      this.focused = false;
      this.focusedItemInfo = {
        index: -1,
        level: 0,
        parentKey: ""
      };
      this.searchValue = "";
      this.dirty = false;
      this.$emit("blur", event);
    }, "onBlur"),
    onKeyDown: /* @__PURE__ */ __name(function onKeyDown7(event) {
      if (this.disabled) {
        event.preventDefault();
        return;
      }
      var metaKey = event.metaKey || event.ctrlKey;
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
        case "Space":
          this.onSpaceKey(event);
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
        case "PageDown":
        case "PageUp":
        case "Backspace":
        case "ShiftLeft":
        case "ShiftRight":
          break;
        default:
          if (!metaKey && isPrintableCharacter(event.key)) {
            this.searchItems(event, event.key);
          }
          break;
      }
    }, "onKeyDown"),
    onItemChange: /* @__PURE__ */ __name(function onItemChange3(event) {
      var processedItem = event.processedItem, isFocus = event.isFocus;
      if (isEmpty(processedItem)) return;
      var index2 = processedItem.index, key = processedItem.key, level = processedItem.level, parentKey = processedItem.parentKey, items = processedItem.items;
      var grouped = isNotEmpty(items);
      var activeItemPath4 = this.activeItemPath.filter(function(p) {
        return p.parentKey !== parentKey && p.parentKey !== key;
      });
      if (grouped) {
        activeItemPath4.push(processedItem);
        this.submenuVisible = true;
      }
      this.focusedItemInfo = {
        index: index2,
        level,
        parentKey
      };
      this.activeItemPath = activeItemPath4;
      grouped && (this.dirty = true);
      isFocus && focus(this.menubar);
    }, "onItemChange"),
    onOverlayClick: /* @__PURE__ */ __name(function onOverlayClick5(event) {
      OverlayEventBus.emit("overlay-click", {
        originalEvent: event,
        target: this.target
      });
    }, "onOverlayClick"),
    onItemClick: /* @__PURE__ */ __name(function onItemClick7(event) {
      var originalEvent = event.originalEvent, processedItem = event.processedItem;
      var grouped = this.isProccessedItemGroup(processedItem);
      var root15 = isEmpty(processedItem.parent);
      var selected2 = this.isSelected(processedItem);
      if (selected2) {
        var index2 = processedItem.index, key = processedItem.key, level = processedItem.level, parentKey = processedItem.parentKey;
        this.activeItemPath = this.activeItemPath.filter(function(p) {
          return key !== p.key && key.startsWith(p.key);
        });
        this.focusedItemInfo = {
          index: index2,
          level,
          parentKey
        };
        this.dirty = !root15;
        focus(this.menubar);
      } else {
        if (grouped) {
          this.onItemChange(event);
        } else {
          var rootProcessedItem = root15 ? processedItem : this.activeItemPath.find(function(p) {
            return p.parentKey === "";
          });
          this.hide(originalEvent);
          this.changeFocusedItemIndex(originalEvent, rootProcessedItem ? rootProcessedItem.index : -1);
          focus(this.menubar);
        }
      }
    }, "onItemClick"),
    onItemMouseEnter: /* @__PURE__ */ __name(function onItemMouseEnter6(event) {
      if (this.dirty) {
        this.onItemChange(event);
      }
    }, "onItemMouseEnter"),
    onItemMouseMove: /* @__PURE__ */ __name(function onItemMouseMove6(event) {
      if (this.focused) {
        this.changeFocusedItemIndex(event, event.processedItem.index);
      }
    }, "onItemMouseMove"),
    onArrowDownKey: /* @__PURE__ */ __name(function onArrowDownKey4(event) {
      var itemIndex = this.focusedItemInfo.index !== -1 ? this.findNextItemIndex(this.focusedItemInfo.index) : this.findFirstFocusedItemIndex();
      this.changeFocusedItemIndex(event, itemIndex);
      event.preventDefault();
    }, "onArrowDownKey"),
    onArrowUpKey: /* @__PURE__ */ __name(function onArrowUpKey4(event) {
      if (event.altKey) {
        if (this.focusedItemInfo.index !== -1) {
          var processedItem = this.visibleItems[this.focusedItemInfo.index];
          var grouped = this.isProccessedItemGroup(processedItem);
          !grouped && this.onItemChange({
            originalEvent: event,
            processedItem
          });
        }
        this.popup && this.hide(event, true);
        event.preventDefault();
      } else {
        var itemIndex = this.focusedItemInfo.index !== -1 ? this.findPrevItemIndex(this.focusedItemInfo.index) : this.findLastFocusedItemIndex();
        this.changeFocusedItemIndex(event, itemIndex);
        event.preventDefault();
      }
    }, "onArrowUpKey"),
    onArrowLeftKey: /* @__PURE__ */ __name(function onArrowLeftKey4(event) {
      var _this = this;
      var processedItem = this.visibleItems[this.focusedItemInfo.index];
      var parentItem = this.activeItemPath.find(function(p) {
        return p.key === processedItem.parentKey;
      });
      var root15 = isEmpty(processedItem.parent);
      if (!root15) {
        this.focusedItemInfo = {
          index: -1,
          parentKey: parentItem ? parentItem.parentKey : ""
        };
        this.searchValue = "";
        this.onArrowDownKey(event);
      }
      this.activeItemPath = this.activeItemPath.filter(function(p) {
        return p.parentKey !== _this.focusedItemInfo.parentKey;
      });
      event.preventDefault();
    }, "onArrowLeftKey"),
    onArrowRightKey: /* @__PURE__ */ __name(function onArrowRightKey4(event) {
      var processedItem = this.visibleItems[this.focusedItemInfo.index];
      var grouped = this.isProccessedItemGroup(processedItem);
      if (grouped) {
        this.onItemChange({
          originalEvent: event,
          processedItem
        });
        this.focusedItemInfo = {
          index: -1,
          parentKey: processedItem.key
        };
        this.searchValue = "";
        this.onArrowDownKey(event);
      }
      event.preventDefault();
    }, "onArrowRightKey"),
    onHomeKey: /* @__PURE__ */ __name(function onHomeKey6(event) {
      this.changeFocusedItemIndex(event, this.findFirstItemIndex());
      event.preventDefault();
    }, "onHomeKey"),
    onEndKey: /* @__PURE__ */ __name(function onEndKey6(event) {
      this.changeFocusedItemIndex(event, this.findLastItemIndex());
      event.preventDefault();
    }, "onEndKey"),
    onEnterKey: /* @__PURE__ */ __name(function onEnterKey5(event) {
      if (this.focusedItemInfo.index !== -1) {
        var element = findSingle(this.menubar, 'li[id="'.concat("".concat(this.focusedItemId), '"]'));
        var anchorElement = element && findSingle(element, '[data-pc-section="itemlink"]');
        anchorElement ? anchorElement.click() : element && element.click();
        if (!this.popup) {
          var processedItem = this.visibleItems[this.focusedItemInfo.index];
          var grouped = this.isProccessedItemGroup(processedItem);
          !grouped && (this.focusedItemInfo.index = this.findFirstFocusedItemIndex());
        }
      }
      event.preventDefault();
    }, "onEnterKey"),
    onSpaceKey: /* @__PURE__ */ __name(function onSpaceKey3(event) {
      this.onEnterKey(event);
    }, "onSpaceKey"),
    onEscapeKey: /* @__PURE__ */ __name(function onEscapeKey4(event) {
      if (this.popup || this.focusedItemInfo.level !== 0) {
        var _focusedItemInfo = this.focusedItemInfo;
        this.hide(event, false);
        this.focusedItemInfo = {
          index: Number(_focusedItemInfo.parentKey.split("_")[0]),
          level: 0,
          parentKey: ""
        };
        this.popup && focus(this.target);
      }
      event.preventDefault();
    }, "onEscapeKey"),
    onTabKey: /* @__PURE__ */ __name(function onTabKey7(event) {
      if (this.focusedItemInfo.index !== -1) {
        var processedItem = this.visibleItems[this.focusedItemInfo.index];
        var grouped = this.isProccessedItemGroup(processedItem);
        !grouped && this.onItemChange({
          originalEvent: event,
          processedItem
        });
      }
      this.hide();
    }, "onTabKey"),
    onEnter: /* @__PURE__ */ __name(function onEnter8(el) {
      if (this.autoZIndex) {
        ZIndex.set("menu", el, this.baseZIndex + this.$primevue.config.zIndex.menu);
      }
      addStyle(el, {
        position: "absolute",
        top: "0",
        left: "0"
      });
      this.alignOverlay();
      focus(this.menubar);
      this.scrollInView();
    }, "onEnter"),
    onAfterEnter: /* @__PURE__ */ __name(function onAfterEnter3() {
      this.bindOutsideClickListener();
      this.bindScrollListener();
      this.bindResizeListener();
      this.$emit("show");
    }, "onAfterEnter"),
    onLeave: /* @__PURE__ */ __name(function onLeave5() {
      this.unbindOutsideClickListener();
      this.unbindScrollListener();
      this.unbindResizeListener();
      this.$emit("hide");
      this.container = null;
      this.dirty = false;
    }, "onLeave"),
    onAfterLeave: /* @__PURE__ */ __name(function onAfterLeave5(el) {
      if (this.autoZIndex) {
        ZIndex.clear(el);
      }
    }, "onAfterLeave"),
    alignOverlay: /* @__PURE__ */ __name(function alignOverlay5() {
      absolutePosition(this.container, this.target);
      var targetWidth = getOuterWidth(this.target);
      if (targetWidth > getOuterWidth(this.container)) {
        this.container.style.minWidth = getOuterWidth(this.target) + "px";
      }
    }, "alignOverlay"),
    bindOutsideClickListener: /* @__PURE__ */ __name(function bindOutsideClickListener7() {
      var _this2 = this;
      if (!this.outsideClickListener) {
        this.outsideClickListener = function(event) {
          var isOutsideContainer = _this2.container && !_this2.container.contains(event.target);
          var isOutsideTarget = _this2.popup ? !(_this2.target && (_this2.target === event.target || _this2.target.contains(event.target))) : true;
          if (isOutsideContainer && isOutsideTarget) {
            _this2.hide();
          }
        };
        document.addEventListener("click", this.outsideClickListener);
      }
    }, "bindOutsideClickListener"),
    unbindOutsideClickListener: /* @__PURE__ */ __name(function unbindOutsideClickListener7() {
      if (this.outsideClickListener) {
        document.removeEventListener("click", this.outsideClickListener);
        this.outsideClickListener = null;
      }
    }, "unbindOutsideClickListener"),
    bindScrollListener: /* @__PURE__ */ __name(function bindScrollListener5() {
      var _this3 = this;
      if (!this.scrollHandler) {
        this.scrollHandler = new ConnectedOverlayScrollHandler(this.target, function(event) {
          _this3.hide(event, true);
        });
      }
      this.scrollHandler.bindScrollListener();
    }, "bindScrollListener"),
    unbindScrollListener: /* @__PURE__ */ __name(function unbindScrollListener5() {
      if (this.scrollHandler) {
        this.scrollHandler.unbindScrollListener();
      }
    }, "unbindScrollListener"),
    bindResizeListener: /* @__PURE__ */ __name(function bindResizeListener7() {
      var _this4 = this;
      if (!this.resizeListener) {
        this.resizeListener = function(event) {
          if (!isTouchDevice()) {
            _this4.hide(event, true);
          }
        };
        window.addEventListener("resize", this.resizeListener);
      }
    }, "bindResizeListener"),
    unbindResizeListener: /* @__PURE__ */ __name(function unbindResizeListener7() {
      if (this.resizeListener) {
        window.removeEventListener("resize", this.resizeListener);
        this.resizeListener = null;
      }
    }, "unbindResizeListener"),
    isItemMatched: /* @__PURE__ */ __name(function isItemMatched3(processedItem) {
      var _this$getProccessedIt;
      return this.isValidItem(processedItem) && ((_this$getProccessedIt = this.getProccessedItemLabel(processedItem)) === null || _this$getProccessedIt === void 0 ? void 0 : _this$getProccessedIt.toLocaleLowerCase().startsWith(this.searchValue.toLocaleLowerCase()));
    }, "isItemMatched"),
    isValidItem: /* @__PURE__ */ __name(function isValidItem3(processedItem) {
      return !!processedItem && !this.isItemDisabled(processedItem.item) && !this.isItemSeparator(processedItem.item) && this.isItemVisible(processedItem.item);
    }, "isValidItem"),
    isValidSelectedItem: /* @__PURE__ */ __name(function isValidSelectedItem3(processedItem) {
      return this.isValidItem(processedItem) && this.isSelected(processedItem);
    }, "isValidSelectedItem"),
    isSelected: /* @__PURE__ */ __name(function isSelected5(processedItem) {
      return this.activeItemPath.some(function(p) {
        return p.key === processedItem.key;
      });
    }, "isSelected"),
    findFirstItemIndex: /* @__PURE__ */ __name(function findFirstItemIndex3() {
      var _this5 = this;
      return this.visibleItems.findIndex(function(processedItem) {
        return _this5.isValidItem(processedItem);
      });
    }, "findFirstItemIndex"),
    findLastItemIndex: /* @__PURE__ */ __name(function findLastItemIndex3() {
      var _this6 = this;
      return findLastIndex(this.visibleItems, function(processedItem) {
        return _this6.isValidItem(processedItem);
      });
    }, "findLastItemIndex"),
    findNextItemIndex: /* @__PURE__ */ __name(function findNextItemIndex3(index2) {
      var _this7 = this;
      var matchedItemIndex = index2 < this.visibleItems.length - 1 ? this.visibleItems.slice(index2 + 1).findIndex(function(processedItem) {
        return _this7.isValidItem(processedItem);
      }) : -1;
      return matchedItemIndex > -1 ? matchedItemIndex + index2 + 1 : index2;
    }, "findNextItemIndex"),
    findPrevItemIndex: /* @__PURE__ */ __name(function findPrevItemIndex3(index2) {
      var _this8 = this;
      var matchedItemIndex = index2 > 0 ? findLastIndex(this.visibleItems.slice(0, index2), function(processedItem) {
        return _this8.isValidItem(processedItem);
      }) : -1;
      return matchedItemIndex > -1 ? matchedItemIndex : index2;
    }, "findPrevItemIndex"),
    findSelectedItemIndex: /* @__PURE__ */ __name(function findSelectedItemIndex3() {
      var _this9 = this;
      return this.visibleItems.findIndex(function(processedItem) {
        return _this9.isValidSelectedItem(processedItem);
      });
    }, "findSelectedItemIndex"),
    findFirstFocusedItemIndex: /* @__PURE__ */ __name(function findFirstFocusedItemIndex3() {
      var selectedIndex = this.findSelectedItemIndex();
      return selectedIndex < 0 ? this.findFirstItemIndex() : selectedIndex;
    }, "findFirstFocusedItemIndex"),
    findLastFocusedItemIndex: /* @__PURE__ */ __name(function findLastFocusedItemIndex3() {
      var selectedIndex = this.findSelectedItemIndex();
      return selectedIndex < 0 ? this.findLastItemIndex() : selectedIndex;
    }, "findLastFocusedItemIndex"),
    searchItems: /* @__PURE__ */ __name(function searchItems3(event, _char) {
      var _this10 = this;
      this.searchValue = (this.searchValue || "") + _char;
      var itemIndex = -1;
      var matched = false;
      if (this.focusedItemInfo.index !== -1) {
        itemIndex = this.visibleItems.slice(this.focusedItemInfo.index).findIndex(function(processedItem) {
          return _this10.isItemMatched(processedItem);
        });
        itemIndex = itemIndex === -1 ? this.visibleItems.slice(0, this.focusedItemInfo.index).findIndex(function(processedItem) {
          return _this10.isItemMatched(processedItem);
        }) : itemIndex + this.focusedItemInfo.index;
      } else {
        itemIndex = this.visibleItems.findIndex(function(processedItem) {
          return _this10.isItemMatched(processedItem);
        });
      }
      if (itemIndex !== -1) {
        matched = true;
      }
      if (itemIndex === -1 && this.focusedItemInfo.index === -1) {
        itemIndex = this.findFirstFocusedItemIndex();
      }
      if (itemIndex !== -1) {
        this.changeFocusedItemIndex(event, itemIndex);
      }
      if (this.searchTimeout) {
        clearTimeout(this.searchTimeout);
      }
      this.searchTimeout = setTimeout(function() {
        _this10.searchValue = "";
        _this10.searchTimeout = null;
      }, 500);
      return matched;
    }, "searchItems"),
    changeFocusedItemIndex: /* @__PURE__ */ __name(function changeFocusedItemIndex3(event, index2) {
      if (this.focusedItemInfo.index !== index2) {
        this.focusedItemInfo.index = index2;
        this.scrollInView();
      }
    }, "changeFocusedItemIndex"),
    scrollInView: /* @__PURE__ */ __name(function scrollInView4() {
      var index2 = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : -1;
      var id = index2 !== -1 ? "".concat(this.id, "_").concat(index2) : this.focusedItemId;
      var element = findSingle(this.menubar, 'li[id="'.concat(id, '"]'));
      if (element) {
        element.scrollIntoView && element.scrollIntoView({
          block: "nearest",
          inline: "start"
        });
      }
    }, "scrollInView"),
    createProcessedItems: /* @__PURE__ */ __name(function createProcessedItems3(items) {
      var _this11 = this;
      var level = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : 0;
      var parent = arguments.length > 2 && arguments[2] !== void 0 ? arguments[2] : {};
      var parentKey = arguments.length > 3 && arguments[3] !== void 0 ? arguments[3] : "";
      var processedItems4 = [];
      items && items.forEach(function(item4, index2) {
        var key = (parentKey !== "" ? parentKey + "_" : "") + index2;
        var newItem = {
          item: item4,
          index: index2,
          level,
          key,
          parent,
          parentKey
        };
        newItem["items"] = _this11.createProcessedItems(item4.items, level + 1, newItem, key);
        processedItems4.push(newItem);
      });
      return processedItems4;
    }, "createProcessedItems"),
    containerRef: /* @__PURE__ */ __name(function containerRef7(el) {
      this.container = el;
    }, "containerRef"),
    menubarRef: /* @__PURE__ */ __name(function menubarRef2(el) {
      this.menubar = el ? el.$el : void 0;
    }, "menubarRef")
  },
  computed: {
    processedItems: /* @__PURE__ */ __name(function processedItems3() {
      return this.createProcessedItems(this.model || []);
    }, "processedItems"),
    visibleItems: /* @__PURE__ */ __name(function visibleItems3() {
      var _this12 = this;
      var processedItem = this.activeItemPath.find(function(p) {
        return p.key === _this12.focusedItemInfo.parentKey;
      });
      return processedItem ? processedItem.items : this.processedItems;
    }, "visibleItems"),
    focusedItemId: /* @__PURE__ */ __name(function focusedItemId2() {
      return this.focusedItemInfo.index !== -1 ? "".concat(this.id).concat(isNotEmpty(this.focusedItemInfo.parentKey) ? "_" + this.focusedItemInfo.parentKey : "", "_").concat(this.focusedItemInfo.index) : null;
    }, "focusedItemId")
  },
  components: {
    TieredMenuSub: script$1$1,
    Portal: script$r
  }
};
var _hoisted_1$3 = ["id"];
function render$2(_ctx, _cache, $props, $setup, $data, $options) {
  var _component_TieredMenuSub = resolveComponent("TieredMenuSub");
  var _component_Portal = resolveComponent("Portal");
  return openBlock(), createBlock(_component_Portal, {
    appendTo: _ctx.appendTo,
    disabled: !_ctx.popup
  }, {
    "default": withCtx(function() {
      return [createVNode(Transition, mergeProps({
        name: "p-connected-overlay",
        onEnter: $options.onEnter,
        onAfterEnter: $options.onAfterEnter,
        onLeave: $options.onLeave,
        onAfterLeave: $options.onAfterLeave
      }, _ctx.ptm("transition")), {
        "default": withCtx(function() {
          return [$data.visible ? (openBlock(), createElementBlock("div", mergeProps({
            key: 0,
            ref: $options.containerRef,
            id: $data.id,
            "class": _ctx.cx("root"),
            onClick: _cache[0] || (_cache[0] = function() {
              return $options.onOverlayClick && $options.onOverlayClick.apply($options, arguments);
            })
          }, _ctx.ptmi("root")), [_ctx.$slots.start ? (openBlock(), createElementBlock("div", mergeProps({
            key: 0,
            "class": _ctx.cx("start")
          }, _ctx.ptm("start")), [renderSlot(_ctx.$slots, "start")], 16)) : createCommentVNode("", true), createVNode(_component_TieredMenuSub, {
            ref: $options.menubarRef,
            id: $data.id + "_list",
            tabindex: !_ctx.disabled ? _ctx.tabindex : -1,
            role: "menubar",
            "aria-label": _ctx.ariaLabel,
            "aria-labelledby": _ctx.ariaLabelledby,
            "aria-disabled": _ctx.disabled || void 0,
            "aria-orientation": "vertical",
            "aria-activedescendant": $data.focused ? $options.focusedItemId : void 0,
            menuId: $data.id,
            focusedItemId: $data.focused ? $options.focusedItemId : void 0,
            items: $options.processedItems,
            templates: _ctx.$slots,
            activeItemPath: $data.activeItemPath,
            level: 0,
            visible: $data.submenuVisible,
            pt: _ctx.pt,
            unstyled: _ctx.unstyled,
            onFocus: $options.onFocus,
            onBlur: $options.onBlur,
            onKeydown: $options.onKeyDown,
            onItemClick: $options.onItemClick,
            onItemMouseenter: $options.onItemMouseEnter,
            onItemMousemove: $options.onItemMouseMove
          }, null, 8, ["id", "tabindex", "aria-label", "aria-labelledby", "aria-disabled", "aria-activedescendant", "menuId", "focusedItemId", "items", "templates", "activeItemPath", "visible", "pt", "unstyled", "onFocus", "onBlur", "onKeydown", "onItemClick", "onItemMouseenter", "onItemMousemove"]), _ctx.$slots.end ? (openBlock(), createElementBlock("div", mergeProps({
            key: 1,
            "class": _ctx.cx("end")
          }, _ctx.ptm("end")), [renderSlot(_ctx.$slots, "end")], 16)) : createCommentVNode("", true)], 16, _hoisted_1$3)) : createCommentVNode("", true)];
        }),
        _: 3
      }, 16, ["onEnter", "onAfterEnter", "onLeave", "onAfterLeave"])];
    }),
    _: 3
  }, 8, ["appendTo", "disabled"]);
}
__name(render$2, "render$2");
script$3.render = render$2;
var theme18 = /* @__PURE__ */ __name(function theme19(_ref) {
  var dt = _ref.dt;
  return "\n.p-splitbutton {\n    display: inline-flex;\n    position: relative;\n    border-radius: ".concat(dt("splitbutton.border.radius"), ";\n}\n\n.p-splitbutton-button {\n    border-top-right-radius: 0;\n    border-bottom-right-radius: 0;\n    border-right: 0 none;\n}\n\n.p-splitbutton-button:focus-visible,\n.p-splitbutton-dropdown:focus-visible {\n    z-index: 1;\n}\n\n.p-splitbutton-button:not(:disabled):hover,\n.p-splitbutton-button:not(:disabled):active {\n    border-right: 0 none;\n}\n\n.p-splitbutton-dropdown {\n    border-top-left-radius: 0;\n    border-bottom-left-radius: 0;\n}\n\n.p-splitbutton .p-menu {\n    min-width: 100%;\n}\n\n.p-splitbutton-fluid {\n    display: flex;\n}\n\n.p-splitbutton-rounded .p-splitbutton-dropdown {\n    border-top-right-radius: ").concat(dt("splitbutton.rounded.border.radius"), ";\n    border-bottom-right-radius: ").concat(dt("splitbutton.rounded.border.radius"), ";\n}\n\n.p-splitbutton-rounded .p-splitbutton-button {\n    border-top-left-radius: ").concat(dt("splitbutton.rounded.border.radius"), ";\n    border-bottom-left-radius: ").concat(dt("splitbutton.rounded.border.radius"), ";\n}\n\n.p-splitbutton-raised {\n    box-shadow: ").concat(dt("splitbutton.raised.shadow"), ";\n}\n");
}, "theme");
var classes = {
  root: /* @__PURE__ */ __name(function root14(_ref2) {
    var instance = _ref2.instance, props = _ref2.props;
    return ["p-splitbutton p-component", {
      "p-splitbutton-raised": props.raised,
      "p-splitbutton-rounded": props.rounded,
      "p-splitbutton-fluid": instance.hasFluid
    }];
  }, "root"),
  pcButton: "p-splitbutton-button",
  pcDropdown: "p-splitbutton-dropdown"
};
var SplitButtonStyle = BaseStyle.extend({
  name: "splitbutton",
  theme: theme18,
  classes
});
var script$1 = {
  name: "BaseSplitButton",
  "extends": script$p,
  props: {
    label: {
      type: String,
      "default": null
    },
    icon: {
      type: String,
      "default": null
    },
    model: {
      type: Array,
      "default": null
    },
    autoZIndex: {
      type: Boolean,
      "default": true
    },
    baseZIndex: {
      type: Number,
      "default": 0
    },
    appendTo: {
      type: [String, Object],
      "default": "body"
    },
    disabled: {
      type: Boolean,
      "default": false
    },
    fluid: {
      type: Boolean,
      "default": null
    },
    "class": {
      type: null,
      "default": null
    },
    style: {
      type: null,
      "default": null
    },
    buttonProps: {
      type: null,
      "default": null
    },
    menuButtonProps: {
      type: null,
      "default": null
    },
    menuButtonIcon: {
      type: String,
      "default": void 0
    },
    dropdownIcon: {
      type: String,
      "default": void 0
    },
    severity: {
      type: String,
      "default": null
    },
    raised: {
      type: Boolean,
      "default": false
    },
    rounded: {
      type: Boolean,
      "default": false
    },
    text: {
      type: Boolean,
      "default": false
    },
    outlined: {
      type: Boolean,
      "default": false
    },
    size: {
      type: String,
      "default": null
    },
    plain: {
      type: Boolean,
      "default": false
    }
  },
  style: SplitButtonStyle,
  provide: /* @__PURE__ */ __name(function provide20() {
    return {
      $pcSplitButton: this,
      $parentInstance: this
    };
  }, "provide")
};
var script = {
  name: "SplitButton",
  "extends": script$1,
  inheritAttrs: false,
  emits: ["click"],
  inject: {
    $pcFluid: {
      "default": null
    }
  },
  data: /* @__PURE__ */ __name(function data16() {
    return {
      id: this.$attrs.id,
      isExpanded: false
    };
  }, "data"),
  watch: {
    "$attrs.id": /* @__PURE__ */ __name(function $attrsId7(newValue) {
      this.id = newValue || UniqueComponentId();
    }, "$attrsId")
  },
  mounted: /* @__PURE__ */ __name(function mounted16() {
    var _this = this;
    this.id = this.id || UniqueComponentId();
    this.$watch("$refs.menu.visible", function(newValue) {
      _this.isExpanded = newValue;
    });
  }, "mounted"),
  methods: {
    onDropdownButtonClick: /* @__PURE__ */ __name(function onDropdownButtonClick(event) {
      if (event) {
        event.preventDefault();
      }
      this.$refs.menu.toggle({
        currentTarget: this.$el,
        relatedTarget: this.$refs.button.$el
      });
      this.isExpanded = this.$refs.menu.visible;
    }, "onDropdownButtonClick"),
    onDropdownKeydown: /* @__PURE__ */ __name(function onDropdownKeydown(event) {
      if (event.code === "ArrowDown" || event.code === "ArrowUp") {
        this.onDropdownButtonClick();
        event.preventDefault();
      }
    }, "onDropdownKeydown"),
    onDefaultButtonClick: /* @__PURE__ */ __name(function onDefaultButtonClick(event) {
      if (this.isExpanded) {
        this.$refs.menu.hide(event);
      }
      this.$emit("click", event);
    }, "onDefaultButtonClick")
  },
  computed: {
    containerClass: /* @__PURE__ */ __name(function containerClass() {
      return [this.cx("root"), this["class"]];
    }, "containerClass"),
    hasFluid: /* @__PURE__ */ __name(function hasFluid2() {
      return isEmpty(this.fluid) ? !!this.$pcFluid : this.fluid;
    }, "hasFluid")
  },
  components: {
    PVSButton: script$o,
    PVSMenu: script$3,
    ChevronDownIcon: script$s
  }
};
var _hoisted_1$2 = ["data-p-severity"];
function render(_ctx, _cache, $props, $setup, $data, $options) {
  var _component_PVSButton = resolveComponent("PVSButton");
  var _component_PVSMenu = resolveComponent("PVSMenu");
  return openBlock(), createElementBlock("div", mergeProps({
    "class": $options.containerClass,
    style: _ctx.style
  }, _ctx.ptmi("root"), {
    "data-p-severity": _ctx.severity
  }), [createVNode(_component_PVSButton, mergeProps({
    type: "button",
    "class": _ctx.cx("pcButton"),
    label: _ctx.label,
    disabled: _ctx.disabled,
    severity: _ctx.severity,
    text: _ctx.text,
    icon: _ctx.icon,
    outlined: _ctx.outlined,
    size: _ctx.size,
    fluid: _ctx.fluid,
    "aria-label": _ctx.label,
    onClick: $options.onDefaultButtonClick
  }, _ctx.buttonProps, {
    pt: _ctx.ptm("pcButton"),
    unstyled: _ctx.unstyled
  }), createSlots({
    "default": withCtx(function() {
      return [renderSlot(_ctx.$slots, "default")];
    }),
    _: 2
  }, [_ctx.$slots.icon ? {
    name: "icon",
    fn: withCtx(function(slotProps) {
      return [renderSlot(_ctx.$slots, "icon", {
        "class": normalizeClass(slotProps["class"])
      }, function() {
        return [createBaseVNode("span", mergeProps({
          "class": [_ctx.icon, slotProps["class"]]
        }, _ctx.ptm("pcButton")["icon"], {
          "data-pc-section": "buttonicon"
        }), null, 16)];
      })];
    }),
    key: "0"
  } : void 0]), 1040, ["class", "label", "disabled", "severity", "text", "icon", "outlined", "size", "fluid", "aria-label", "onClick", "pt", "unstyled"]), createVNode(_component_PVSButton, mergeProps({
    ref: "button",
    type: "button",
    "class": _ctx.cx("pcDropdown"),
    disabled: _ctx.disabled,
    "aria-haspopup": "true",
    "aria-expanded": $data.isExpanded,
    "aria-controls": $data.id + "_overlay",
    onClick: $options.onDropdownButtonClick,
    onKeydown: $options.onDropdownKeydown,
    severity: _ctx.severity,
    text: _ctx.text,
    outlined: _ctx.outlined,
    size: _ctx.size,
    unstyled: _ctx.unstyled
  }, _ctx.menuButtonProps, {
    pt: _ctx.ptm("pcDropdown")
  }), {
    icon: withCtx(function(slotProps) {
      return [renderSlot(_ctx.$slots, _ctx.$slots.dropdownicon ? "dropdownicon" : "menubuttonicon", {
        "class": normalizeClass(slotProps["class"])
      }, function() {
        return [(openBlock(), createBlock(resolveDynamicComponent(_ctx.menuButtonIcon || _ctx.dropdownIcon ? "span" : "ChevronDownIcon"), mergeProps({
          "class": [_ctx.dropdownIcon || _ctx.menuButtonIcon, slotProps["class"]]
        }, _ctx.ptm("pcDropdown")["icon"], {
          "data-pc-section": "menubuttonicon"
        }), null, 16, ["class"]))];
      })];
    }),
    _: 3
  }, 16, ["class", "disabled", "aria-expanded", "aria-controls", "onClick", "onKeydown", "severity", "text", "outlined", "size", "unstyled", "pt"]), createVNode(_component_PVSMenu, {
    ref: "menu",
    id: $data.id + "_overlay",
    model: _ctx.model,
    popup: true,
    autoZIndex: _ctx.autoZIndex,
    baseZIndex: _ctx.baseZIndex,
    appendTo: _ctx.appendTo,
    unstyled: _ctx.unstyled,
    pt: _ctx.ptm("pcMenu")
  }, createSlots({
    _: 2
  }, [_ctx.$slots.menuitemicon ? {
    name: "itemicon",
    fn: withCtx(function(slotProps) {
      return [renderSlot(_ctx.$slots, "menuitemicon", {
        item: slotProps.item,
        "class": normalizeClass(slotProps["class"])
      })];
    }),
    key: "0"
  } : void 0, _ctx.$slots.item ? {
    name: "item",
    fn: withCtx(function(slotProps) {
      return [renderSlot(_ctx.$slots, "item", {
        item: slotProps.item,
        hasSubmenu: slotProps.hasSubmenu,
        label: slotProps.label,
        props: slotProps.props
      })];
    }),
    key: "1"
  } : void 0]), 1032, ["id", "model", "autoZIndex", "baseZIndex", "appendTo", "unstyled", "pt"])], 16, _hoisted_1$2);
}
__name(render, "render");
script.render = render;
const minQueueCount = 1;
const _sfc_main$4 = /* @__PURE__ */ defineComponent({
  __name: "BatchCountEdit",
  props: {
    class: { default: "" }
  },
  setup(__props) {
    const props = __props;
    const queueSettingsStore = useQueueSettingsStore();
    const { batchCount } = storeToRefs(queueSettingsStore);
    const settingStore = useSettingStore();
    const maxQueueCount = computed(
      () => settingStore.get("Comfy.QueueButton.BatchCountLimit")
    );
    const handleClick = /* @__PURE__ */ __name((increment) => {
      let newCount;
      if (increment) {
        const originalCount = batchCount.value - 1;
        newCount = originalCount * 2;
      } else {
        const originalCount = batchCount.value + 1;
        newCount = Math.floor(originalCount / 2);
      }
      batchCount.value = newCount;
    }, "handleClick");
    return (_ctx, _cache) => {
      const _directive_tooltip = resolveDirective("tooltip");
      return withDirectives((openBlock(), createElementBlock("div", {
        class: normalizeClass(["batch-count", props.class])
      }, [
        createVNode(unref(script$R), {
          class: "w-14",
          modelValue: unref(batchCount),
          "onUpdate:modelValue": _cache[0] || (_cache[0] = ($event) => isRef(batchCount) ? batchCount.value = $event : null),
          min: minQueueCount,
          max: maxQueueCount.value,
          fluid: "",
          showButtons: "",
          pt: {
            incrementButton: {
              class: "w-6",
              onmousedown: /* @__PURE__ */ __name(() => {
                handleClick(true);
              }, "onmousedown")
            },
            decrementButton: {
              class: "w-6",
              onmousedown: /* @__PURE__ */ __name(() => {
                handleClick(false);
              }, "onmousedown")
            }
          }
        }, null, 8, ["modelValue", "max", "pt"])
      ], 2)), [
        [
          _directive_tooltip,
          _ctx.$t("menu.batchCount"),
          void 0,
          { bottom: true }
        ]
      ]);
    };
  }
});
const BatchCountEdit = /* @__PURE__ */ _export_sfc(_sfc_main$4, [["__scopeId", "data-v-713442be"]]);
const _withScopeId$1 = /* @__PURE__ */ __name((n) => (pushScopeId("data-v-fcd3efcd"), n = n(), popScopeId(), n), "_withScopeId$1");
const _hoisted_1$1 = { class: "queue-button-group flex" };
const _sfc_main$3 = /* @__PURE__ */ defineComponent({
  __name: "ComfyQueueButton",
  setup(__props) {
    const queueCountStore = storeToRefs(useQueuePendingTaskCountStore());
    const { mode: queueMode } = storeToRefs(useQueueSettingsStore());
    const { t } = useI18n();
    const queueModeMenuItemLookup = {
      disabled: {
        key: "disabled",
        label: "Queue",
        icon: "pi pi-play",
        tooltip: t("menu.disabledTooltip"),
        command: /* @__PURE__ */ __name(() => {
          queueMode.value = "disabled";
        }, "command")
      },
      instant: {
        key: "instant",
        label: "Queue (Instant)",
        icon: "pi pi-forward",
        tooltip: t("menu.instantTooltip"),
        command: /* @__PURE__ */ __name(() => {
          queueMode.value = "instant";
        }, "command")
      },
      change: {
        key: "change",
        label: "Queue (Change)",
        icon: "pi pi-step-forward-alt",
        tooltip: t("menu.changeTooltip"),
        command: /* @__PURE__ */ __name(() => {
          queueMode.value = "change";
        }, "command")
      }
    };
    const activeQueueModeMenuItem = computed(
      () => queueModeMenuItemLookup[queueMode.value]
    );
    const queueModeMenuItems = computed(
      () => Object.values(queueModeMenuItemLookup)
    );
    const executingPrompt = computed(() => !!queueCountStore.count.value);
    const hasPendingTasks = computed(() => queueCountStore.count.value > 1);
    const commandStore = useCommandStore();
    const queuePrompt = /* @__PURE__ */ __name((e) => {
      const commandId = e.shiftKey ? "Comfy.QueuePromptFront" : "Comfy.QueuePrompt";
      commandStore.execute(commandId);
    }, "queuePrompt");
    return (_ctx, _cache) => {
      const _directive_tooltip = resolveDirective("tooltip");
      return openBlock(), createElementBlock("div", _hoisted_1$1, [
        withDirectives((openBlock(), createBlock(unref(script), {
          class: "comfyui-queue-button",
          label: activeQueueModeMenuItem.value.label,
          icon: activeQueueModeMenuItem.value.icon,
          severity: "primary",
          onClick: queuePrompt,
          model: queueModeMenuItems.value,
          "data-testid": "queue-button"
        }, {
          item: withCtx(({ item: item4 }) => [
            withDirectives(createVNode(unref(script$o), {
              label: item4.label,
              icon: item4.icon,
              severity: item4.key === unref(queueMode) ? "primary" : "secondary",
              text: ""
            }, null, 8, ["label", "icon", "severity"]), [
              [_directive_tooltip, item4.tooltip]
            ])
          ]),
          _: 1
        }, 8, ["label", "icon", "model"])), [
          [
            _directive_tooltip,
            _ctx.$t("menu.queueWorkflow"),
            void 0,
            { bottom: true }
          ]
        ]),
        createVNode(BatchCountEdit),
        createVNode(unref(script$f), { class: "execution-actions flex flex-nowrap" }, {
          default: withCtx(() => [
            withDirectives(createVNode(unref(script$o), {
              icon: "pi pi-times",
              severity: executingPrompt.value ? "danger" : "secondary",
              disabled: !executingPrompt.value,
              text: "",
              onClick: _cache[0] || (_cache[0] = () => unref(commandStore).execute("Comfy.Interrupt"))
            }, null, 8, ["severity", "disabled"]), [
              [
                _directive_tooltip,
                _ctx.$t("menu.interrupt"),
                void 0,
                { bottom: true }
              ]
            ]),
            withDirectives(createVNode(unref(script$o), {
              icon: "pi pi-stop",
              severity: hasPendingTasks.value ? "danger" : "secondary",
              disabled: !hasPendingTasks.value,
              text: "",
              onClick: _cache[1] || (_cache[1] = () => unref(commandStore).execute("Comfy.ClearPendingTasks"))
            }, null, 8, ["severity", "disabled"]), [
              [
                _directive_tooltip,
                _ctx.$t("sideToolbar.queueTab.clearPendingTasks"),
                void 0,
                { bottom: true }
              ]
            ])
          ]),
          _: 1
        })
      ]);
    };
  }
});
const ComfyQueueButton = /* @__PURE__ */ _export_sfc(_sfc_main$3, [["__scopeId", "data-v-fcd3efcd"]]);
const overlapThreshold = 20;
const _sfc_main$2 = /* @__PURE__ */ defineComponent({
  __name: "ComfyActionbar",
  setup(__props) {
    const settingsStore = useSettingStore();
    const commandStore = useCommandStore();
    const visible = computed(
      () => settingsStore.get("Comfy.UseNewMenu") !== "Disabled"
    );
    const panelRef = ref(null);
    const dragHandleRef = ref(null);
    const isDocked = useLocalStorage("Comfy.MenuPosition.Docked", false);
    const storedPosition = useLocalStorage("Comfy.MenuPosition.Floating", {
      x: 0,
      y: 0
    });
    const {
      x,
      y,
      style,
      isDragging
    } = useDraggable(panelRef, {
      initialValue: { x: 0, y: 0 },
      handle: dragHandleRef,
      containerElement: document.body
    });
    watchDebounced(
      [x, y],
      ([newX, newY]) => {
        storedPosition.value = { x: newX, y: newY };
      },
      { debounce: 300 }
    );
    const setInitialPosition = /* @__PURE__ */ __name(() => {
      if (x.value !== 0 || y.value !== 0) {
        return;
      }
      if (storedPosition.value.x !== 0 || storedPosition.value.y !== 0) {
        x.value = storedPosition.value.x;
        y.value = storedPosition.value.y;
        return;
      }
      if (panelRef.value) {
        const screenWidth = window.innerWidth;
        const screenHeight = window.innerHeight;
        const menuWidth = panelRef.value.offsetWidth;
        const menuHeight = panelRef.value.offsetHeight;
        if (menuWidth === 0 || menuHeight === 0) {
          return;
        }
        x.value = (screenWidth - menuWidth) / 2;
        y.value = screenHeight - menuHeight - 10;
      }
    }, "setInitialPosition");
    onMounted(setInitialPosition);
    watch(visible, (newVisible) => {
      if (newVisible) {
        nextTick(setInitialPosition);
      }
    });
    const adjustMenuPosition = /* @__PURE__ */ __name(() => {
      if (panelRef.value) {
        const screenWidth = window.innerWidth;
        const screenHeight = window.innerHeight;
        const menuWidth = panelRef.value.offsetWidth;
        const menuHeight = panelRef.value.offsetHeight;
        x.value = lodashExports.clamp(x.value, 0, screenWidth - menuWidth);
        y.value = lodashExports.clamp(y.value, 0, screenHeight - menuHeight);
      }
    }, "adjustMenuPosition");
    useEventListener(window, "resize", adjustMenuPosition);
    const topMenuRef = inject("topMenuRef");
    const topMenuBounds = useElementBounding(topMenuRef);
    const isOverlappingWithTopMenu = computed(() => {
      if (!panelRef.value) {
        return false;
      }
      const { height } = panelRef.value.getBoundingClientRect();
      const actionbarBottom = y.value + height;
      const topMenuBottom = topMenuBounds.bottom.value;
      const overlapPixels = Math.min(actionbarBottom, topMenuBottom) - Math.max(y.value, topMenuBounds.top.value);
      return overlapPixels > overlapThreshold;
    });
    watch(isDragging, (newIsDragging) => {
      if (!newIsDragging) {
        isDocked.value = isOverlappingWithTopMenu.value;
      } else {
        isDocked.value = false;
      }
    });
    const eventBus = useEventBus("topMenu");
    watch([isDragging, isOverlappingWithTopMenu], ([dragging, overlapping]) => {
      eventBus.emit("updateHighlight", {
        isDragging: dragging,
        isOverlapping: overlapping
      });
    });
    return (_ctx, _cache) => {
      const _directive_tooltip = resolveDirective("tooltip");
      return openBlock(), createBlock(unref(script$4), {
        class: normalizeClass(["actionbar w-fit", { "is-dragging": unref(isDragging), "is-docked": unref(isDocked) }]),
        style: normalizeStyle(unref(style))
      }, {
        default: withCtx(() => [
          createBaseVNode("div", {
            class: "actionbar-content flex items-center",
            ref_key: "panelRef",
            ref: panelRef
          }, [
            createBaseVNode("span", {
              class: "drag-handle cursor-move mr-2 p-0!",
              ref_key: "dragHandleRef",
              ref: dragHandleRef
            }, null, 512),
            createVNode(ComfyQueueButton),
            createVNode(unref(script$K), {
              layout: "vertical",
              class: "mx-1"
            }),
            createVNode(unref(script$f), { class: "flex flex-nowrap" }, {
              default: withCtx(() => [
                withDirectives(createVNode(unref(script$o), {
                  icon: "pi pi-refresh",
                  severity: "secondary",
                  text: "",
                  onClick: _cache[0] || (_cache[0] = () => unref(commandStore).execute("Comfy.RefreshNodeDefinitions"))
                }, null, 512), [
                  [
                    _directive_tooltip,
                    _ctx.$t("menu.refresh"),
                    void 0,
                    { bottom: true }
                  ]
                ])
              ]),
              _: 1
            })
          ], 512)
        ]),
        _: 1
      }, 8, ["style", "class"]);
    };
  }
});
const Actionbar = /* @__PURE__ */ _export_sfc(_sfc_main$2, [["__scopeId", "data-v-bc6c78dd"]]);
const _withScopeId = /* @__PURE__ */ __name((n) => (pushScopeId("data-v-b13fdc92"), n = n(), popScopeId(), n), "_withScopeId");
const _hoisted_1 = /* @__PURE__ */ _withScopeId(() => /* @__PURE__ */ createBaseVNode("h1", { class: "comfyui-logo mx-2" }, "ComfyUI", -1));
const _hoisted_2 = { class: "flex-grow" };
const _sfc_main$1 = /* @__PURE__ */ defineComponent({
  __name: "TopMenubar",
  setup(__props) {
    const settingStore = useSettingStore();
    const workflowTabsPosition = computed(
      () => settingStore.get("Comfy.Workflow.WorkflowTabsPosition")
    );
    const betaMenuEnabled = computed(
      () => settingStore.get("Comfy.UseNewMenu") !== "Disabled"
    );
    const teleportTarget = computed(
      () => settingStore.get("Comfy.UseNewMenu") === "Top" ? ".comfyui-body-top" : ".comfyui-body-bottom"
    );
    const menuRight = ref(null);
    onMounted(() => {
      if (menuRight.value) {
        menuRight.value.appendChild(app.menu.element);
      }
    });
    const topMenuRef = ref(null);
    provide("topMenuRef", topMenuRef);
    const eventBus = useEventBus("topMenu");
    const isDropZone = ref(false);
    const isDroppable = ref(false);
    eventBus.on((event, payload) => {
      if (event === "updateHighlight") {
        isDropZone.value = payload.isDragging;
        isDroppable.value = payload.isOverlapping && payload.isDragging;
      }
    });
    return (_ctx, _cache) => {
      return openBlock(), createBlock(Teleport, { to: teleportTarget.value }, [
        withDirectives(createBaseVNode("div", {
          ref_key: "topMenuRef",
          ref: topMenuRef,
          class: normalizeClass(["comfyui-menu flex items-center", { dropzone: isDropZone.value, "dropzone-active": isDroppable.value }])
        }, [
          _hoisted_1,
          createVNode(CommandMenubar),
          createVNode(unref(script$K), {
            layout: "vertical",
            class: "mx-2"
          }),
          createBaseVNode("div", _hoisted_2, [
            workflowTabsPosition.value === "Topbar" ? (openBlock(), createBlock(WorkflowTabs, { key: 0 })) : createCommentVNode("", true)
          ]),
          createBaseVNode("div", {
            class: "comfyui-menu-right",
            ref_key: "menuRight",
            ref: menuRight
          }, null, 512),
          createVNode(Actionbar)
        ], 2), [
          [vShow, betaMenuEnabled.value]
        ])
      ], 8, ["to"]);
    };
  }
});
const TopMenubar = /* @__PURE__ */ _export_sfc(_sfc_main$1, [["__scopeId", "data-v-b13fdc92"]]);
function setupAutoQueueHandler() {
  const queueCountStore = useQueuePendingTaskCountStore();
  const queueSettingsStore = useQueueSettingsStore();
  let graphHasChanged = false;
  let internalCount = 0;
  api.addEventListener("graphChanged", () => {
    if (queueSettingsStore.mode === "change") {
      if (internalCount) {
        graphHasChanged = true;
      } else {
        graphHasChanged = false;
        app.queuePrompt(0, queueSettingsStore.batchCount);
        internalCount++;
      }
    }
  });
  queueCountStore.$subscribe(
    () => {
      internalCount = queueCountStore.count;
      if (!internalCount && !app.lastExecutionError) {
        if (queueSettingsStore.mode === "instant" || queueSettingsStore.mode === "change" && graphHasChanged) {
          graphHasChanged = false;
          app.queuePrompt(0, queueSettingsStore.batchCount);
        }
      }
    },
    { detached: true }
  );
}
__name(setupAutoQueueHandler, "setupAutoQueueHandler");
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "GraphView",
  setup(__props) {
    setupAutoQueueHandler();
    const { t } = useI18n();
    const toast = useToast();
    const settingStore = useSettingStore();
    const executionStore = useExecutionStore();
    const theme20 = computed(() => settingStore.get("Comfy.ColorPalette"));
    watch(
      theme20,
      (newTheme) => {
        const DARK_THEME_CLASS = "dark-theme";
        const isDarkTheme = newTheme !== "light";
        if (isDarkTheme) {
          document.body.classList.add(DARK_THEME_CLASS);
        } else {
          document.body.classList.remove(DARK_THEME_CLASS);
        }
      },
      { immediate: true }
    );
    watchEffect(() => {
      const fontSize = settingStore.get("Comfy.TextareaWidget.FontSize");
      document.documentElement.style.setProperty(
        "--comfy-textarea-font-size",
        `${fontSize}px`
      );
    });
    watchEffect(() => {
      const padding = settingStore.get("Comfy.TreeExplorer.ItemPadding");
      document.documentElement.style.setProperty(
        "--comfy-tree-explorer-item-padding",
        `${padding}px`
      );
    });
    watchEffect(() => {
      const locale = settingStore.get("Comfy.Locale");
      if (locale) {
        i18n.global.locale.value = locale;
      }
    });
    watchEffect(() => {
      const useNewMenu = settingStore.get("Comfy.UseNewMenu");
      if (useNewMenu === "Disabled") {
        app.ui.menuContainer.style.removeProperty("display");
        app.ui.restoreMenuPosition();
      } else {
        app.ui.menuContainer.style.setProperty("display", "none");
      }
    });
    const init = /* @__PURE__ */ __name(() => {
      settingStore.addSettings(app.ui.settings);
      useKeybindingStore().loadCoreKeybindings();
      app.extensionManager = useWorkspaceStore();
      app.extensionManager.registerSidebarTab({
        id: "queue",
        icon: "pi pi-history",
        iconBadge: /* @__PURE__ */ __name(() => {
          const value = useQueuePendingTaskCountStore().count.toString();
          return value === "0" ? null : value;
        }, "iconBadge"),
        title: t("sideToolbar.queue"),
        tooltip: t("sideToolbar.queue"),
        component: markRaw(QueueSidebarTab),
        type: "vue"
      });
      app.extensionManager.registerSidebarTab({
        id: "node-library",
        icon: "pi pi-book",
        title: t("sideToolbar.nodeLibrary"),
        tooltip: t("sideToolbar.nodeLibrary"),
        component: markRaw(NodeLibrarySidebarTab),
        type: "vue"
      });
      app.extensionManager.registerSidebarTab({
        id: "model-library",
        icon: "pi pi-box",
        title: t("sideToolbar.modelLibrary"),
        tooltip: t("sideToolbar.modelLibrary"),
        component: markRaw(ModelLibrarySidebarTab),
        type: "vue"
      });
      app.extensionManager.registerSidebarTab({
        id: "workflows",
        icon: "pi pi-folder-open",
        iconBadge: /* @__PURE__ */ __name(() => {
          if (settingStore.get("Comfy.Workflow.WorkflowTabsPosition") !== "Sidebar") {
            return null;
          }
          const value = useWorkflowStore().openWorkflows.length.toString();
          return value === "0" ? null : value;
        }, "iconBadge"),
        title: t("sideToolbar.workflows"),
        tooltip: t("sideToolbar.workflows"),
        component: markRaw(WorkflowsSidebarTab),
        type: "vue"
      });
    }, "init");
    const queuePendingTaskCountStore = useQueuePendingTaskCountStore();
    const onStatus = /* @__PURE__ */ __name((e) => {
      queuePendingTaskCountStore.update(e);
    }, "onStatus");
    const reconnectingMessage = {
      severity: "error",
      summary: t("reconnecting")
    };
    const onReconnecting = /* @__PURE__ */ __name(() => {
      toast.remove(reconnectingMessage);
      toast.add(reconnectingMessage);
    }, "onReconnecting");
    const onReconnected = /* @__PURE__ */ __name(() => {
      toast.remove(reconnectingMessage);
      toast.add({
        severity: "success",
        summary: t("reconnected"),
        life: 2e3
      });
    }, "onReconnected");
    const workflowStore = useWorkflowStore();
    const workflowBookmarkStore = useWorkflowBookmarkStore();
    app.workflowManager.executionStore = executionStore;
    app.workflowManager.workflowStore = workflowStore;
    app.workflowManager.workflowBookmarkStore = workflowBookmarkStore;
    onMounted(() => {
      api.addEventListener("status", onStatus);
      api.addEventListener("reconnecting", onReconnecting);
      api.addEventListener("reconnected", onReconnected);
      executionStore.bindExecutionEvents();
      try {
        init();
      } catch (e) {
        console.error("Failed to init ComfyUI frontend", e);
      }
    });
    onBeforeUnmount(() => {
      api.removeEventListener("status", onStatus);
      api.removeEventListener("reconnecting", onReconnecting);
      api.removeEventListener("reconnected", onReconnected);
      executionStore.unbindExecutionEvents();
    });
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock(Fragment, null, [
        createVNode(TopMenubar),
        createVNode(_sfc_main$t),
        createVNode(_sfc_main$c),
        createVNode(_sfc_main$b),
        createVNode(_sfc_main$a)
      ], 64);
    };
  }
});
export {
  _sfc_main as default
};
//# sourceMappingURL=GraphView-CVV2XJjS.js.map
