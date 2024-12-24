var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });
import { d as defineStore, s as shallowRef, a as defineComponent, u as useSettingStore, r as ref, w as watch, L as LGraphGroup, b as app, c as LGraphNode, o as onMounted, e as onUnmounted, f as openBlock, g as createElementBlock, h as createVNode, E as EditableText, n as normalizeStyle, i as createCommentVNode, j as LiteGraph, _ as _export_sfc, B as BaseStyle, k as script$e, l as resolveComponent, m as mergeProps, p as renderSlot, q as computed, t as resolveDirective, v as withDirectives, x as createBlock, y as withCtx, z as unref, A as createBaseVNode, C as normalizeClass, D as script$f, F as useCommandStore, G as useDialogStore, S as SettingDialogHeader, H as SettingDialogContent, I as useI18n, J as useUserStore, K as onBeforeUnmount, M as resolveDynamicComponent, N as useWorkspaceStore, O as useKeybindingStore, P as Fragment, Q as renderList, T as Teleport, R as pushScopeId, U as popScopeId, V as script$g, W as getWidth, X as findSingle, Y as getOuterHeight, Z as getOffset, $ as getOuterWidth, a0 as getHeight, a1 as script$h, a2 as script$i, a3 as Ripple, a4 as getAttribute, a5 as focus, a6 as equals, a7 as useBottomPanelStore, a8 as toDisplayString, a9 as script$j, aa as getVNodeProp, ab as isArray, ac as useSidebarTabStore, ad as vShow, ae as isNotEmpty, af as UniqueComponentId, ag as ZIndex, ah as resolveFieldData, ai as OverlayEventBus, aj as isEmpty, ak as addStyle, al as relativePosition, am as absolutePosition, an as ConnectedOverlayScrollHandler, ao as isTouchDevice, ap as findLastIndex, aq as script$k, ar as script$l, as as script$m, at as script$n, au as script$o, av as script$p, aw as Transition, ax as createSlots, ay as createTextVNode, az as useNodeFrequencyStore, aA as useNodeBookmarkStore, aB as highlightQuery, aC as script$q, aD as formatNumberWithSuffix, aE as NodeSourceType, aF as useNodeDefStore, aG as NodePreview, aH as NodeSearchFilter, aI as script$r, aJ as SearchFilterChip, aK as storeToRefs, aL as watchEffect, aM as useEventListener, aN as isRef, aO as toRaw, aP as LinkReleaseTriggerAction, aQ as nextTick, aR as st, aS as normalizeI18nKey, aT as getColorPalette, aU as BadgePosition, aV as LGraphBadge, aW as _, aX as defaultColorPalette, aY as NodeBadgeMode, aZ as markRaw, a_ as useModelToNodeStore, a$ as CanvasPointer, b0 as useWorkflowStore, b1 as setStorageValue, b2 as api, b3 as usePragmaticDroppable, b4 as ComfyNodeDefImpl, b5 as ComfyModelDef, b6 as LGraph, b7 as LLink, b8 as DragAndScale, b9 as LGraphCanvas, ba as ContextMenu, bb as ChangeTracker, bc as workflowService, bd as showNativeMenu, be as script$s, bf as script$t, bg as script$u, bh as script$v, bi as script$w, bj as normalizeProps, bk as ToastEventBus, bl as setAttribute, bm as TransitionGroup, bn as useToast, bo as useToastStore, bp as useExecutionStore, bq as useTitle, br as withModifiers, bs as script$x, bt as script$y, bu as resolve, bv as script$z, bw as script$A, bx as isPrintableCharacter, by as guardReactiveProps, bz as useMenuItemStore, bA as script$D, bB as nestedPosition, bC as useQueueSettingsStore, bD as script$E, bE as useQueuePendingTaskCountStore, bF as useLocalStorage, bG as useDraggable, bH as watchDebounced, bI as inject, bJ as useElementBounding, bK as lodashExports, bL as useEventBus, bM as provide, bN as script$F, bO as LGraphEventMode, bP as useQueueStore, bQ as showTemplateWorkflowsDialog, bR as showSettingsDialog, bS as i18n, bT as useModelStore } from "./index-DIU5yZe9.js";
import { s as script$B, a as script$C } from "./index-d698Brhb.js";
import { u as useServerConfigStore } from "./serverConfigStore-DYv7_Nld.js";
const useTitleEditorStore = defineStore("titleEditor", () => {
  const titleEditorTarget = shallowRef(null);
  return {
    titleEditorTarget
  };
});
const useCanvasStore = defineStore("canvas", () => {
  const canvas = shallowRef(null);
  return {
    canvas
  };
});
const _sfc_main$s = /* @__PURE__ */ defineComponent({
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
const TitleEditor = /* @__PURE__ */ _export_sfc(_sfc_main$s, [["__scopeId", "data-v-8a100d5a"]]);
var theme$8 = /* @__PURE__ */ __name(function theme(_ref) {
  var dt = _ref.dt;
  return "\n.p-overlaybadge {\n    position: relative;\n}\n\n.p-overlaybadge .p-badge {\n    position: absolute;\n    top: 0;\n    right: 0;\n    transform: translate(50%, -50%);\n    transform-origin: 100% 0;\n    margin: 0;\n    outline-width: ".concat(dt("overlaybadge.outline.width"), ";\n    outline-style: solid;\n    outline-color: ").concat(dt("overlaybadge.outline.color"), ";\n}\n");
}, "theme");
var classes$b = {
  root: "p-overlaybadge"
};
var OverlayBadgeStyle = BaseStyle.extend({
  name: "overlaybadge",
  theme: theme$8,
  classes: classes$b
});
var script$1$b = {
  name: "OverlayBadge",
  "extends": script$e,
  style: OverlayBadgeStyle,
  provide: /* @__PURE__ */ __name(function provide2() {
    return {
      $pcOverlayBadge: this,
      $parentInstance: this
    };
  }, "provide")
};
var script$d = {
  name: "OverlayBadge",
  "extends": script$1$b,
  inheritAttrs: false,
  components: {
    Badge: script$e
  }
};
function render$j(_ctx, _cache, $props, $setup, $data, $options) {
  var _component_Badge = resolveComponent("Badge");
  return openBlock(), createElementBlock("div", mergeProps({
    "class": _ctx.cx("root")
  }, _ctx.ptmi("root")), [renderSlot(_ctx.$slots, "default"), createVNode(_component_Badge, mergeProps(_ctx.$props, {
    pt: _ctx.ptm("pcBadge")
  }), null, 16, ["pt"])], 16);
}
__name(render$j, "render$j");
script$d.render = render$j;
const _sfc_main$r = /* @__PURE__ */ defineComponent({
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
      return withDirectives((openBlock(), createBlock(unref(script$f), {
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
          shouldShowBadge.value ? (openBlock(), createBlock(unref(script$d), {
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
const SidebarIcon = /* @__PURE__ */ _export_sfc(_sfc_main$r, [["__scopeId", "data-v-caa3ee9c"]]);
const _sfc_main$q = /* @__PURE__ */ defineComponent({
  __name: "SidebarThemeToggleIcon",
  setup(__props) {
    const settingStore = useSettingStore();
    const currentTheme = computed(() => settingStore.get("Comfy.ColorPalette"));
    const icon = computed(
      () => currentTheme.value !== "light" ? "pi pi-moon" : "pi pi-sun"
    );
    const commandStore = useCommandStore();
    const toggleTheme = /* @__PURE__ */ __name(() => {
      commandStore.execute("Comfy.ToggleTheme");
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
const _sfc_main$p = /* @__PURE__ */ defineComponent({
  __name: "SidebarSettingsToggleIcon",
  setup(__props) {
    const dialogStore = useDialogStore();
    const showSetting = /* @__PURE__ */ __name(() => {
      dialogStore.showDialog({
        key: "global-settings",
        headerComponent: SettingDialogHeader,
        component: SettingDialogContent
      });
    }, "showSetting");
    return (_ctx, _cache) => {
      return openBlock(), createBlock(SidebarIcon, {
        icon: "pi pi-cog",
        class: "comfy-settings-btn",
        onClick: showSetting,
        tooltip: _ctx.$t("g.settings")
      }, null, 8, ["tooltip"]);
    };
  }
});
const _sfc_main$o = /* @__PURE__ */ defineComponent({
  __name: "SidebarLogoutIcon",
  setup(__props) {
    const { t } = useI18n();
    const userStore = useUserStore();
    const tooltip = computed(
      () => `${t("sideToolbar.logout")} (${userStore.currentUser?.username})`
    );
    const logout = /* @__PURE__ */ __name(() => {
      userStore.logout();
      window.location.reload();
    }, "logout");
    return (_ctx, _cache) => {
      return openBlock(), createBlock(SidebarIcon, {
        icon: "pi pi-sign-out",
        tooltip: tooltip.value,
        onClick: logout
      }, null, 8, ["tooltip"]);
    };
  }
});
const _sfc_main$n = /* @__PURE__ */ defineComponent({
  __name: "ExtensionSlot",
  props: {
    extension: {}
  },
  setup(__props) {
    const props = __props;
    const mountCustomExtension = /* @__PURE__ */ __name((extension, el) => {
      extension.render(el);
    }, "mountCustomExtension");
    onBeforeUnmount(() => {
      if (props.extension.type === "custom" && props.extension.destroy) {
        props.extension.destroy();
      }
    });
    return (_ctx, _cache) => {
      return _ctx.extension.type === "vue" ? (openBlock(), createBlock(resolveDynamicComponent(_ctx.extension.component), { key: 0 })) : (openBlock(), createElementBlock("div", {
        key: 1,
        ref: /* @__PURE__ */ __name((el) => {
          if (el)
            mountCustomExtension(
              props.extension,
              el
            );
        }, "ref")
      }, null, 512));
    };
  }
});
const _withScopeId$5 = /* @__PURE__ */ __name((n) => (pushScopeId("data-v-7851c166"), n = n(), popScopeId(), n), "_withScopeId$5");
const _hoisted_1$n = { class: "side-tool-bar-end" };
const _hoisted_2$k = {
  key: 0,
  class: "sidebar-content-container h-full overflow-y-auto overflow-x-hidden"
};
const _sfc_main$m = /* @__PURE__ */ defineComponent({
  __name: "SideToolbar",
  setup(__props) {
    const workspaceStore = useWorkspaceStore();
    const settingStore = useSettingStore();
    const userStore = useUserStore();
    const teleportTarget = computed(
      () => settingStore.get("Comfy.Sidebar.Location") === "left" ? ".comfyui-body-left" : ".comfyui-body-right"
    );
    const isSmall = computed(
      () => settingStore.get("Comfy.Sidebar.Size") === "small"
    );
    const tabs = computed(() => workspaceStore.getSidebarTabs());
    const selectedTab = computed(() => workspaceStore.sidebarTab.activeSidebarTab);
    const onTabClick = /* @__PURE__ */ __name((item3) => {
      workspaceStore.sidebarTab.toggleSidebarTab(item3.id);
    }, "onTabClick");
    const keybindingStore = useKeybindingStore();
    const getTabTooltipSuffix = /* @__PURE__ */ __name((tab) => {
      const keybinding = keybindingStore.getKeybindingByCommandId(
        `Workspace.ToggleSidebarTab.${tab.id}`
      );
      return keybinding ? ` (${keybinding.combo.toString()})` : "";
    }, "getTabTooltipSuffix");
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
                tooltip: tab.tooltip + getTabTooltipSuffix(tab),
                selected: tab.id === selectedTab.value?.id,
                class: normalizeClass(tab.id + "-tab-button"),
                onClick: /* @__PURE__ */ __name(($event) => onTabClick(tab), "onClick")
              }, null, 8, ["icon", "iconBadge", "tooltip", "selected", "class", "onClick"]);
            }), 128)),
            createBaseVNode("div", _hoisted_1$n, [
              unref(userStore).isMultiUserServer ? (openBlock(), createBlock(_sfc_main$o, { key: 0 })) : createCommentVNode("", true),
              createVNode(_sfc_main$q),
              createVNode(_sfc_main$p)
            ])
          ], 2)
        ], 8, ["to"])),
        selectedTab.value ? (openBlock(), createElementBlock("div", _hoisted_2$k, [
          createVNode(_sfc_main$n, { extension: selectedTab.value }, null, 8, ["extension"])
        ])) : createCommentVNode("", true)
      ], 64);
    };
  }
});
const SideToolbar = /* @__PURE__ */ _export_sfc(_sfc_main$m, [["__scopeId", "data-v-7851c166"]]);
var classes$a = {
  root: "p-tablist",
  content: /* @__PURE__ */ __name(function content(_ref) {
    var instance = _ref.instance;
    return ["p-tablist-content", {
      "p-tablist-viewport": instance.$pcTabs.scrollable
    }];
  }, "content"),
  tabList: "p-tablist-tab-list",
  activeBar: "p-tablist-active-bar",
  prevButton: "p-tablist-prev-button p-tablist-nav-button",
  nextButton: "p-tablist-next-button p-tablist-nav-button"
};
var TabListStyle = BaseStyle.extend({
  name: "tablist",
  classes: classes$a
});
var script$1$a = {
  name: "BaseTabList",
  "extends": script$g,
  props: {},
  style: TabListStyle,
  provide: /* @__PURE__ */ __name(function provide3() {
    return {
      $pcTabList: this,
      $parentInstance: this
    };
  }, "provide")
};
var script$c = {
  name: "TabList",
  "extends": script$1$a,
  inheritAttrs: false,
  inject: ["$pcTabs"],
  data: /* @__PURE__ */ __name(function data() {
    return {
      isPrevButtonEnabled: false,
      isNextButtonEnabled: true
    };
  }, "data"),
  resizeObserver: void 0,
  watch: {
    showNavigators: /* @__PURE__ */ __name(function showNavigators(newValue) {
      newValue ? this.bindResizeObserver() : this.unbindResizeObserver();
    }, "showNavigators"),
    activeValue: {
      flush: "post",
      handler: /* @__PURE__ */ __name(function handler() {
        this.updateInkBar();
      }, "handler")
    }
  },
  mounted: /* @__PURE__ */ __name(function mounted() {
    var _this = this;
    this.$nextTick(function() {
      _this.updateInkBar();
    });
    if (this.showNavigators) {
      this.updateButtonState();
      this.bindResizeObserver();
    }
  }, "mounted"),
  updated: /* @__PURE__ */ __name(function updated() {
    this.showNavigators && this.updateButtonState();
  }, "updated"),
  beforeUnmount: /* @__PURE__ */ __name(function beforeUnmount() {
    this.unbindResizeObserver();
  }, "beforeUnmount"),
  methods: {
    onScroll: /* @__PURE__ */ __name(function onScroll(event) {
      this.showNavigators && this.updateButtonState();
      event.preventDefault();
    }, "onScroll"),
    onPrevButtonClick: /* @__PURE__ */ __name(function onPrevButtonClick() {
      var content2 = this.$refs.content;
      var width = getWidth(content2);
      var pos = content2.scrollLeft - width;
      content2.scrollLeft = pos <= 0 ? 0 : pos;
    }, "onPrevButtonClick"),
    onNextButtonClick: /* @__PURE__ */ __name(function onNextButtonClick() {
      var content2 = this.$refs.content;
      var width = getWidth(content2) - this.getVisibleButtonWidths();
      var pos = content2.scrollLeft + width;
      var lastPos = content2.scrollWidth - width;
      content2.scrollLeft = pos >= lastPos ? lastPos : pos;
    }, "onNextButtonClick"),
    bindResizeObserver: /* @__PURE__ */ __name(function bindResizeObserver() {
      var _this2 = this;
      this.resizeObserver = new ResizeObserver(function() {
        return _this2.updateButtonState();
      });
      this.resizeObserver.observe(this.$refs.list);
    }, "bindResizeObserver"),
    unbindResizeObserver: /* @__PURE__ */ __name(function unbindResizeObserver() {
      var _this$resizeObserver;
      (_this$resizeObserver = this.resizeObserver) === null || _this$resizeObserver === void 0 || _this$resizeObserver.unobserve(this.$refs.list);
      this.resizeObserver = void 0;
    }, "unbindResizeObserver"),
    updateInkBar: /* @__PURE__ */ __name(function updateInkBar() {
      var _this$$refs = this.$refs, content2 = _this$$refs.content, inkbar = _this$$refs.inkbar, tabs = _this$$refs.tabs;
      var activeTab = findSingle(content2, '[data-pc-name="tab"][data-p-active="true"]');
      if (this.$pcTabs.isVertical()) {
        inkbar.style.height = getOuterHeight(activeTab) + "px";
        inkbar.style.top = getOffset(activeTab).top - getOffset(tabs).top + "px";
      } else {
        inkbar.style.width = getOuterWidth(activeTab) + "px";
        inkbar.style.left = getOffset(activeTab).left - getOffset(tabs).left + "px";
      }
    }, "updateInkBar"),
    updateButtonState: /* @__PURE__ */ __name(function updateButtonState() {
      var _this$$refs2 = this.$refs, list = _this$$refs2.list, content2 = _this$$refs2.content;
      var scrollLeft = content2.scrollLeft, scrollTop = content2.scrollTop, scrollWidth = content2.scrollWidth, scrollHeight = content2.scrollHeight, offsetWidth = content2.offsetWidth, offsetHeight = content2.offsetHeight;
      var _ref = [getWidth(content2), getHeight(content2)], width = _ref[0], height = _ref[1];
      if (this.$pcTabs.isVertical()) {
        this.isPrevButtonEnabled = scrollTop !== 0;
        this.isNextButtonEnabled = list.offsetHeight >= offsetHeight && parseInt(scrollTop) !== scrollHeight - height;
      } else {
        this.isPrevButtonEnabled = scrollLeft !== 0;
        this.isNextButtonEnabled = list.offsetWidth >= offsetWidth && parseInt(scrollLeft) !== scrollWidth - width;
      }
    }, "updateButtonState"),
    getVisibleButtonWidths: /* @__PURE__ */ __name(function getVisibleButtonWidths() {
      var _this$$refs3 = this.$refs, prevBtn = _this$$refs3.prevBtn, nextBtn = _this$$refs3.nextBtn;
      return [prevBtn, nextBtn].reduce(function(acc, el) {
        return el ? acc + getWidth(el) : acc;
      }, 0);
    }, "getVisibleButtonWidths")
  },
  computed: {
    templates: /* @__PURE__ */ __name(function templates() {
      return this.$pcTabs.$slots;
    }, "templates"),
    activeValue: /* @__PURE__ */ __name(function activeValue() {
      return this.$pcTabs.d_value;
    }, "activeValue"),
    showNavigators: /* @__PURE__ */ __name(function showNavigators2() {
      return this.$pcTabs.scrollable && this.$pcTabs.showNavigators;
    }, "showNavigators"),
    prevButtonAriaLabel: /* @__PURE__ */ __name(function prevButtonAriaLabel() {
      return this.$primevue.config.locale.aria ? this.$primevue.config.locale.aria.previous : void 0;
    }, "prevButtonAriaLabel"),
    nextButtonAriaLabel: /* @__PURE__ */ __name(function nextButtonAriaLabel() {
      return this.$primevue.config.locale.aria ? this.$primevue.config.locale.aria.next : void 0;
    }, "nextButtonAriaLabel")
  },
  components: {
    ChevronLeftIcon: script$h,
    ChevronRightIcon: script$i
  },
  directives: {
    ripple: Ripple
  }
};
var _hoisted_1$m = ["aria-label", "tabindex"];
var _hoisted_2$j = ["aria-orientation"];
var _hoisted_3$g = ["aria-label", "tabindex"];
function render$i(_ctx, _cache, $props, $setup, $data, $options) {
  var _directive_ripple = resolveDirective("ripple");
  return openBlock(), createElementBlock("div", mergeProps({
    ref: "list",
    "class": _ctx.cx("root")
  }, _ctx.ptmi("root")), [$options.showNavigators && $data.isPrevButtonEnabled ? withDirectives((openBlock(), createElementBlock("button", mergeProps({
    key: 0,
    ref: "prevButton",
    "class": _ctx.cx("prevButton"),
    "aria-label": $options.prevButtonAriaLabel,
    tabindex: $options.$pcTabs.tabindex,
    onClick: _cache[0] || (_cache[0] = function() {
      return $options.onPrevButtonClick && $options.onPrevButtonClick.apply($options, arguments);
    })
  }, _ctx.ptm("prevButton"), {
    "data-pc-group-section": "navigator"
  }), [(openBlock(), createBlock(resolveDynamicComponent($options.templates.previcon || "ChevronLeftIcon"), mergeProps({
    "aria-hidden": "true"
  }, _ctx.ptm("prevIcon")), null, 16))], 16, _hoisted_1$m)), [[_directive_ripple]]) : createCommentVNode("", true), createBaseVNode("div", mergeProps({
    ref: "content",
    "class": _ctx.cx("content"),
    onScroll: _cache[1] || (_cache[1] = function() {
      return $options.onScroll && $options.onScroll.apply($options, arguments);
    })
  }, _ctx.ptm("content")), [createBaseVNode("div", mergeProps({
    ref: "tabs",
    "class": _ctx.cx("tabList"),
    role: "tablist",
    "aria-orientation": $options.$pcTabs.orientation || "horizontal"
  }, _ctx.ptm("tabList")), [renderSlot(_ctx.$slots, "default"), createBaseVNode("span", mergeProps({
    ref: "inkbar",
    "class": _ctx.cx("activeBar"),
    role: "presentation",
    "aria-hidden": "true"
  }, _ctx.ptm("activeBar")), null, 16)], 16, _hoisted_2$j)], 16), $options.showNavigators && $data.isNextButtonEnabled ? withDirectives((openBlock(), createElementBlock("button", mergeProps({
    key: 1,
    ref: "nextButton",
    "class": _ctx.cx("nextButton"),
    "aria-label": $options.nextButtonAriaLabel,
    tabindex: $options.$pcTabs.tabindex,
    onClick: _cache[2] || (_cache[2] = function() {
      return $options.onNextButtonClick && $options.onNextButtonClick.apply($options, arguments);
    })
  }, _ctx.ptm("nextButton"), {
    "data-pc-group-section": "navigator"
  }), [(openBlock(), createBlock(resolveDynamicComponent($options.templates.nexticon || "ChevronRightIcon"), mergeProps({
    "aria-hidden": "true"
  }, _ctx.ptm("nextIcon")), null, 16))], 16, _hoisted_3$g)), [[_directive_ripple]]) : createCommentVNode("", true)], 16);
}
__name(render$i, "render$i");
script$c.render = render$i;
var classes$9 = {
  root: /* @__PURE__ */ __name(function root(_ref) {
    var instance = _ref.instance, props = _ref.props;
    return ["p-tab", {
      "p-tab-active": instance.active,
      "p-disabled": props.disabled
    }];
  }, "root")
};
var TabStyle = BaseStyle.extend({
  name: "tab",
  classes: classes$9
});
var script$1$9 = {
  name: "BaseTab",
  "extends": script$g,
  props: {
    value: {
      type: [String, Number],
      "default": void 0
    },
    disabled: {
      type: Boolean,
      "default": false
    },
    as: {
      type: [String, Object],
      "default": "BUTTON"
    },
    asChild: {
      type: Boolean,
      "default": false
    }
  },
  style: TabStyle,
  provide: /* @__PURE__ */ __name(function provide4() {
    return {
      $pcTab: this,
      $parentInstance: this
    };
  }, "provide")
};
var script$b = {
  name: "Tab",
  "extends": script$1$9,
  inheritAttrs: false,
  inject: ["$pcTabs", "$pcTabList"],
  methods: {
    onFocus: /* @__PURE__ */ __name(function onFocus() {
      this.$pcTabs.selectOnFocus && this.changeActiveValue();
    }, "onFocus"),
    onClick: /* @__PURE__ */ __name(function onClick() {
      this.changeActiveValue();
    }, "onClick"),
    onKeydown: /* @__PURE__ */ __name(function onKeydown(event) {
      switch (event.code) {
        case "ArrowRight":
          this.onArrowRightKey(event);
          break;
        case "ArrowLeft":
          this.onArrowLeftKey(event);
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
        case "Space":
          this.onEnterKey(event);
          break;
      }
    }, "onKeydown"),
    onArrowRightKey: /* @__PURE__ */ __name(function onArrowRightKey(event) {
      var nextTab = this.findNextTab(event.currentTarget);
      nextTab ? this.changeFocusedTab(event, nextTab) : this.onHomeKey(event);
      event.preventDefault();
    }, "onArrowRightKey"),
    onArrowLeftKey: /* @__PURE__ */ __name(function onArrowLeftKey(event) {
      var prevTab = this.findPrevTab(event.currentTarget);
      prevTab ? this.changeFocusedTab(event, prevTab) : this.onEndKey(event);
      event.preventDefault();
    }, "onArrowLeftKey"),
    onHomeKey: /* @__PURE__ */ __name(function onHomeKey(event) {
      var firstTab = this.findFirstTab();
      this.changeFocusedTab(event, firstTab);
      event.preventDefault();
    }, "onHomeKey"),
    onEndKey: /* @__PURE__ */ __name(function onEndKey(event) {
      var lastTab = this.findLastTab();
      this.changeFocusedTab(event, lastTab);
      event.preventDefault();
    }, "onEndKey"),
    onPageDownKey: /* @__PURE__ */ __name(function onPageDownKey(event) {
      this.scrollInView(this.findLastTab());
      event.preventDefault();
    }, "onPageDownKey"),
    onPageUpKey: /* @__PURE__ */ __name(function onPageUpKey(event) {
      this.scrollInView(this.findFirstTab());
      event.preventDefault();
    }, "onPageUpKey"),
    onEnterKey: /* @__PURE__ */ __name(function onEnterKey(event) {
      this.changeActiveValue();
      event.preventDefault();
    }, "onEnterKey"),
    findNextTab: /* @__PURE__ */ __name(function findNextTab(tabElement) {
      var selfCheck = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : false;
      var element = selfCheck ? tabElement : tabElement.nextElementSibling;
      return element ? getAttribute(element, "data-p-disabled") || getAttribute(element, "data-pc-section") === "inkbar" ? this.findNextTab(element) : findSingle(element, '[data-pc-name="tab"]') : null;
    }, "findNextTab"),
    findPrevTab: /* @__PURE__ */ __name(function findPrevTab(tabElement) {
      var selfCheck = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : false;
      var element = selfCheck ? tabElement : tabElement.previousElementSibling;
      return element ? getAttribute(element, "data-p-disabled") || getAttribute(element, "data-pc-section") === "inkbar" ? this.findPrevTab(element) : findSingle(element, '[data-pc-name="tab"]') : null;
    }, "findPrevTab"),
    findFirstTab: /* @__PURE__ */ __name(function findFirstTab() {
      return this.findNextTab(this.$pcTabList.$refs.content.firstElementChild, true);
    }, "findFirstTab"),
    findLastTab: /* @__PURE__ */ __name(function findLastTab() {
      return this.findPrevTab(this.$pcTabList.$refs.content.lastElementChild, true);
    }, "findLastTab"),
    changeActiveValue: /* @__PURE__ */ __name(function changeActiveValue() {
      this.$pcTabs.updateValue(this.value);
    }, "changeActiveValue"),
    changeFocusedTab: /* @__PURE__ */ __name(function changeFocusedTab(event, element) {
      focus(element);
      this.scrollInView(element);
    }, "changeFocusedTab"),
    scrollInView: /* @__PURE__ */ __name(function scrollInView(element) {
      var _element$scrollIntoVi;
      element === null || element === void 0 || (_element$scrollIntoVi = element.scrollIntoView) === null || _element$scrollIntoVi === void 0 || _element$scrollIntoVi.call(element, {
        block: "nearest"
      });
    }, "scrollInView")
  },
  computed: {
    active: /* @__PURE__ */ __name(function active() {
      var _this$$pcTabs;
      return equals((_this$$pcTabs = this.$pcTabs) === null || _this$$pcTabs === void 0 ? void 0 : _this$$pcTabs.d_value, this.value);
    }, "active"),
    id: /* @__PURE__ */ __name(function id() {
      var _this$$pcTabs2;
      return "".concat((_this$$pcTabs2 = this.$pcTabs) === null || _this$$pcTabs2 === void 0 ? void 0 : _this$$pcTabs2.id, "_tab_").concat(this.value);
    }, "id"),
    ariaControls: /* @__PURE__ */ __name(function ariaControls() {
      var _this$$pcTabs3;
      return "".concat((_this$$pcTabs3 = this.$pcTabs) === null || _this$$pcTabs3 === void 0 ? void 0 : _this$$pcTabs3.id, "_tabpanel_").concat(this.value);
    }, "ariaControls"),
    attrs: /* @__PURE__ */ __name(function attrs() {
      return mergeProps(this.asAttrs, this.a11yAttrs, this.ptmi("root", this.ptParams));
    }, "attrs"),
    asAttrs: /* @__PURE__ */ __name(function asAttrs() {
      return this.as === "BUTTON" ? {
        type: "button",
        disabled: this.disabled
      } : void 0;
    }, "asAttrs"),
    a11yAttrs: /* @__PURE__ */ __name(function a11yAttrs() {
      return {
        id: this.id,
        tabindex: this.active ? this.$pcTabs.tabindex : -1,
        role: "tab",
        "aria-selected": this.active,
        "aria-controls": this.ariaControls,
        "data-pc-name": "tab",
        "data-p-disabled": this.disabled,
        "data-p-active": this.active,
        onFocus: this.onFocus,
        onKeydown: this.onKeydown
      };
    }, "a11yAttrs"),
    ptParams: /* @__PURE__ */ __name(function ptParams() {
      return {
        context: {
          active: this.active
        }
      };
    }, "ptParams")
  },
  directives: {
    ripple: Ripple
  }
};
function render$h(_ctx, _cache, $props, $setup, $data, $options) {
  var _directive_ripple = resolveDirective("ripple");
  return !_ctx.asChild ? withDirectives((openBlock(), createBlock(resolveDynamicComponent(_ctx.as), mergeProps({
    key: 0,
    "class": _ctx.cx("root"),
    onClick: $options.onClick
  }, $options.attrs), {
    "default": withCtx(function() {
      return [renderSlot(_ctx.$slots, "default")];
    }),
    _: 3
  }, 16, ["class", "onClick"])), [[_directive_ripple]]) : renderSlot(_ctx.$slots, "default", {
    key: 1,
    "class": normalizeClass(_ctx.cx("root")),
    active: $options.active,
    a11yAttrs: $options.a11yAttrs,
    onClick: $options.onClick
  });
}
__name(render$h, "render$h");
script$b.render = render$h;
const _hoisted_1$l = { class: "flex flex-col h-full" };
const _hoisted_2$i = { class: "w-full flex justify-between" };
const _hoisted_3$f = { class: "tabs-container" };
const _hoisted_4$6 = { class: "font-bold" };
const _hoisted_5$4 = { class: "flex-grow h-0" };
const _sfc_main$l = /* @__PURE__ */ defineComponent({
  __name: "BottomPanel",
  setup(__props) {
    const bottomPanelStore = useBottomPanelStore();
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", _hoisted_1$l, [
        createVNode(unref(script$j), {
          value: unref(bottomPanelStore).activeBottomPanelTabId,
          "onUpdate:value": _cache[1] || (_cache[1] = ($event) => unref(bottomPanelStore).activeBottomPanelTabId = $event)
        }, {
          default: withCtx(() => [
            createVNode(unref(script$c), { "pt:tabList": "border-none" }, {
              default: withCtx(() => [
                createBaseVNode("div", _hoisted_2$i, [
                  createBaseVNode("div", _hoisted_3$f, [
                    (openBlock(true), createElementBlock(Fragment, null, renderList(unref(bottomPanelStore).bottomPanelTabs, (tab) => {
                      return openBlock(), createBlock(unref(script$b), {
                        key: tab.id,
                        value: tab.id,
                        class: "p-3 border-none"
                      }, {
                        default: withCtx(() => [
                          createBaseVNode("span", _hoisted_4$6, toDisplayString(tab.title.toUpperCase()), 1)
                        ]),
                        _: 2
                      }, 1032, ["value"]);
                    }), 128))
                  ]),
                  createVNode(unref(script$f), {
                    class: "justify-self-end",
                    icon: "pi pi-times",
                    severity: "secondary",
                    size: "small",
                    text: "",
                    onClick: _cache[0] || (_cache[0] = ($event) => unref(bottomPanelStore).bottomPanelVisible = false)
                  })
                ])
              ]),
              _: 1
            })
          ]),
          _: 1
        }, 8, ["value"]),
        createBaseVNode("div", _hoisted_5$4, [
          unref(bottomPanelStore).bottomPanelVisible && unref(bottomPanelStore).activeBottomPanelTab ? (openBlock(), createBlock(_sfc_main$n, {
            key: 0,
            extension: unref(bottomPanelStore).activeBottomPanelTab
          }, null, 8, ["extension"])) : createCommentVNode("", true)
        ])
      ]);
    };
  }
});
var theme$7 = /* @__PURE__ */ __name(function theme2(_ref) {
  var dt = _ref.dt;
  return "\n.p-splitter {\n    display: flex;\n    flex-wrap: nowrap;\n    border: 1px solid ".concat(dt("splitter.border.color"), ";\n    background: ").concat(dt("splitter.background"), ";\n    border-radius: ").concat(dt("border.radius.md"), ";\n    color: ").concat(dt("splitter.color"), ";\n}\n\n.p-splitter-vertical {\n    flex-direction: column;\n}\n\n.p-splitter-gutter {\n    flex-grow: 0;\n    flex-shrink: 0;\n    display: flex;\n    align-items: center;\n    justify-content: center;\n    z-index: 1;\n    background: ").concat(dt("splitter.gutter.background"), ";\n}\n\n.p-splitter-gutter-handle {\n    border-radius: ").concat(dt("splitter.handle.border.radius"), ";\n    background: ").concat(dt("splitter.handle.background"), ";\n    transition: outline-color ").concat(dt("splitter.transition.duration"), ", box-shadow ").concat(dt("splitter.transition.duration"), ";\n    outline-color: transparent;\n}\n\n.p-splitter-gutter-handle:focus-visible {\n    box-shadow: ").concat(dt("splitter.handle.focus.ring.shadow"), ";\n    outline: ").concat(dt("splitter.handle.focus.ring.width"), " ").concat(dt("splitter.handle.focus.ring.style"), " ").concat(dt("splitter.handle.focus.ring.color"), ";\n    outline-offset: ").concat(dt("splitter.handle.focus.ring.offset"), ";\n}\n\n.p-splitter-horizontal.p-splitter-resizing {\n    cursor: col-resize;\n    user-select: none;\n}\n\n.p-splitter-vertical.p-splitter-resizing {\n    cursor: row-resize;\n    user-select: none;\n}\n\n.p-splitter-horizontal > .p-splitter-gutter > .p-splitter-gutter-handle {\n    height: ").concat(dt("splitter.handle.size"), ";\n    width: 100%;\n}\n\n.p-splitter-vertical > .p-splitter-gutter > .p-splitter-gutter-handle {\n    width: ").concat(dt("splitter.handle.size"), ";\n    height: 100%;\n}\n\n.p-splitter-horizontal > .p-splitter-gutter {\n    cursor: col-resize;\n}\n\n.p-splitter-vertical > .p-splitter-gutter {\n    cursor: row-resize;\n}\n\n.p-splitterpanel {\n    flex-grow: 1;\n    overflow: hidden;\n}\n\n.p-splitterpanel-nested {\n    display: flex;\n}\n\n.p-splitterpanel .p-splitter {\n    flex-grow: 1;\n    border: 0 none;\n}\n");
}, "theme");
var classes$8 = {
  root: /* @__PURE__ */ __name(function root2(_ref2) {
    var props = _ref2.props;
    return ["p-splitter p-component", "p-splitter-" + props.layout];
  }, "root"),
  gutter: "p-splitter-gutter",
  gutterHandle: "p-splitter-gutter-handle"
};
var inlineStyles$4 = {
  root: /* @__PURE__ */ __name(function root3(_ref3) {
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
  theme: theme$7,
  classes: classes$8,
  inlineStyles: inlineStyles$4
});
var script$1$8 = {
  name: "BaseSplitter",
  "extends": script$g,
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
  provide: /* @__PURE__ */ __name(function provide5() {
    return {
      $pcSplitter: this,
      $parentInstance: this
    };
  }, "provide")
};
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
var script$a = {
  name: "Splitter",
  "extends": script$1$8,
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
  data: /* @__PURE__ */ __name(function data2() {
    return {
      prevSize: null
    };
  }, "data"),
  mounted: /* @__PURE__ */ __name(function mounted2() {
    var _this = this;
    if (this.panels && this.panels.length) {
      var initialized = false;
      if (this.isStateful()) {
        initialized = this.restoreState();
      }
      if (!initialized) {
        var children = _toConsumableArray$2(this.$el.children).filter(function(child) {
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
  beforeUnmount: /* @__PURE__ */ __name(function beforeUnmount2() {
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
        var children = _toConsumableArray$2(this.$el.children).filter(function(child) {
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
var _hoisted_1$k = ["onMousedown", "onTouchstart", "onTouchmove", "onTouchend"];
var _hoisted_2$h = ["aria-orientation", "aria-valuenow", "onKeydown"];
function render$g(_ctx, _cache, $props, $setup, $data, $options) {
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
      onKeydown: /* @__PURE__ */ __name(function onKeydown2($event) {
        return $options.onGutterKeyDown($event, i);
      }, "onKeydown"),
      ref_for: true
    }, _ctx.ptm("gutterHandle")), null, 16, _hoisted_2$h)], 16, _hoisted_1$k)) : createCommentVNode("", true)], 64);
  }), 128))], 16);
}
__name(render$g, "render$g");
script$a.render = render$g;
var classes$7 = {
  root: /* @__PURE__ */ __name(function root4(_ref) {
    var instance = _ref.instance;
    return ["p-splitterpanel", {
      "p-splitterpanel-nested": instance.isNested
    }];
  }, "root")
};
var SplitterPanelStyle = BaseStyle.extend({
  name: "splitterpanel",
  classes: classes$7
});
var script$1$7 = {
  name: "BaseSplitterPanel",
  "extends": script$g,
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
  provide: /* @__PURE__ */ __name(function provide6() {
    return {
      $pcSplitterPanel: this,
      $parentInstance: this
    };
  }, "provide")
};
var script$9 = {
  name: "SplitterPanel",
  "extends": script$1$7,
  inheritAttrs: false,
  data: /* @__PURE__ */ __name(function data3() {
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
function render$f(_ctx, _cache, $props, $setup, $data, $options) {
  return openBlock(), createElementBlock("div", mergeProps({
    ref: "container",
    "class": _ctx.cx("root")
  }, _ctx.ptmi("root", $options.getPTOptions)), [renderSlot(_ctx.$slots, "default")], 16);
}
__name(render$f, "render$f");
script$9.render = render$f;
const _sfc_main$k = /* @__PURE__ */ defineComponent({
  __name: "LiteGraphCanvasSplitterOverlay",
  setup(__props) {
    const settingStore = useSettingStore();
    const sidebarLocation = computed(
      () => settingStore.get("Comfy.Sidebar.Location")
    );
    const sidebarPanelVisible = computed(
      () => useSidebarTabStore().activeSidebarTab !== null
    );
    const bottomPanelVisible = computed(
      () => useBottomPanelStore().bottomPanelVisible
    );
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(script$a), {
        class: "splitter-overlay-root splitter-overlay",
        "pt:gutter": sidebarPanelVisible.value ? "" : "hidden",
        stateKey: "sidebar-splitter",
        stateStorage: "local"
      }, {
        default: withCtx(() => [
          sidebarLocation.value === "left" ? withDirectives((openBlock(), createBlock(unref(script$9), {
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
          createVNode(unref(script$9), { size: 100 }, {
            default: withCtx(() => [
              createVNode(unref(script$a), {
                class: "splitter-overlay max-w-full",
                layout: "vertical",
                "pt:gutter": bottomPanelVisible.value ? "" : "hidden",
                stateKey: "bottom-panel-splitter",
                stateStorage: "local"
              }, {
                default: withCtx(() => [
                  createVNode(unref(script$9), { class: "graph-canvas-panel relative" }, {
                    default: withCtx(() => [
                      renderSlot(_ctx.$slots, "graph-canvas-panel", {}, void 0, true)
                    ]),
                    _: 3
                  }),
                  withDirectives(createVNode(unref(script$9), { class: "bottom-panel" }, {
                    default: withCtx(() => [
                      renderSlot(_ctx.$slots, "bottom-panel", {}, void 0, true)
                    ]),
                    _: 3
                  }, 512), [
                    [vShow, bottomPanelVisible.value]
                  ])
                ]),
                _: 3
              }, 8, ["pt:gutter"])
            ]),
            _: 3
          }),
          sidebarLocation.value === "right" ? withDirectives((openBlock(), createBlock(unref(script$9), {
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
const LiteGraphCanvasSplitterOverlay = /* @__PURE__ */ _export_sfc(_sfc_main$k, [["__scopeId", "data-v-7c3279c1"]]);
var theme$6 = /* @__PURE__ */ __name(function theme3(_ref) {
  var dt = _ref.dt;
  return "\n.p-autocomplete {\n    display: inline-flex;\n}\n\n.p-autocomplete-loader {\n    position: absolute;\n    top: 50%;\n    margin-top: -0.5rem;\n    right: ".concat(dt("autocomplete.padding.x"), ";\n}\n\n.p-autocomplete:has(.p-autocomplete-dropdown) .p-autocomplete-loader {\n    right: calc(").concat(dt("autocomplete.dropdown.width"), " + ").concat(dt("autocomplete.padding.x"), ");\n}\n\n.p-autocomplete:has(.p-autocomplete-dropdown) .p-autocomplete-input {\n    flex: 1 1 auto;\n    width: 1%;\n}\n\n.p-autocomplete:has(.p-autocomplete-dropdown) .p-autocomplete-input,\n.p-autocomplete:has(.p-autocomplete-dropdown) .p-autocomplete-input-multiple {\n    border-top-right-radius: 0;\n    border-bottom-right-radius: 0;\n}\n\n.p-autocomplete-dropdown {\n    cursor: pointer;\n    display: inline-flex;\n    cursor: pointer;\n    user-select: none;\n    align-items: center;\n    justify-content: center;\n    overflow: hidden;\n    position: relative;\n    width: ").concat(dt("autocomplete.dropdown.width"), ";\n    border-top-right-radius: ").concat(dt("autocomplete.dropdown.border.radius"), ";\n    border-bottom-right-radius: ").concat(dt("autocomplete.dropdown.border.radius"), ";\n    background: ").concat(dt("autocomplete.dropdown.background"), ";\n    border: 1px solid ").concat(dt("autocomplete.dropdown.border.color"), ";\n    border-left: 0 none;\n    color: ").concat(dt("autocomplete.dropdown.color"), ";\n    transition: background ").concat(dt("autocomplete.transition.duration"), ", color ").concat(dt("autocomplete.transition.duration"), ", border-color ").concat(dt("autocomplete.transition.duration"), ", outline-color ").concat(dt("autocomplete.transition.duration"), ", box-shadow ").concat(dt("autocomplete.transition.duration"), ";\n    outline-color: transparent;\n}\n\n.p-autocomplete-dropdown:not(:disabled):hover {\n    background: ").concat(dt("autocomplete.dropdown.hover.background"), ";\n    border-color: ").concat(dt("autocomplete.dropdown.hover.border.color"), ";\n    color: ").concat(dt("autocomplete.dropdown.hover.color"), ";\n}\n\n.p-autocomplete-dropdown:not(:disabled):active {\n    background: ").concat(dt("autocomplete.dropdown.active.background"), ";\n    border-color: ").concat(dt("autocomplete.dropdown.active.border.color"), ";\n    color: ").concat(dt("autocomplete.dropdown.active.color"), ";\n}\n\n.p-autocomplete-dropdown:focus-visible {\n    box-shadow: ").concat(dt("autocomplete.dropdown.focus.ring.shadow"), ";\n    outline: ").concat(dt("autocomplete.dropdown.focus.ring.width"), " ").concat(dt("autocomplete.dropdown.focus.ring.style"), " ").concat(dt("autocomplete.dropdown.focus.ring.color"), ";\n    outline-offset: ").concat(dt("autocomplete.dropdown.focus.ring.offset"), ";\n}\n\n.p-autocomplete .p-autocomplete-overlay {\n    min-width: 100%;\n}\n\n.p-autocomplete-overlay {\n    position: absolute;\n    overflow: auto;\n    top: 0;\n    left: 0;\n    background: ").concat(dt("autocomplete.overlay.background"), ";\n    color: ").concat(dt("autocomplete.overlay.color"), ";\n    border: 1px solid ").concat(dt("autocomplete.overlay.border.color"), ";\n    border-radius: ").concat(dt("autocomplete.overlay.border.radius"), ";\n    box-shadow: ").concat(dt("autocomplete.overlay.shadow"), ";\n}\n\n.p-autocomplete-list {\n    margin: 0;\n    padding: 0;\n    list-style-type: none;\n    display: flex;\n    flex-direction: column;\n    gap: ").concat(dt("autocomplete.list.gap"), ";\n    padding: ").concat(dt("autocomplete.list.padding"), ";\n}\n\n.p-autocomplete-option {\n    cursor: pointer;\n    white-space: nowrap;\n    position: relative;\n    overflow: hidden;\n    display: flex;\n    align-items: center;\n    padding: ").concat(dt("autocomplete.option.padding"), ";\n    border: 0 none;\n    color: ").concat(dt("autocomplete.option.color"), ";\n    background: transparent;\n    transition: background ").concat(dt("autocomplete.transition.duration"), ", color ").concat(dt("autocomplete.transition.duration"), ", border-color ").concat(dt("autocomplete.transition.duration"), ";\n    border-radius: ").concat(dt("autocomplete.option.border.radius"), ";\n}\n\n.p-autocomplete-option:not(.p-autocomplete-option-selected):not(.p-disabled).p-focus {\n    background: ").concat(dt("autocomplete.option.focus.background"), ";\n    color: ").concat(dt("autocomplete.option.focus.color"), ";\n}\n\n.p-autocomplete-option-selected {\n    background: ").concat(dt("autocomplete.option.selected.background"), ";\n    color: ").concat(dt("autocomplete.option.selected.color"), ";\n}\n\n.p-autocomplete-option-selected.p-focus {\n    background: ").concat(dt("autocomplete.option.selected.focus.background"), ";\n    color: ").concat(dt("autocomplete.option.selected.focus.color"), ";\n}\n\n.p-autocomplete-option-group {\n    margin: 0;\n    padding: ").concat(dt("autocomplete.option.group.padding"), ";\n    color: ").concat(dt("autocomplete.option.group.color"), ";\n    background: ").concat(dt("autocomplete.option.group.background"), ";\n    font-weight: ").concat(dt("autocomplete.option.group.font.weight"), ";\n}\n\n.p-autocomplete-input-multiple {\n    margin: 0;\n    list-style-type: none;\n    cursor: text;\n    overflow: hidden;\n    display: flex;\n    align-items: center;\n    flex-wrap: wrap;\n    padding: calc(").concat(dt("autocomplete.padding.y"), " / 2) ").concat(dt("autocomplete.padding.x"), ";\n    gap: calc(").concat(dt("autocomplete.padding.y"), " / 2);\n    color: ").concat(dt("autocomplete.color"), ";\n    background: ").concat(dt("autocomplete.background"), ";\n    border: 1px solid ").concat(dt("autocomplete.border.color"), ";\n    border-radius: ").concat(dt("autocomplete.border.radius"), ";\n    width: 100%;\n    transition: background ").concat(dt("autocomplete.transition.duration"), ", color ").concat(dt("autocomplete.transition.duration"), ", border-color ").concat(dt("autocomplete.transition.duration"), ", outline-color ").concat(dt("autocomplete.transition.duration"), ", box-shadow ").concat(dt("autocomplete.transition.duration"), ";\n    outline-color: transparent;\n    box-shadow: ").concat(dt("autocomplete.shadow"), ";\n}\n\n.p-autocomplete:not(.p-disabled):hover .p-autocomplete-input-multiple {\n    border-color: ").concat(dt("autocomplete.hover.border.color"), ";\n}\n\n.p-autocomplete:not(.p-disabled).p-focus .p-autocomplete-input-multiple {\n    border-color: ").concat(dt("autocomplete.focus.border.color"), ";\n    box-shadow: ").concat(dt("autocomplete.focus.ring.shadow"), ";\n    outline: ").concat(dt("autocomplete.focus.ring.width"), " ").concat(dt("autocomplete.focus.ring.style"), " ").concat(dt("autocomplete.focus.ring.color"), ";\n    outline-offset: ").concat(dt("autocomplete.focus.ring.offset"), ";\n}\n\n.p-autocomplete.p-invalid .p-autocomplete-input-multiple {\n    border-color: ").concat(dt("autocomplete.invalid.border.color"), ";\n}\n\n.p-variant-filled.p-autocomplete-input-multiple {\n    background: ").concat(dt("autocomplete.filled.background"), ";\n}\n\n.p-autocomplete:not(.p-disabled).p-focus .p-variant-filled.p-autocomplete-input-multiple  {\n    background: ").concat(dt("autocomplete.filled.focus.background"), ";\n}\n\n.p-autocomplete.p-disabled .p-autocomplete-input-multiple {\n    opacity: 1;\n    background: ").concat(dt("autocomplete.disabled.background"), ";\n    color: ").concat(dt("autocomplete.disabled.color"), ";\n}\n\n.p-autocomplete-chip.p-chip {\n    padding-top: calc(").concat(dt("autocomplete.padding.y"), " / 2);\n    padding-bottom: calc(").concat(dt("autocomplete.padding.y"), " / 2);\n    border-radius: ").concat(dt("autocomplete.chip.border.radius"), ";\n}\n\n.p-autocomplete-input-multiple:has(.p-autocomplete-chip) {\n    padding-left: calc(").concat(dt("autocomplete.padding.y"), " / 2);\n    padding-right: calc(").concat(dt("autocomplete.padding.y"), " / 2);\n}\n\n.p-autocomplete-chip-item.p-focus .p-autocomplete-chip {\n    background: ").concat(dt("inputchips.chip.focus.background"), ";\n    color: ").concat(dt("inputchips.chip.focus.color"), ";\n}\n\n.p-autocomplete-input-chip {\n    flex: 1 1 auto;\n    display: inline-flex;\n    padding-top: calc(").concat(dt("autocomplete.padding.y"), " / 2);\n    padding-bottom: calc(").concat(dt("autocomplete.padding.y"), " / 2);\n}\n\n.p-autocomplete-input-chip input {\n    border: 0 none;\n    outline: 0 none;\n    background: transparent;\n    margin: 0;\n    padding: 0;\n    box-shadow: none;\n    border-radius: 0;\n    width: 100%;\n    font-family: inherit;\n    font-feature-settings: inherit;\n    font-size: 1rem;\n    color: inherit;\n}\n\n.p-autocomplete-input-chip input::placeholder {\n    color: ").concat(dt("autocomplete.placeholder.color"), ";\n}\n\n.p-autocomplete-empty-message {\n    padding: ").concat(dt("autocomplete.empty.message.padding"), ";\n}\n\n.p-autocomplete-fluid {\n    display: flex;\n}\n\n.p-autocomplete-fluid:has(.p-autocomplete-dropdown) .p-autocomplete-input {\n    width: 1%;\n}\n");
}, "theme");
var inlineStyles$3 = {
  root: {
    position: "relative"
  }
};
var classes$6 = {
  root: /* @__PURE__ */ __name(function root5(_ref2) {
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
  theme: theme$6,
  classes: classes$6,
  inlineStyles: inlineStyles$3
});
var script$1$6 = {
  name: "BaseAutoComplete",
  "extends": script$g,
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
  provide: /* @__PURE__ */ __name(function provide7() {
    return {
      $pcAutoComplete: this,
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
var script$8 = {
  name: "AutoComplete",
  "extends": script$1$6,
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
  data: /* @__PURE__ */ __name(function data4() {
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
  mounted: /* @__PURE__ */ __name(function mounted3() {
    this.id = this.id || UniqueComponentId();
    this.autoUpdateModel();
  }, "mounted"),
  updated: /* @__PURE__ */ __name(function updated2() {
    if (this.overlayVisible) {
      this.alignOverlay();
    }
  }, "updated"),
  beforeUnmount: /* @__PURE__ */ __name(function beforeUnmount3() {
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
    onFocus: /* @__PURE__ */ __name(function onFocus2(event) {
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
          this.updateModel(event, [].concat(_toConsumableArray$1(this.modelValue || []), [value]));
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
    onArrowLeftKey: /* @__PURE__ */ __name(function onArrowLeftKey2(event) {
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
    onArrowRightKey: /* @__PURE__ */ __name(function onArrowRightKey2(event) {
      this.focusedOptionIndex = -1;
      this.multiple && event.stopPropagation();
    }, "onArrowRightKey"),
    onHomeKey: /* @__PURE__ */ __name(function onHomeKey2(event) {
      var currentTarget = event.currentTarget;
      var len = currentTarget.value.length;
      currentTarget.setSelectionRange(0, event.shiftKey ? len : 0);
      this.focusedOptionIndex = -1;
      event.preventDefault();
    }, "onHomeKey"),
    onEndKey: /* @__PURE__ */ __name(function onEndKey2(event) {
      var currentTarget = event.currentTarget;
      var len = currentTarget.value.length;
      currentTarget.setSelectionRange(event.shiftKey ? 0 : len, len);
      this.focusedOptionIndex = -1;
      event.preventDefault();
    }, "onEndKey"),
    onPageUpKey: /* @__PURE__ */ __name(function onPageUpKey2(event) {
      this.scrollInView(0);
      event.preventDefault();
    }, "onPageUpKey"),
    onPageDownKey: /* @__PURE__ */ __name(function onPageDownKey2(event) {
      this.scrollInView(this.visibleOptions.length - 1);
      event.preventDefault();
    }, "onPageDownKey"),
    onEnterKey: /* @__PURE__ */ __name(function onEnterKey2(event) {
      if (!this.typeahead) {
        if (this.multiple) {
          this.updateModel(event, [].concat(_toConsumableArray$1(this.modelValue || []), [event.target.value]));
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
      var value = this.modelValue.filter(function(_2, i) {
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
    scrollInView: /* @__PURE__ */ __name(function scrollInView2() {
      var _this15 = this;
      var index = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : -1;
      this.$nextTick(function() {
        var id2 = index !== -1 ? "".concat(_this15.id, "_").concat(index) : _this15.focusedOptionId;
        var element = findSingle(_this15.list, 'li[id="'.concat(id2, '"]'));
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
        if (_typeof$1$1(this.modelValue) === "object") {
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
    InputText: script$k,
    VirtualScroller: script$l,
    Portal: script$m,
    ChevronDownIcon: script$n,
    SpinnerIcon: script$o,
    Chip: script$p
  },
  directives: {
    ripple: Ripple
  }
};
function _typeof$4(o) {
  "@babel/helpers - typeof";
  return _typeof$4 = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function(o2) {
    return typeof o2;
  } : function(o2) {
    return o2 && "function" == typeof Symbol && o2.constructor === Symbol && o2 !== Symbol.prototype ? "symbol" : typeof o2;
  }, _typeof$4(o);
}
__name(_typeof$4, "_typeof$4");
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
var _hoisted_1$j = ["aria-activedescendant"];
var _hoisted_2$g = ["id", "aria-label", "aria-setsize", "aria-posinset"];
var _hoisted_3$e = ["id", "placeholder", "tabindex", "disabled", "aria-label", "aria-labelledby", "aria-expanded", "aria-controls", "aria-activedescendant", "aria-invalid"];
var _hoisted_4$5 = ["disabled", "aria-expanded", "aria-controls"];
var _hoisted_5$3 = ["id"];
var _hoisted_6$2 = ["id", "aria-label"];
var _hoisted_7$1 = ["id"];
var _hoisted_8$1 = ["id", "aria-label", "aria-selected", "aria-disabled", "aria-setsize", "aria-posinset", "onClick", "onMousemove", "data-p-selected", "data-p-focus", "data-p-disabled"];
function render$e(_ctx, _cache, $props, $setup, $data, $options) {
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
    })], 16, _hoisted_2$g);
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
  }, _ctx.ptm("input")), null, 16, _hoisted_3$e)], 16)], 16, _hoisted_1$j)) : createCommentVNode("", true), $data.searching || _ctx.loading ? renderSlot(_ctx.$slots, _ctx.$slots.loader ? "loader" : "loadingicon", {
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
    })], 16, _hoisted_4$5)) : createCommentVNode("", true)];
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
            style: _objectSpread$3(_objectSpread$3(_objectSpread$3({}, _ctx.panelStyle), _ctx.overlayStyle), {}, {
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
                })], 16, _hoisted_8$1)), [[_directive_ripple]])], 64);
              }), 128)), !items || items && items.length === 0 ? (openBlock(), createElementBlock("li", mergeProps({
                key: 0,
                "class": _ctx.cx("emptyMessage"),
                role: "option"
              }, _ctx.ptm("emptyMessage")), [renderSlot(_ctx.$slots, "empty", {}, function() {
                return [createTextVNode(toDisplayString($options.searchResultMessageText), 1)];
              })], 16)) : createCommentVNode("", true)], 16, _hoisted_6$2)];
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
          }), toDisplayString($options.selectedMessageText), 17)], 16, _hoisted_5$3)) : createCommentVNode("", true)];
        }),
        _: 3
      }, 16, ["onEnter", "onAfterEnter", "onLeave", "onAfterLeave"])];
    }),
    _: 3
  }, 8, ["appendTo"])], 16);
}
__name(render$e, "render$e");
script$8.render = render$e;
const _sfc_main$j = {
  name: "AutoCompletePlus",
  extends: script$8,
  emits: ["focused-option-changed"],
  mounted() {
    if (typeof script$8.mounted === "function") {
      script$8.mounted.call(this);
    }
    this.$watch(
      () => this.focusedOptionIndex,
      (newVal, oldVal) => {
        this.$emit("focused-option-changed", newVal);
      }
    );
  }
};
const _withScopeId$4 = /* @__PURE__ */ __name((n) => (pushScopeId("data-v-d7cc0bce"), n = n(), popScopeId(), n), "_withScopeId$4");
const _hoisted_1$i = { class: "option-container flex justify-between items-center px-2 py-0 cursor-pointer overflow-hidden w-full" };
const _hoisted_2$f = { class: "option-display-name font-semibold flex flex-col" };
const _hoisted_3$d = { key: 0 };
const _hoisted_4$4 = /* @__PURE__ */ _withScopeId$4(() => /* @__PURE__ */ createBaseVNode("i", { class: "pi pi-bookmark-fill text-sm mr-1" }, null, -1));
const _hoisted_5$2 = [
  _hoisted_4$4
];
const _hoisted_6$1 = ["innerHTML"];
const _hoisted_7 = /* @__PURE__ */ _withScopeId$4(() => /* @__PURE__ */ createBaseVNode("span", null, " ", -1));
const _hoisted_8 = ["innerHTML"];
const _hoisted_9 = {
  key: 0,
  class: "option-category font-light text-sm text-gray-400 overflow-hidden text-ellipsis whitespace-nowrap"
};
const _hoisted_10 = { class: "option-badges" };
const _sfc_main$i = /* @__PURE__ */ defineComponent({
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
      return openBlock(), createElementBlock("div", _hoisted_1$i, [
        createBaseVNode("div", _hoisted_2$f, [
          createBaseVNode("div", null, [
            isBookmarked.value ? (openBlock(), createElementBlock("span", _hoisted_3$d, _hoisted_5$2)) : createCommentVNode("", true),
            createBaseVNode("span", {
              innerHTML: unref(highlightQuery)(_ctx.nodeDef.display_name, _ctx.currentQuery)
            }, null, 8, _hoisted_6$1),
            _hoisted_7,
            showIdName.value ? (openBlock(), createBlock(unref(script$q), {
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
          _ctx.nodeDef.experimental ? (openBlock(), createBlock(unref(script$q), {
            key: 0,
            value: _ctx.$t("g.experimental"),
            severity: "primary"
          }, null, 8, ["value"])) : createCommentVNode("", true),
          _ctx.nodeDef.deprecated ? (openBlock(), createBlock(unref(script$q), {
            key: 1,
            value: _ctx.$t("g.deprecated"),
            severity: "danger"
          }, null, 8, ["value"])) : createCommentVNode("", true),
          showNodeFrequency.value && nodeFrequency.value > 0 ? (openBlock(), createBlock(unref(script$q), {
            key: 2,
            value: unref(formatNumberWithSuffix)(nodeFrequency.value, { roundToInt: true }),
            severity: "secondary"
          }, null, 8, ["value"])) : createCommentVNode("", true),
          _ctx.nodeDef.nodeSource.type !== unref(NodeSourceType).Unknown ? (openBlock(), createBlock(unref(script$p), {
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
const NodeSearchItem = /* @__PURE__ */ _export_sfc(_sfc_main$i, [["__scopeId", "data-v-d7cc0bce"]]);
const _hoisted_1$h = { class: "comfy-vue-node-search-container flex justify-center items-center w-full min-w-96 pointer-events-auto" };
const _hoisted_2$e = {
  key: 0,
  class: "comfy-vue-node-preview-container absolute left-[-350px] top-[50px]"
};
const _hoisted_3$c = /* @__PURE__ */ createBaseVNode("h3", null, "Add node filter condition", -1);
const _hoisted_4$3 = { class: "_dialog-body" };
const _sfc_main$h = /* @__PURE__ */ defineComponent({
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
      return props.filters.length === 0 ? t("g.searchNodes") + "..." : "";
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
      return openBlock(), createElementBlock("div", _hoisted_1$h, [
        enableNodePreview.value ? (openBlock(), createElementBlock("div", _hoisted_2$e, [
          hoveredSuggestion.value ? (openBlock(), createBlock(NodePreview, {
            nodeDef: hoveredSuggestion.value,
            key: hoveredSuggestion.value?.name || ""
          }, null, 8, ["nodeDef"])) : createCommentVNode("", true)
        ])) : createCommentVNode("", true),
        createVNode(unref(script$f), {
          icon: "pi pi-filter",
          severity: "secondary",
          class: "filter-button z-10",
          onClick: _cache[0] || (_cache[0] = ($event) => nodeSearchFilterVisible.value = true)
        }),
        createVNode(unref(script$r), {
          visible: nodeSearchFilterVisible.value,
          "onUpdate:visible": _cache[1] || (_cache[1] = ($event) => nodeSearchFilterVisible.value = $event),
          class: "min-w-96"
        }, {
          header: withCtx(() => [
            _hoisted_3$c
          ]),
          default: withCtx(() => [
            createBaseVNode("div", _hoisted_4$3, [
              createVNode(NodeSearchFilter, { onAddFilter })
            ])
          ]),
          _: 1
        }, 8, ["visible"]),
        createVNode(_sfc_main$j, {
          "model-value": props.filters,
          class: "comfy-vue-node-search-box z-10 flex-grow",
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
class ConnectingLinkImpl {
  static {
    __name(this, "ConnectingLinkImpl");
  }
  constructor(node, slot, input, output, pos, afterRerouteId) {
    this.node = node;
    this.slot = slot;
    this.input = input;
    this.output = output;
    this.pos = pos;
    this.afterRerouteId = afterRerouteId;
  }
  static createFromPlainObject(obj) {
    return new ConnectingLinkImpl(
      obj.node,
      obj.slot,
      obj.input,
      obj.output,
      obj.pos,
      obj.afterRerouteId
    );
  }
  get type() {
    const result = this.input ? this.input.type : this.output?.type ?? null;
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
      this.node.connect(this.slot, newNode, newNodeSlot, this.afterRerouteId);
    } else {
      newNode.connect(newNodeSlot, this.node, this.slot, this.afterRerouteId);
    }
  }
}
const useSearchBoxStore = defineStore("searchBox", () => {
  const visible = ref(false);
  function toggleVisible() {
    visible.value = !visible.value;
  }
  __name(toggleVisible, "toggleVisible");
  return {
    visible,
    toggleVisible
  };
});
const _sfc_main$g = /* @__PURE__ */ defineComponent({
  __name: "NodeSearchBoxPopover",
  setup(__props) {
    const settingStore = useSettingStore();
    const { visible } = storeToRefs(useSearchBoxStore());
    const dismissable = ref(true);
    const triggerEvent = ref(null);
    const getNewNodeLocation = /* @__PURE__ */ __name(() => {
      if (!triggerEvent.value) {
        return app.getCanvasCenter();
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
      const eventDetail = triggerEvent.value?.detail;
      if (eventDetail && eventDetail.subType === "empty-release") {
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
      const detail = e.detail;
      if (newSearchBoxEnabled.value) {
        if (detail.originalEvent?.pointerType === "touch") {
          setTimeout(() => {
            showNewSearchBox(e);
          }, 128);
        } else {
          showNewSearchBox(e);
        }
      } else {
        canvasStore.canvas.showSearchBox(detail.originalEvent);
      }
    }, "showSearchBox");
    const nodeDefStore = useNodeDefStore();
    const showNewSearchBox = /* @__PURE__ */ __name((e) => {
      if (e.detail.subType === "empty-release") {
        const links = e.detail.linkReleaseContext.links;
        if (links.length === 0) {
          console.warn("Empty release with no links! This should never happen");
          return;
        }
        const firstLink = ConnectingLinkImpl.createFromPlainObject(links[0]);
        const filter = nodeDefStore.nodeSearchService.getFilterById(
          firstLink.releaseSlotType
        );
        const dataType = firstLink.type.toString();
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
      if (e.detail.subType !== "empty-release") {
        return;
      }
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
      const connectionOptions = firstLink.output ? {
        nodeFrom: firstLink.node,
        slotFrom: firstLink.output,
        afterRerouteId: firstLink.afterRerouteId
      } : {
        nodeTo: firstLink.node,
        slotTo: firstLink.input,
        afterRerouteId: firstLink.afterRerouteId
      };
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
      const detail = e.detail;
      const shiftPressed = detail.originalEvent.shiftKey;
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
    useEventListener(document, "litegraph:canvas", canvasEventHandler);
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", null, [
        createVNode(unref(script$r), {
          visible: unref(visible),
          "onUpdate:visible": _cache[0] || (_cache[0] = ($event) => isRef(visible) ? visible.value = $event : null),
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
            createVNode(_sfc_main$h, {
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
const _sfc_main$f = /* @__PURE__ */ defineComponent({
  __name: "NodeTooltip",
  setup(__props) {
    let idleTimeout;
    const nodeDefStore = useNodeDefStore();
    const tooltipRef = ref();
    const tooltipText = ref("");
    const left = ref();
    const top = ref();
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
        const translatedTooltip = st(
          `nodeDefs.${normalizeI18nKey(node.type)}.inputs.${normalizeI18nKey(inputName)}.tooltip`,
          nodeDef.inputs.getInput(inputName)?.tooltip
        );
        return showTooltip(translatedTooltip);
      }
      const outputSlot = canvas.isOverNodeOutput(
        node,
        canvas.graph_mouse[0],
        canvas.graph_mouse[1],
        [0, 0]
      );
      if (outputSlot !== -1) {
        const translatedTooltip = st(
          `nodeDefs.${normalizeI18nKey(node.type)}.outputs.${outputSlot}.tooltip`,
          nodeDef.outputs.all?.[outputSlot]?.tooltip
        );
        return showTooltip(translatedTooltip);
      }
      const widget = app.canvas.getWidgetAtCursor();
      if (widget && !widget.element) {
        const translatedTooltip = st(
          `nodeDefs.${normalizeI18nKey(node.type)}.inputs.${normalizeI18nKey(widget.name)}.tooltip`,
          nodeDef.inputs.getInput(widget.name)?.tooltip
        );
        return showTooltip(widget.tooltip ?? translatedTooltip);
      }
    }, "onIdle");
    const onMouseMove = /* @__PURE__ */ __name((e) => {
      hideTooltip();
      clearTimeout(idleTimeout);
      if (e.target.nodeName !== "CANVAS") return;
      idleTimeout = window.setTimeout(onIdle, 500);
    }, "onMouseMove");
    useEventListener(window, "mousemove", onMouseMove);
    useEventListener(window, "click", hideTooltip);
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
const NodeTooltip = /* @__PURE__ */ _export_sfc(_sfc_main$f, [["__scopeId", "data-v-9ecc8adc"]]);
const _sfc_main$e = /* @__PURE__ */ defineComponent({
  __name: "NodeBadge",
  setup(__props) {
    const settingStore = useSettingStore();
    const nodeSourceBadgeMode = computed(
      () => settingStore.get("Comfy.NodeBadge.NodeSourceBadgeMode")
    );
    const nodeIdBadgeMode = computed(
      () => settingStore.get("Comfy.NodeBadge.NodeIdBadgeMode")
    );
    const nodeLifeCycleBadgeMode = computed(
      () => settingStore.get("Comfy.NodeBadge.NodeLifeCycleBadgeMode")
    );
    watch([nodeSourceBadgeMode, nodeIdBadgeMode, nodeLifeCycleBadgeMode], () => {
      app.graph?.setDirtyCanvas(true, true);
    });
    const colorPalette = computed(
      () => getColorPalette(settingStore.get("Comfy.ColorPalette"))
    );
    const nodeDefStore = useNodeDefStore();
    function badgeTextVisible(nodeDef, badgeMode) {
      return !(badgeMode === NodeBadgeMode.None || nodeDef?.isCoreNode && badgeMode === NodeBadgeMode.HideBuiltIn);
    }
    __name(badgeTextVisible, "badgeTextVisible");
    onMounted(() => {
      app.registerExtension({
        name: "Comfy.NodeBadge",
        nodeCreated(node) {
          node.badgePosition = BadgePosition.TopRight;
          const badge = computed(() => {
            const nodeDef = nodeDefStore.fromLGraphNode(node);
            return new LGraphBadge({
              text: _.truncate(
                [
                  badgeTextVisible(nodeDef, nodeIdBadgeMode.value) ? `#${node.id}` : "",
                  badgeTextVisible(nodeDef, nodeLifeCycleBadgeMode.value) ? nodeDef?.nodeLifeCycleBadgeText ?? "" : "",
                  badgeTextVisible(nodeDef, nodeSourceBadgeMode.value) ? nodeDef?.nodeSource?.badgeText ?? "" : ""
                ].filter((s) => s.length > 0).join(" "),
                {
                  length: 31
                }
              ),
              fgColor: colorPalette.value?.colors?.litegraph_base?.BADGE_FG_COLOR || defaultColorPalette.colors.litegraph_base.BADGE_FG_COLOR,
              bgColor: colorPalette.value?.colors?.litegraph_base?.BADGE_BG_COLOR || defaultColorPalette.colors.litegraph_base.BADGE_BG_COLOR
            });
          });
          node.badges.push(() => badge.value);
        }
      });
    });
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div");
    };
  }
});
const _hoisted_1$g = {
  viewBox: "0 0 1024 1024",
  width: "1.2em",
  height: "1.2em"
};
const _hoisted_2$d = /* @__PURE__ */ createBaseVNode("path", {
  fill: "currentColor",
  d: "M921.088 103.232L584.832 889.024L465.52 544.512L121.328 440.48zM1004.46.769c-6.096 0-13.52 1.728-22.096 5.36L27.708 411.2c-34.383 14.592-36.56 42.704-4.847 62.464l395.296 123.584l129.36 403.264c9.28 15.184 20.496 22.72 31.263 22.72c11.936 0 23.296-9.152 31.04-27.248l408.272-953.728C1029.148 16.368 1022.86.769 1004.46.769"
}, null, -1);
const _hoisted_3$b = [
  _hoisted_2$d
];
function render$d(_ctx, _cache) {
  return openBlock(), createElementBlock("svg", _hoisted_1$g, [..._hoisted_3$b]);
}
__name(render$d, "render$d");
const __unplugin_components_1$2 = markRaw({ name: "simple-line-icons-cursor", render: render$d });
const _hoisted_1$f = {
  viewBox: "0 0 24 24",
  width: "1.2em",
  height: "1.2em"
};
const _hoisted_2$c = /* @__PURE__ */ createBaseVNode("path", {
  fill: "currentColor",
  d: "M10.05 23q-.75 0-1.4-.337T7.575 21.7L1.2 12.375l.6-.575q.475-.475 1.125-.55t1.175.3L7 13.575V4q0-.425.288-.712T8 3t.713.288T9 4v13.425l-3.7-2.6l3.925 5.725q.125.2.35.325t.475.125H17q.825 0 1.413-.587T19 19V5q0-.425.288-.712T20 4t.713.288T21 5v14q0 1.65-1.175 2.825T17 23zM11 12V2q0-.425.288-.712T12 1t.713.288T13 2v10zm4 0V3q0-.425.288-.712T16 2t.713.288T17 3v9zm-2.85 4.5"
}, null, -1);
const _hoisted_3$a = [
  _hoisted_2$c
];
function render$c(_ctx, _cache) {
  return openBlock(), createElementBlock("svg", _hoisted_1$f, [..._hoisted_3$a]);
}
__name(render$c, "render$c");
const __unplugin_components_0$2 = markRaw({ name: "material-symbols-pan-tool-outline", render: render$c });
var theme$5 = /* @__PURE__ */ __name(function theme4(_ref) {
  _ref.dt;
  return "\n.p-buttongroup .p-button {\n    margin: 0;\n}\n\n.p-buttongroup .p-button:not(:last-child),\n.p-buttongroup .p-button:not(:last-child):hover {\n    border-right: 0 none;\n}\n\n.p-buttongroup .p-button:not(:first-of-type):not(:last-of-type) {\n    border-radius: 0;\n}\n\n.p-buttongroup .p-button:first-of-type:not(:only-of-type) {\n    border-top-right-radius: 0;\n    border-bottom-right-radius: 0;\n}\n\n.p-buttongroup .p-button:last-of-type:not(:only-of-type) {\n    border-top-left-radius: 0;\n    border-bottom-left-radius: 0;\n}\n\n.p-buttongroup .p-button:focus {\n    position: relative;\n    z-index: 1;\n}\n";
}, "theme");
var classes$5 = {
  root: "p-buttongroup p-component"
};
var ButtonGroupStyle = BaseStyle.extend({
  name: "buttongroup",
  theme: theme$5,
  classes: classes$5
});
var script$1$5 = {
  name: "BaseButtonGroup",
  "extends": script$g,
  style: ButtonGroupStyle,
  provide: /* @__PURE__ */ __name(function provide8() {
    return {
      $pcButtonGroup: this,
      $parentInstance: this
    };
  }, "provide")
};
var script$7 = {
  name: "ButtonGroup",
  "extends": script$1$5,
  inheritAttrs: false
};
function render$b(_ctx, _cache, $props, $setup, $data, $options) {
  return openBlock(), createElementBlock("span", mergeProps({
    "class": _ctx.cx("root"),
    role: "group"
  }, _ctx.ptmi("root")), [renderSlot(_ctx.$slots, "default")], 16);
}
__name(render$b, "render$b");
script$7.render = render$b;
const _sfc_main$d = /* @__PURE__ */ defineComponent({
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
      const _component_i_material_symbols58pan_tool_outline = __unplugin_components_0$2;
      const _component_i_simple_line_icons58cursor = __unplugin_components_1$2;
      const _directive_tooltip = resolveDirective("tooltip");
      return openBlock(), createBlock(unref(script$7), { class: "p-buttongroup-vertical absolute bottom-[10px] right-[10px] z-[1000] pointer-events-auto" }, {
        default: withCtx(() => [
          withDirectives(createVNode(unref(script$f), {
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
          withDirectives(createVNode(unref(script$f), {
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
          withDirectives(createVNode(unref(script$f), {
            severity: "secondary",
            icon: "pi pi-expand",
            onClick: _cache[2] || (_cache[2] = () => unref(commandStore).execute("Comfy.Canvas.FitView"))
          }, null, 512), [
            [
              _directive_tooltip,
              unref(t)("graphCanvasMenu.fitView"),
              void 0,
              { left: true }
            ]
          ]),
          withDirectives((openBlock(), createBlock(unref(script$f), {
            severity: "secondary",
            onClick: _cache[3] || (_cache[3] = () => unref(commandStore).execute("Comfy.Canvas.ToggleLock"))
          }, {
            icon: withCtx(() => [
              unref(canvasStore).canvas?.read_only ? (openBlock(), createBlock(_component_i_material_symbols58pan_tool_outline, { key: 0 })) : (openBlock(), createBlock(_component_i_simple_line_icons58cursor, { key: 1 }))
            ]),
            _: 1
          })), [
            [
              _directive_tooltip,
              unref(t)(
                "graphCanvasMenu." + (unref(canvasStore).canvas?.read_only ? "panMode" : "selectMode")
              ) + " (Space)",
              void 0,
              { left: true }
            ]
          ]),
          withDirectives(createVNode(unref(script$f), {
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
const GraphCanvasMenu = /* @__PURE__ */ _export_sfc(_sfc_main$d, [["__scopeId", "data-v-94481f39"]]);
const _sfc_main$c = /* @__PURE__ */ defineComponent({
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
    const tooltipEnabled = computed(() => settingStore.get("Comfy.EnableTooltips"));
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
      LiteGraph.snaps_for_comfy = settingStore.get("Comfy.Node.AutoSnapLinkToSlot");
    });
    watchEffect(() => {
      LiteGraph.snap_highlights_node = settingStore.get(
        "Comfy.Node.SnapHighlightsNode"
      );
    });
    watchEffect(() => {
      LGraphNode.keepAllLinksOnBypass = settingStore.get(
        "Comfy.Node.BypassAllLinksOnDelete"
      );
    });
    watchEffect(() => {
      LiteGraph.middle_click_slot_add_default_node = settingStore.get(
        "Comfy.Node.MiddleClickRerouteNode"
      );
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
      const linkRenderMode = settingStore.get("Comfy.LinkRenderMode");
      if (canvasStore.canvas) {
        canvasStore.canvas.links_render_mode = linkRenderMode;
        canvasStore.canvas.setDirty(
          /* fg */
          false,
          /* bg */
          true
        );
      }
    });
    watchEffect(() => {
      const linkMarkerShape = settingStore.get("Comfy.Graph.LinkMarkers");
      const { canvas } = canvasStore;
      if (canvas) {
        canvas.linkMarkerShape = linkMarkerShape;
        canvas.setDirty(false, true);
      }
    });
    watchEffect(() => {
      const reroutesEnabled = settingStore.get("Comfy.RerouteBeta");
      const { canvas } = canvasStore;
      if (canvas) {
        canvas.reroutesEnabled = reroutesEnabled;
        canvas.setDirty(false, true);
      }
    });
    watchEffect(() => {
      const maximumFps = settingStore.get("LiteGraph.Canvas.MaximumFps");
      const { canvas } = canvasStore;
      if (canvas) canvas.maximumFps = maximumFps;
    });
    watchEffect(() => {
      CanvasPointer.doubleClickTime = settingStore.get(
        "Comfy.Pointer.DoubleClickTime"
      );
    });
    watchEffect(() => {
      CanvasPointer.bufferTime = settingStore.get("Comfy.Pointer.ClickBufferTime");
    });
    watchEffect(() => {
      CanvasPointer.maxClickDrift = settingStore.get("Comfy.Pointer.ClickDrift");
    });
    watchEffect(() => {
      LiteGraph.CANVAS_GRID_SIZE = settingStore.get("Comfy.SnapToGrid.GridSize");
    });
    watchEffect(() => {
      LiteGraph.alwaysSnapToGrid = settingStore.get("pysssss.SnapToGrid");
    });
    watchEffect(() => {
      if (!canvasStore.canvas) return;
      if (canvasStore.canvas.state.draggingCanvas) {
        canvasStore.canvas.canvas.style.cursor = "grabbing";
        return;
      }
      if (canvasStore.canvas.state.readOnly) {
        canvasStore.canvas.canvas.style.cursor = "grab";
        return;
      }
      canvasStore.canvas.canvas.style.cursor = "default";
    });
    const workflowStore = useWorkflowStore();
    const persistCurrentWorkflow = /* @__PURE__ */ __name(() => {
      const workflow = JSON.stringify(app.serializeGraph());
      localStorage.setItem("workflow", workflow);
      if (api.clientId) {
        sessionStorage.setItem(`workflow:${api.clientId}`, workflow);
      }
    }, "persistCurrentWorkflow");
    watchEffect(() => {
      if (workflowStore.activeWorkflow) {
        const workflow = workflowStore.activeWorkflow;
        setStorageValue("Comfy.PreviousWorkflow", workflow.key);
        persistCurrentWorkflow();
      }
    });
    api.addEventListener("graphChanged", persistCurrentWorkflow);
    usePragmaticDroppable(() => canvasRef.value, {
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
          } else if (node.data instanceof ComfyModelDef) {
            const model = node.data;
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
    const comfyAppReady = ref(false);
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
      ChangeTracker.init(app);
      await app.setup(canvasRef.value);
      canvasStore.canvas = app.canvas;
      canvasStore.canvas.render_canvas_border = false;
      workspaceStore.spinner = false;
      window["app"] = app;
      window["graph"] = app.graph;
      comfyAppReady.value = true;
      watch(
        () => settingStore.get("Comfy.Locale"),
        async () => {
          await useCommandStore().execute("Comfy.RefreshNodeDefinitions");
          workflowService.reloadCurrentWorkflow();
        }
      );
      emit("ready");
    });
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock(Fragment, null, [
        (openBlock(), createBlock(Teleport, { to: ".graph-canvas-container" }, [
          comfyAppReady.value && betaMenuEnabled.value && !unref(workspaceStore).focusMode ? (openBlock(), createBlock(LiteGraphCanvasSplitterOverlay, { key: 0 }, {
            "side-bar-panel": withCtx(() => [
              createVNode(SideToolbar)
            ]),
            "bottom-panel": withCtx(() => [
              createVNode(_sfc_main$l)
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
        createVNode(_sfc_main$g),
        tooltipEnabled.value ? (openBlock(), createBlock(NodeTooltip, { key: 0 })) : createCommentVNode("", true),
        createVNode(_sfc_main$e)
      ], 64);
    };
  }
});
const _sfc_main$b = /* @__PURE__ */ defineComponent({
  __name: "MenuHamburger",
  setup(__props) {
    const workspaceState = useWorkspaceStore();
    const settingStore = useSettingStore();
    const exitFocusMode = /* @__PURE__ */ __name(() => {
      workspaceState.focusMode = false;
    }, "exitFocusMode");
    watchEffect(() => {
      if (settingStore.get("Comfy.UseNewMenu") !== "Disabled") {
        return;
      }
      if (workspaceState.focusMode) {
        app.ui.menuContainer.style.display = "none";
      } else {
        app.ui.menuContainer.style.display = "block";
      }
    });
    const menuSetting = computed(() => settingStore.get("Comfy.UseNewMenu"));
    const positionCSS = computed(
      () => (
        // 'Bottom' menuSetting shows the hamburger button in the bottom right corner
        // 'Disabled', 'Top' menuSetting shows the hamburger button in the top right corner
        menuSetting.value === "Bottom" ? { bottom: "0px", right: "0px" } : { top: "0px", right: "0px" }
      )
    );
    return (_ctx, _cache) => {
      const _directive_tooltip = resolveDirective("tooltip");
      return withDirectives((openBlock(), createBlock(unref(script$f), {
        class: "comfy-menu-hamburger",
        style: normalizeStyle(positionCSS.value),
        icon: "pi pi-bars",
        severity: "secondary",
        text: "",
        size: "large",
        onClick: exitFocusMode,
        onContextmenu: unref(showNativeMenu)
      }, null, 8, ["style", "onContextmenu"])), [
        [vShow, unref(workspaceState).focusMode],
        [_directive_tooltip, { value: _ctx.$t("menu.showMenu"), showDelay: 300 }]
      ]);
    };
  }
});
const MenuHamburger = /* @__PURE__ */ _export_sfc(_sfc_main$b, [["__scopeId", "data-v-962c4073"]]);
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
var theme$4 = /* @__PURE__ */ __name(function theme5(_ref) {
  var dt = _ref.dt;
  return "\n.p-toast {\n    width: ".concat(dt("toast.width"), ";\n    white-space: pre-line;\n    word-break: break-word;\n}\n\n.p-toast-message {\n    margin: 0 0 1rem 0;\n}\n\n.p-toast-message-icon {\n    flex-shrink: 0;\n    font-size: ").concat(dt("toast.icon.size"), ";\n    width: ").concat(dt("toast.icon.size"), ";\n    height: ").concat(dt("toast.icon.size"), ";\n}\n\n.p-toast-message-content {\n    display: flex;\n    align-items: flex-start;\n    padding: ").concat(dt("toast.content.padding"), ";\n    gap: ").concat(dt("toast.content.gap"), ";\n}\n\n.p-toast-message-text {\n    flex: 1 1 auto;\n    display: flex;\n    flex-direction: column;\n    gap: ").concat(dt("toast.text.gap"), ";\n}\n\n.p-toast-summary {\n    font-weight: ").concat(dt("toast.summary.font.weight"), ";\n    font-size: ").concat(dt("toast.summary.font.size"), ";\n}\n\n.p-toast-detail {\n    font-weight: ").concat(dt("toast.detail.font.weight"), ";\n    font-size: ").concat(dt("toast.detail.font.size"), ";\n}\n\n.p-toast-close-button {\n    display: flex;\n    align-items: center;\n    justify-content: center;\n    overflow: hidden;\n    position: relative;\n    cursor: pointer;\n    background: transparent;\n    transition: background ").concat(dt("toast.transition.duration"), ", color ").concat(dt("toast.transition.duration"), ", outline-color ").concat(dt("toast.transition.duration"), ", box-shadow ").concat(dt("toast.transition.duration"), ";\n    outline-color: transparent;\n    color: inherit;\n    width: ").concat(dt("toast.close.button.width"), ";\n    height: ").concat(dt("toast.close.button.height"), ";\n    border-radius: ").concat(dt("toast.close.button.border.radius"), ";\n    margin: -25% 0 0 0;\n    right: -25%;\n    padding: 0;\n    border: none;\n    user-select: none;\n}\n\n.p-toast-message-info,\n.p-toast-message-success,\n.p-toast-message-warn,\n.p-toast-message-error,\n.p-toast-message-secondary,\n.p-toast-message-contrast {\n    border-width: ").concat(dt("toast.border.width"), ";\n    border-style: solid;\n    backdrop-filter: blur(").concat(dt("toast.blur"), ");\n    border-radius: ").concat(dt("toast.border.radius"), ";\n}\n\n.p-toast-close-icon {\n    font-size: ").concat(dt("toast.close.icon.size"), ";\n    width: ").concat(dt("toast.close.icon.size"), ";\n    height: ").concat(dt("toast.close.icon.size"), ";\n}\n\n.p-toast-close-button:focus-visible {\n    outline-width: ").concat(dt("focus.ring.width"), ";\n    outline-style: ").concat(dt("focus.ring.style"), ";\n    outline-offset: ").concat(dt("focus.ring.offset"), ";\n}\n\n.p-toast-message-info {\n    background: ").concat(dt("toast.info.background"), ";\n    border-color: ").concat(dt("toast.info.border.color"), ";\n    color: ").concat(dt("toast.info.color"), ";\n    box-shadow: ").concat(dt("toast.info.shadow"), ";\n}\n\n.p-toast-message-info .p-toast-detail {\n    color: ").concat(dt("toast.info.detail.color"), ";\n}\n\n.p-toast-message-info .p-toast-close-button:focus-visible {\n    outline-color: ").concat(dt("toast.info.close.button.focus.ring.color"), ";\n    box-shadow: ").concat(dt("toast.info.close.button.focus.ring.shadow"), ";\n}\n\n.p-toast-message-info .p-toast-close-button:hover {\n    background: ").concat(dt("toast.info.close.button.hover.background"), ";\n}\n\n.p-toast-message-success {\n    background: ").concat(dt("toast.success.background"), ";\n    border-color: ").concat(dt("toast.success.border.color"), ";\n    color: ").concat(dt("toast.success.color"), ";\n    box-shadow: ").concat(dt("toast.success.shadow"), ";\n}\n\n.p-toast-message-success .p-toast-detail {\n    color: ").concat(dt("toast.success.detail.color"), ";\n}\n\n.p-toast-message-success .p-toast-close-button:focus-visible {\n    outline-color: ").concat(dt("toast.success.close.button.focus.ring.color"), ";\n    box-shadow: ").concat(dt("toast.success.close.button.focus.ring.shadow"), ";\n}\n\n.p-toast-message-success .p-toast-close-button:hover {\n    background: ").concat(dt("toast.success.close.button.hover.background"), ";\n}\n\n.p-toast-message-warn {\n    background: ").concat(dt("toast.warn.background"), ";\n    border-color: ").concat(dt("toast.warn.border.color"), ";\n    color: ").concat(dt("toast.warn.color"), ";\n    box-shadow: ").concat(dt("toast.warn.shadow"), ";\n}\n\n.p-toast-message-warn .p-toast-detail {\n    color: ").concat(dt("toast.warn.detail.color"), ";\n}\n\n.p-toast-message-warn .p-toast-close-button:focus-visible {\n    outline-color: ").concat(dt("toast.warn.close.button.focus.ring.color"), ";\n    box-shadow: ").concat(dt("toast.warn.close.button.focus.ring.shadow"), ";\n}\n\n.p-toast-message-warn .p-toast-close-button:hover {\n    background: ").concat(dt("toast.warn.close.button.hover.background"), ";\n}\n\n.p-toast-message-error {\n    background: ").concat(dt("toast.error.background"), ";\n    border-color: ").concat(dt("toast.error.border.color"), ";\n    color: ").concat(dt("toast.error.color"), ";\n    box-shadow: ").concat(dt("toast.error.shadow"), ";\n}\n\n.p-toast-message-error .p-toast-detail {\n    color: ").concat(dt("toast.error.detail.color"), ";\n}\n\n.p-toast-message-error .p-toast-close-button:focus-visible {\n    outline-color: ").concat(dt("toast.error.close.button.focus.ring.color"), ";\n    box-shadow: ").concat(dt("toast.error.close.button.focus.ring.shadow"), ";\n}\n\n.p-toast-message-error .p-toast-close-button:hover {\n    background: ").concat(dt("toast.error.close.button.hover.background"), ";\n}\n\n.p-toast-message-secondary {\n    background: ").concat(dt("toast.secondary.background"), ";\n    border-color: ").concat(dt("toast.secondary.border.color"), ";\n    color: ").concat(dt("toast.secondary.color"), ";\n    box-shadow: ").concat(dt("toast.secondary.shadow"), ";\n}\n\n.p-toast-message-secondary .p-toast-detail {\n    color: ").concat(dt("toast.secondary.detail.color"), ";\n}\n\n.p-toast-message-secondary .p-toast-close-button:focus-visible {\n    outline-color: ").concat(dt("toast.secondary.close.button.focus.ring.color"), ";\n    box-shadow: ").concat(dt("toast.secondary.close.button.focus.ring.shadow"), ";\n}\n\n.p-toast-message-secondary .p-toast-close-button:hover {\n    background: ").concat(dt("toast.secondary.close.button.hover.background"), ";\n}\n\n.p-toast-message-contrast {\n    background: ").concat(dt("toast.contrast.background"), ";\n    border-color: ").concat(dt("toast.contrast.border.color"), ";\n    color: ").concat(dt("toast.contrast.color"), ";\n    box-shadow: ").concat(dt("toast.contrast.shadow"), ";\n}\n\n.p-toast-message-contrast .p-toast-detail {\n    color: ").concat(dt("toast.contrast.detail.color"), ";\n}\n\n.p-toast-message-contrast .p-toast-close-button:focus-visible {\n    outline-color: ").concat(dt("toast.contrast.close.button.focus.ring.color"), ";\n    box-shadow: ").concat(dt("toast.contrast.close.button.focus.ring.shadow"), ";\n}\n\n.p-toast-message-contrast .p-toast-close-button:hover {\n    background: ").concat(dt("toast.contrast.close.button.hover.background"), ";\n}\n\n.p-toast-top-center {\n    transform: translateX(-50%);\n}\n\n.p-toast-bottom-center {\n    transform: translateX(-50%);\n}\n\n.p-toast-center {\n    min-width: 20vw;\n    transform: translate(-50%, -50%);\n}\n\n.p-toast-message-enter-from {\n    opacity: 0;\n    transform: translateY(50%);\n}\n\n.p-toast-message-leave-from {\n    max-height: 1000px;\n}\n\n.p-toast .p-toast-message.p-toast-message-leave-to {\n    max-height: 0;\n    opacity: 0;\n    margin-bottom: 0;\n    overflow: hidden;\n}\n\n.p-toast-message-enter-active {\n    transition: transform 0.3s, opacity 0.3s;\n}\n\n.p-toast-message-leave-active {\n    transition: max-height 0.45s cubic-bezier(0, 1, 0, 1), opacity 0.3s, margin-bottom 0.3s;\n}\n");
}, "theme");
var inlineStyles$2 = {
  root: /* @__PURE__ */ __name(function root6(_ref2) {
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
var classes$4 = {
  root: /* @__PURE__ */ __name(function root7(_ref3) {
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
  "extends": script$g,
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
  provide: /* @__PURE__ */ __name(function provide9() {
    return {
      $pcToast: this,
      $parentInstance: this
    };
  }, "provide")
};
var script$1$4 = {
  name: "ToastMessage",
  hostName: "Toast",
  "extends": script$g,
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
  mounted: /* @__PURE__ */ __name(function mounted4() {
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
  beforeUnmount: /* @__PURE__ */ __name(function beforeUnmount4() {
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
        info: !this.infoIcon && script$s,
        success: !this.successIcon && script$t,
        warn: !this.warnIcon && script$u,
        error: !this.errorIcon && script$v
      }[this.message.severity];
    }, "iconComponent"),
    closeAriaLabel: /* @__PURE__ */ __name(function closeAriaLabel() {
      return this.$primevue.config.locale.aria ? this.$primevue.config.locale.aria.close : void 0;
    }, "closeAriaLabel")
  },
  components: {
    TimesIcon: script$w,
    InfoCircleIcon: script$s,
    CheckIcon: script$t,
    ExclamationTriangleIcon: script$u,
    TimesCircleIcon: script$v
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
var _hoisted_1$e = ["aria-label"];
function render$1$3(_ctx, _cache, $props, $setup, $data, $options) {
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
  }, _ctx.ptm("closeIcon")), null, 16, ["class"]))], 16, _hoisted_1$e)), [[_directive_ripple]])], 16)) : createCommentVNode("", true)], 16))], 16);
}
__name(render$1$3, "render$1$3");
script$1$4.render = render$1$3;
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
  data: /* @__PURE__ */ __name(function data5() {
    return {
      messages: []
    };
  }, "data"),
  styleElement: null,
  mounted: /* @__PURE__ */ __name(function mounted5() {
    ToastEventBus.on("add", this.onAdd);
    ToastEventBus.on("remove", this.onRemove);
    ToastEventBus.on("remove-group", this.onRemoveGroup);
    ToastEventBus.on("remove-all-groups", this.onRemoveAllGroups);
    if (this.breakpoints) {
      this.createStyle();
    }
  }, "mounted"),
  beforeUnmount: /* @__PURE__ */ __name(function beforeUnmount5() {
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
      this.$refs.container.setAttribute(this.attributeSelector, "");
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
          innerHTML += "\n                        @media screen and (max-width: ".concat(breakpoint, ") {\n                            .p-toast[").concat(this.attributeSelector, "] {\n                                ").concat(breakpointStyle, "\n                            }\n                        }\n                    ");
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
  computed: {
    attributeSelector: /* @__PURE__ */ __name(function attributeSelector() {
      return UniqueComponentId();
    }, "attributeSelector")
  },
  components: {
    ToastMessage: script$1$4,
    Portal: script$m
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
function render$a(_ctx, _cache, $props, $setup, $data, $options) {
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
__name(render$a, "render$a");
script$6.render = render$a;
const _sfc_main$a = /* @__PURE__ */ defineComponent({
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
        newMessages.forEach((message2) => {
          toast.add(message2);
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
        messagesToRemove.forEach((message2) => {
          toast.remove(message2);
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
const _sfc_main$9 = /* @__PURE__ */ defineComponent({
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
const _sfc_main$8 = /* @__PURE__ */ defineComponent({
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
      () => workflowStore.activeWorkflow?.isModified || !workflowStore.activeWorkflow?.isPersisted ? " *" : ""
    );
    const workflowNameText = computed(() => {
      const workflowName = workflowStore.activeWorkflow?.filename;
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
const _withScopeId$3 = /* @__PURE__ */ __name((n) => (pushScopeId("data-v-4cb762cb"), n = n(), popScopeId(), n), "_withScopeId$3");
const _hoisted_1$d = ["onContextmenu", "onMouseup"];
const _hoisted_2$b = { class: "workflow-label text-sm max-w-[150px] truncate inline-block" };
const _hoisted_3$9 = { class: "relative" };
const _hoisted_4$2 = {
  key: 0,
  class: "status-indicator"
};
const _sfc_main$7 = /* @__PURE__ */ defineComponent({
  __name: "WorkflowTabs",
  props: {
    class: {}
  },
  setup(__props) {
    const props = __props;
    const { t } = useI18n();
    const workspaceStore = useWorkspaceStore();
    const workflowStore = useWorkflowStore();
    const rightClickedTab = ref(null);
    const menu = ref();
    const showContextMenu = /* @__PURE__ */ __name((event, option2) => {
      rightClickedTab.value = option2;
      menu.value.show(event);
    }, "showContextMenu");
    const workflowToOption = /* @__PURE__ */ __name((workflow) => ({
      value: workflow.path,
      workflow
    }), "workflowToOption");
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
      workflowService.openWorkflow(option2.workflow);
    }, "onWorkflowChange");
    const closeWorkflows = /* @__PURE__ */ __name(async (options2) => {
      for (const opt of options2) {
        if (!await workflowService.closeWorkflow(opt.workflow, {
          warnIfUnsaved: !workspaceStore.shiftDown
        })) {
          break;
        }
      }
    }, "closeWorkflows");
    const onCloseWorkflow = /* @__PURE__ */ __name((option2) => {
      closeWorkflows([option2]);
    }, "onCloseWorkflow");
    const contextMenuItems = computed(() => {
      const tab = rightClickedTab.value;
      if (!tab) return [];
      const index = options.value.findIndex((v) => v.workflow === tab.workflow);
      return [
        {
          label: t("tabMenu.duplicateTab"),
          command: /* @__PURE__ */ __name(() => {
            workflowService.duplicateWorkflow(tab.workflow);
          }, "command")
        },
        {
          separator: true
        },
        {
          label: t("tabMenu.closeTab"),
          command: /* @__PURE__ */ __name(() => onCloseWorkflow(tab), "command")
        },
        {
          label: t("tabMenu.closeTabsToLeft"),
          command: /* @__PURE__ */ __name(() => closeWorkflows(options.value.slice(0, index)), "command"),
          disabled: index <= 0
        },
        {
          label: t("tabMenu.closeTabsToRight"),
          command: /* @__PURE__ */ __name(() => closeWorkflows(options.value.slice(index + 1)), "command"),
          disabled: index === options.value.length - 1
        },
        {
          label: t("tabMenu.closeOtherTabs"),
          command: /* @__PURE__ */ __name(() => closeWorkflows([
            ...options.value.slice(index + 1),
            ...options.value.slice(0, index)
          ]), "command"),
          disabled: options.value.length <= 1
        }
      ];
    });
    const commandStore = useCommandStore();
    return (_ctx, _cache) => {
      const _directive_tooltip = resolveDirective("tooltip");
      return openBlock(), createElementBlock(Fragment, null, [
        createVNode(unref(script$x), {
          class: normalizeClass(["workflow-tabs bg-transparent inline", props.class]),
          modelValue: selectedWorkflow.value,
          "onUpdate:modelValue": onWorkflowChange,
          options: options.value,
          optionLabel: "label",
          dataKey: "value"
        }, {
          option: withCtx(({ option: option2 }) => [
            createBaseVNode("div", {
              class: "flex p-2 gap-2",
              onContextmenu: /* @__PURE__ */ __name(($event) => showContextMenu($event, option2), "onContextmenu"),
              onMouseup: withModifiers(($event) => onCloseWorkflow(option2), ["middle"])
            }, [
              withDirectives((openBlock(), createElementBlock("span", _hoisted_2$b, [
                createTextVNode(toDisplayString(option2.workflow.filename), 1)
              ])), [
                [
                  _directive_tooltip,
                  option2.workflow.key,
                  void 0,
                  { bottom: true }
                ]
              ]),
              createBaseVNode("div", _hoisted_3$9, [
                !unref(workspaceStore).shiftDown && (option2.workflow.isModified || !option2.workflow.isPersisted) ? (openBlock(), createElementBlock("span", _hoisted_4$2, "•")) : createCommentVNode("", true),
                createVNode(unref(script$f), {
                  class: "close-button p-0 w-auto",
                  icon: "pi pi-times",
                  text: "",
                  severity: "secondary",
                  size: "small",
                  onClick: withModifiers(($event) => onCloseWorkflow(option2), ["stop"])
                }, null, 8, ["onClick"])
              ])
            ], 40, _hoisted_1$d)
          ]),
          _: 1
        }, 8, ["class", "modelValue", "options"]),
        createVNode(unref(script$f), {
          class: "new-blank-workflow-button",
          icon: "pi pi-plus",
          text: "",
          severity: "secondary",
          onClick: _cache[0] || (_cache[0] = () => unref(commandStore).execute("Comfy.NewBlankWorkflow"))
        }),
        createVNode(unref(script$y), {
          ref_key: "menu",
          ref: menu,
          model: contextMenuItems.value
        }, null, 8, ["model"])
      ], 64);
    };
  }
});
const WorkflowTabs = /* @__PURE__ */ _export_sfc(_sfc_main$7, [["__scopeId", "data-v-4cb762cb"]]);
var theme$3 = /* @__PURE__ */ __name(function theme6(_ref) {
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
  root: /* @__PURE__ */ __name(function root8(_ref3) {
    var instance = _ref3.instance;
    return ["p-menubar p-component", {
      "p-menubar-mobile": instance.queryMatches,
      "p-menubar-mobile-active": instance.mobileActive
    }];
  }, "root"),
  start: "p-menubar-start",
  button: "p-menubar-button",
  rootList: "p-menubar-root-list",
  item: /* @__PURE__ */ __name(function item(_ref4) {
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
  "extends": script$g,
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
  provide: /* @__PURE__ */ __name(function provide10() {
    return {
      $pcMenubar: this,
      $parentInstance: this
    };
  }, "provide")
};
var script$1$3 = {
  name: "MenubarSub",
  hostName: "Menubar",
  "extends": script$g,
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
    getPTOptions: /* @__PURE__ */ __name(function getPTOptions4(processedItem, index, key) {
      return this.ptm(key, {
        context: {
          item: processedItem.item,
          index,
          active: this.isItemActive(processedItem),
          focused: this.isItemFocused(processedItem),
          disabled: this.isItemDisabled(processedItem),
          level: this.level
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
        processedItem
      });
    }, "onItemMouseMove"),
    getAriaPosInset: /* @__PURE__ */ __name(function getAriaPosInset2(index) {
      return index - this.calculateAriaSetSize.slice(0, index).length + 1;
    }, "getAriaPosInset"),
    getMenuItemProps: /* @__PURE__ */ __name(function getMenuItemProps(processedItem, index) {
      return {
        action: mergeProps({
          "class": this.cx("itemLink"),
          tabindex: -1,
          "aria-hidden": true
        }, this.getPTOptions(processedItem, index, "itemLink")),
        icon: mergeProps({
          "class": [this.cx("itemIcon"), this.getItemProp(processedItem, "icon")]
        }, this.getPTOptions(processedItem, index, "itemIcon")),
        label: mergeProps({
          "class": this.cx("itemLabel")
        }, this.getPTOptions(processedItem, index, "itemLabel")),
        submenuicon: mergeProps({
          "class": this.cx("submenuIcon")
        }, this.getPTOptions(processedItem, index, "submenuIcon"))
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
    getAriaSetSize: /* @__PURE__ */ __name(function getAriaSetSize() {
      var _this2 = this;
      return this.items.filter(function(processedItem) {
        return _this2.isItemVisible(processedItem) && !_this2.getItemProp(processedItem, "separator");
      }).length;
    }, "getAriaSetSize")
  },
  components: {
    AngleRightIcon: script$z,
    AngleDownIcon: script$A
  },
  directives: {
    ripple: Ripple
  }
};
var _hoisted_1$1$2 = ["id", "aria-label", "aria-disabled", "aria-expanded", "aria-haspopup", "aria-level", "aria-setsize", "aria-posinset", "data-p-active", "data-p-focused", "data-p-disabled"];
var _hoisted_2$a = ["onClick", "onMouseenter", "onMousemove"];
var _hoisted_3$8 = ["href", "target"];
var _hoisted_4$1 = ["id"];
var _hoisted_5$1 = ["id"];
function render$1$2(_ctx, _cache, $props, $setup, $data, $options) {
  var _component_MenubarSub = resolveComponent("MenubarSub", true);
  var _directive_ripple = resolveDirective("ripple");
  return openBlock(), createElementBlock("ul", mergeProps({
    "class": $props.level === 0 ? _ctx.cx("rootList") : _ctx.cx("submenu")
  }, $props.level === 0 ? _ctx.ptm("rootList") : _ctx.ptm("submenu")), [(openBlock(true), createElementBlock(Fragment, null, renderList($props.items, function(processedItem, index) {
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
      "aria-posinset": $options.getAriaPosInset(index),
      ref_for: true
    }, $options.getPTOptions(processedItem, index, "item"), {
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
    }, $options.getPTOptions(processedItem, index, "itemContent")), [!$props.templates.item ? withDirectives((openBlock(), createElementBlock("a", mergeProps({
      key: 0,
      href: $options.getItemProp(processedItem, "url"),
      "class": _ctx.cx("itemLink"),
      target: $options.getItemProp(processedItem, "target"),
      tabindex: "-1",
      ref_for: true
    }, $options.getPTOptions(processedItem, index, "itemLink")), [$props.templates.itemicon ? (openBlock(), createBlock(resolveDynamicComponent($props.templates.itemicon), {
      key: 0,
      item: processedItem.item,
      "class": normalizeClass(_ctx.cx("itemIcon"))
    }, null, 8, ["item", "class"])) : $options.getItemProp(processedItem, "icon") ? (openBlock(), createElementBlock("span", mergeProps({
      key: 1,
      "class": [_ctx.cx("itemIcon"), $options.getItemProp(processedItem, "icon")],
      ref_for: true
    }, $options.getPTOptions(processedItem, index, "itemIcon")), null, 16)) : createCommentVNode("", true), createBaseVNode("span", mergeProps({
      id: $options.getItemLabelId(processedItem),
      "class": _ctx.cx("itemLabel"),
      ref_for: true
    }, $options.getPTOptions(processedItem, index, "itemLabel")), toDisplayString($options.getItemLabel(processedItem)), 17, _hoisted_4$1), $options.getItemProp(processedItem, "items") ? (openBlock(), createElementBlock(Fragment, {
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
    }, $options.getPTOptions(processedItem, index, "submenuIcon")), null, 16, ["class"]))], 64)) : createCommentVNode("", true)], 16, _hoisted_3$8)), [[_directive_ripple]]) : (openBlock(), createBlock(resolveDynamicComponent($props.templates.item), {
      key: 1,
      item: processedItem.item,
      root: $props.root,
      hasSubmenu: $options.getItemProp(processedItem, "items"),
      label: $options.getItemLabel(processedItem),
      props: $options.getMenuItemProps(processedItem, index)
    }, null, 8, ["item", "root", "hasSubmenu", "label", "props"]))], 16, _hoisted_2$a), $options.isItemVisible(processedItem) && $options.isItemGroup(processedItem) ? (openBlock(), createBlock(_component_MenubarSub, {
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
__name(render$1$2, "render$1$2");
script$1$3.render = render$1$2;
var script$5 = {
  name: "Menubar",
  "extends": script$2$1,
  inheritAttrs: false,
  emits: ["focus", "blur"],
  matchMediaListener: null,
  data: /* @__PURE__ */ __name(function data6() {
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
    "$attrs.id": /* @__PURE__ */ __name(function $attrsId2(newValue) {
      this.id = newValue || UniqueComponentId();
    }, "$attrsId"),
    activeItemPath: /* @__PURE__ */ __name(function activeItemPath(newPath) {
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
  mounted: /* @__PURE__ */ __name(function mounted6() {
    this.id = this.id || UniqueComponentId();
    this.bindMatchMediaListener();
  }, "mounted"),
  beforeUnmount: /* @__PURE__ */ __name(function beforeUnmount6() {
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
    getItemProp: /* @__PURE__ */ __name(function getItemProp2(item3, name) {
      return item3 ? resolve(item3[name]) : void 0;
    }, "getItemProp"),
    getItemLabel: /* @__PURE__ */ __name(function getItemLabel2(item3) {
      return this.getItemProp(item3, "label");
    }, "getItemLabel"),
    isItemDisabled: /* @__PURE__ */ __name(function isItemDisabled2(item3) {
      return this.getItemProp(item3, "disabled");
    }, "isItemDisabled"),
    isItemVisible: /* @__PURE__ */ __name(function isItemVisible2(item3) {
      return this.getItemProp(item3, "visible") !== false;
    }, "isItemVisible"),
    isItemGroup: /* @__PURE__ */ __name(function isItemGroup2(item3) {
      return isNotEmpty(this.getItemProp(item3, "items"));
    }, "isItemGroup"),
    isItemSeparator: /* @__PURE__ */ __name(function isItemSeparator(item3) {
      return this.getItemProp(item3, "separator");
    }, "isItemSeparator"),
    getProccessedItemLabel: /* @__PURE__ */ __name(function getProccessedItemLabel(processedItem) {
      return processedItem ? this.getItemLabel(processedItem.item) : void 0;
    }, "getProccessedItemLabel"),
    isProccessedItemGroup: /* @__PURE__ */ __name(function isProccessedItemGroup(processedItem) {
      return processedItem && isNotEmpty(processedItem.items);
    }, "isProccessedItemGroup"),
    toggle: /* @__PURE__ */ __name(function toggle(event) {
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
    show: /* @__PURE__ */ __name(function show2() {
      focus(this.menubar);
    }, "show"),
    hide: /* @__PURE__ */ __name(function hide2(event, isFocus) {
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
    onBlur: /* @__PURE__ */ __name(function onBlur2(event) {
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
      var index = processedItem.index, key = processedItem.key, level = processedItem.level, parentKey = processedItem.parentKey, items = processedItem.items;
      var grouped = isNotEmpty(items);
      var activeItemPath3 = this.activeItemPath.filter(function(p) {
        return p.parentKey !== parentKey && p.parentKey !== key;
      });
      grouped && activeItemPath3.push(processedItem);
      this.focusedItemInfo = {
        index,
        level,
        parentKey
      };
      this.activeItemPath = activeItemPath3;
      grouped && (this.dirty = true);
      isFocus && focus(this.menubar);
    }, "onItemChange"),
    onItemClick: /* @__PURE__ */ __name(function onItemClick2(event) {
      var originalEvent = event.originalEvent, processedItem = event.processedItem;
      var grouped = this.isProccessedItemGroup(processedItem);
      var root12 = isEmpty(processedItem.parent);
      var selected = this.isSelected(processedItem);
      if (selected) {
        var index = processedItem.index, key = processedItem.key, level = processedItem.level, parentKey = processedItem.parentKey;
        this.activeItemPath = this.activeItemPath.filter(function(p) {
          return key !== p.key && key.startsWith(p.key);
        });
        this.focusedItemInfo = {
          index,
          level,
          parentKey
        };
        this.dirty = !root12;
        focus(this.menubar);
      } else {
        if (grouped) {
          this.onItemChange(event);
        } else {
          var rootProcessedItem = root12 ? processedItem : this.activeItemPath.find(function(p) {
            return p.parentKey === "";
          });
          this.hide(originalEvent);
          this.changeFocusedItemIndex(originalEvent, rootProcessedItem ? rootProcessedItem.index : -1);
          this.mobileActive = false;
          focus(this.menubar);
        }
      }
    }, "onItemClick"),
    onItemMouseEnter: /* @__PURE__ */ __name(function onItemMouseEnter2(event) {
      if (this.dirty) {
        this.onItemChange(event);
      }
    }, "onItemMouseEnter"),
    onItemMouseMove: /* @__PURE__ */ __name(function onItemMouseMove2(event) {
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
    onArrowDownKey: /* @__PURE__ */ __name(function onArrowDownKey2(event) {
      var processedItem = this.visibleItems[this.focusedItemInfo.index];
      var root12 = processedItem ? isEmpty(processedItem.parent) : null;
      if (root12) {
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
    onArrowUpKey: /* @__PURE__ */ __name(function onArrowUpKey2(event) {
      var _this3 = this;
      var processedItem = this.visibleItems[this.focusedItemInfo.index];
      var root12 = isEmpty(processedItem.parent);
      if (root12) {
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
    onHomeKey: /* @__PURE__ */ __name(function onHomeKey3(event) {
      this.changeFocusedItemIndex(event, this.findFirstItemIndex());
      event.preventDefault();
    }, "onHomeKey"),
    onEndKey: /* @__PURE__ */ __name(function onEndKey3(event) {
      this.changeFocusedItemIndex(event, this.findLastItemIndex());
      event.preventDefault();
    }, "onEndKey"),
    onEnterKey: /* @__PURE__ */ __name(function onEnterKey3(event) {
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
    onSpaceKey: /* @__PURE__ */ __name(function onSpaceKey(event) {
      this.onEnterKey(event);
    }, "onSpaceKey"),
    onEscapeKey: /* @__PURE__ */ __name(function onEscapeKey2(event) {
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
    bindOutsideClickListener: /* @__PURE__ */ __name(function bindOutsideClickListener2() {
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
    unbindOutsideClickListener: /* @__PURE__ */ __name(function unbindOutsideClickListener2() {
      if (this.outsideClickListener) {
        document.removeEventListener("click", this.outsideClickListener);
        this.outsideClickListener = null;
      }
    }, "unbindOutsideClickListener"),
    bindResizeListener: /* @__PURE__ */ __name(function bindResizeListener2() {
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
    unbindResizeListener: /* @__PURE__ */ __name(function unbindResizeListener2() {
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
    isSelected: /* @__PURE__ */ __name(function isSelected2(processedItem) {
      return this.activeItemPath.some(function(p) {
        return p.key === processedItem.key;
      });
    }, "isSelected"),
    findFirstItemIndex: /* @__PURE__ */ __name(function findFirstItemIndex() {
      var _this8 = this;
      return this.visibleItems.findIndex(function(processedItem) {
        return _this8.isValidItem(processedItem);
      });
    }, "findFirstItemIndex"),
    findLastItemIndex: /* @__PURE__ */ __name(function findLastItemIndex() {
      var _this9 = this;
      return findLastIndex(this.visibleItems, function(processedItem) {
        return _this9.isValidItem(processedItem);
      });
    }, "findLastItemIndex"),
    findNextItemIndex: /* @__PURE__ */ __name(function findNextItemIndex(index) {
      var _this10 = this;
      var matchedItemIndex = index < this.visibleItems.length - 1 ? this.visibleItems.slice(index + 1).findIndex(function(processedItem) {
        return _this10.isValidItem(processedItem);
      }) : -1;
      return matchedItemIndex > -1 ? matchedItemIndex + index + 1 : index;
    }, "findNextItemIndex"),
    findPrevItemIndex: /* @__PURE__ */ __name(function findPrevItemIndex(index) {
      var _this11 = this;
      var matchedItemIndex = index > 0 ? findLastIndex(this.visibleItems.slice(0, index), function(processedItem) {
        return _this11.isValidItem(processedItem);
      }) : -1;
      return matchedItemIndex > -1 ? matchedItemIndex : index;
    }, "findPrevItemIndex"),
    findSelectedItemIndex: /* @__PURE__ */ __name(function findSelectedItemIndex() {
      var _this12 = this;
      return this.visibleItems.findIndex(function(processedItem) {
        return _this12.isValidSelectedItem(processedItem);
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
    changeFocusedItemIndex: /* @__PURE__ */ __name(function changeFocusedItemIndex(event, index) {
      if (this.focusedItemInfo.index !== index) {
        this.focusedItemInfo.index = index;
        this.scrollInView();
      }
    }, "changeFocusedItemIndex"),
    scrollInView: /* @__PURE__ */ __name(function scrollInView3() {
      var index = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : -1;
      var id2 = index !== -1 ? "".concat(this.id, "_").concat(index) : this.focusedItemId;
      var element = findSingle(this.menubar, 'li[id="'.concat(id2, '"]'));
      if (element) {
        element.scrollIntoView && element.scrollIntoView({
          block: "nearest",
          inline: "start"
        });
      }
    }, "scrollInView"),
    createProcessedItems: /* @__PURE__ */ __name(function createProcessedItems(items) {
      var _this14 = this;
      var level = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : 0;
      var parent = arguments.length > 2 && arguments[2] !== void 0 ? arguments[2] : {};
      var parentKey = arguments.length > 3 && arguments[3] !== void 0 ? arguments[3] : "";
      var processedItems3 = [];
      items && items.forEach(function(item3, index) {
        var key = (parentKey !== "" ? parentKey + "_" : "") + index;
        var newItem = {
          item: item3,
          index,
          level,
          key,
          parent,
          parentKey
        };
        newItem["items"] = _this14.createProcessedItems(item3.items, level + 1, newItem, key);
        processedItems3.push(newItem);
      });
      return processedItems3;
    }, "createProcessedItems"),
    containerRef: /* @__PURE__ */ __name(function containerRef(el) {
      this.container = el;
    }, "containerRef"),
    menubarRef: /* @__PURE__ */ __name(function menubarRef(el) {
      this.menubar = el ? el.$el : void 0;
    }, "menubarRef")
  },
  computed: {
    processedItems: /* @__PURE__ */ __name(function processedItems() {
      return this.createProcessedItems(this.model || []);
    }, "processedItems"),
    visibleItems: /* @__PURE__ */ __name(function visibleItems() {
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
    BarsIcon: script$B
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
var _hoisted_1$c = ["aria-haspopup", "aria-expanded", "aria-controls", "aria-label"];
function render$9(_ctx, _cache, $props, $setup, $data, $options) {
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
    })], 16, _hoisted_1$c)) : createCommentVNode("", true)];
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
__name(render$9, "render$9");
script$5.render = render$9;
const _withScopeId$2 = /* @__PURE__ */ __name((n) => (pushScopeId("data-v-a2b12676"), n = n(), popScopeId(), n), "_withScopeId$2");
const _hoisted_1$b = ["href"];
const _hoisted_2$9 = { class: "p-menubar-item-label" };
const _hoisted_3$7 = {
  key: 1,
  class: "ml-auto border border-surface rounded text-muted text-xs p-1 keybinding-tag"
};
const _sfc_main$6 = /* @__PURE__ */ defineComponent({
  __name: "CommandMenubar",
  setup(__props) {
    const settingStore = useSettingStore();
    const dropdownDirection = computed(
      () => settingStore.get("Comfy.UseNewMenu") === "Top" ? "down" : "up"
    );
    const menuItemsStore = useMenuItemStore();
    const { t } = useI18n();
    const translateMenuItem = /* @__PURE__ */ __name((item3) => {
      const label = typeof item3.label === "function" ? item3.label() : item3.label;
      const translatedLabel = label ? t(`menuLabels.${normalizeI18nKey(label)}`, label) : void 0;
      return {
        ...item3,
        label: translatedLabel,
        items: item3.items?.map(translateMenuItem)
      };
    }, "translateMenuItem");
    const translatedItems = computed(
      () => menuItemsStore.menuItems.map(translateMenuItem)
    );
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(script$5), {
        model: translatedItems.value,
        class: "top-menubar border-none p-0 bg-transparent",
        pt: {
          rootList: "gap-0 flex-nowrap w-auto",
          submenu: `dropdown-direction-${dropdownDirection.value}`,
          item: "relative"
        }
      }, {
        item: withCtx(({ item: item3, props }) => [
          createBaseVNode("a", mergeProps({ class: "p-menubar-item-link" }, props.action, {
            href: item3.url,
            target: "_blank"
          }), [
            item3.icon ? (openBlock(), createElementBlock("span", {
              key: 0,
              class: normalizeClass(["p-menubar-item-icon", item3.icon])
            }, null, 2)) : createCommentVNode("", true),
            createBaseVNode("span", _hoisted_2$9, toDisplayString(item3.label), 1),
            item3?.comfyCommand?.keybinding ? (openBlock(), createElementBlock("span", _hoisted_3$7, toDisplayString(item3.comfyCommand.keybinding.combo.toString()), 1)) : createCommentVNode("", true)
          ], 16, _hoisted_1$b)
        ]),
        _: 1
      }, 8, ["model", "pt"]);
    };
  }
});
const CommandMenubar = /* @__PURE__ */ _export_sfc(_sfc_main$6, [["__scopeId", "data-v-a2b12676"]]);
var theme$2 = /* @__PURE__ */ __name(function theme7(_ref) {
  var dt = _ref.dt;
  return "\n.p-panel {\n    border: 1px solid ".concat(dt("panel.border.color"), ";\n    border-radius: ").concat(dt("panel.border.radius"), ";\n    background: ").concat(dt("panel.background"), ";\n    color: ").concat(dt("panel.color"), ";\n}\n\n.p-panel-header {\n    display: flex;\n    justify-content: space-between;\n    align-items: center;\n    padding: ").concat(dt("panel.header.padding"), ";\n    background: ").concat(dt("panel.header.background"), ";\n    color: ").concat(dt("panel.header.color"), ";\n    border-style: solid;\n    border-width: ").concat(dt("panel.header.border.width"), ";\n    border-color: ").concat(dt("panel.header.border.color"), ";\n    border-radius: ").concat(dt("panel.header.border.radius"), ";\n}\n\n.p-panel-toggleable .p-panel-header {\n    padding: ").concat(dt("panel.toggleable.header.padding"), ";\n}\n\n.p-panel-title {\n    line-height: 1;\n    font-weight: ").concat(dt("panel.title.font.weight"), ";\n}\n\n.p-panel-content {\n    padding: ").concat(dt("panel.content.padding"), ";\n}\n\n.p-panel-footer {\n    padding: ").concat(dt("panel.footer.padding"), ";\n}\n");
}, "theme");
var classes$2 = {
  root: /* @__PURE__ */ __name(function root9(_ref2) {
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
  "extends": script$g,
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
  provide: /* @__PURE__ */ __name(function provide11() {
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
  data: /* @__PURE__ */ __name(function data7() {
    return {
      id: this.$attrs.id,
      d_collapsed: this.collapsed
    };
  }, "data"),
  watch: {
    "$attrs.id": /* @__PURE__ */ __name(function $attrsId3(newValue) {
      this.id = newValue || UniqueComponentId();
    }, "$attrsId"),
    collapsed: /* @__PURE__ */ __name(function collapsed(newValue) {
      this.d_collapsed = newValue;
    }, "collapsed")
  },
  mounted: /* @__PURE__ */ __name(function mounted7() {
    this.id = this.id || UniqueComponentId();
  }, "mounted"),
  methods: {
    toggle: /* @__PURE__ */ __name(function toggle2(event) {
      this.d_collapsed = !this.d_collapsed;
      this.$emit("update:collapsed", this.d_collapsed);
      this.$emit("toggle", {
        originalEvent: event,
        value: this.d_collapsed
      });
    }, "toggle"),
    onKeyDown: /* @__PURE__ */ __name(function onKeyDown3(event) {
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
    PlusIcon: script$C,
    MinusIcon: script$D,
    Button: script$f
  },
  directives: {
    ripple: Ripple
  }
};
var _hoisted_1$a = ["id"];
var _hoisted_2$8 = ["id", "aria-labelledby"];
function render$8(_ctx, _cache, $props, $setup, $data, $options) {
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
    }, _ctx.ptm("title")), toDisplayString(_ctx.header), 17, _hoisted_1$a)) : createCommentVNode("", true)];
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
      }, _ctx.ptm("footer")), [renderSlot(_ctx.$slots, "footer")], 16)) : createCommentVNode("", true)], 16, _hoisted_2$8), [[vShow, !$data.d_collapsed]])];
    }),
    _: 3
  }, 16)], 16);
}
__name(render$8, "render$8");
script$4.render = render$8;
const _hoisted_1$9 = {
  viewBox: "0 0 24 24",
  width: "1.2em",
  height: "1.2em"
};
const _hoisted_2$7 = /* @__PURE__ */ createBaseVNode("path", {
  fill: "none",
  stroke: "currentColor",
  "stroke-linecap": "round",
  "stroke-linejoin": "round",
  "stroke-width": "2",
  d: "M6 4v16m4-16l10 8l-10 8z"
}, null, -1);
const _hoisted_3$6 = [
  _hoisted_2$7
];
function render$7(_ctx, _cache) {
  return openBlock(), createElementBlock("svg", _hoisted_1$9, [..._hoisted_3$6]);
}
__name(render$7, "render$7");
const __unplugin_components_3 = markRaw({ name: "lucide-step-forward", render: render$7 });
const _hoisted_1$8 = {
  viewBox: "0 0 24 24",
  width: "1.2em",
  height: "1.2em"
};
const _hoisted_2$6 = /* @__PURE__ */ createBaseVNode("path", {
  fill: "none",
  stroke: "currentColor",
  "stroke-linecap": "round",
  "stroke-linejoin": "round",
  "stroke-width": "2",
  d: "m13 19l9-7l-9-7zM2 19l9-7l-9-7z"
}, null, -1);
const _hoisted_3$5 = [
  _hoisted_2$6
];
function render$6(_ctx, _cache) {
  return openBlock(), createElementBlock("svg", _hoisted_1$8, [..._hoisted_3$5]);
}
__name(render$6, "render$6");
const __unplugin_components_2 = markRaw({ name: "lucide-fast-forward", render: render$6 });
const _hoisted_1$7 = {
  viewBox: "0 0 24 24",
  width: "1.2em",
  height: "1.2em"
};
const _hoisted_2$5 = /* @__PURE__ */ createBaseVNode("path", {
  fill: "none",
  stroke: "currentColor",
  "stroke-linecap": "round",
  "stroke-linejoin": "round",
  "stroke-width": "2",
  d: "m6 3l14 9l-14 9z"
}, null, -1);
const _hoisted_3$4 = [
  _hoisted_2$5
];
function render$5(_ctx, _cache) {
  return openBlock(), createElementBlock("svg", _hoisted_1$7, [..._hoisted_3$4]);
}
__name(render$5, "render$5");
const __unplugin_components_1$1 = markRaw({ name: "lucide-play", render: render$5 });
const _hoisted_1$6 = {
  viewBox: "0 0 24 24",
  width: "1.2em",
  height: "1.2em"
};
const _hoisted_2$4 = /* @__PURE__ */ createBaseVNode("g", {
  fill: "none",
  stroke: "currentColor",
  "stroke-linecap": "round",
  "stroke-linejoin": "round",
  "stroke-width": "2"
}, [
  /* @__PURE__ */ createBaseVNode("path", { d: "M16 12H3m13 6H3m7-12H3m18 12V8a2 2 0 0 0-2-2h-5" }),
  /* @__PURE__ */ createBaseVNode("path", { d: "m16 8l-2-2l2-2" })
], -1);
const _hoisted_3$3 = [
  _hoisted_2$4
];
function render$4(_ctx, _cache) {
  return openBlock(), createElementBlock("svg", _hoisted_1$6, [..._hoisted_3$3]);
}
__name(render$4, "render$4");
const __unplugin_components_0$1 = markRaw({ name: "lucide-list-start", render: render$4 });
var theme$1 = /* @__PURE__ */ __name(function theme8(_ref) {
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
  root: /* @__PURE__ */ __name(function root10(_ref3) {
    _ref3.instance;
    var props = _ref3.props;
    return ["p-tieredmenu p-component", {
      "p-tieredmenu-overlay": props.popup
    }];
  }, "root"),
  start: "p-tieredmenu-start",
  rootList: "p-tieredmenu-root-list",
  item: /* @__PURE__ */ __name(function item2(_ref4) {
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
  "extends": script$g,
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
  provide: /* @__PURE__ */ __name(function provide12() {
    return {
      $pcTieredMenu: this,
      $parentInstance: this
    };
  }, "provide")
};
var script$1$1 = {
  name: "TieredMenuSub",
  hostName: "TieredMenu",
  "extends": script$g,
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
    getPTOptions: /* @__PURE__ */ __name(function getPTOptions5(processedItem, index, key) {
      return this.ptm(key, {
        context: {
          item: processedItem.item,
          index,
          active: this.isItemActive(processedItem),
          focused: this.isItemFocused(processedItem),
          disabled: this.isItemDisabled(processedItem)
        }
      });
    }, "getPTOptions"),
    isItemActive: /* @__PURE__ */ __name(function isItemActive2(processedItem) {
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
    onEnter: /* @__PURE__ */ __name(function onEnter2() {
      nestedPosition(this.container, this.level);
    }, "onEnter"),
    onItemClick: /* @__PURE__ */ __name(function onItemClick3(event, processedItem) {
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
    getAriaSetSize: /* @__PURE__ */ __name(function getAriaSetSize2() {
      var _this = this;
      return this.items.filter(function(processedItem) {
        return _this.isItemVisible(processedItem) && !_this.getItemProp(processedItem, "separator");
      }).length;
    }, "getAriaSetSize"),
    getAriaPosInset: /* @__PURE__ */ __name(function getAriaPosInset3(index) {
      var _this2 = this;
      return index - this.items.slice(0, index).filter(function(processedItem) {
        return _this2.isItemVisible(processedItem) && _this2.getItemProp(processedItem, "separator");
      }).length + 1;
    }, "getAriaPosInset"),
    getMenuItemProps: /* @__PURE__ */ __name(function getMenuItemProps2(processedItem, index) {
      return {
        action: mergeProps({
          "class": this.cx("itemLink"),
          tabindex: -1,
          "aria-hidden": true
        }, this.getPTOptions(processedItem, index, "itemLink")),
        icon: mergeProps({
          "class": [this.cx("itemIcon"), this.getItemProp(processedItem, "icon")]
        }, this.getPTOptions(processedItem, index, "itemIcon")),
        label: mergeProps({
          "class": this.cx("itemLabel")
        }, this.getPTOptions(processedItem, index, "itemLabel")),
        submenuicon: mergeProps({
          "class": this.cx("submenuIcon")
        }, this.getPTOptions(processedItem, index, "submenuIcon"))
      };
    }, "getMenuItemProps"),
    containerRef: /* @__PURE__ */ __name(function containerRef2(el) {
      this.container = el;
    }, "containerRef")
  },
  components: {
    AngleRightIcon: script$z
  },
  directives: {
    ripple: Ripple
  }
};
var _hoisted_1$1$1 = ["tabindex"];
var _hoisted_2$3 = ["id", "aria-label", "aria-disabled", "aria-expanded", "aria-haspopup", "aria-level", "aria-setsize", "aria-posinset", "data-p-active", "data-p-focused", "data-p-disabled"];
var _hoisted_3$2 = ["onClick", "onMouseenter", "onMousemove"];
var _hoisted_4 = ["href", "target"];
var _hoisted_5 = ["id"];
var _hoisted_6 = ["id"];
function render$1$1(_ctx, _cache, $props, $setup, $data, $options) {
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
      }, $props.level === 0 ? _ctx.ptm("rootList") : _ctx.ptm("submenu")), [(openBlock(true), createElementBlock(Fragment, null, renderList($props.items, function(processedItem, index) {
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
          "aria-posinset": $options.getAriaPosInset(index),
          ref_for: true
        }, $options.getPTOptions(processedItem, index, "item"), {
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
        }, $options.getPTOptions(processedItem, index, "itemContent")), [!$props.templates.item ? withDirectives((openBlock(), createElementBlock("a", mergeProps({
          key: 0,
          href: $options.getItemProp(processedItem, "url"),
          "class": _ctx.cx("itemLink"),
          target: $options.getItemProp(processedItem, "target"),
          tabindex: "-1",
          ref_for: true
        }, $options.getPTOptions(processedItem, index, "itemLink")), [$props.templates.itemicon ? (openBlock(), createBlock(resolveDynamicComponent($props.templates.itemicon), {
          key: 0,
          item: processedItem.item,
          "class": normalizeClass(_ctx.cx("itemIcon"))
        }, null, 8, ["item", "class"])) : $options.getItemProp(processedItem, "icon") ? (openBlock(), createElementBlock("span", mergeProps({
          key: 1,
          "class": [_ctx.cx("itemIcon"), $options.getItemProp(processedItem, "icon")],
          ref_for: true
        }, $options.getPTOptions(processedItem, index, "itemIcon")), null, 16)) : createCommentVNode("", true), createBaseVNode("span", mergeProps({
          id: $options.getItemLabelId(processedItem),
          "class": _ctx.cx("itemLabel"),
          ref_for: true
        }, $options.getPTOptions(processedItem, index, "itemLabel")), toDisplayString($options.getItemLabel(processedItem)), 17, _hoisted_5), $options.getItemProp(processedItem, "items") ? (openBlock(), createElementBlock(Fragment, {
          key: 2
        }, [$props.templates.submenuicon ? (openBlock(), createBlock(resolveDynamicComponent($props.templates.submenuicon), mergeProps({
          key: 0,
          "class": _ctx.cx("submenuIcon"),
          active: $options.isItemActive(processedItem),
          ref_for: true
        }, $options.getPTOptions(processedItem, index, "submenuIcon")), null, 16, ["class", "active"])) : (openBlock(), createBlock(_component_AngleRightIcon, mergeProps({
          key: 1,
          "class": _ctx.cx("submenuIcon"),
          ref_for: true
        }, $options.getPTOptions(processedItem, index, "submenuIcon")), null, 16, ["class"]))], 64)) : createCommentVNode("", true)], 16, _hoisted_4)), [[_directive_ripple]]) : (openBlock(), createBlock(resolveDynamicComponent($props.templates.item), {
          key: 1,
          item: processedItem.item,
          hasSubmenu: $options.getItemProp(processedItem, "items"),
          label: $options.getItemLabel(processedItem),
          props: $options.getMenuItemProps(processedItem, index)
        }, null, 8, ["item", "hasSubmenu", "label", "props"]))], 16, _hoisted_3$2), $options.isItemVisible(processedItem) && $options.isItemGroup(processedItem) ? (openBlock(), createBlock(_component_TieredMenuSub, {
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
        }, null, 8, ["id", "style", "aria-labelledby", "menuId", "focusedItemId", "items", "templates", "activeItemPath", "level", "visible", "pt", "unstyled"])) : createCommentVNode("", true)], 16, _hoisted_2$3)) : createCommentVNode("", true), $options.isItemVisible(processedItem) && $options.getItemProp(processedItem, "separator") ? (openBlock(), createElementBlock("li", mergeProps({
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
__name(render$1$1, "render$1$1");
script$1$1.render = render$1$1;
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
  data: /* @__PURE__ */ __name(function data8() {
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
    "$attrs.id": /* @__PURE__ */ __name(function $attrsId4(newValue) {
      this.id = newValue || UniqueComponentId();
    }, "$attrsId"),
    activeItemPath: /* @__PURE__ */ __name(function activeItemPath2(newPath) {
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
  mounted: /* @__PURE__ */ __name(function mounted8() {
    this.id = this.id || UniqueComponentId();
  }, "mounted"),
  beforeUnmount: /* @__PURE__ */ __name(function beforeUnmount7() {
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
    getItemProp: /* @__PURE__ */ __name(function getItemProp4(item3, name) {
      return item3 ? resolve(item3[name]) : void 0;
    }, "getItemProp"),
    getItemLabel: /* @__PURE__ */ __name(function getItemLabel4(item3) {
      return this.getItemProp(item3, "label");
    }, "getItemLabel"),
    isItemDisabled: /* @__PURE__ */ __name(function isItemDisabled4(item3) {
      return this.getItemProp(item3, "disabled");
    }, "isItemDisabled"),
    isItemVisible: /* @__PURE__ */ __name(function isItemVisible4(item3) {
      return this.getItemProp(item3, "visible") !== false;
    }, "isItemVisible"),
    isItemGroup: /* @__PURE__ */ __name(function isItemGroup4(item3) {
      return isNotEmpty(this.getItemProp(item3, "items"));
    }, "isItemGroup"),
    isItemSeparator: /* @__PURE__ */ __name(function isItemSeparator2(item3) {
      return this.getItemProp(item3, "separator");
    }, "isItemSeparator"),
    getProccessedItemLabel: /* @__PURE__ */ __name(function getProccessedItemLabel2(processedItem) {
      return processedItem ? this.getItemLabel(processedItem.item) : void 0;
    }, "getProccessedItemLabel"),
    isProccessedItemGroup: /* @__PURE__ */ __name(function isProccessedItemGroup2(processedItem) {
      return processedItem && isNotEmpty(processedItem.items);
    }, "isProccessedItemGroup"),
    toggle: /* @__PURE__ */ __name(function toggle3(event) {
      this.visible ? this.hide(event, true) : this.show(event);
    }, "toggle"),
    show: /* @__PURE__ */ __name(function show3(event, isFocus) {
      if (this.popup) {
        this.$emit("before-show");
        this.visible = true;
        this.target = this.target || event.currentTarget;
        this.relatedTarget = event.relatedTarget || null;
      }
      isFocus && focus(this.menubar);
    }, "show"),
    hide: /* @__PURE__ */ __name(function hide3(event, isFocus) {
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
    onKeyDown: /* @__PURE__ */ __name(function onKeyDown4(event) {
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
    onItemChange: /* @__PURE__ */ __name(function onItemChange2(event) {
      var processedItem = event.processedItem, isFocus = event.isFocus;
      if (isEmpty(processedItem)) return;
      var index = processedItem.index, key = processedItem.key, level = processedItem.level, parentKey = processedItem.parentKey, items = processedItem.items;
      var grouped = isNotEmpty(items);
      var activeItemPath3 = this.activeItemPath.filter(function(p) {
        return p.parentKey !== parentKey && p.parentKey !== key;
      });
      if (grouped) {
        activeItemPath3.push(processedItem);
        this.submenuVisible = true;
      }
      this.focusedItemInfo = {
        index,
        level,
        parentKey
      };
      this.activeItemPath = activeItemPath3;
      grouped && (this.dirty = true);
      isFocus && focus(this.menubar);
    }, "onItemChange"),
    onOverlayClick: /* @__PURE__ */ __name(function onOverlayClick2(event) {
      OverlayEventBus.emit("overlay-click", {
        originalEvent: event,
        target: this.target
      });
    }, "onOverlayClick"),
    onItemClick: /* @__PURE__ */ __name(function onItemClick4(event) {
      var originalEvent = event.originalEvent, processedItem = event.processedItem;
      var grouped = this.isProccessedItemGroup(processedItem);
      var root12 = isEmpty(processedItem.parent);
      var selected = this.isSelected(processedItem);
      if (selected) {
        var index = processedItem.index, key = processedItem.key, level = processedItem.level, parentKey = processedItem.parentKey;
        this.activeItemPath = this.activeItemPath.filter(function(p) {
          return key !== p.key && key.startsWith(p.key);
        });
        this.focusedItemInfo = {
          index,
          level,
          parentKey
        };
        this.dirty = !root12;
        focus(this.menubar);
      } else {
        if (grouped) {
          this.onItemChange(event);
        } else {
          var rootProcessedItem = root12 ? processedItem : this.activeItemPath.find(function(p) {
            return p.parentKey === "";
          });
          this.hide(originalEvent);
          this.changeFocusedItemIndex(originalEvent, rootProcessedItem ? rootProcessedItem.index : -1);
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
    onArrowDownKey: /* @__PURE__ */ __name(function onArrowDownKey3(event) {
      var itemIndex = this.focusedItemInfo.index !== -1 ? this.findNextItemIndex(this.focusedItemInfo.index) : this.findFirstFocusedItemIndex();
      this.changeFocusedItemIndex(event, itemIndex);
      event.preventDefault();
    }, "onArrowDownKey"),
    onArrowUpKey: /* @__PURE__ */ __name(function onArrowUpKey3(event) {
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
      var root12 = isEmpty(processedItem.parent);
      if (!root12) {
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
    onHomeKey: /* @__PURE__ */ __name(function onHomeKey4(event) {
      this.changeFocusedItemIndex(event, this.findFirstItemIndex());
      event.preventDefault();
    }, "onHomeKey"),
    onEndKey: /* @__PURE__ */ __name(function onEndKey4(event) {
      this.changeFocusedItemIndex(event, this.findLastItemIndex());
      event.preventDefault();
    }, "onEndKey"),
    onEnterKey: /* @__PURE__ */ __name(function onEnterKey4(event) {
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
    onSpaceKey: /* @__PURE__ */ __name(function onSpaceKey2(event) {
      this.onEnterKey(event);
    }, "onSpaceKey"),
    onEscapeKey: /* @__PURE__ */ __name(function onEscapeKey3(event) {
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
    onTabKey: /* @__PURE__ */ __name(function onTabKey3(event) {
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
    onAfterEnter: /* @__PURE__ */ __name(function onAfterEnter() {
      this.bindOutsideClickListener();
      this.bindScrollListener();
      this.bindResizeListener();
      this.$emit("show");
    }, "onAfterEnter"),
    onLeave: /* @__PURE__ */ __name(function onLeave2() {
      this.unbindOutsideClickListener();
      this.unbindScrollListener();
      this.unbindResizeListener();
      this.$emit("hide");
      this.container = null;
      this.dirty = false;
    }, "onLeave"),
    onAfterLeave: /* @__PURE__ */ __name(function onAfterLeave(el) {
      if (this.autoZIndex) {
        ZIndex.clear(el);
      }
    }, "onAfterLeave"),
    alignOverlay: /* @__PURE__ */ __name(function alignOverlay2() {
      absolutePosition(this.container, this.target);
      var targetWidth = getOuterWidth(this.target);
      if (targetWidth > getOuterWidth(this.container)) {
        this.container.style.minWidth = getOuterWidth(this.target) + "px";
      }
    }, "alignOverlay"),
    bindOutsideClickListener: /* @__PURE__ */ __name(function bindOutsideClickListener3() {
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
    unbindOutsideClickListener: /* @__PURE__ */ __name(function unbindOutsideClickListener3() {
      if (this.outsideClickListener) {
        document.removeEventListener("click", this.outsideClickListener);
        this.outsideClickListener = null;
      }
    }, "unbindOutsideClickListener"),
    bindScrollListener: /* @__PURE__ */ __name(function bindScrollListener2() {
      var _this3 = this;
      if (!this.scrollHandler) {
        this.scrollHandler = new ConnectedOverlayScrollHandler(this.target, function(event) {
          _this3.hide(event, true);
        });
      }
      this.scrollHandler.bindScrollListener();
    }, "bindScrollListener"),
    unbindScrollListener: /* @__PURE__ */ __name(function unbindScrollListener2() {
      if (this.scrollHandler) {
        this.scrollHandler.unbindScrollListener();
      }
    }, "unbindScrollListener"),
    bindResizeListener: /* @__PURE__ */ __name(function bindResizeListener3() {
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
    unbindResizeListener: /* @__PURE__ */ __name(function unbindResizeListener3() {
      if (this.resizeListener) {
        window.removeEventListener("resize", this.resizeListener);
        this.resizeListener = null;
      }
    }, "unbindResizeListener"),
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
    isSelected: /* @__PURE__ */ __name(function isSelected3(processedItem) {
      return this.activeItemPath.some(function(p) {
        return p.key === processedItem.key;
      });
    }, "isSelected"),
    findFirstItemIndex: /* @__PURE__ */ __name(function findFirstItemIndex2() {
      var _this5 = this;
      return this.visibleItems.findIndex(function(processedItem) {
        return _this5.isValidItem(processedItem);
      });
    }, "findFirstItemIndex"),
    findLastItemIndex: /* @__PURE__ */ __name(function findLastItemIndex2() {
      var _this6 = this;
      return findLastIndex(this.visibleItems, function(processedItem) {
        return _this6.isValidItem(processedItem);
      });
    }, "findLastItemIndex"),
    findNextItemIndex: /* @__PURE__ */ __name(function findNextItemIndex2(index) {
      var _this7 = this;
      var matchedItemIndex = index < this.visibleItems.length - 1 ? this.visibleItems.slice(index + 1).findIndex(function(processedItem) {
        return _this7.isValidItem(processedItem);
      }) : -1;
      return matchedItemIndex > -1 ? matchedItemIndex + index + 1 : index;
    }, "findNextItemIndex"),
    findPrevItemIndex: /* @__PURE__ */ __name(function findPrevItemIndex2(index) {
      var _this8 = this;
      var matchedItemIndex = index > 0 ? findLastIndex(this.visibleItems.slice(0, index), function(processedItem) {
        return _this8.isValidItem(processedItem);
      }) : -1;
      return matchedItemIndex > -1 ? matchedItemIndex : index;
    }, "findPrevItemIndex"),
    findSelectedItemIndex: /* @__PURE__ */ __name(function findSelectedItemIndex2() {
      var _this9 = this;
      return this.visibleItems.findIndex(function(processedItem) {
        return _this9.isValidSelectedItem(processedItem);
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
    changeFocusedItemIndex: /* @__PURE__ */ __name(function changeFocusedItemIndex2(event, index) {
      if (this.focusedItemInfo.index !== index) {
        this.focusedItemInfo.index = index;
        this.scrollInView();
      }
    }, "changeFocusedItemIndex"),
    scrollInView: /* @__PURE__ */ __name(function scrollInView4() {
      var index = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : -1;
      var id2 = index !== -1 ? "".concat(this.id, "_").concat(index) : this.focusedItemId;
      var element = findSingle(this.menubar, 'li[id="'.concat(id2, '"]'));
      if (element) {
        element.scrollIntoView && element.scrollIntoView({
          block: "nearest",
          inline: "start"
        });
      }
    }, "scrollInView"),
    createProcessedItems: /* @__PURE__ */ __name(function createProcessedItems2(items) {
      var _this11 = this;
      var level = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : 0;
      var parent = arguments.length > 2 && arguments[2] !== void 0 ? arguments[2] : {};
      var parentKey = arguments.length > 3 && arguments[3] !== void 0 ? arguments[3] : "";
      var processedItems3 = [];
      items && items.forEach(function(item3, index) {
        var key = (parentKey !== "" ? parentKey + "_" : "") + index;
        var newItem = {
          item: item3,
          index,
          level,
          key,
          parent,
          parentKey
        };
        newItem["items"] = _this11.createProcessedItems(item3.items, level + 1, newItem, key);
        processedItems3.push(newItem);
      });
      return processedItems3;
    }, "createProcessedItems"),
    containerRef: /* @__PURE__ */ __name(function containerRef3(el) {
      this.container = el;
    }, "containerRef"),
    menubarRef: /* @__PURE__ */ __name(function menubarRef2(el) {
      this.menubar = el ? el.$el : void 0;
    }, "menubarRef")
  },
  computed: {
    processedItems: /* @__PURE__ */ __name(function processedItems2() {
      return this.createProcessedItems(this.model || []);
    }, "processedItems"),
    visibleItems: /* @__PURE__ */ __name(function visibleItems2() {
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
    Portal: script$m
  }
};
var _hoisted_1$5 = ["id"];
function render$3(_ctx, _cache, $props, $setup, $data, $options) {
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
          }, _ctx.ptm("end")), [renderSlot(_ctx.$slots, "end")], 16)) : createCommentVNode("", true)], 16, _hoisted_1$5)) : createCommentVNode("", true)];
        }),
        _: 3
      }, 16, ["onEnter", "onAfterEnter", "onLeave", "onAfterLeave"])];
    }),
    _: 3
  }, 8, ["appendTo", "disabled"]);
}
__name(render$3, "render$3");
script$3.render = render$3;
var theme9 = /* @__PURE__ */ __name(function theme10(_ref) {
  var dt = _ref.dt;
  return "\n.p-splitbutton {\n    display: inline-flex;\n    position: relative;\n    border-radius: ".concat(dt("splitbutton.border.radius"), ";\n}\n\n.p-splitbutton-button {\n    border-top-right-radius: 0;\n    border-bottom-right-radius: 0;\n    border-right: 0 none;\n}\n\n.p-splitbutton-button:focus-visible,\n.p-splitbutton-dropdown:focus-visible {\n    z-index: 1;\n}\n\n.p-splitbutton-button:not(:disabled):hover,\n.p-splitbutton-button:not(:disabled):active {\n    border-right: 0 none;\n}\n\n.p-splitbutton-dropdown {\n    border-top-left-radius: 0;\n    border-bottom-left-radius: 0;\n}\n\n.p-splitbutton .p-menu {\n    min-width: 100%;\n}\n\n.p-splitbutton-fluid {\n    display: flex;\n}\n\n.p-splitbutton-rounded .p-splitbutton-dropdown {\n    border-top-right-radius: ").concat(dt("splitbutton.rounded.border.radius"), ";\n    border-bottom-right-radius: ").concat(dt("splitbutton.rounded.border.radius"), ";\n}\n\n.p-splitbutton-rounded .p-splitbutton-button {\n    border-top-left-radius: ").concat(dt("splitbutton.rounded.border.radius"), ";\n    border-bottom-left-radius: ").concat(dt("splitbutton.rounded.border.radius"), ";\n}\n\n.p-splitbutton-raised {\n    box-shadow: ").concat(dt("splitbutton.raised.shadow"), ";\n}\n");
}, "theme");
var classes = {
  root: /* @__PURE__ */ __name(function root11(_ref2) {
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
  theme: theme9,
  classes
});
var script$1 = {
  name: "BaseSplitButton",
  "extends": script$g,
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
  provide: /* @__PURE__ */ __name(function provide13() {
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
  data: /* @__PURE__ */ __name(function data9() {
    return {
      id: this.$attrs.id,
      isExpanded: false
    };
  }, "data"),
  watch: {
    "$attrs.id": /* @__PURE__ */ __name(function $attrsId5(newValue) {
      this.id = newValue || UniqueComponentId();
    }, "$attrsId")
  },
  mounted: /* @__PURE__ */ __name(function mounted9() {
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
    PVSButton: script$f,
    PVSMenu: script$3,
    ChevronDownIcon: script$n
  }
};
var _hoisted_1$4 = ["data-p-severity"];
function render$2(_ctx, _cache, $props, $setup, $data, $options) {
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
  } : void 0]), 1032, ["id", "model", "autoZIndex", "baseZIndex", "appendTo", "unstyled", "pt"])], 16, _hoisted_1$4);
}
__name(render$2, "render$2");
script.render = render$2;
const minQueueCount = 1;
const _sfc_main$5 = /* @__PURE__ */ defineComponent({
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
        createVNode(unref(script$E), {
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
const BatchCountEdit = /* @__PURE__ */ _export_sfc(_sfc_main$5, [["__scopeId", "data-v-713442be"]]);
const _withScopeId$1 = /* @__PURE__ */ __name((n) => (pushScopeId("data-v-d3897845"), n = n(), popScopeId(), n), "_withScopeId$1");
const _hoisted_1$3 = { class: "queue-button-group flex" };
const _sfc_main$4 = /* @__PURE__ */ defineComponent({
  __name: "ComfyQueueButton",
  setup(__props) {
    const workspaceStore = useWorkspaceStore();
    const queueCountStore = storeToRefs(useQueuePendingTaskCountStore());
    const { mode: queueMode } = storeToRefs(useQueueSettingsStore());
    const { t } = useI18n();
    const queueModeMenuItemLookup = computed(() => ({
      disabled: {
        key: "disabled",
        label: t("menu.queue"),
        tooltip: t("menu.disabledTooltip"),
        command: /* @__PURE__ */ __name(() => {
          queueMode.value = "disabled";
        }, "command")
      },
      instant: {
        key: "instant",
        label: `${t("menu.queue")} (${t("menu.instant")})`,
        tooltip: t("menu.instantTooltip"),
        command: /* @__PURE__ */ __name(() => {
          queueMode.value = "instant";
        }, "command")
      },
      change: {
        key: "change",
        label: `${t("menu.queue")} (${t("menu.onChange")})`,
        tooltip: t("menu.onChangeTooltip"),
        command: /* @__PURE__ */ __name(() => {
          queueMode.value = "change";
        }, "command")
      }
    }));
    const activeQueueModeMenuItem = computed(
      () => queueModeMenuItemLookup.value[queueMode.value]
    );
    const queueModeMenuItems = computed(
      () => Object.values(queueModeMenuItemLookup.value)
    );
    const executingPrompt = computed(() => !!queueCountStore.count.value);
    const hasPendingTasks = computed(() => queueCountStore.count.value > 1);
    const commandStore = useCommandStore();
    const queuePrompt = /* @__PURE__ */ __name((e) => {
      const commandId = e.shiftKey ? "Comfy.QueuePromptFront" : "Comfy.QueuePrompt";
      commandStore.execute(commandId);
    }, "queuePrompt");
    return (_ctx, _cache) => {
      const _component_i_lucide58list_start = __unplugin_components_0$1;
      const _component_i_lucide58play = __unplugin_components_1$1;
      const _component_i_lucide58fast_forward = __unplugin_components_2;
      const _component_i_lucide58step_forward = __unplugin_components_3;
      const _directive_tooltip = resolveDirective("tooltip");
      return openBlock(), createElementBlock("div", _hoisted_1$3, [
        withDirectives((openBlock(), createBlock(unref(script), {
          class: "comfyui-queue-button",
          label: activeQueueModeMenuItem.value.label,
          severity: "primary",
          size: "small",
          onClick: queuePrompt,
          model: queueModeMenuItems.value,
          "data-testid": "queue-button"
        }, {
          icon: withCtx(() => [
            unref(workspaceStore).shiftDown ? (openBlock(), createBlock(_component_i_lucide58list_start, { key: 0 })) : unref(queueMode) === "disabled" ? (openBlock(), createBlock(_component_i_lucide58play, { key: 1 })) : unref(queueMode) === "instant" ? (openBlock(), createBlock(_component_i_lucide58fast_forward, { key: 2 })) : unref(queueMode) === "change" ? (openBlock(), createBlock(_component_i_lucide58step_forward, { key: 3 })) : createCommentVNode("", true)
          ]),
          item: withCtx(({ item: item3 }) => [
            withDirectives(createVNode(unref(script$f), {
              label: item3.label,
              icon: item3.icon,
              severity: item3.key === unref(queueMode) ? "primary" : "secondary",
              size: "small",
              text: ""
            }, null, 8, ["label", "icon", "severity"]), [
              [_directive_tooltip, item3.tooltip]
            ])
          ]),
          _: 1
        }, 8, ["label", "model"])), [
          [
            _directive_tooltip,
            unref(workspaceStore).shiftDown ? _ctx.$t("menu.queueWorkflowFront") : _ctx.$t("menu.queueWorkflow"),
            void 0,
            { bottom: true }
          ]
        ]),
        createVNode(BatchCountEdit),
        createVNode(unref(script$7), { class: "execution-actions flex flex-nowrap" }, {
          default: withCtx(() => [
            withDirectives(createVNode(unref(script$f), {
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
            withDirectives(createVNode(unref(script$f), {
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
const ComfyQueueButton = /* @__PURE__ */ _export_sfc(_sfc_main$4, [["__scopeId", "data-v-d3897845"]]);
const overlapThreshold = 20;
const _sfc_main$3 = /* @__PURE__ */ defineComponent({
  __name: "ComfyActionbar",
  setup(__props) {
    const settingsStore = useSettingStore();
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
        captureLastDragState();
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
        captureLastDragState();
      }
    }, "setInitialPosition");
    onMounted(setInitialPosition);
    watch(visible, (newVisible) => {
      if (newVisible) {
        nextTick(setInitialPosition);
      }
    });
    const lastDragState = ref({
      x: x.value,
      y: y.value,
      windowWidth: window.innerWidth,
      windowHeight: window.innerHeight
    });
    const captureLastDragState = /* @__PURE__ */ __name(() => {
      lastDragState.value = {
        x: x.value,
        y: y.value,
        windowWidth: window.innerWidth,
        windowHeight: window.innerHeight
      };
    }, "captureLastDragState");
    watch(
      isDragging,
      (newIsDragging) => {
        if (!newIsDragging) {
          captureLastDragState();
        }
      },
      { immediate: true }
    );
    const adjustMenuPosition = /* @__PURE__ */ __name(() => {
      if (panelRef.value) {
        const screenWidth = window.innerWidth;
        const screenHeight = window.innerHeight;
        const menuWidth = panelRef.value.offsetWidth;
        const menuHeight = panelRef.value.offsetHeight;
        const distanceLeft = lastDragState.value.x;
        const distanceRight = lastDragState.value.windowWidth - (lastDragState.value.x + menuWidth);
        const distanceTop = lastDragState.value.y;
        const distanceBottom = lastDragState.value.windowHeight - (lastDragState.value.y + menuHeight);
        const distances = [
          { edge: "left", distance: distanceLeft },
          { edge: "right", distance: distanceRight },
          { edge: "top", distance: distanceTop },
          { edge: "bottom", distance: distanceBottom }
        ];
        const closestEdge = distances.reduce(
          (min, curr) => curr.distance < min.distance ? curr : min
        );
        const verticalRatio = lastDragState.value.y / lastDragState.value.windowHeight;
        const horizontalRatio = lastDragState.value.x / lastDragState.value.windowWidth;
        if (closestEdge.edge === "left") {
          x.value = closestEdge.distance;
          y.value = verticalRatio * screenHeight;
        } else if (closestEdge.edge === "right") {
          x.value = screenWidth - menuWidth - closestEdge.distance;
          y.value = verticalRatio * screenHeight;
        } else if (closestEdge.edge === "top") {
          x.value = horizontalRatio * screenWidth;
          y.value = closestEdge.distance;
        } else {
          x.value = horizontalRatio * screenWidth;
          y.value = screenHeight - menuHeight - closestEdge.distance;
        }
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
            createVNode(ComfyQueueButton)
          ], 512)
        ]),
        _: 1
      }, 8, ["style", "class"]);
    };
  }
});
const Actionbar = /* @__PURE__ */ _export_sfc(_sfc_main$3, [["__scopeId", "data-v-542a7001"]]);
const _hoisted_1$2 = {
  viewBox: "0 0 24 24",
  width: "1.2em",
  height: "1.2em"
};
const _hoisted_2$2 = /* @__PURE__ */ createBaseVNode("path", {
  fill: "currentColor",
  d: "M5 21q-.825 0-1.412-.587T3 19V5q0-.825.588-1.412T5 3h14q.825 0 1.413.588T21 5v14q0 .825-.587 1.413T19 21zm0-5v3h14v-3zm0-2h14V5H5zm0 2v3z"
}, null, -1);
const _hoisted_3$1 = [
  _hoisted_2$2
];
function render$1(_ctx, _cache) {
  return openBlock(), createElementBlock("svg", _hoisted_1$2, [..._hoisted_3$1]);
}
__name(render$1, "render$1");
const __unplugin_components_1 = markRaw({ name: "material-symbols-dock-to-bottom-outline", render: render$1 });
const _hoisted_1$1 = {
  viewBox: "0 0 24 24",
  width: "1.2em",
  height: "1.2em"
};
const _hoisted_2$1 = /* @__PURE__ */ createBaseVNode("path", {
  fill: "currentColor",
  d: "M5 21q-.825 0-1.412-.587T3 19V5q0-.825.588-1.412T5 3h14q.825 0 1.413.588T21 5v14q0 .825-.587 1.413T19 21zm0-7h14V5H5z"
}, null, -1);
const _hoisted_3 = [
  _hoisted_2$1
];
function render(_ctx, _cache) {
  return openBlock(), createElementBlock("svg", _hoisted_1$1, [..._hoisted_3]);
}
__name(render, "render");
const __unplugin_components_0 = markRaw({ name: "material-symbols-dock-to-bottom", render });
const _sfc_main$2 = /* @__PURE__ */ defineComponent({
  __name: "BottomPanelToggleButton",
  setup(__props) {
    const bottomPanelStore = useBottomPanelStore();
    return (_ctx, _cache) => {
      const _component_i_material_symbols58dock_to_bottom = __unplugin_components_0;
      const _component_i_material_symbols58dock_to_bottom_outline = __unplugin_components_1;
      const _directive_tooltip = resolveDirective("tooltip");
      return withDirectives((openBlock(), createBlock(unref(script$f), {
        severity: "secondary",
        text: "",
        onClick: unref(bottomPanelStore).toggleBottomPanel
      }, {
        icon: withCtx(() => [
          unref(bottomPanelStore).bottomPanelVisible ? (openBlock(), createBlock(_component_i_material_symbols58dock_to_bottom, { key: 0 })) : (openBlock(), createBlock(_component_i_material_symbols58dock_to_bottom_outline, { key: 1 }))
        ]),
        _: 1
      }, 8, ["onClick"])), [
        [vShow, unref(bottomPanelStore).bottomPanelTabs.length > 0],
        [_directive_tooltip, { value: _ctx.$t("menu.toggleBottomPanel"), showDelay: 300 }]
      ]);
    };
  }
});
const _withScopeId = /* @__PURE__ */ __name((n) => (pushScopeId("data-v-d792da31"), n = n(), popScopeId(), n), "_withScopeId");
const _hoisted_1 = /* @__PURE__ */ _withScopeId(() => /* @__PURE__ */ createBaseVNode("h1", { class: "comfyui-logo mx-2" }, "ComfyUI", -1));
const _hoisted_2 = { class: "flex-grow" };
const _sfc_main$1 = /* @__PURE__ */ defineComponent({
  __name: "TopMenubar",
  setup(__props) {
    const workspaceState = useWorkspaceStore();
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
      const _directive_tooltip = resolveDirective("tooltip");
      return openBlock(), createBlock(Teleport, { to: teleportTarget.value }, [
        withDirectives(createBaseVNode("div", {
          ref_key: "topMenuRef",
          ref: topMenuRef,
          class: normalizeClass(["comfyui-menu flex items-center", { dropzone: isDropZone.value, "dropzone-active": isDroppable.value }])
        }, [
          _hoisted_1,
          createVNode(CommandMenubar),
          createVNode(unref(script$F), {
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
          createVNode(Actionbar),
          createVNode(_sfc_main$2),
          withDirectives(createVNode(unref(script$f), {
            icon: "pi pi-bars",
            severity: "secondary",
            text: "",
            onClick: _cache[0] || (_cache[0] = ($event) => unref(workspaceState).focusMode = true),
            onContextmenu: unref(showNativeMenu)
          }, null, 8, ["onContextmenu"]), [
            [_directive_tooltip, { value: _ctx.$t("menu.hideMenu"), showDelay: 300 }]
          ])
        ], 2), [
          [vShow, betaMenuEnabled.value && !unref(workspaceState).focusMode]
        ])
      ], 8, ["to"]);
    };
  }
});
const TopMenubar = /* @__PURE__ */ _export_sfc(_sfc_main$1, [["__scopeId", "data-v-d792da31"]]);
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
var LatentPreviewMethod = /* @__PURE__ */ ((LatentPreviewMethod2) => {
  LatentPreviewMethod2["NoPreviews"] = "none";
  LatentPreviewMethod2["Auto"] = "auto";
  LatentPreviewMethod2["Latent2RGB"] = "latent2rgb";
  LatentPreviewMethod2["TAESD"] = "taesd";
  return LatentPreviewMethod2;
})(LatentPreviewMethod || {});
var LogLevel = /* @__PURE__ */ ((LogLevel2) => {
  LogLevel2["DEBUG"] = "DEBUG";
  LogLevel2["INFO"] = "INFO";
  LogLevel2["WARNING"] = "WARNING";
  LogLevel2["ERROR"] = "ERROR";
  LogLevel2["CRITICAL"] = "CRITICAL";
  return LogLevel2;
})(LogLevel || {});
var HashFunction = /* @__PURE__ */ ((HashFunction2) => {
  HashFunction2["MD5"] = "md5";
  HashFunction2["SHA1"] = "sha1";
  HashFunction2["SHA256"] = "sha256";
  HashFunction2["SHA512"] = "sha512";
  return HashFunction2;
})(HashFunction || {});
var AutoLaunch = /* @__PURE__ */ ((AutoLaunch2) => {
  AutoLaunch2["Auto"] = "auto";
  AutoLaunch2["Disable"] = "disable";
  AutoLaunch2["Enable"] = "enable";
  return AutoLaunch2;
})(AutoLaunch || {});
var CudaMalloc = /* @__PURE__ */ ((CudaMalloc2) => {
  CudaMalloc2["Auto"] = "auto";
  CudaMalloc2["Disable"] = "disable";
  CudaMalloc2["Enable"] = "enable";
  return CudaMalloc2;
})(CudaMalloc || {});
var FloatingPointPrecision = /* @__PURE__ */ ((FloatingPointPrecision2) => {
  FloatingPointPrecision2["AUTO"] = "auto";
  FloatingPointPrecision2["FP64"] = "fp64";
  FloatingPointPrecision2["FP32"] = "fp32";
  FloatingPointPrecision2["FP16"] = "fp16";
  FloatingPointPrecision2["BF16"] = "bf16";
  FloatingPointPrecision2["FP8E4M3FN"] = "fp8_e4m3fn";
  FloatingPointPrecision2["FP8E5M2"] = "fp8_e5m2";
  return FloatingPointPrecision2;
})(FloatingPointPrecision || {});
var CrossAttentionMethod = /* @__PURE__ */ ((CrossAttentionMethod2) => {
  CrossAttentionMethod2["Auto"] = "auto";
  CrossAttentionMethod2["Split"] = "split";
  CrossAttentionMethod2["Quad"] = "quad";
  CrossAttentionMethod2["Pytorch"] = "pytorch";
  return CrossAttentionMethod2;
})(CrossAttentionMethod || {});
var VramManagement = /* @__PURE__ */ ((VramManagement2) => {
  VramManagement2["Auto"] = "auto";
  VramManagement2["GPUOnly"] = "gpu-only";
  VramManagement2["HighVram"] = "highvram";
  VramManagement2["NormalVram"] = "normalvram";
  VramManagement2["LowVram"] = "lowvram";
  VramManagement2["NoVram"] = "novram";
  VramManagement2["CPU"] = "cpu";
  return VramManagement2;
})(VramManagement || {});
const WEB_ONLY_CONFIG_ITEMS = [
  // Launch behavior
  {
    id: "auto-launch",
    name: "Automatically opens in the browser on startup",
    category: ["Launch"],
    type: "combo",
    options: Object.values(AutoLaunch),
    defaultValue: AutoLaunch.Auto,
    getValue: /* @__PURE__ */ __name((value) => {
      switch (value) {
        case AutoLaunch.Auto:
          return {};
        case AutoLaunch.Enable:
          return {
            ["auto-launch"]: true
          };
        case AutoLaunch.Disable:
          return {
            ["disable-auto-launch"]: true
          };
      }
    }, "getValue")
  }
];
const SERVER_CONFIG_ITEMS = [
  // Network settings
  {
    id: "listen",
    name: "Host: The IP address to listen on",
    category: ["Network"],
    type: "text",
    defaultValue: "127.0.0.1"
  },
  {
    id: "port",
    name: "Port: The port to listen on",
    category: ["Network"],
    type: "number",
    // The default launch port for desktop app is 8000 instead of 8188.
    defaultValue: 8e3
  },
  {
    id: "tls-keyfile",
    name: "TLS Key File: Path to TLS key file for HTTPS",
    category: ["Network"],
    type: "text",
    defaultValue: ""
  },
  {
    id: "tls-certfile",
    name: "TLS Certificate File: Path to TLS certificate file for HTTPS",
    category: ["Network"],
    type: "text",
    defaultValue: ""
  },
  {
    id: "enable-cors-header",
    name: 'Enable CORS header: Use "*" for all origins or specify domain',
    category: ["Network"],
    type: "text",
    defaultValue: ""
  },
  {
    id: "max-upload-size",
    name: "Maximum upload size (MB)",
    category: ["Network"],
    type: "number",
    defaultValue: 100
  },
  // CUDA settings
  {
    id: "cuda-device",
    name: "CUDA device index to use",
    category: ["CUDA"],
    type: "number",
    defaultValue: null
  },
  {
    id: "cuda-malloc",
    name: "Use CUDA malloc for memory allocation",
    category: ["CUDA"],
    type: "combo",
    options: Object.values(CudaMalloc),
    defaultValue: CudaMalloc.Auto,
    getValue: /* @__PURE__ */ __name((value) => {
      switch (value) {
        case CudaMalloc.Auto:
          return {};
        case CudaMalloc.Enable:
          return {
            ["cuda-malloc"]: true
          };
        case CudaMalloc.Disable:
          return {
            ["disable-cuda-malloc"]: true
          };
      }
    }, "getValue")
  },
  // Precision settings
  {
    id: "global-precision",
    name: "Global floating point precision",
    category: ["Inference"],
    type: "combo",
    options: [
      FloatingPointPrecision.AUTO,
      FloatingPointPrecision.FP32,
      FloatingPointPrecision.FP16
    ],
    defaultValue: FloatingPointPrecision.AUTO,
    tooltip: "Global floating point precision",
    getValue: /* @__PURE__ */ __name((value) => {
      switch (value) {
        case FloatingPointPrecision.AUTO:
          return {};
        case FloatingPointPrecision.FP32:
          return {
            ["force-fp32"]: true
          };
        case FloatingPointPrecision.FP16:
          return {
            ["force-fp16"]: true
          };
        default:
          return {};
      }
    }, "getValue")
  },
  // UNET precision
  {
    id: "unet-precision",
    name: "UNET precision",
    category: ["Inference"],
    type: "combo",
    options: [
      FloatingPointPrecision.AUTO,
      FloatingPointPrecision.FP64,
      FloatingPointPrecision.FP32,
      FloatingPointPrecision.FP16,
      FloatingPointPrecision.BF16,
      FloatingPointPrecision.FP8E4M3FN,
      FloatingPointPrecision.FP8E5M2
    ],
    defaultValue: FloatingPointPrecision.AUTO,
    tooltip: "UNET precision",
    getValue: /* @__PURE__ */ __name((value) => {
      switch (value) {
        case FloatingPointPrecision.AUTO:
          return {};
        default:
          return {
            [`${value.toLowerCase()}-unet`]: true
          };
      }
    }, "getValue")
  },
  // VAE settings
  {
    id: "vae-precision",
    name: "VAE precision",
    category: ["Inference"],
    type: "combo",
    options: [
      FloatingPointPrecision.AUTO,
      FloatingPointPrecision.FP16,
      FloatingPointPrecision.FP32,
      FloatingPointPrecision.BF16
    ],
    defaultValue: FloatingPointPrecision.AUTO,
    tooltip: "VAE precision",
    getValue: /* @__PURE__ */ __name((value) => {
      switch (value) {
        case FloatingPointPrecision.AUTO:
          return {};
        default:
          return {
            [`${value.toLowerCase()}-vae`]: true
          };
      }
    }, "getValue")
  },
  {
    id: "cpu-vae",
    name: "Run VAE on CPU",
    category: ["Inference"],
    type: "boolean",
    defaultValue: false
  },
  // Text Encoder settings
  {
    id: "text-encoder-precision",
    name: "Text Encoder precision",
    category: ["Inference"],
    type: "combo",
    options: [
      FloatingPointPrecision.AUTO,
      FloatingPointPrecision.FP8E4M3FN,
      FloatingPointPrecision.FP8E5M2,
      FloatingPointPrecision.FP16,
      FloatingPointPrecision.FP32
    ],
    defaultValue: FloatingPointPrecision.AUTO,
    tooltip: "Text Encoder precision",
    getValue: /* @__PURE__ */ __name((value) => {
      switch (value) {
        case FloatingPointPrecision.AUTO:
          return {};
        default:
          return {
            [`${value.toLowerCase()}-text-enc`]: true
          };
      }
    }, "getValue")
  },
  // Memory and performance settings
  {
    id: "force-channels-last",
    name: "Force channels-last memory format",
    category: ["Memory"],
    type: "boolean",
    defaultValue: false
  },
  {
    id: "directml",
    name: "DirectML device index",
    category: ["Memory"],
    type: "number",
    defaultValue: null
  },
  {
    id: "disable-ipex-optimize",
    name: "Disable IPEX optimization",
    category: ["Memory"],
    type: "boolean",
    defaultValue: false
  },
  // Preview settings
  {
    id: "preview-method",
    name: "Method used for latent previews",
    category: ["Preview"],
    type: "combo",
    options: Object.values(LatentPreviewMethod),
    defaultValue: LatentPreviewMethod.NoPreviews
  },
  {
    id: "preview-size",
    name: "Size of preview images",
    category: ["Preview"],
    type: "slider",
    defaultValue: 512,
    attrs: {
      min: 128,
      max: 2048,
      step: 128
    }
  },
  // Cache settings
  {
    id: "cache-classic",
    name: "Use classic cache system",
    category: ["Cache"],
    type: "boolean",
    defaultValue: false
  },
  {
    id: "cache-lru",
    name: "Use LRU caching with a maximum of N node results cached.",
    category: ["Cache"],
    type: "number",
    defaultValue: null,
    tooltip: "May use more RAM/VRAM."
  },
  // Attention settings
  {
    id: "cross-attention-method",
    name: "Cross attention method",
    category: ["Attention"],
    type: "combo",
    options: Object.values(CrossAttentionMethod),
    defaultValue: CrossAttentionMethod.Auto,
    getValue: /* @__PURE__ */ __name((value) => {
      switch (value) {
        case CrossAttentionMethod.Auto:
          return {};
        default:
          return {
            [`use-${value.toLowerCase()}-cross-attention`]: true
          };
      }
    }, "getValue")
  },
  {
    id: "disable-xformers",
    name: "Disable xFormers optimization",
    type: "boolean",
    defaultValue: false
  },
  {
    id: "force-upcast-attention",
    name: "Force attention upcast",
    category: ["Attention"],
    type: "boolean",
    defaultValue: false
  },
  {
    id: "dont-upcast-attention",
    name: "Prevent attention upcast",
    category: ["Attention"],
    type: "boolean",
    defaultValue: false
  },
  // VRAM management
  {
    id: "vram-management",
    name: "VRAM management mode",
    category: ["Memory"],
    type: "combo",
    options: Object.values(VramManagement),
    defaultValue: VramManagement.Auto,
    getValue: /* @__PURE__ */ __name((value) => {
      switch (value) {
        case VramManagement.Auto:
          return {};
        default:
          return {
            [value]: true
          };
      }
    }, "getValue")
  },
  {
    id: "reserve-vram",
    name: "Reserved VRAM (GB)",
    category: ["Memory"],
    type: "number",
    defaultValue: null,
    tooltip: "Set the amount of vram in GB you want to reserve for use by your OS/other software. By default some amount is reverved depending on your OS."
  },
  // Misc settings
  {
    id: "default-hashing-function",
    name: "Default hashing function for model files",
    type: "combo",
    options: Object.values(HashFunction),
    defaultValue: HashFunction.SHA256
  },
  {
    id: "disable-smart-memory",
    name: "Disable smart memory management",
    tooltip: "Force ComfyUI to aggressively offload to regular ram instead of keeping models in vram when it can.",
    category: ["Memory"],
    type: "boolean",
    defaultValue: false
  },
  {
    id: "deterministic",
    name: "Make pytorch use slower deterministic algorithms when it can.",
    type: "boolean",
    defaultValue: false,
    tooltip: "Note that this might not make images deterministic in all cases."
  },
  {
    id: "fast",
    name: "Enable some untested and potentially quality deteriorating optimizations.",
    type: "boolean",
    defaultValue: false
  },
  {
    id: "dont-print-server",
    name: "Don't print server output to console.",
    type: "boolean",
    defaultValue: false
  },
  {
    id: "disable-metadata",
    name: "Disable saving prompt metadata in files.",
    type: "boolean",
    defaultValue: false
  },
  {
    id: "disable-all-custom-nodes",
    name: "Disable loading all custom nodes.",
    type: "boolean",
    defaultValue: false
  },
  {
    id: "log-level",
    name: "Logging verbosity level",
    type: "combo",
    options: Object.values(LogLevel),
    defaultValue: LogLevel.INFO,
    getValue: /* @__PURE__ */ __name((value) => {
      return {
        verbose: value
      };
    }, "getValue")
  },
  // Directories
  {
    id: "input-directory",
    name: "Input directory",
    category: ["Directories"],
    type: "text",
    defaultValue: ""
  },
  {
    id: "output-directory",
    name: "Output directory",
    category: ["Directories"],
    type: "text",
    defaultValue: ""
  }
];
function useCoreCommands() {
  const getTracker = /* @__PURE__ */ __name(() => useWorkflowStore()?.activeWorkflow?.changeTracker, "getTracker");
  const getSelectedNodes = /* @__PURE__ */ __name(() => {
    const selectedNodes = app.canvas.selected_nodes;
    const result = [];
    if (selectedNodes) {
      for (const i in selectedNodes) {
        const node = selectedNodes[i];
        result.push(node);
      }
    }
    return result;
  }, "getSelectedNodes");
  const toggleSelectedNodesMode = /* @__PURE__ */ __name((mode) => {
    getSelectedNodes().forEach((node) => {
      if (node.mode === mode) {
        node.mode = LGraphEventMode.ALWAYS;
      } else {
        node.mode = mode;
      }
    });
  }, "toggleSelectedNodesMode");
  return [
    {
      id: "Comfy.NewBlankWorkflow",
      icon: "pi pi-plus",
      label: "New Blank Workflow",
      menubarLabel: "New",
      function: /* @__PURE__ */ __name(() => workflowService.loadBlankWorkflow(), "function")
    },
    {
      id: "Comfy.OpenWorkflow",
      icon: "pi pi-folder-open",
      label: "Open Workflow",
      menubarLabel: "Open",
      function: /* @__PURE__ */ __name(() => {
        app.ui.loadFile();
      }, "function")
    },
    {
      id: "Comfy.LoadDefaultWorkflow",
      icon: "pi pi-code",
      label: "Load Default Workflow",
      function: /* @__PURE__ */ __name(() => workflowService.loadDefaultWorkflow(), "function")
    },
    {
      id: "Comfy.SaveWorkflow",
      icon: "pi pi-save",
      label: "Save Workflow",
      menubarLabel: "Save",
      function: /* @__PURE__ */ __name(async () => {
        const workflow = useWorkflowStore().activeWorkflow;
        if (!workflow) return;
        await workflowService.saveWorkflow(workflow);
      }, "function")
    },
    {
      id: "Comfy.SaveWorkflowAs",
      icon: "pi pi-save",
      label: "Save Workflow As",
      menubarLabel: "Save As",
      function: /* @__PURE__ */ __name(async () => {
        const workflow = useWorkflowStore().activeWorkflow;
        if (!workflow) return;
        await workflowService.saveWorkflowAs(workflow);
      }, "function")
    },
    {
      id: "Comfy.ExportWorkflow",
      icon: "pi pi-download",
      label: "Export Workflow",
      menubarLabel: "Export",
      function: /* @__PURE__ */ __name(() => {
        workflowService.exportWorkflow("workflow", "workflow");
      }, "function")
    },
    {
      id: "Comfy.ExportWorkflowAPI",
      icon: "pi pi-download",
      label: "Export Workflow (API Format)",
      menubarLabel: "Export (API)",
      function: /* @__PURE__ */ __name(() => {
        workflowService.exportWorkflow("workflow_api", "output");
      }, "function")
    },
    {
      id: "Comfy.Undo",
      icon: "pi pi-undo",
      label: "Undo",
      function: /* @__PURE__ */ __name(async () => {
        await getTracker()?.undo?.();
      }, "function")
    },
    {
      id: "Comfy.Redo",
      icon: "pi pi-refresh",
      label: "Redo",
      function: /* @__PURE__ */ __name(async () => {
        await getTracker()?.redo?.();
      }, "function")
    },
    {
      id: "Comfy.ClearWorkflow",
      icon: "pi pi-trash",
      label: "Clear Workflow",
      function: /* @__PURE__ */ __name(() => {
        const settingStore = useSettingStore();
        if (!settingStore.get("Comfy.ComfirmClear") || confirm("Clear workflow?")) {
          app.clean();
          app.graph.clear();
          api.dispatchCustomEvent("graphCleared");
        }
      }, "function")
    },
    {
      id: "Comfy.Canvas.ResetView",
      icon: "pi pi-expand",
      label: "Reset View",
      function: /* @__PURE__ */ __name(() => {
        app.resetView();
      }, "function")
    },
    {
      id: "Comfy.OpenClipspace",
      icon: "pi pi-clipboard",
      label: "Clipspace",
      function: /* @__PURE__ */ __name(() => {
        app.openClipspace();
      }, "function")
    },
    {
      id: "Comfy.RefreshNodeDefinitions",
      icon: "pi pi-refresh",
      label: "Refresh Node Definitions",
      function: /* @__PURE__ */ __name(async () => {
        await app.refreshComboInNodes();
      }, "function")
    },
    {
      id: "Comfy.Interrupt",
      icon: "pi pi-stop",
      label: "Interrupt",
      function: /* @__PURE__ */ __name(async () => {
        await api.interrupt();
        useToastStore().add({
          severity: "info",
          summary: "Interrupted",
          detail: "Execution has been interrupted",
          life: 1e3
        });
      }, "function")
    },
    {
      id: "Comfy.ClearPendingTasks",
      icon: "pi pi-stop",
      label: "Clear Pending Tasks",
      function: /* @__PURE__ */ __name(async () => {
        await useQueueStore().clear(["queue"]);
        useToastStore().add({
          severity: "info",
          summary: "Confirmed",
          detail: "Pending tasks deleted",
          life: 3e3
        });
      }, "function")
    },
    {
      id: "Comfy.BrowseTemplates",
      icon: "pi pi-folder-open",
      label: "Browse Templates",
      function: showTemplateWorkflowsDialog
    },
    {
      id: "Comfy.Canvas.ZoomIn",
      icon: "pi pi-plus",
      label: "Zoom In",
      function: /* @__PURE__ */ __name(() => {
        const ds = app.canvas.ds;
        ds.changeScale(
          ds.scale * 1.1,
          ds.element ? [ds.element.width / 2, ds.element.height / 2] : void 0
        );
        app.canvas.setDirty(true, true);
      }, "function")
    },
    {
      id: "Comfy.Canvas.ZoomOut",
      icon: "pi pi-minus",
      label: "Zoom Out",
      function: /* @__PURE__ */ __name(() => {
        const ds = app.canvas.ds;
        ds.changeScale(
          ds.scale / 1.1,
          ds.element ? [ds.element.width / 2, ds.element.height / 2] : void 0
        );
        app.canvas.setDirty(true, true);
      }, "function")
    },
    {
      id: "Comfy.Canvas.FitView",
      icon: "pi pi-expand",
      label: "Fit view to selected nodes",
      function: /* @__PURE__ */ __name(() => {
        if (app.canvas.empty) {
          useToastStore().add({
            severity: "error",
            summary: "Empty canvas",
            life: 3e3
          });
          return;
        }
        app.canvas.fitViewToSelectionAnimated();
      }, "function")
    },
    {
      id: "Comfy.Canvas.ToggleLock",
      icon: "pi pi-lock",
      label: "Canvas Toggle Lock",
      function: /* @__PURE__ */ __name(() => {
        app.canvas["read_only"] = !app.canvas["read_only"];
      }, "function")
    },
    {
      id: "Comfy.Canvas.ToggleLinkVisibility",
      icon: "pi pi-eye",
      label: "Canvas Toggle Link Visibility",
      versionAdded: "1.3.6",
      function: (() => {
        const settingStore = useSettingStore();
        let lastLinksRenderMode = LiteGraph.SPLINE_LINK;
        return () => {
          const currentMode = settingStore.get("Comfy.LinkRenderMode");
          if (currentMode === LiteGraph.HIDDEN_LINK) {
            settingStore.set("Comfy.LinkRenderMode", lastLinksRenderMode);
          } else {
            lastLinksRenderMode = currentMode;
            settingStore.set("Comfy.LinkRenderMode", LiteGraph.HIDDEN_LINK);
          }
        };
      })()
    },
    {
      id: "Comfy.QueuePrompt",
      icon: "pi pi-play",
      label: "Queue Prompt",
      versionAdded: "1.3.7",
      function: /* @__PURE__ */ __name(() => {
        const batchCount = useQueueSettingsStore().batchCount;
        app.queuePrompt(0, batchCount);
      }, "function")
    },
    {
      id: "Comfy.QueuePromptFront",
      icon: "pi pi-play",
      label: "Queue Prompt (Front)",
      versionAdded: "1.3.7",
      function: /* @__PURE__ */ __name(() => {
        const batchCount = useQueueSettingsStore().batchCount;
        app.queuePrompt(-1, batchCount);
      }, "function")
    },
    {
      id: "Comfy.ShowSettingsDialog",
      icon: "pi pi-cog",
      label: "Show Settings Dialog",
      versionAdded: "1.3.7",
      function: /* @__PURE__ */ __name(() => {
        showSettingsDialog();
      }, "function")
    },
    {
      id: "Comfy.Graph.GroupSelectedNodes",
      icon: "pi pi-sitemap",
      label: "Group Selected Nodes",
      versionAdded: "1.3.7",
      function: /* @__PURE__ */ __name(() => {
        const { canvas } = app;
        if (!canvas.selectedItems?.size) {
          useToastStore().add({
            severity: "error",
            summary: "Nothing to group",
            detail: "Please select the nodes (or other groups) to create a group for",
            life: 3e3
          });
          return;
        }
        const group = new LGraphGroup();
        const padding = useSettingStore().get(
          "Comfy.GroupSelectedNodes.Padding"
        );
        group.resizeTo(canvas.selectedItems, padding);
        canvas.graph.add(group);
        useTitleEditorStore().titleEditorTarget = group;
      }, "function")
    },
    {
      id: "Workspace.NextOpenedWorkflow",
      icon: "pi pi-step-forward",
      label: "Next Opened Workflow",
      versionAdded: "1.3.9",
      function: /* @__PURE__ */ __name(() => {
        workflowService.loadNextOpenedWorkflow();
      }, "function")
    },
    {
      id: "Workspace.PreviousOpenedWorkflow",
      icon: "pi pi-step-backward",
      label: "Previous Opened Workflow",
      versionAdded: "1.3.9",
      function: /* @__PURE__ */ __name(() => {
        workflowService.loadPreviousOpenedWorkflow();
      }, "function")
    },
    {
      id: "Comfy.Canvas.ToggleSelectedNodes.Mute",
      icon: "pi pi-volume-off",
      label: "Mute/Unmute Selected Nodes",
      versionAdded: "1.3.11",
      function: /* @__PURE__ */ __name(() => {
        toggleSelectedNodesMode(LGraphEventMode.NEVER);
      }, "function")
    },
    {
      id: "Comfy.Canvas.ToggleSelectedNodes.Bypass",
      icon: "pi pi-shield",
      label: "Bypass/Unbypass Selected Nodes",
      versionAdded: "1.3.11",
      function: /* @__PURE__ */ __name(() => {
        toggleSelectedNodesMode(LGraphEventMode.BYPASS);
      }, "function")
    },
    {
      id: "Comfy.Canvas.ToggleSelectedNodes.Pin",
      icon: "pi pi-pin",
      label: "Pin/Unpin Selected Nodes",
      versionAdded: "1.3.11",
      function: /* @__PURE__ */ __name(() => {
        getSelectedNodes().forEach((node) => {
          node.pin(!node.pinned);
        });
      }, "function")
    },
    {
      id: "Comfy.Canvas.ToggleSelected.Pin",
      icon: "pi pi-pin",
      label: "Pin/Unpin Selected Items",
      versionAdded: "1.3.33",
      function: /* @__PURE__ */ __name(() => {
        for (const item3 of app.canvas.selectedItems) {
          if (item3 instanceof LGraphNode || item3 instanceof LGraphGroup) {
            item3.pin(!item3.pinned);
          }
        }
      }, "function")
    },
    {
      id: "Comfy.Canvas.ToggleSelectedNodes.Collapse",
      icon: "pi pi-minus",
      label: "Collapse/Expand Selected Nodes",
      versionAdded: "1.3.11",
      function: /* @__PURE__ */ __name(() => {
        getSelectedNodes().forEach((node) => {
          node.collapse();
        });
      }, "function")
    },
    {
      id: "Comfy.ToggleTheme",
      icon: "pi pi-moon",
      label: "Toggle Theme (Dark/Light)",
      versionAdded: "1.3.12",
      function: /* @__PURE__ */ (() => {
        let previousDarkTheme = "dark";
        const isDarkMode = /* @__PURE__ */ __name((themeId) => themeId !== "light", "isDarkMode");
        return () => {
          const settingStore = useSettingStore();
          const currentTheme = settingStore.get("Comfy.ColorPalette");
          if (isDarkMode(currentTheme)) {
            previousDarkTheme = currentTheme;
            settingStore.set("Comfy.ColorPalette", "light");
          } else {
            settingStore.set("Comfy.ColorPalette", previousDarkTheme);
          }
        };
      })()
    },
    {
      id: "Workspace.ToggleBottomPanel",
      icon: "pi pi-list",
      label: "Toggle Bottom Panel",
      versionAdded: "1.3.22",
      function: /* @__PURE__ */ __name(() => {
        useBottomPanelStore().toggleBottomPanel();
      }, "function")
    },
    {
      id: "Workspace.ToggleFocusMode",
      icon: "pi pi-eye",
      label: "Toggle Focus Mode",
      versionAdded: "1.3.27",
      function: /* @__PURE__ */ __name(() => {
        useWorkspaceStore().toggleFocusMode();
      }, "function")
    },
    {
      id: "Comfy.Graph.FitGroupToContents",
      icon: "pi pi-expand",
      label: "Fit Group To Contents",
      versionAdded: "1.4.9",
      function: /* @__PURE__ */ __name(() => {
        for (const group of app.canvas.selectedItems) {
          if (group instanceof LGraphGroup) {
            group.recomputeInsideNodes();
            const padding = useSettingStore().get(
              "Comfy.GroupSelectedNodes.Padding"
            );
            group.resizeTo(group.children, padding);
            app.graph.change();
          }
        }
      }, "function")
    },
    {
      id: "Comfy.Help.OpenComfyUIIssues",
      icon: "pi pi-github",
      label: "Open ComfyUI Issues",
      menubarLabel: "ComfyUI Issues",
      versionAdded: "1.5.5",
      function: /* @__PURE__ */ __name(() => {
        window.open(
          "https://github.com/comfyanonymous/ComfyUI/issues",
          "_blank"
        );
      }, "function")
    },
    {
      id: "Comfy.Help.OpenComfyUIDocs",
      icon: "pi pi-info-circle",
      label: "Open ComfyUI Docs",
      menubarLabel: "ComfyUI Docs",
      versionAdded: "1.5.5",
      function: /* @__PURE__ */ __name(() => {
        window.open("https://docs.comfy.org/", "_blank");
      }, "function")
    },
    {
      id: "Comfy.Help.OpenComfyOrgDiscord",
      icon: "pi pi-discord",
      label: "Open Comfy-Org Discord",
      menubarLabel: "Comfy-Org Discord",
      versionAdded: "1.5.5",
      function: /* @__PURE__ */ __name(() => {
        window.open("https://www.comfy.org/discord", "_blank");
      }, "function")
    },
    {
      id: "Workspace.SearchBox.Toggle",
      icon: "pi pi-search",
      label: "Toggle Search Box",
      versionAdded: "1.5.7",
      function: /* @__PURE__ */ __name(() => {
        useSearchBoxStore().toggleVisible();
      }, "function")
    }
  ];
}
__name(useCoreCommands, "useCoreCommands");
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "GraphView",
  setup(__props) {
    setupAutoQueueHandler();
    const { t } = useI18n();
    const toast = useToast();
    const settingStore = useSettingStore();
    const executionStore = useExecutionStore();
    const theme11 = computed(() => settingStore.get("Comfy.ColorPalette"));
    watch(
      theme11,
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
    watchEffect(() => {
      useQueueStore().maxHistoryItems = settingStore.get(
        "Comfy.Queue.MaxHistoryItems"
      );
    });
    const init = /* @__PURE__ */ __name(() => {
      settingStore.addSettings(app.ui.settings);
      const coreCommands = useCoreCommands();
      useCommandStore().registerCommands(coreCommands);
      useMenuItemStore().registerCoreMenuCommands();
      useKeybindingStore().loadCoreKeybindings();
      useSidebarTabStore().registerCoreSidebarTabs();
      useBottomPanelStore().registerCoreBottomPanelTabs();
      app.extensionManager = useWorkspaceStore();
    }, "init");
    const queuePendingTaskCountStore = useQueuePendingTaskCountStore();
    const onStatus = /* @__PURE__ */ __name((e) => {
      queuePendingTaskCountStore.update(e);
    }, "onStatus");
    const reconnectingMessage = {
      severity: "error",
      summary: t("g.reconnecting")
    };
    const onReconnecting = /* @__PURE__ */ __name(() => {
      toast.remove(reconnectingMessage);
      toast.add(reconnectingMessage);
    }, "onReconnecting");
    const onReconnected = /* @__PURE__ */ __name(() => {
      toast.remove(reconnectingMessage);
      toast.add({
        severity: "success",
        summary: t("g.reconnected"),
        life: 2e3
      });
    }, "onReconnected");
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
    const onGraphReady = /* @__PURE__ */ __name(() => {
      requestIdleCallback(
        () => {
          useKeybindingStore().loadUserKeybindings();
          useServerConfigStore().loadServerConfig(
            SERVER_CONFIG_ITEMS,
            settingStore.get("Comfy.Server.ServerConfigValues")
          );
          useModelStore().loadModelFolders();
          useNodeDefStore().nodeSearchService.endsWithFilterStartSequence("");
          useNodeFrequencyStore().loadNodeFrequencies();
        },
        { timeout: 1e3 }
      );
    }, "onGraphReady");
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock(Fragment, null, [
        createVNode(TopMenubar),
        createVNode(_sfc_main$c, { onReady: onGraphReady }),
        createVNode(_sfc_main$a),
        createVNode(_sfc_main$9),
        createVNode(_sfc_main$8),
        createVNode(MenuHamburger)
      ], 64);
    };
  }
});
export {
  _sfc_main as default
};
//# sourceMappingURL=GraphView-BWxgNrh6.js.map
