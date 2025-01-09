var __defProp = Object.defineProperty;
var __name = (target, value2) => __defProp(target, "name", { value: value2, configurable: true });
import { B as BaseStyle, q as script$6, o as openBlock, f as createElementBlock, D as mergeProps, c1 as findIndexInList, c2 as find, aB as resolveComponent, k as createBlock, G as resolveDynamicComponent, M as withCtx, H as createBaseVNode, X as toDisplayString, J as renderSlot, I as createCommentVNode, T as normalizeClass, P as findSingle, F as Fragment, aC as Transition, i as withDirectives, v as vShow, ak as UniqueComponentId, d as defineComponent, ab as ref, c3 as useModel, N as createVNode, j as unref, c4 as script$7, bQ as script$8, bM as withModifiers, aP as script$9, a1 as useI18n, c as computed, aI as script$a, aE as createTextVNode, c0 as electronAPI, m as onMounted, r as resolveDirective, av as script$b, c5 as script$c, c6 as script$d, l as script$e, bZ as script$f, c7 as MigrationItems, w as watchEffect, E as renderList, c8 as script$g, bW as useRouter, aL as pushScopeId, aM as popScopeId, aU as toRaw, _ as _export_sfc } from "./index-DjNHn37O.js";
import { _ as _sfc_main$5 } from "./BaseViewTemplate-BNGF4K22.js";
var classes$4 = {
  root: /* @__PURE__ */ __name(function root(_ref) {
    var instance = _ref.instance;
    return ["p-step", {
      "p-step-active": instance.active,
      "p-disabled": instance.isStepDisabled
    }];
  }, "root"),
  header: "p-step-header",
  number: "p-step-number",
  title: "p-step-title"
};
var StepStyle = BaseStyle.extend({
  name: "step",
  classes: classes$4
});
var script$2$2 = {
  name: "StepperSeparator",
  hostName: "Stepper",
  "extends": script$6
};
function render$1$2(_ctx, _cache, $props, $setup, $data, $options) {
  return openBlock(), createElementBlock("span", mergeProps({
    "class": _ctx.cx("separator")
  }, _ctx.ptm("separator")), null, 16);
}
__name(render$1$2, "render$1$2");
script$2$2.render = render$1$2;
var script$1$4 = {
  name: "BaseStep",
  "extends": script$6,
  props: {
    value: {
      type: [String, Number],
      "default": void 0
    },
    disabled: {
      type: Boolean,
      "default": false
    },
    asChild: {
      type: Boolean,
      "default": false
    },
    as: {
      type: [String, Object],
      "default": "DIV"
    }
  },
  style: StepStyle,
  provide: /* @__PURE__ */ __name(function provide() {
    return {
      $pcStep: this,
      $parentInstance: this
    };
  }, "provide")
};
var script$5 = {
  name: "Step",
  "extends": script$1$4,
  inheritAttrs: false,
  inject: {
    $pcStepper: {
      "default": null
    },
    $pcStepList: {
      "default": null
    },
    $pcStepItem: {
      "default": null
    }
  },
  data: /* @__PURE__ */ __name(function data() {
    return {
      isSeparatorVisible: false
    };
  }, "data"),
  mounted: /* @__PURE__ */ __name(function mounted() {
    if (this.$el && this.$pcStepList) {
      var index = findIndexInList(this.$el, find(this.$pcStepper.$el, '[data-pc-name="step"]'));
      var stepLen = find(this.$pcStepper.$el, '[data-pc-name="step"]').length;
      this.isSeparatorVisible = index !== stepLen - 1;
    }
  }, "mounted"),
  methods: {
    getPTOptions: /* @__PURE__ */ __name(function getPTOptions(key) {
      var _ptm = key === "root" ? this.ptmi : this.ptm;
      return _ptm(key, {
        context: {
          active: this.active,
          disabled: this.isStepDisabled
        }
      });
    }, "getPTOptions"),
    onStepClick: /* @__PURE__ */ __name(function onStepClick() {
      this.$pcStepper.updateValue(this.activeValue);
    }, "onStepClick")
  },
  computed: {
    active: /* @__PURE__ */ __name(function active() {
      return this.$pcStepper.isStepActive(this.activeValue);
    }, "active"),
    activeValue: /* @__PURE__ */ __name(function activeValue() {
      var _this$$pcStepItem;
      return !!this.$pcStepItem ? (_this$$pcStepItem = this.$pcStepItem) === null || _this$$pcStepItem === void 0 ? void 0 : _this$$pcStepItem.value : this.value;
    }, "activeValue"),
    isStepDisabled: /* @__PURE__ */ __name(function isStepDisabled() {
      return !this.active && (this.$pcStepper.isStepDisabled() || this.disabled);
    }, "isStepDisabled"),
    id: /* @__PURE__ */ __name(function id() {
      var _this$$pcStepper;
      return "".concat((_this$$pcStepper = this.$pcStepper) === null || _this$$pcStepper === void 0 ? void 0 : _this$$pcStepper.id, "_step_").concat(this.activeValue);
    }, "id"),
    ariaControls: /* @__PURE__ */ __name(function ariaControls() {
      var _this$$pcStepper2;
      return "".concat((_this$$pcStepper2 = this.$pcStepper) === null || _this$$pcStepper2 === void 0 ? void 0 : _this$$pcStepper2.id, "_steppanel_").concat(this.activeValue);
    }, "ariaControls"),
    a11yAttrs: /* @__PURE__ */ __name(function a11yAttrs() {
      return {
        root: {
          role: "presentation",
          "aria-current": this.active ? "step" : void 0,
          "data-pc-name": "step",
          "data-pc-section": "root",
          "data-p-disabled": this.disabled,
          "data-p-active": this.active
        },
        header: {
          id: this.id,
          role: "tab",
          taindex: this.disabled ? -1 : void 0,
          "aria-controls": this.ariaControls,
          "data-pc-section": "header",
          disabled: this.disabled,
          onClick: this.onStepClick
        }
      };
    }, "a11yAttrs")
  },
  components: {
    StepperSeparator: script$2$2
  }
};
var _hoisted_1$5 = ["id", "tabindex", "aria-controls", "disabled"];
function render$4(_ctx, _cache, $props, $setup, $data, $options) {
  var _component_StepperSeparator = resolveComponent("StepperSeparator");
  return !_ctx.asChild ? (openBlock(), createBlock(resolveDynamicComponent(_ctx.as), mergeProps({
    key: 0,
    "class": _ctx.cx("root"),
    "aria-current": $options.active ? "step" : void 0,
    role: "presentation",
    "data-p-active": $options.active,
    "data-p-disabled": $options.isStepDisabled
  }, $options.getPTOptions("root")), {
    "default": withCtx(function() {
      return [createBaseVNode("button", mergeProps({
        id: $options.id,
        "class": _ctx.cx("header"),
        role: "tab",
        type: "button",
        tabindex: $options.isStepDisabled ? -1 : void 0,
        "aria-controls": $options.ariaControls,
        disabled: $options.isStepDisabled,
        onClick: _cache[0] || (_cache[0] = function() {
          return $options.onStepClick && $options.onStepClick.apply($options, arguments);
        })
      }, $options.getPTOptions("header")), [createBaseVNode("span", mergeProps({
        "class": _ctx.cx("number")
      }, $options.getPTOptions("number")), toDisplayString($options.activeValue), 17), createBaseVNode("span", mergeProps({
        "class": _ctx.cx("title")
      }, $options.getPTOptions("title")), [renderSlot(_ctx.$slots, "default")], 16)], 16, _hoisted_1$5), $data.isSeparatorVisible ? (openBlock(), createBlock(_component_StepperSeparator, {
        key: 0
      })) : createCommentVNode("", true)];
    }),
    _: 3
  }, 16, ["class", "aria-current", "data-p-active", "data-p-disabled"])) : renderSlot(_ctx.$slots, "default", {
    key: 1,
    "class": normalizeClass(_ctx.cx("root")),
    active: $options.active,
    value: _ctx.value,
    a11yAttrs: $options.a11yAttrs,
    activateCallback: $options.onStepClick
  });
}
__name(render$4, "render$4");
script$5.render = render$4;
var classes$3 = {
  root: "p-steplist"
};
var StepListStyle = BaseStyle.extend({
  name: "steplist",
  classes: classes$3
});
var script$1$3 = {
  name: "BaseStepList",
  "extends": script$6,
  style: StepListStyle,
  provide: /* @__PURE__ */ __name(function provide2() {
    return {
      $pcStepList: this,
      $parentInstance: this
    };
  }, "provide")
};
var script$4 = {
  name: "StepList",
  "extends": script$1$3,
  inheritAttrs: false
};
function render$3(_ctx, _cache, $props, $setup, $data, $options) {
  return openBlock(), createElementBlock("div", mergeProps({
    "class": _ctx.cx("root")
  }, _ctx.ptmi("root")), [renderSlot(_ctx.$slots, "default")], 16);
}
__name(render$3, "render$3");
script$4.render = render$3;
var classes$2 = {
  root: /* @__PURE__ */ __name(function root2(_ref) {
    var instance = _ref.instance;
    return ["p-steppanel", {
      "p-steppanel-active": instance.isVertical && instance.active
    }];
  }, "root"),
  content: "p-steppanel-content"
};
var StepPanelStyle = BaseStyle.extend({
  name: "steppanel",
  classes: classes$2
});
var script$2$1 = {
  name: "StepperSeparator",
  hostName: "Stepper",
  "extends": script$6
};
function render$1$1(_ctx, _cache, $props, $setup, $data, $options) {
  return openBlock(), createElementBlock("span", mergeProps({
    "class": _ctx.cx("separator")
  }, _ctx.ptm("separator")), null, 16);
}
__name(render$1$1, "render$1$1");
script$2$1.render = render$1$1;
var script$1$2 = {
  name: "BaseStepPanel",
  "extends": script$6,
  props: {
    value: {
      type: [String, Number],
      "default": void 0
    },
    asChild: {
      type: Boolean,
      "default": false
    },
    as: {
      type: [String, Object],
      "default": "DIV"
    }
  },
  style: StepPanelStyle,
  provide: /* @__PURE__ */ __name(function provide3() {
    return {
      $pcStepPanel: this,
      $parentInstance: this
    };
  }, "provide")
};
var script$3 = {
  name: "StepPanel",
  "extends": script$1$2,
  inheritAttrs: false,
  inject: {
    $pcStepper: {
      "default": null
    },
    $pcStepItem: {
      "default": null
    },
    $pcStepList: {
      "default": null
    }
  },
  data: /* @__PURE__ */ __name(function data2() {
    return {
      isSeparatorVisible: false
    };
  }, "data"),
  mounted: /* @__PURE__ */ __name(function mounted2() {
    if (this.$el) {
      var _this$$pcStepItem, _this$$pcStepList;
      var stepElements = find(this.$pcStepper.$el, '[data-pc-name="step"]');
      var stepPanelEl = findSingle(this.isVertical ? (_this$$pcStepItem = this.$pcStepItem) === null || _this$$pcStepItem === void 0 ? void 0 : _this$$pcStepItem.$el : (_this$$pcStepList = this.$pcStepList) === null || _this$$pcStepList === void 0 ? void 0 : _this$$pcStepList.$el, '[data-pc-name="step"]');
      var stepPanelIndex = findIndexInList(stepPanelEl, stepElements);
      this.isSeparatorVisible = this.isVertical && stepPanelIndex !== stepElements.length - 1;
    }
  }, "mounted"),
  methods: {
    getPTOptions: /* @__PURE__ */ __name(function getPTOptions2(key) {
      var _ptm = key === "root" ? this.ptmi : this.ptm;
      return _ptm(key, {
        context: {
          active: this.active
        }
      });
    }, "getPTOptions"),
    updateValue: /* @__PURE__ */ __name(function updateValue(val) {
      this.$pcStepper.updateValue(val);
    }, "updateValue")
  },
  computed: {
    active: /* @__PURE__ */ __name(function active2() {
      var _this$$pcStepItem2, _this$$pcStepper;
      var activeValue3 = !!this.$pcStepItem ? (_this$$pcStepItem2 = this.$pcStepItem) === null || _this$$pcStepItem2 === void 0 ? void 0 : _this$$pcStepItem2.value : this.value;
      return activeValue3 === ((_this$$pcStepper = this.$pcStepper) === null || _this$$pcStepper === void 0 ? void 0 : _this$$pcStepper.d_value);
    }, "active"),
    isVertical: /* @__PURE__ */ __name(function isVertical() {
      return !!this.$pcStepItem;
    }, "isVertical"),
    activeValue: /* @__PURE__ */ __name(function activeValue2() {
      var _this$$pcStepItem3;
      return this.isVertical ? (_this$$pcStepItem3 = this.$pcStepItem) === null || _this$$pcStepItem3 === void 0 ? void 0 : _this$$pcStepItem3.value : this.value;
    }, "activeValue"),
    id: /* @__PURE__ */ __name(function id2() {
      var _this$$pcStepper2;
      return "".concat((_this$$pcStepper2 = this.$pcStepper) === null || _this$$pcStepper2 === void 0 ? void 0 : _this$$pcStepper2.id, "_steppanel_").concat(this.activeValue);
    }, "id"),
    ariaControls: /* @__PURE__ */ __name(function ariaControls2() {
      var _this$$pcStepper3;
      return "".concat((_this$$pcStepper3 = this.$pcStepper) === null || _this$$pcStepper3 === void 0 ? void 0 : _this$$pcStepper3.id, "_step_").concat(this.activeValue);
    }, "ariaControls"),
    a11yAttrs: /* @__PURE__ */ __name(function a11yAttrs2() {
      return {
        id: this.id,
        role: "tabpanel",
        "aria-controls": this.ariaControls,
        "data-pc-name": "steppanel",
        "data-p-active": this.active
      };
    }, "a11yAttrs")
  },
  components: {
    StepperSeparator: script$2$1
  }
};
function render$2(_ctx, _cache, $props, $setup, $data, $options) {
  var _component_StepperSeparator = resolveComponent("StepperSeparator");
  return $options.isVertical ? (openBlock(), createElementBlock(Fragment, {
    key: 0
  }, [!_ctx.asChild ? (openBlock(), createBlock(Transition, mergeProps({
    key: 0,
    name: "p-toggleable-content"
  }, _ctx.ptm("transition")), {
    "default": withCtx(function() {
      return [withDirectives((openBlock(), createBlock(resolveDynamicComponent(_ctx.as), mergeProps({
        id: $options.id,
        "class": _ctx.cx("root"),
        role: "tabpanel",
        "aria-controls": $options.ariaControls
      }, $options.getPTOptions("root")), {
        "default": withCtx(function() {
          return [$data.isSeparatorVisible ? (openBlock(), createBlock(_component_StepperSeparator, {
            key: 0
          })) : createCommentVNode("", true), createBaseVNode("div", mergeProps({
            "class": _ctx.cx("content")
          }, $options.getPTOptions("content")), [renderSlot(_ctx.$slots, "default", {
            active: $options.active,
            activateCallback: /* @__PURE__ */ __name(function activateCallback(val) {
              return $options.updateValue(val);
            }, "activateCallback")
          })], 16)];
        }),
        _: 3
      }, 16, ["id", "class", "aria-controls"])), [[vShow, $options.active]])];
    }),
    _: 3
  }, 16)) : renderSlot(_ctx.$slots, "default", {
    key: 1,
    active: $options.active,
    a11yAttrs: $options.a11yAttrs,
    activateCallback: /* @__PURE__ */ __name(function activateCallback(val) {
      return $options.updateValue(val);
    }, "activateCallback")
  })], 64)) : (openBlock(), createElementBlock(Fragment, {
    key: 1
  }, [!_ctx.asChild ? withDirectives((openBlock(), createBlock(resolveDynamicComponent(_ctx.as), mergeProps({
    key: 0,
    id: $options.id,
    "class": _ctx.cx("root"),
    role: "tabpanel",
    "aria-controls": $options.ariaControls
  }, $options.getPTOptions("root")), {
    "default": withCtx(function() {
      return [renderSlot(_ctx.$slots, "default", {
        active: $options.active,
        activateCallback: /* @__PURE__ */ __name(function activateCallback(val) {
          return $options.updateValue(val);
        }, "activateCallback")
      })];
    }),
    _: 3
  }, 16, ["id", "class", "aria-controls"])), [[vShow, $options.active]]) : _ctx.asChild && $options.active ? renderSlot(_ctx.$slots, "default", {
    key: 1,
    active: $options.active,
    a11yAttrs: $options.a11yAttrs,
    activateCallback: /* @__PURE__ */ __name(function activateCallback(val) {
      return $options.updateValue(val);
    }, "activateCallback")
  }) : createCommentVNode("", true)], 64));
}
__name(render$2, "render$2");
script$3.render = render$2;
var classes$1 = {
  root: "p-steppanels"
};
var StepPanelsStyle = BaseStyle.extend({
  name: "steppanels",
  classes: classes$1
});
var script$1$1 = {
  name: "BaseStepPanels",
  "extends": script$6,
  style: StepPanelsStyle,
  provide: /* @__PURE__ */ __name(function provide4() {
    return {
      $pcStepPanels: this,
      $parentInstance: this
    };
  }, "provide")
};
var script$2 = {
  name: "StepPanels",
  "extends": script$1$1,
  inheritAttrs: false
};
function render$1(_ctx, _cache, $props, $setup, $data, $options) {
  return openBlock(), createElementBlock("div", mergeProps({
    "class": _ctx.cx("root")
  }, _ctx.ptmi("root")), [renderSlot(_ctx.$slots, "default")], 16);
}
__name(render$1, "render$1");
script$2.render = render$1;
var theme = /* @__PURE__ */ __name(function theme2(_ref) {
  var dt = _ref.dt;
  return "\n.p-steplist {\n    position: relative;\n    display: flex;\n    justify-content: space-between;\n    align-items: center;\n    margin: 0;\n    padding: 0;\n    list-style-type: none;\n    overflow-x: auto;\n}\n\n.p-step {\n    position: relative;\n    display: flex;\n    flex: 1 1 auto;\n    align-items: center;\n    gap: ".concat(dt("stepper.step.gap"), ";\n    padding: ").concat(dt("stepper.step.padding"), ";\n}\n\n.p-step:last-of-type {\n    flex: initial;\n}\n\n.p-step-header {\n    border: 0 none;\n    display: inline-flex;\n    align-items: center;\n    text-decoration: none;\n    cursor: pointer;\n    transition: background ").concat(dt("stepper.transition.duration"), ", color ").concat(dt("stepper.transition.duration"), ", border-color ").concat(dt("stepper.transition.duration"), ", outline-color ").concat(dt("stepper.transition.duration"), ", box-shadow ").concat(dt("stepper.transition.duration"), ";\n    border-radius: ").concat(dt("stepper.step.header.border.radius"), ";\n    outline-color: transparent;\n    background: transparent;\n    padding: ").concat(dt("stepper.step.header.padding"), ";\n    gap: ").concat(dt("stepper.step.header.gap"), ";\n}\n\n.p-step-header:focus-visible {\n    box-shadow: ").concat(dt("stepper.step.header.focus.ring.shadow"), ";\n    outline: ").concat(dt("stepper.step.header.focus.ring.width"), " ").concat(dt("stepper.step.header.focus.ring.style"), " ").concat(dt("stepper.step.header.focus.ring.color"), ";\n    outline-offset: ").concat(dt("stepper.step.header.focus.ring.offset"), ";\n}\n\n.p-stepper.p-stepper-readonly .p-step {\n    cursor: auto;\n}\n\n.p-step-title {\n    display: block;\n    white-space: nowrap;\n    overflow: hidden;\n    text-overflow: ellipsis;\n    max-width: 100%;\n    color: ").concat(dt("stepper.step.title.color"), ";\n    font-weight: ").concat(dt("stepper.step.title.font.weight"), ";\n    transition: background ").concat(dt("stepper.transition.duration"), ", color ").concat(dt("stepper.transition.duration"), ", border-color ").concat(dt("stepper.transition.duration"), ", box-shadow ").concat(dt("stepper.transition.duration"), ", outline-color ").concat(dt("stepper.transition.duration"), ";\n}\n\n.p-step-number {\n    display: flex;\n    align-items: center;\n    justify-content: center;\n    color: ").concat(dt("stepper.step.number.color"), ";\n    border: 2px solid ").concat(dt("stepper.step.number.border.color"), ";\n    background: ").concat(dt("stepper.step.number.background"), ";\n    min-width: ").concat(dt("stepper.step.number.size"), ";\n    height: ").concat(dt("stepper.step.number.size"), ";\n    line-height: ").concat(dt("stepper.step.number.size"), ";\n    font-size: ").concat(dt("stepper.step.number.font.size"), ";\n    z-index: 1;\n    border-radius: ").concat(dt("stepper.step.number.border.radius"), ";\n    position: relative;\n    font-weight: ").concat(dt("stepper.step.number.font.weight"), ';\n}\n\n.p-step-number::after {\n    content: " ";\n    position: absolute;\n    width: 100%;\n    height: 100%;\n    border-radius: ').concat(dt("stepper.step.number.border.radius"), ";\n    box-shadow: ").concat(dt("stepper.step.number.shadow"), ";\n}\n\n.p-step-active .p-step-header {\n    cursor: default;\n}\n\n.p-step-active .p-step-number {\n    background: ").concat(dt("stepper.step.number.active.background"), ";\n    border-color: ").concat(dt("stepper.step.number.active.border.color"), ";\n    color: ").concat(dt("stepper.step.number.active.color"), ";\n}\n\n.p-step-active .p-step-title {\n    color: ").concat(dt("stepper.step.title.active.color"), ";\n}\n\n.p-step:not(.p-disabled):focus-visible {\n    outline: ").concat(dt("focus.ring.width"), " ").concat(dt("focus.ring.style"), " ").concat(dt("focus.ring.color"), ";\n    outline-offset: ").concat(dt("focus.ring.offset"), ";\n}\n\n.p-step:has(~ .p-step-active) .p-stepper-separator {\n    background: ").concat(dt("stepper.separator.active.background"), ";\n}\n\n.p-stepper-separator {\n    flex: 1 1 0;\n    background: ").concat(dt("stepper.separator.background"), ";\n    width: 100%;\n    height: ").concat(dt("stepper.separator.size"), ";\n    transition: background ").concat(dt("stepper.transition.duration"), ", color ").concat(dt("stepper.transition.duration"), ", border-color ").concat(dt("stepper.transition.duration"), ", box-shadow ").concat(dt("stepper.transition.duration"), ", outline-color ").concat(dt("stepper.transition.duration"), ";\n}\n\n.p-steppanels {\n    padding: ").concat(dt("stepper.steppanels.padding"), ";\n}\n\n.p-steppanel {\n    background: ").concat(dt("stepper.steppanel.background"), ";\n    color: ").concat(dt("stepper.steppanel.color"), ";\n}\n\n.p-stepper:has(.p-stepitem) {\n    display: flex;\n    flex-direction: column;\n}\n\n.p-stepitem {\n    display: flex;\n    flex-direction: column;\n    flex: initial;\n}\n\n.p-stepitem.p-stepitem-active {\n    flex: 1 1 auto;\n}\n\n.p-stepitem .p-step {\n    flex: initial;\n}\n\n.p-stepitem .p-steppanel-content {\n    width: 100%;\n    padding: ").concat(dt("stepper.steppanel.padding"), ";\n}\n\n.p-stepitem .p-steppanel {\n    display: flex;\n    flex: 1 1 auto;\n}\n\n.p-stepitem .p-stepper-separator {\n    flex: 0 0 auto;\n    width: ").concat(dt("stepper.separator.size"), ";\n    height: auto;\n    margin: ").concat(dt("stepper.separator.margin"), ";\n    position: relative;\n    left: calc(-1 * ").concat(dt("stepper.separator.size"), ");\n}\n\n.p-stepitem:has(~ .p-stepitem-active) .p-stepper-separator {\n    background: ").concat(dt("stepper.separator.active.background"), ";\n}\n\n.p-stepitem:last-of-type .p-steppanel {\n    padding-inline-start: ").concat(dt("stepper.step.number.size"), ";\n}\n");
}, "theme");
var classes = {
  root: /* @__PURE__ */ __name(function root3(_ref2) {
    var props = _ref2.props;
    return ["p-stepper p-component", {
      "p-readonly": props.linear
    }];
  }, "root"),
  separator: "p-stepper-separator"
};
var StepperStyle = BaseStyle.extend({
  name: "stepper",
  theme,
  classes
});
var script$1 = {
  name: "BaseStepper",
  "extends": script$6,
  props: {
    value: {
      type: [String, Number],
      "default": void 0
    },
    linear: {
      type: Boolean,
      "default": false
    }
  },
  style: StepperStyle,
  provide: /* @__PURE__ */ __name(function provide5() {
    return {
      $pcStepper: this,
      $parentInstance: this
    };
  }, "provide")
};
var script = {
  name: "Stepper",
  "extends": script$1,
  inheritAttrs: false,
  emits: ["update:value"],
  data: /* @__PURE__ */ __name(function data3() {
    return {
      id: this.$attrs.id,
      d_value: this.value
    };
  }, "data"),
  watch: {
    "$attrs.id": /* @__PURE__ */ __name(function $attrsId(newValue) {
      this.id = newValue || UniqueComponentId();
    }, "$attrsId"),
    value: /* @__PURE__ */ __name(function value(newValue) {
      this.d_value = newValue;
    }, "value")
  },
  mounted: /* @__PURE__ */ __name(function mounted3() {
    this.id = this.id || UniqueComponentId();
  }, "mounted"),
  methods: {
    updateValue: /* @__PURE__ */ __name(function updateValue2(newValue) {
      if (this.d_value !== newValue) {
        this.d_value = newValue;
        this.$emit("update:value", newValue);
      }
    }, "updateValue"),
    isStepActive: /* @__PURE__ */ __name(function isStepActive(value2) {
      return this.d_value === value2;
    }, "isStepActive"),
    isStepDisabled: /* @__PURE__ */ __name(function isStepDisabled2() {
      return this.linear;
    }, "isStepDisabled")
  }
};
function render(_ctx, _cache, $props, $setup, $data, $options) {
  return openBlock(), createElementBlock("div", mergeProps({
    "class": _ctx.cx("root"),
    role: "tablist"
  }, _ctx.ptmi("root")), [_ctx.$slots.start ? renderSlot(_ctx.$slots, "start", {
    key: 0
  }) : createCommentVNode("", true), renderSlot(_ctx.$slots, "default"), _ctx.$slots.end ? renderSlot(_ctx.$slots, "end", {
    key: 1
  }) : createCommentVNode("", true)], 16);
}
__name(render, "render");
script.render = render;
const _hoisted_1$4 = { class: "flex flex-col gap-6 w-[600px]" };
const _hoisted_2$4 = { class: "flex flex-col gap-4" };
const _hoisted_3$4 = { class: "text-2xl font-semibold text-neutral-100" };
const _hoisted_4$4 = { class: "text-neutral-400 my-0" };
const _hoisted_5$3 = { class: "flex flex-col bg-neutral-800 p-4 rounded-lg" };
const _hoisted_6$3 = { class: "flex items-center gap-4" };
const _hoisted_7$3 = { class: "flex-1" };
const _hoisted_8$3 = { class: "text-lg font-medium text-neutral-100" };
const _hoisted_9$3 = { class: "text-sm text-neutral-400 mt-1" };
const _hoisted_10$3 = { class: "flex items-center gap-4" };
const _hoisted_11$3 = { class: "flex-1" };
const _hoisted_12$3 = { class: "text-lg font-medium text-neutral-100" };
const _hoisted_13$2 = { class: "text-sm text-neutral-400 mt-1" };
const _hoisted_14$2 = { class: "text-neutral-300" };
const _hoisted_15$2 = { class: "font-medium mb-2" };
const _hoisted_16$2 = { class: "list-disc pl-6 space-y-1" };
const _hoisted_17$2 = { class: "font-medium mt-4 mb-2" };
const _hoisted_18$2 = { class: "list-disc pl-6 space-y-1" };
const _sfc_main$4 = /* @__PURE__ */ defineComponent({
  __name: "DesktopSettingsConfiguration",
  props: {
    "autoUpdate": { required: true },
    "autoUpdateModifiers": {},
    "allowMetrics": { required: true },
    "allowMetricsModifiers": {}
  },
  emits: ["update:autoUpdate", "update:allowMetrics"],
  setup(__props) {
    const showDialog = ref(false);
    const autoUpdate = useModel(__props, "autoUpdate");
    const allowMetrics = useModel(__props, "allowMetrics");
    const showMetricsInfo = /* @__PURE__ */ __name(() => {
      showDialog.value = true;
    }, "showMetricsInfo");
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", _hoisted_1$4, [
        createBaseVNode("div", _hoisted_2$4, [
          createBaseVNode("h2", _hoisted_3$4, toDisplayString(_ctx.$t("install.desktopAppSettings")), 1),
          createBaseVNode("p", _hoisted_4$4, toDisplayString(_ctx.$t("install.desktopAppSettingsDescription")), 1)
        ]),
        createBaseVNode("div", _hoisted_5$3, [
          createBaseVNode("div", _hoisted_6$3, [
            createBaseVNode("div", _hoisted_7$3, [
              createBaseVNode("h3", _hoisted_8$3, toDisplayString(_ctx.$t("install.settings.autoUpdate")), 1),
              createBaseVNode("p", _hoisted_9$3, toDisplayString(_ctx.$t("install.settings.autoUpdateDescription")), 1)
            ]),
            createVNode(unref(script$7), {
              modelValue: autoUpdate.value,
              "onUpdate:modelValue": _cache[0] || (_cache[0] = ($event) => autoUpdate.value = $event)
            }, null, 8, ["modelValue"])
          ]),
          createVNode(unref(script$8)),
          createBaseVNode("div", _hoisted_10$3, [
            createBaseVNode("div", _hoisted_11$3, [
              createBaseVNode("h3", _hoisted_12$3, toDisplayString(_ctx.$t("install.settings.allowMetrics")), 1),
              createBaseVNode("p", _hoisted_13$2, toDisplayString(_ctx.$t("install.settings.allowMetricsDescription")), 1),
              createBaseVNode("a", {
                href: "#",
                class: "text-sm text-blue-400 hover:text-blue-300 mt-1 inline-block",
                onClick: withModifiers(showMetricsInfo, ["prevent"])
              }, toDisplayString(_ctx.$t("install.settings.learnMoreAboutData")), 1)
            ]),
            createVNode(unref(script$7), {
              modelValue: allowMetrics.value,
              "onUpdate:modelValue": _cache[1] || (_cache[1] = ($event) => allowMetrics.value = $event)
            }, null, 8, ["modelValue"])
          ])
        ]),
        createVNode(unref(script$9), {
          visible: showDialog.value,
          "onUpdate:visible": _cache[2] || (_cache[2] = ($event) => showDialog.value = $event),
          modal: "",
          header: _ctx.$t("install.settings.dataCollectionDialog.title")
        }, {
          default: withCtx(() => [
            createBaseVNode("div", _hoisted_14$2, [
              createBaseVNode("h4", _hoisted_15$2, toDisplayString(_ctx.$t("install.settings.dataCollectionDialog.whatWeCollect")), 1),
              createBaseVNode("ul", _hoisted_16$2, [
                createBaseVNode("li", null, toDisplayString(_ctx.$t("install.settings.dataCollectionDialog.errorReports")), 1),
                createBaseVNode("li", null, toDisplayString(_ctx.$t("install.settings.dataCollectionDialog.systemInfo")), 1)
              ]),
              createBaseVNode("h4", _hoisted_17$2, toDisplayString(_ctx.$t("install.settings.dataCollectionDialog.whatWeDoNotCollect")), 1),
              createBaseVNode("ul", _hoisted_18$2, [
                createBaseVNode("li", null, toDisplayString(_ctx.$t("install.settings.dataCollectionDialog.personalInformation")), 1),
                createBaseVNode("li", null, toDisplayString(_ctx.$t("install.settings.dataCollectionDialog.workflowContents")), 1),
                createBaseVNode("li", null, toDisplayString(_ctx.$t("install.settings.dataCollectionDialog.fileSystemInformation")), 1),
                createBaseVNode("li", null, toDisplayString(_ctx.$t(
                  "install.settings.dataCollectionDialog.customNodeConfigurations"
                )), 1)
              ])
            ])
          ]),
          _: 1
        }, 8, ["visible", "header"])
      ]);
    };
  }
});
const _imports_0 = "" + new URL("images/nvidia-logo.svg", import.meta.url).href;
const _imports_1 = "" + new URL("images/apple-mps-logo.png", import.meta.url).href;
const _imports_2 = "" + new URL("images/manual-configuration.svg", import.meta.url).href;
const _hoisted_1$3 = { class: "flex flex-col gap-6 w-[600px] h-[30rem] select-none" };
const _hoisted_2$3 = { class: "grow flex flex-col gap-4 text-neutral-300" };
const _hoisted_3$3 = { class: "text-2xl font-semibold text-neutral-100" };
const _hoisted_4$3 = { class: "m-1 text-neutral-400" };
const _hoisted_5$2 = /* @__PURE__ */ createBaseVNode("img", {
  class: "m-12",
  alt: "NVIDIA logo",
  width: "196",
  height: "32",
  src: _imports_0
}, null, -1);
const _hoisted_6$2 = [
  _hoisted_5$2
];
const _hoisted_7$2 = /* @__PURE__ */ createBaseVNode("img", {
  class: "rounded-lg hover-brighten",
  alt: "Apple Metal Performance Shaders Logo",
  width: "292",
  ratio: "",
  src: _imports_1
}, null, -1);
const _hoisted_8$2 = [
  _hoisted_7$2
];
const _hoisted_9$2 = /* @__PURE__ */ createBaseVNode("img", {
  class: "m-12",
  alt: "Manual configuration",
  width: "196",
  src: _imports_2
}, null, -1);
const _hoisted_10$2 = [
  _hoisted_9$2
];
const _hoisted_11$2 = {
  key: 0,
  class: "m-1"
};
const _hoisted_12$2 = {
  key: 1,
  class: "m-1"
};
const _hoisted_13$1 = {
  key: 2,
  class: "text-neutral-300"
};
const _hoisted_14$1 = { class: "m-1" };
const _hoisted_15$1 = { key: 3 };
const _hoisted_16$1 = { class: "m-1" };
const _hoisted_17$1 = { class: "m-1" };
const _hoisted_18$1 = {
  for: "cpu-mode",
  class: "select-none"
};
const _sfc_main$3 = /* @__PURE__ */ defineComponent({
  __name: "GpuPicker",
  props: {
    "device": {
      required: true
    },
    "deviceModifiers": {}
  },
  emits: ["update:device"],
  setup(__props) {
    const { t } = useI18n();
    const cpuMode = computed({
      get: /* @__PURE__ */ __name(() => selected.value === "cpu", "get"),
      set: /* @__PURE__ */ __name((value2) => {
        selected.value = value2 ? "cpu" : null;
      }, "set")
    });
    const selected = useModel(__props, "device");
    const electron = electronAPI();
    const platform = electron.getPlatform();
    const pickGpu = /* @__PURE__ */ __name((value2) => {
      const newValue = selected.value === value2 ? null : value2;
      selected.value = newValue;
    }, "pickGpu");
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", _hoisted_1$3, [
        createBaseVNode("div", _hoisted_2$3, [
          createBaseVNode("h2", _hoisted_3$3, toDisplayString(_ctx.$t("install.gpuSelection.selectGpu")), 1),
          createBaseVNode("p", _hoisted_4$3, toDisplayString(_ctx.$t("install.gpuSelection.selectGpuDescription")) + ": ", 1),
          createBaseVNode("div", {
            class: normalizeClass(["flex gap-2 text-center transition-opacity", { selected: selected.value }])
          }, [
            unref(platform) !== "darwin" ? (openBlock(), createElementBlock("div", {
              key: 0,
              class: normalizeClass(["gpu-button", { selected: selected.value === "nvidia" }]),
              role: "button",
              onClick: _cache[0] || (_cache[0] = ($event) => pickGpu("nvidia"))
            }, _hoisted_6$2, 2)) : createCommentVNode("", true),
            unref(platform) === "darwin" ? (openBlock(), createElementBlock("div", {
              key: 1,
              class: normalizeClass(["gpu-button", { selected: selected.value === "mps" }]),
              role: "button",
              onClick: _cache[1] || (_cache[1] = ($event) => pickGpu("mps"))
            }, _hoisted_8$2, 2)) : createCommentVNode("", true),
            createBaseVNode("div", {
              class: normalizeClass(["gpu-button", { selected: selected.value === "unsupported" }]),
              role: "button",
              onClick: _cache[2] || (_cache[2] = ($event) => pickGpu("unsupported"))
            }, _hoisted_10$2, 2)
          ], 2),
          selected.value === "nvidia" ? (openBlock(), createElementBlock("p", _hoisted_11$2, [
            createVNode(unref(script$a), {
              icon: "pi pi-check",
              severity: "success",
              value: "CUDA"
            }),
            createTextVNode(" " + toDisplayString(_ctx.$t("install.gpuSelection.nvidiaDescription")), 1)
          ])) : createCommentVNode("", true),
          selected.value === "mps" ? (openBlock(), createElementBlock("p", _hoisted_12$2, [
            createVNode(unref(script$a), {
              icon: "pi pi-check",
              severity: "success",
              value: "MPS"
            }),
            createTextVNode(" " + toDisplayString(_ctx.$t("install.gpuSelection.mpsDescription")), 1)
          ])) : createCommentVNode("", true),
          selected.value === "unsupported" ? (openBlock(), createElementBlock("div", _hoisted_13$1, [
            createBaseVNode("p", _hoisted_14$1, [
              createVNode(unref(script$a), {
                icon: "pi pi-exclamation-triangle",
                severity: "warn",
                value: unref(t)("icon.exclamation-triangle")
              }, null, 8, ["value"]),
              createTextVNode(" " + toDisplayString(_ctx.$t("install.gpuSelection.customSkipsPython")), 1)
            ]),
            createBaseVNode("ul", null, [
              createBaseVNode("li", null, [
                createBaseVNode("strong", null, toDisplayString(_ctx.$t("install.gpuSelection.customComfyNeedsPython")), 1)
              ]),
              createBaseVNode("li", null, toDisplayString(_ctx.$t("install.gpuSelection.customManualVenv")), 1),
              createBaseVNode("li", null, toDisplayString(_ctx.$t("install.gpuSelection.customInstallRequirements")), 1),
              createBaseVNode("li", null, toDisplayString(_ctx.$t("install.gpuSelection.customMayNotWork")), 1)
            ])
          ])) : createCommentVNode("", true),
          selected.value === "cpu" ? (openBlock(), createElementBlock("div", _hoisted_15$1, [
            createBaseVNode("p", _hoisted_16$1, [
              createVNode(unref(script$a), {
                icon: "pi pi-exclamation-triangle",
                severity: "warn",
                value: unref(t)("icon.exclamation-triangle")
              }, null, 8, ["value"]),
              createTextVNode(" " + toDisplayString(_ctx.$t("install.gpuSelection.cpuModeDescription")), 1)
            ]),
            createBaseVNode("p", _hoisted_17$1, toDisplayString(_ctx.$t("install.gpuSelection.cpuModeDescription2")), 1)
          ])) : createCommentVNode("", true)
        ]),
        createBaseVNode("div", {
          class: normalizeClass(["transition-opacity flex gap-3 h-0", {
            "opacity-40": selected.value && selected.value !== "cpu"
          }])
        }, [
          createVNode(unref(script$7), {
            modelValue: cpuMode.value,
            "onUpdate:modelValue": _cache[3] || (_cache[3] = ($event) => cpuMode.value = $event),
            inputId: "cpu-mode",
            class: "-translate-y-40"
          }, null, 8, ["modelValue"]),
          createBaseVNode("label", _hoisted_18$1, toDisplayString(_ctx.$t("install.gpuSelection.enableCpuMode")), 1)
        ], 2)
      ]);
    };
  }
});
const _hoisted_1$2 = { class: "flex flex-col gap-6 w-[600px]" };
const _hoisted_2$2 = { class: "flex flex-col gap-4" };
const _hoisted_3$2 = { class: "text-2xl font-semibold text-neutral-100" };
const _hoisted_4$2 = { class: "text-neutral-400 my-0" };
const _hoisted_5$1 = { class: "flex gap-2" };
const _hoisted_6$1 = { class: "bg-neutral-800 p-4 rounded-lg" };
const _hoisted_7$1 = { class: "text-lg font-medium mt-0 mb-3 text-neutral-100" };
const _hoisted_8$1 = { class: "flex flex-col gap-2" };
const _hoisted_9$1 = { class: "flex items-center gap-2" };
const _hoisted_10$1 = /* @__PURE__ */ createBaseVNode("i", { class: "pi pi-folder text-neutral-400" }, null, -1);
const _hoisted_11$1 = /* @__PURE__ */ createBaseVNode("span", { class: "text-neutral-400" }, "App Data:", -1);
const _hoisted_12$1 = { class: "text-neutral-200" };
const _hoisted_13 = { class: "pi pi-info-circle" };
const _hoisted_14 = { class: "flex items-center gap-2" };
const _hoisted_15 = /* @__PURE__ */ createBaseVNode("i", { class: "pi pi-desktop text-neutral-400" }, null, -1);
const _hoisted_16 = /* @__PURE__ */ createBaseVNode("span", { class: "text-neutral-400" }, "App Path:", -1);
const _hoisted_17 = { class: "text-neutral-200" };
const _hoisted_18 = { class: "pi pi-info-circle" };
const _sfc_main$2 = /* @__PURE__ */ defineComponent({
  __name: "InstallLocationPicker",
  props: {
    "installPath": { required: true },
    "installPathModifiers": {},
    "pathError": { required: true },
    "pathErrorModifiers": {}
  },
  emits: ["update:installPath", "update:pathError"],
  setup(__props) {
    const { t } = useI18n();
    const installPath = useModel(__props, "installPath");
    const pathError = useModel(__props, "pathError");
    const pathExists = ref(false);
    const appData = ref("");
    const appPath = ref("");
    const electron = electronAPI();
    onMounted(async () => {
      const paths = await electron.getSystemPaths();
      appData.value = paths.appData;
      appPath.value = paths.appPath;
      installPath.value = paths.defaultInstallPath;
      await validatePath(paths.defaultInstallPath);
    });
    const validatePath = /* @__PURE__ */ __name(async (path) => {
      try {
        pathError.value = "";
        pathExists.value = false;
        const validation = await electron.validateInstallPath(path);
        if (!validation.isValid) {
          const errors = [];
          if (validation.cannotWrite) errors.push(t("install.cannotWrite"));
          if (validation.freeSpace < validation.requiredSpace) {
            const requiredGB = validation.requiredSpace / 1024 / 1024 / 1024;
            errors.push(`${t("install.insufficientFreeSpace")}: ${requiredGB} GB`);
          }
          if (validation.parentMissing) errors.push(t("install.parentMissing"));
          if (validation.error)
            errors.push(`${t("install.unhandledError")}: ${validation.error}`);
          pathError.value = errors.join("\n");
        }
        if (validation.exists) pathExists.value = true;
      } catch (error) {
        pathError.value = t("install.pathValidationFailed");
      }
    }, "validatePath");
    const browsePath = /* @__PURE__ */ __name(async () => {
      try {
        const result = await electron.showDirectoryPicker();
        if (result) {
          installPath.value = result;
          await validatePath(result);
        }
      } catch (error) {
        pathError.value = t("install.failedToSelectDirectory");
      }
    }, "browsePath");
    return (_ctx, _cache) => {
      const _directive_tooltip = resolveDirective("tooltip");
      return openBlock(), createElementBlock("div", _hoisted_1$2, [
        createBaseVNode("div", _hoisted_2$2, [
          createBaseVNode("h2", _hoisted_3$2, toDisplayString(_ctx.$t("install.chooseInstallationLocation")), 1),
          createBaseVNode("p", _hoisted_4$2, toDisplayString(_ctx.$t("install.installLocationDescription")), 1),
          createBaseVNode("div", _hoisted_5$1, [
            createVNode(unref(script$d), { class: "flex-1" }, {
              default: withCtx(() => [
                createVNode(unref(script$b), {
                  modelValue: installPath.value,
                  "onUpdate:modelValue": [
                    _cache[0] || (_cache[0] = ($event) => installPath.value = $event),
                    validatePath
                  ],
                  class: normalizeClass(["w-full", { "p-invalid": pathError.value }])
                }, null, 8, ["modelValue", "class"]),
                withDirectives(createVNode(unref(script$c), { class: "pi pi-info-circle" }, null, 512), [
                  [_directive_tooltip, _ctx.$t("install.installLocationTooltip")]
                ])
              ]),
              _: 1
            }),
            createVNode(unref(script$e), {
              icon: "pi pi-folder",
              onClick: browsePath,
              class: "w-12"
            })
          ]),
          pathError.value ? (openBlock(), createBlock(unref(script$f), {
            key: 0,
            severity: "error",
            class: "whitespace-pre-line"
          }, {
            default: withCtx(() => [
              createTextVNode(toDisplayString(pathError.value), 1)
            ]),
            _: 1
          })) : createCommentVNode("", true),
          pathExists.value ? (openBlock(), createBlock(unref(script$f), {
            key: 1,
            severity: "warn"
          }, {
            default: withCtx(() => [
              createTextVNode(toDisplayString(_ctx.$t("install.pathExists")), 1)
            ]),
            _: 1
          })) : createCommentVNode("", true)
        ]),
        createBaseVNode("div", _hoisted_6$1, [
          createBaseVNode("h3", _hoisted_7$1, toDisplayString(_ctx.$t("install.systemLocations")), 1),
          createBaseVNode("div", _hoisted_8$1, [
            createBaseVNode("div", _hoisted_9$1, [
              _hoisted_10$1,
              _hoisted_11$1,
              createBaseVNode("span", _hoisted_12$1, toDisplayString(appData.value), 1),
              withDirectives(createBaseVNode("span", _hoisted_13, null, 512), [
                [_directive_tooltip, _ctx.$t("install.appDataLocationTooltip")]
              ])
            ]),
            createBaseVNode("div", _hoisted_14, [
              _hoisted_15,
              _hoisted_16,
              createBaseVNode("span", _hoisted_17, toDisplayString(appPath.value), 1),
              withDirectives(createBaseVNode("span", _hoisted_18, null, 512), [
                [_directive_tooltip, _ctx.$t("install.appPathLocationTooltip")]
              ])
            ])
          ])
        ])
      ]);
    };
  }
});
const _hoisted_1$1 = { class: "flex flex-col gap-6 w-[600px]" };
const _hoisted_2$1 = { class: "flex flex-col gap-4" };
const _hoisted_3$1 = { class: "text-2xl font-semibold text-neutral-100" };
const _hoisted_4$1 = { class: "text-neutral-400 my-0" };
const _hoisted_5 = { class: "flex gap-2" };
const _hoisted_6 = {
  key: 0,
  class: "flex flex-col gap-4 bg-neutral-800 p-4 rounded-lg"
};
const _hoisted_7 = { class: "text-lg mt-0 font-medium text-neutral-100" };
const _hoisted_8 = { class: "flex flex-col gap-3" };
const _hoisted_9 = ["onClick"];
const _hoisted_10 = ["for"];
const _hoisted_11 = { class: "text-sm text-neutral-400 my-1" };
const _hoisted_12 = {
  key: 1,
  class: "text-neutral-400 italic"
};
const _sfc_main$1 = /* @__PURE__ */ defineComponent({
  __name: "MigrationPicker",
  props: {
    "sourcePath": { required: false },
    "sourcePathModifiers": {},
    "migrationItemIds": {
      required: false
    },
    "migrationItemIdsModifiers": {}
  },
  emits: ["update:sourcePath", "update:migrationItemIds"],
  setup(__props) {
    const { t } = useI18n();
    const electron = electronAPI();
    const sourcePath = useModel(__props, "sourcePath");
    const migrationItemIds = useModel(__props, "migrationItemIds");
    const migrationItems = ref(
      MigrationItems.map((item) => ({
        ...item,
        selected: true
      }))
    );
    const pathError = ref("");
    const isValidSource = computed(
      () => sourcePath.value !== "" && pathError.value === ""
    );
    const validateSource = /* @__PURE__ */ __name(async (sourcePath2) => {
      if (!sourcePath2) {
        pathError.value = "";
        return;
      }
      try {
        pathError.value = "";
        const validation = await electron.validateComfyUISource(sourcePath2);
        if (!validation.isValid) pathError.value = validation.error;
      } catch (error) {
        console.error(error);
        pathError.value = t("install.pathValidationFailed");
      }
    }, "validateSource");
    const browsePath = /* @__PURE__ */ __name(async () => {
      try {
        const result = await electron.showDirectoryPicker();
        if (result) {
          sourcePath.value = result;
          await validateSource(result);
        }
      } catch (error) {
        console.error(error);
        pathError.value = t("install.failedToSelectDirectory");
      }
    }, "browsePath");
    watchEffect(() => {
      migrationItemIds.value = migrationItems.value.filter((item) => item.selected).map((item) => item.id);
    });
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", _hoisted_1$1, [
        createBaseVNode("div", _hoisted_2$1, [
          createBaseVNode("h2", _hoisted_3$1, toDisplayString(_ctx.$t("install.migrateFromExistingInstallation")), 1),
          createBaseVNode("p", _hoisted_4$1, toDisplayString(_ctx.$t("install.migrationSourcePathDescription")), 1),
          createBaseVNode("div", _hoisted_5, [
            createVNode(unref(script$b), {
              modelValue: sourcePath.value,
              "onUpdate:modelValue": [
                _cache[0] || (_cache[0] = ($event) => sourcePath.value = $event),
                validateSource
              ],
              placeholder: "Select existing ComfyUI installation (optional)",
              class: normalizeClass(["flex-1", { "p-invalid": pathError.value }])
            }, null, 8, ["modelValue", "class"]),
            createVNode(unref(script$e), {
              icon: "pi pi-folder",
              onClick: browsePath,
              class: "w-12"
            })
          ]),
          pathError.value ? (openBlock(), createBlock(unref(script$f), {
            key: 0,
            severity: "error"
          }, {
            default: withCtx(() => [
              createTextVNode(toDisplayString(pathError.value), 1)
            ]),
            _: 1
          })) : createCommentVNode("", true)
        ]),
        isValidSource.value ? (openBlock(), createElementBlock("div", _hoisted_6, [
          createBaseVNode("h3", _hoisted_7, toDisplayString(_ctx.$t("install.selectItemsToMigrate")), 1),
          createBaseVNode("div", _hoisted_8, [
            (openBlock(true), createElementBlock(Fragment, null, renderList(migrationItems.value, (item) => {
              return openBlock(), createElementBlock("div", {
                key: item.id,
                class: "flex items-center gap-3 p-2 hover:bg-neutral-700 rounded",
                onClick: /* @__PURE__ */ __name(($event) => item.selected = !item.selected, "onClick")
              }, [
                createVNode(unref(script$g), {
                  modelValue: item.selected,
                  "onUpdate:modelValue": /* @__PURE__ */ __name(($event) => item.selected = $event, "onUpdate:modelValue"),
                  inputId: item.id,
                  binary: true,
                  onClick: _cache[1] || (_cache[1] = withModifiers(() => {
                  }, ["stop"]))
                }, null, 8, ["modelValue", "onUpdate:modelValue", "inputId"]),
                createBaseVNode("div", null, [
                  createBaseVNode("label", {
                    for: item.id,
                    class: "text-neutral-200 font-medium"
                  }, toDisplayString(item.label), 9, _hoisted_10),
                  createBaseVNode("p", _hoisted_11, toDisplayString(item.description), 1)
                ])
              ], 8, _hoisted_9);
            }), 128))
          ])
        ])) : (openBlock(), createElementBlock("div", _hoisted_12, toDisplayString(_ctx.$t("install.migrationOptional")), 1))
      ]);
    };
  }
});
const _withScopeId = /* @__PURE__ */ __name((n) => (pushScopeId("data-v-de33872d"), n = n(), popScopeId(), n), "_withScopeId");
const _hoisted_1 = { class: "flex pt-6 justify-end" };
const _hoisted_2 = { class: "flex pt-6 justify-between" };
const _hoisted_3 = { class: "flex pt-6 justify-between" };
const _hoisted_4 = { class: "flex pt-6 justify-between" };
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "InstallView",
  setup(__props) {
    const device = ref(null);
    const installPath = ref("");
    const pathError = ref("");
    const migrationSourcePath = ref("");
    const migrationItemIds = ref([]);
    const autoUpdate = ref(true);
    const allowMetrics = ref(true);
    const highestStep = ref(0);
    const setHighestStep = /* @__PURE__ */ __name((value2) => {
      const int = typeof value2 === "number" ? value2 : parseInt(value2, 10);
      if (!isNaN(int) && int > highestStep.value) highestStep.value = int;
    }, "setHighestStep");
    const hasError = computed(() => pathError.value !== "");
    const noGpu = computed(() => typeof device.value !== "string");
    const electron = electronAPI();
    const router = useRouter();
    const install = /* @__PURE__ */ __name(() => {
      const options = {
        installPath: installPath.value,
        autoUpdate: autoUpdate.value,
        allowMetrics: allowMetrics.value,
        migrationSourcePath: migrationSourcePath.value,
        migrationItemIds: toRaw(migrationItemIds.value),
        device: device.value
      };
      electron.installComfyUI(options);
      const nextPage = options.device === "unsupported" ? "/manual-configuration" : "/server-start";
      router.push(nextPage);
    }, "install");
    onMounted(async () => {
      if (!electron) return;
      const detectedGpu = await electron.Config.getDetectedGpu();
      if (detectedGpu === "mps" || detectedGpu === "nvidia")
        device.value = detectedGpu;
    });
    return (_ctx, _cache) => {
      return openBlock(), createBlock(_sfc_main$5, { dark: "" }, {
        default: withCtx(() => [
          createVNode(unref(script), {
            class: "h-full p-8 2xl:p-16",
            value: "0",
            "onUpdate:value": setHighestStep
          }, {
            default: withCtx(() => [
              createVNode(unref(script$4), { class: "select-none" }, {
                default: withCtx(() => [
                  createVNode(unref(script$5), { value: "0" }, {
                    default: withCtx(() => [
                      createTextVNode(toDisplayString(_ctx.$t("install.gpu")), 1)
                    ]),
                    _: 1
                  }),
                  createVNode(unref(script$5), {
                    value: "1",
                    disabled: noGpu.value
                  }, {
                    default: withCtx(() => [
                      createTextVNode(toDisplayString(_ctx.$t("install.installLocation")), 1)
                    ]),
                    _: 1
                  }, 8, ["disabled"]),
                  createVNode(unref(script$5), {
                    value: "2",
                    disabled: noGpu.value || hasError.value || highestStep.value < 1
                  }, {
                    default: withCtx(() => [
                      createTextVNode(toDisplayString(_ctx.$t("install.migration")), 1)
                    ]),
                    _: 1
                  }, 8, ["disabled"]),
                  createVNode(unref(script$5), {
                    value: "3",
                    disabled: noGpu.value || hasError.value || highestStep.value < 2
                  }, {
                    default: withCtx(() => [
                      createTextVNode(toDisplayString(_ctx.$t("install.desktopSettings")), 1)
                    ]),
                    _: 1
                  }, 8, ["disabled"])
                ]),
                _: 1
              }),
              createVNode(unref(script$2), null, {
                default: withCtx(() => [
                  createVNode(unref(script$3), { value: "0" }, {
                    default: withCtx(({ activateCallback }) => [
                      createVNode(_sfc_main$3, {
                        device: device.value,
                        "onUpdate:device": _cache[0] || (_cache[0] = ($event) => device.value = $event)
                      }, null, 8, ["device"]),
                      createBaseVNode("div", _hoisted_1, [
                        createVNode(unref(script$e), {
                          label: _ctx.$t("g.next"),
                          icon: "pi pi-arrow-right",
                          iconPos: "right",
                          onClick: /* @__PURE__ */ __name(($event) => activateCallback("1"), "onClick"),
                          disabled: typeof device.value !== "string"
                        }, null, 8, ["label", "onClick", "disabled"])
                      ])
                    ]),
                    _: 1
                  }),
                  createVNode(unref(script$3), { value: "1" }, {
                    default: withCtx(({ activateCallback }) => [
                      createVNode(_sfc_main$2, {
                        installPath: installPath.value,
                        "onUpdate:installPath": _cache[1] || (_cache[1] = ($event) => installPath.value = $event),
                        pathError: pathError.value,
                        "onUpdate:pathError": _cache[2] || (_cache[2] = ($event) => pathError.value = $event)
                      }, null, 8, ["installPath", "pathError"]),
                      createBaseVNode("div", _hoisted_2, [
                        createVNode(unref(script$e), {
                          label: _ctx.$t("g.back"),
                          severity: "secondary",
                          icon: "pi pi-arrow-left",
                          onClick: /* @__PURE__ */ __name(($event) => activateCallback("0"), "onClick")
                        }, null, 8, ["label", "onClick"]),
                        createVNode(unref(script$e), {
                          label: _ctx.$t("g.next"),
                          icon: "pi pi-arrow-right",
                          iconPos: "right",
                          onClick: /* @__PURE__ */ __name(($event) => activateCallback("2"), "onClick"),
                          disabled: pathError.value !== ""
                        }, null, 8, ["label", "onClick", "disabled"])
                      ])
                    ]),
                    _: 1
                  }),
                  createVNode(unref(script$3), { value: "2" }, {
                    default: withCtx(({ activateCallback }) => [
                      createVNode(_sfc_main$1, {
                        sourcePath: migrationSourcePath.value,
                        "onUpdate:sourcePath": _cache[3] || (_cache[3] = ($event) => migrationSourcePath.value = $event),
                        migrationItemIds: migrationItemIds.value,
                        "onUpdate:migrationItemIds": _cache[4] || (_cache[4] = ($event) => migrationItemIds.value = $event)
                      }, null, 8, ["sourcePath", "migrationItemIds"]),
                      createBaseVNode("div", _hoisted_3, [
                        createVNode(unref(script$e), {
                          label: _ctx.$t("g.back"),
                          severity: "secondary",
                          icon: "pi pi-arrow-left",
                          onClick: /* @__PURE__ */ __name(($event) => activateCallback("1"), "onClick")
                        }, null, 8, ["label", "onClick"]),
                        createVNode(unref(script$e), {
                          label: _ctx.$t("g.next"),
                          icon: "pi pi-arrow-right",
                          iconPos: "right",
                          onClick: /* @__PURE__ */ __name(($event) => activateCallback("3"), "onClick")
                        }, null, 8, ["label", "onClick"])
                      ])
                    ]),
                    _: 1
                  }),
                  createVNode(unref(script$3), { value: "3" }, {
                    default: withCtx(({ activateCallback }) => [
                      createVNode(_sfc_main$4, {
                        autoUpdate: autoUpdate.value,
                        "onUpdate:autoUpdate": _cache[5] || (_cache[5] = ($event) => autoUpdate.value = $event),
                        allowMetrics: allowMetrics.value,
                        "onUpdate:allowMetrics": _cache[6] || (_cache[6] = ($event) => allowMetrics.value = $event)
                      }, null, 8, ["autoUpdate", "allowMetrics"]),
                      createBaseVNode("div", _hoisted_4, [
                        createVNode(unref(script$e), {
                          label: _ctx.$t("g.back"),
                          severity: "secondary",
                          icon: "pi pi-arrow-left",
                          onClick: /* @__PURE__ */ __name(($event) => activateCallback("2"), "onClick")
                        }, null, 8, ["label", "onClick"]),
                        createVNode(unref(script$e), {
                          label: _ctx.$t("g.install"),
                          icon: "pi pi-check",
                          iconPos: "right",
                          disabled: hasError.value,
                          onClick: _cache[7] || (_cache[7] = ($event) => install())
                        }, null, 8, ["label", "disabled"])
                      ])
                    ]),
                    _: 1
                  })
                ]),
                _: 1
              })
            ]),
            _: 1
          })
        ]),
        _: 1
      });
    };
  }
});
const InstallView = /* @__PURE__ */ _export_sfc(_sfc_main, [["__scopeId", "data-v-de33872d"]]);
export {
  InstallView as default
};
//# sourceMappingURL=InstallView-CAcYt0HL.js.map
