var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });
import { o as openBlock, f as createElementBlock, m as createBaseVNode, H as markRaw, d as defineComponent, a as useSettingStore, ae as storeToRefs, O as watch, dy as useCopyToClipboard, K as useI18n, y as createBlock, z as withCtx, j as unref, bj as script, E as toDisplayString, D as renderList, F as Fragment, k as createVNode, l as script$1, B as createCommentVNode, bh as script$2, dz as FormItem, dn as _sfc_main$1, b5 as electronAPI } from "./index-DqqhYDnY.js";
import { u as useServerConfigStore } from "./serverConfigStore-Kb5DJVFt.js";
const _hoisted_1$1 = {
  viewBox: "0 0 24 24",
  width: "1.2em",
  height: "1.2em"
};
function render(_ctx, _cache) {
  return openBlock(), createElementBlock("svg", _hoisted_1$1, _cache[0] || (_cache[0] = [
    createBaseVNode("path", {
      fill: "none",
      stroke: "currentColor",
      "stroke-linecap": "round",
      "stroke-linejoin": "round",
      "stroke-width": "2",
      d: "m4 17l6-6l-6-6m8 14h8"
    }, null, -1)
  ]));
}
__name(render, "render");
const __unplugin_components_0 = markRaw({ name: "lucide-terminal", render });
const _hoisted_1 = { class: "flex flex-col gap-2" };
const _hoisted_2 = { class: "flex justify-end gap-2" };
const _hoisted_3 = { class: "flex items-center justify-between" };
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "ServerConfigPanel",
  setup(__props) {
    const settingStore = useSettingStore();
    const serverConfigStore = useServerConfigStore();
    const {
      serverConfigsByCategory,
      serverConfigValues,
      launchArgs,
      commandLineArgs,
      modifiedConfigs
    } = storeToRefs(serverConfigStore);
    const revertChanges = /* @__PURE__ */ __name(() => {
      serverConfigStore.revertChanges();
    }, "revertChanges");
    const restartApp = /* @__PURE__ */ __name(() => {
      electronAPI().restartApp();
    }, "restartApp");
    watch(launchArgs, (newVal) => {
      settingStore.set("Comfy.Server.LaunchArgs", newVal);
    });
    watch(serverConfigValues, (newVal) => {
      settingStore.set("Comfy.Server.ServerConfigValues", newVal);
    });
    const { copyToClipboard } = useCopyToClipboard();
    const copyCommandLineArgs = /* @__PURE__ */ __name(async () => {
      await copyToClipboard(commandLineArgs.value);
    }, "copyCommandLineArgs");
    const { t } = useI18n();
    const translateItem = /* @__PURE__ */ __name((item) => {
      return {
        ...item,
        name: t(`serverConfigItems.${item.id}.name`, item.name),
        tooltip: item.tooltip ? t(`serverConfigItems.${item.id}.tooltip`, item.tooltip) : void 0
      };
    }, "translateItem");
    return (_ctx, _cache) => {
      const _component_i_lucide58terminal = __unplugin_components_0;
      return openBlock(), createBlock(_sfc_main$1, {
        value: "Server-Config",
        class: "server-config-panel"
      }, {
        header: withCtx(() => [
          createBaseVNode("div", _hoisted_1, [
            unref(modifiedConfigs).length > 0 ? (openBlock(), createBlock(unref(script), {
              key: 0,
              severity: "info",
              "pt:text": "w-full"
            }, {
              default: withCtx(() => [
                createBaseVNode("p", null, toDisplayString(_ctx.$t("serverConfig.modifiedConfigs")), 1),
                createBaseVNode("ul", null, [
                  (openBlock(true), createElementBlock(Fragment, null, renderList(unref(modifiedConfigs), (config) => {
                    return openBlock(), createElementBlock("li", {
                      key: config.id
                    }, toDisplayString(config.name) + ": " + toDisplayString(config.initialValue) + " → " + toDisplayString(config.value), 1);
                  }), 128))
                ]),
                createBaseVNode("div", _hoisted_2, [
                  createVNode(unref(script$1), {
                    label: _ctx.$t("serverConfig.revertChanges"),
                    onClick: revertChanges,
                    outlined: ""
                  }, null, 8, ["label"]),
                  createVNode(unref(script$1), {
                    label: _ctx.$t("serverConfig.restart"),
                    onClick: restartApp,
                    outlined: "",
                    severity: "danger"
                  }, null, 8, ["label"])
                ])
              ]),
              _: 1
            })) : createCommentVNode("", true),
            unref(commandLineArgs) ? (openBlock(), createBlock(unref(script), {
              key: 1,
              severity: "secondary",
              "pt:text": "w-full"
            }, {
              icon: withCtx(() => [
                createVNode(_component_i_lucide58terminal, { class: "text-xl font-bold" })
              ]),
              default: withCtx(() => [
                createBaseVNode("div", _hoisted_3, [
                  createBaseVNode("p", null, toDisplayString(unref(commandLineArgs)), 1),
                  createVNode(unref(script$1), {
                    icon: "pi pi-clipboard",
                    onClick: copyCommandLineArgs,
                    severity: "secondary",
                    text: ""
                  })
                ])
              ]),
              _: 1
            })) : createCommentVNode("", true)
          ])
        ]),
        default: withCtx(() => [
          (openBlock(true), createElementBlock(Fragment, null, renderList(Object.entries(unref(serverConfigsByCategory)), ([label, items], i) => {
            return openBlock(), createElementBlock("div", { key: label }, [
              i > 0 ? (openBlock(), createBlock(unref(script$2), { key: 0 })) : createCommentVNode("", true),
              createBaseVNode("h3", null, toDisplayString(_ctx.$t(`serverConfigCategories.${label}`, label)), 1),
              (openBlock(true), createElementBlock(Fragment, null, renderList(items, (item) => {
                return openBlock(), createElementBlock("div", {
                  key: item.name,
                  class: "mb-4"
                }, [
                  createVNode(FormItem, {
                    item: translateItem(item),
                    formValue: item.value,
                    "onUpdate:formValue": /* @__PURE__ */ __name(($event) => item.value = $event, "onUpdate:formValue"),
                    id: item.id,
                    labelClass: {
                      "text-highlight": item.initialValue !== item.value
                    }
                  }, null, 8, ["item", "formValue", "onUpdate:formValue", "id", "labelClass"])
                ]);
              }), 128))
            ]);
          }), 128))
        ]),
        _: 1
      });
    };
  }
});
export {
  _sfc_main as default
};
//# sourceMappingURL=ServerConfigPanel-B1lI5M9c.js.map
