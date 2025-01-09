var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });
import { d as defineComponent, a1 as useI18n, ab as ref, m as onMounted, o as openBlock, k as createBlock, M as withCtx, H as createBaseVNode, X as toDisplayString, N as createVNode, j as unref, aI as script, l as script$2, c0 as electronAPI } from "./index-DjNHn37O.js";
import { s as script$1 } from "./index-jXPKy3pP.js";
import { _ as _sfc_main$1 } from "./BaseViewTemplate-BNGF4K22.js";
import "./index-5HFeZax4.js";
const _hoisted_1 = { class: "comfy-installer grow flex flex-col gap-4 text-neutral-300 max-w-110" };
const _hoisted_2 = { class: "text-2xl font-semibold text-neutral-100" };
const _hoisted_3 = { class: "m-1 text-neutral-300" };
const _hoisted_4 = { class: "ml-2" };
const _hoisted_5 = { class: "m-1 mb-4" };
const _hoisted_6 = { class: "m-0" };
const _hoisted_7 = { class: "m-1" };
const _hoisted_8 = { class: "font-mono" };
const _hoisted_9 = { class: "m-1" };
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "ManualConfigurationView",
  setup(__props) {
    const { t } = useI18n();
    const electron = electronAPI();
    const basePath = ref(null);
    const sep = ref("/");
    const restartApp = /* @__PURE__ */ __name((message) => electron.restartApp(message), "restartApp");
    onMounted(async () => {
      basePath.value = await electron.getBasePath();
      if (basePath.value.indexOf("/") === -1) sep.value = "\\";
    });
    return (_ctx, _cache) => {
      return openBlock(), createBlock(_sfc_main$1, { dark: "" }, {
        default: withCtx(() => [
          createBaseVNode("div", _hoisted_1, [
            createBaseVNode("h2", _hoisted_2, toDisplayString(_ctx.$t("install.manualConfiguration.title")), 1),
            createBaseVNode("p", _hoisted_3, [
              createVNode(unref(script), {
                icon: "pi pi-exclamation-triangle",
                severity: "warn",
                value: unref(t)("icon.exclamation-triangle")
              }, null, 8, ["value"]),
              createBaseVNode("strong", _hoisted_4, toDisplayString(_ctx.$t("install.gpuSelection.customComfyNeedsPython")), 1)
            ]),
            createBaseVNode("div", null, [
              createBaseVNode("p", _hoisted_5, toDisplayString(_ctx.$t("install.manualConfiguration.requirements")) + ": ", 1),
              createBaseVNode("ul", _hoisted_6, [
                createBaseVNode("li", null, toDisplayString(_ctx.$t("install.gpuSelection.customManualVenv")), 1),
                createBaseVNode("li", null, toDisplayString(_ctx.$t("install.gpuSelection.customInstallRequirements")), 1)
              ])
            ]),
            createBaseVNode("p", _hoisted_7, toDisplayString(_ctx.$t("install.manualConfiguration.createVenv")) + ":", 1),
            createVNode(unref(script$1), {
              header: unref(t)("install.manualConfiguration.virtualEnvironmentPath")
            }, {
              default: withCtx(() => [
                createBaseVNode("span", _hoisted_8, toDisplayString(`${basePath.value}${sep.value}.venv${sep.value}`), 1)
              ]),
              _: 1
            }, 8, ["header"]),
            createBaseVNode("p", _hoisted_9, toDisplayString(_ctx.$t("install.manualConfiguration.restartWhenFinished")), 1),
            createVNode(unref(script$2), {
              class: "place-self-end",
              label: unref(t)("menuLabels.Restart"),
              severity: "warn",
              icon: "pi pi-refresh",
              onClick: _cache[0] || (_cache[0] = ($event) => restartApp("Manual configuration complete"))
            }, null, 8, ["label"])
          ])
        ]),
        _: 1
      });
    };
  }
});
export {
  _sfc_main as default
};
//# sourceMappingURL=ManualConfigurationView-Bi_qHE-n.js.map
