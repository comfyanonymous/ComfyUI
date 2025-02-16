var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });
import { d as defineComponent, T as ref, d8 as onUnmounted, o as openBlock, y as createBlock, z as withCtx, m as createBaseVNode, E as toDisplayString, j as unref, bg as t, k as createVNode, bE as script, l as script$1, b9 as electronAPI, _ as _export_sfc } from "./index-Bv0b06LE.js";
import { s as script$2 } from "./index-A_bXPJCN.js";
import { _ as _sfc_main$1 } from "./TerminalOutputDrawer-CKr7Br7O.js";
import { _ as _sfc_main$2 } from "./BaseViewTemplate-BTbuZf5t.js";
const _hoisted_1 = { class: "h-screen w-screen grid items-center justify-around overflow-y-auto" };
const _hoisted_2 = { class: "relative m-8 text-center" };
const _hoisted_3 = { class: "download-bg pi-download text-4xl font-bold" };
const _hoisted_4 = { class: "m-8" };
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "DesktopUpdateView",
  setup(__props) {
    const electron = electronAPI();
    const terminalVisible = ref(false);
    const toggleConsoleDrawer = /* @__PURE__ */ __name(() => {
      terminalVisible.value = !terminalVisible.value;
    }, "toggleConsoleDrawer");
    onUnmounted(() => electron.Validation.dispose());
    return (_ctx, _cache) => {
      return openBlock(), createBlock(_sfc_main$2, { dark: "" }, {
        default: withCtx(() => [
          createBaseVNode("div", _hoisted_1, [
            createBaseVNode("div", _hoisted_2, [
              createBaseVNode("h1", _hoisted_3, toDisplayString(unref(t)("desktopUpdate.title")), 1),
              createBaseVNode("div", _hoisted_4, [
                createBaseVNode("span", null, toDisplayString(unref(t)("desktopUpdate.description")), 1)
              ]),
              createVNode(unref(script), { class: "m-8 w-48 h-48" }),
              createVNode(unref(script$1), {
                style: { "transform": "translateX(-50%)" },
                class: "fixed bottom-0 left-1/2 my-8",
                label: unref(t)("maintenance.consoleLogs"),
                icon: "pi pi-desktop",
                "icon-pos": "left",
                severity: "secondary",
                onClick: toggleConsoleDrawer
              }, null, 8, ["label"]),
              createVNode(_sfc_main$1, {
                modelValue: terminalVisible.value,
                "onUpdate:modelValue": _cache[0] || (_cache[0] = ($event) => terminalVisible.value = $event),
                header: unref(t)("g.terminal"),
                "default-message": unref(t)("desktopUpdate.terminalDefaultMessage")
              }, null, 8, ["modelValue", "header", "default-message"])
            ])
          ]),
          createVNode(unref(script$2))
        ]),
        _: 1
      });
    };
  }
});
const DesktopUpdateView = /* @__PURE__ */ _export_sfc(_sfc_main, [["__scopeId", "data-v-8d77828d"]]);
export {
  DesktopUpdateView as default
};
//# sourceMappingURL=DesktopUpdateView-C-R0415K.js.map
