var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });
import { d as defineComponent, g as openBlock, h as createElementBlock, A as createBaseVNode, a6 as toDisplayString, i as createVNode, z as unref, D as script, P as pushScopeId, Q as popScopeId, _ as _export_sfc } from "./index-CoOvI8ZH.js";
const _withScopeId = /* @__PURE__ */ __name((n) => (pushScopeId("data-v-12b8b11b"), n = n(), popScopeId(), n), "_withScopeId");
const _hoisted_1 = { class: "font-sans flex flex-col justify-center items-center h-screen m-0 text-neutral-300 bg-neutral-900 dark-theme pointer-events-auto" };
const _hoisted_2 = { class: "flex flex-col items-center justify-center gap-8 p-8" };
const _hoisted_3 = { class: "animated-gradient-text text-glow select-none" };
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "WelcomeView",
  setup(__props) {
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", _hoisted_1, [
        createBaseVNode("div", _hoisted_2, [
          createBaseVNode("h1", _hoisted_3, toDisplayString(_ctx.$t("welcome.title")), 1),
          createVNode(unref(script), {
            label: _ctx.$t("welcome.getStarted"),
            icon: "pi pi-arrow-right",
            iconPos: "right",
            size: "large",
            rounded: "",
            onClick: _cache[0] || (_cache[0] = ($event) => _ctx.$router.push("/install")),
            class: "p-4 text-lg fade-in-up"
          }, null, 8, ["label"])
        ])
      ]);
    };
  }
});
const WelcomeView = /* @__PURE__ */ _export_sfc(_sfc_main, [["__scopeId", "data-v-12b8b11b"]]);
export {
  WelcomeView as default
};
//# sourceMappingURL=WelcomeView-C4D1cggT.js.map
