var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });
import { a as defineComponent, bU as useRouter, f as openBlock, g as createElementBlock, A as createBaseVNode, a8 as toDisplayString, h as createVNode, z as unref, D as script, R as pushScopeId, U as popScopeId, _ as _export_sfc } from "./index-DIU5yZe9.js";
const _withScopeId = /* @__PURE__ */ __name((n) => (pushScopeId("data-v-c4d014c5"), n = n(), popScopeId(), n), "_withScopeId");
const _hoisted_1 = { class: "font-sans flex flex-col justify-center items-center h-screen m-0 text-neutral-300 bg-neutral-900 dark-theme pointer-events-auto" };
const _hoisted_2 = { class: "flex flex-col items-center justify-center gap-8 p-8" };
const _hoisted_3 = { class: "animated-gradient-text text-glow select-none" };
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "WelcomeView",
  setup(__props) {
    const router = useRouter();
    const navigateTo = /* @__PURE__ */ __name((path) => {
      router.push(path);
    }, "navigateTo");
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
            onClick: _cache[0] || (_cache[0] = ($event) => navigateTo("/install")),
            class: "p-4 text-lg fade-in-up"
          }, null, 8, ["label"])
        ])
      ]);
    };
  }
});
const WelcomeView = /* @__PURE__ */ _export_sfc(_sfc_main, [["__scopeId", "data-v-c4d014c5"]]);
export {
  WelcomeView as default
};
//# sourceMappingURL=WelcomeView-Db7ZDfZo.js.map
