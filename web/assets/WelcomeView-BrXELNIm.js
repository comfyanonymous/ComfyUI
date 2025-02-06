var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });
import { d as defineComponent, be as useRouter, o as openBlock, y as createBlock, z as withCtx, m as createBaseVNode, E as toDisplayString, k as createVNode, j as unref, l as script, _ as _export_sfc } from "./index-DqqhYDnY.js";
import { _ as _sfc_main$1 } from "./BaseViewTemplate-Cz111_1A.js";
const _hoisted_1 = { class: "flex flex-col items-center justify-center gap-8 p-8" };
const _hoisted_2 = { class: "animated-gradient-text text-glow select-none" };
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "WelcomeView",
  setup(__props) {
    const router = useRouter();
    const navigateTo = /* @__PURE__ */ __name((path) => {
      router.push(path);
    }, "navigateTo");
    return (_ctx, _cache) => {
      return openBlock(), createBlock(_sfc_main$1, { dark: "" }, {
        default: withCtx(() => [
          createBaseVNode("div", _hoisted_1, [
            createBaseVNode("h1", _hoisted_2, toDisplayString(_ctx.$t("welcome.title")), 1),
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
        ]),
        _: 1
      });
    };
  }
});
const WelcomeView = /* @__PURE__ */ _export_sfc(_sfc_main, [["__scopeId", "data-v-7dfaf74c"]]);
export {
  WelcomeView as default
};
//# sourceMappingURL=WelcomeView-BrXELNIm.js.map
