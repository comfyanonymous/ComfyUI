var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });
<<<<<<<< HEAD:comfy/web/assets/WelcomeView-BmGjncl9.js
import { d as defineComponent, c0 as useRouter, o as openBlock, H as createBlock, N as withCtx, m as createBaseVNode, X as toDisplayString, k as createVNode, j as unref, l as script, _ as _export_sfc } from "./index-Du3ctekX.js";
import { _ as _sfc_main$1 } from "./BaseViewTemplate-BG-_HPdC.js";
========
import { d as defineComponent, c2 as useRouter, o as openBlock, J as createBlock, P as withCtx, m as createBaseVNode, Z as toDisplayString, k as createVNode, j as unref, l as script, p as pushScopeId, q as popScopeId, _ as _export_sfc } from "./index-QvfM__ze.js";
import { _ as _sfc_main$1 } from "./BaseViewTemplate-BhQMaVFP.js";
const _withScopeId = /* @__PURE__ */ __name((n) => (pushScopeId("data-v-7dfaf74c"), n = n(), popScopeId(), n), "_withScopeId");
>>>>>>>> 13fd4d6e45128f3aed95ee66151353e84daf2d13:comfy/web/assets/WelcomeView-C8whKl15.js
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
<<<<<<<< HEAD:comfy/web/assets/WelcomeView-BmGjncl9.js
//# sourceMappingURL=WelcomeView-BmGjncl9.js.map
========
//# sourceMappingURL=WelcomeView-C8whKl15.js.map
>>>>>>>> 13fd4d6e45128f3aed95ee66151353e84daf2d13:comfy/web/assets/WelcomeView-C8whKl15.js
