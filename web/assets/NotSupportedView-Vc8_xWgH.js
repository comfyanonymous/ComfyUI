var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });
import { d as defineComponent, c2 as useRouter, r as resolveDirective, o as openBlock, J as createBlock, P as withCtx, m as createBaseVNode, Z as toDisplayString, k as createVNode, j as unref, l as script, i as withDirectives, p as pushScopeId, q as popScopeId, _ as _export_sfc } from "./index-QvfM__ze.js";
import { _ as _sfc_main$1 } from "./BaseViewTemplate-BhQMaVFP.js";
const _imports_0 = "" + new URL("images/sad_girl.png", import.meta.url).href;
const _withScopeId = /* @__PURE__ */ __name((n) => (pushScopeId("data-v-ebb20958"), n = n(), popScopeId(), n), "_withScopeId");
const _hoisted_1 = { class: "sad-container" };
const _hoisted_2 = /* @__PURE__ */ _withScopeId(() => /* @__PURE__ */ createBaseVNode("img", {
  class: "sad-girl",
  src: _imports_0,
  alt: "Sad girl illustration"
}, null, -1));
const _hoisted_3 = { class: "no-drag sad-text flex items-center" };
const _hoisted_4 = { class: "flex flex-col gap-8 p-8 min-w-110" };
const _hoisted_5 = { class: "text-4xl font-bold text-red-500" };
const _hoisted_6 = { class: "space-y-4" };
const _hoisted_7 = { class: "text-xl" };
const _hoisted_8 = { class: "list-disc list-inside space-y-1 text-neutral-800" };
const _hoisted_9 = { class: "flex gap-4" };
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "NotSupportedView",
  setup(__props) {
    const openDocs = /* @__PURE__ */ __name(() => {
      window.open(
        "https://github.com/Comfy-Org/desktop#currently-supported-platforms",
        "_blank"
      );
    }, "openDocs");
    const reportIssue = /* @__PURE__ */ __name(() => {
      window.open("https://forum.comfy.org/c/v1-feedback/", "_blank");
    }, "reportIssue");
    const router = useRouter();
    const continueToInstall = /* @__PURE__ */ __name(() => {
      router.push("/install");
    }, "continueToInstall");
    return (_ctx, _cache) => {
      const _directive_tooltip = resolveDirective("tooltip");
      return openBlock(), createBlock(_sfc_main$1, null, {
        default: withCtx(() => [
          createBaseVNode("div", _hoisted_1, [
            _hoisted_2,
            createBaseVNode("div", _hoisted_3, [
              createBaseVNode("div", _hoisted_4, [
                createBaseVNode("h1", _hoisted_5, toDisplayString(_ctx.$t("notSupported.title")), 1),
                createBaseVNode("div", _hoisted_6, [
                  createBaseVNode("p", _hoisted_7, toDisplayString(_ctx.$t("notSupported.message")), 1),
                  createBaseVNode("ul", _hoisted_8, [
                    createBaseVNode("li", null, toDisplayString(_ctx.$t("notSupported.supportedDevices.macos")), 1),
                    createBaseVNode("li", null, toDisplayString(_ctx.$t("notSupported.supportedDevices.windows")), 1)
                  ])
                ]),
                createBaseVNode("div", _hoisted_9, [
                  createVNode(unref(script), {
                    label: _ctx.$t("notSupported.learnMore"),
                    icon: "pi pi-github",
                    onClick: openDocs,
                    severity: "secondary"
                  }, null, 8, ["label"]),
                  createVNode(unref(script), {
                    label: _ctx.$t("notSupported.reportIssue"),
                    icon: "pi pi-flag",
                    onClick: reportIssue,
                    severity: "secondary"
                  }, null, 8, ["label"]),
                  withDirectives(createVNode(unref(script), {
                    label: _ctx.$t("notSupported.continue"),
                    icon: "pi pi-arrow-right",
                    iconPos: "right",
                    onClick: continueToInstall,
                    severity: "danger"
                  }, null, 8, ["label"]), [
                    [_directive_tooltip, _ctx.$t("notSupported.continueTooltip")]
                  ])
                ])
              ])
            ])
          ])
        ]),
        _: 1
      });
    };
  }
});
const NotSupportedView = /* @__PURE__ */ _export_sfc(_sfc_main, [["__scopeId", "data-v-ebb20958"]]);
export {
  NotSupportedView as default
};
//# sourceMappingURL=NotSupportedView-Vc8_xWgH.js.map
