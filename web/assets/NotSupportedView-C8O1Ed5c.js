var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });
import { a as defineComponent, bU as useRouter, t as resolveDirective, f as openBlock, g as createElementBlock, A as createBaseVNode, a8 as toDisplayString, h as createVNode, z as unref, D as script, v as withDirectives } from "./index-DIU5yZe9.js";
const _imports_0 = "" + new URL("images/sad_girl.png", import.meta.url).href;
const _hoisted_1 = { class: "font-sans w-screen h-screen flex items-center m-0 text-neutral-900 bg-neutral-300 pointer-events-auto" };
const _hoisted_2 = { class: "flex-grow flex items-center justify-center" };
const _hoisted_3 = { class: "flex flex-col gap-8 p-8" };
const _hoisted_4 = { class: "text-4xl font-bold text-red-500" };
const _hoisted_5 = { class: "space-y-4" };
const _hoisted_6 = { class: "text-xl" };
const _hoisted_7 = { class: "list-disc list-inside space-y-1 text-neutral-800" };
const _hoisted_8 = { class: "flex gap-4" };
const _hoisted_9 = /* @__PURE__ */ createBaseVNode("div", { class: "h-screen flex-grow-0" }, [
  /* @__PURE__ */ createBaseVNode("img", {
    src: _imports_0,
    alt: "Sad girl illustration",
    class: "h-full object-cover"
  })
], -1);
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
      return openBlock(), createElementBlock("div", _hoisted_1, [
        createBaseVNode("div", _hoisted_2, [
          createBaseVNode("div", _hoisted_3, [
            createBaseVNode("h1", _hoisted_4, toDisplayString(_ctx.$t("notSupported.title")), 1),
            createBaseVNode("div", _hoisted_5, [
              createBaseVNode("p", _hoisted_6, toDisplayString(_ctx.$t("notSupported.message")), 1),
              createBaseVNode("ul", _hoisted_7, [
                createBaseVNode("li", null, toDisplayString(_ctx.$t("notSupported.supportedDevices.macos")), 1),
                createBaseVNode("li", null, toDisplayString(_ctx.$t("notSupported.supportedDevices.windows")), 1)
              ])
            ]),
            createBaseVNode("div", _hoisted_8, [
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
        ]),
        _hoisted_9
      ]);
    };
  }
});
export {
  _sfc_main as default
};
//# sourceMappingURL=NotSupportedView-C8O1Ed5c.js.map
