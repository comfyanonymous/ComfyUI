var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });
import { a as defineComponent, f as openBlock, g as createElementBlock, A as createBaseVNode, a8 as toDisplayString, h as createVNode, z as unref, D as script, bU as useRouter } from "./index-DIU5yZe9.js";
const _hoisted_1 = { class: "font-sans w-screen h-screen mx-0 grid place-items-center justify-center items-center text-neutral-900 bg-neutral-300 pointer-events-auto" };
const _hoisted_2 = { class: "col-start-1 h-screen row-start-1 place-content-center mx-auto overflow-y-auto" };
const _hoisted_3 = { class: "max-w-screen-sm flex flex-col gap-8 p-8 bg-[url('/assets/images/Git-Logo-White.svg')] bg-no-repeat bg-right-top bg-origin-padding" };
const _hoisted_4 = { class: "mt-24 text-4xl font-bold text-red-500" };
const _hoisted_5 = { class: "space-y-4" };
const _hoisted_6 = { class: "text-xl" };
const _hoisted_7 = { class: "text-xl" };
const _hoisted_8 = { class: "text-m" };
const _hoisted_9 = { class: "flex gap-4 flex-row-reverse" };
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "DownloadGitView",
  setup(__props) {
    const openGitDownloads = /* @__PURE__ */ __name(() => {
      window.open("https://git-scm.com/downloads/", "_blank");
    }, "openGitDownloads");
    const skipGit = /* @__PURE__ */ __name(() => {
      console.warn("pushing");
      const router = useRouter();
      router.push("install");
    }, "skipGit");
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", _hoisted_1, [
        createBaseVNode("div", _hoisted_2, [
          createBaseVNode("div", _hoisted_3, [
            createBaseVNode("h1", _hoisted_4, toDisplayString(_ctx.$t("downloadGit.title")), 1),
            createBaseVNode("div", _hoisted_5, [
              createBaseVNode("p", _hoisted_6, toDisplayString(_ctx.$t("downloadGit.message")), 1),
              createBaseVNode("p", _hoisted_7, toDisplayString(_ctx.$t("downloadGit.instructions")), 1),
              createBaseVNode("p", _hoisted_8, toDisplayString(_ctx.$t("downloadGit.warning")), 1)
            ]),
            createBaseVNode("div", _hoisted_9, [
              createVNode(unref(script), {
                label: _ctx.$t("downloadGit.gitWebsite"),
                icon: "pi pi-external-link",
                "icon-pos": "right",
                onClick: openGitDownloads,
                severity: "primary"
              }, null, 8, ["label"]),
              createVNode(unref(script), {
                label: _ctx.$t("downloadGit.skip"),
                icon: "pi pi-exclamation-triangle",
                onClick: skipGit,
                severity: "secondary"
              }, null, 8, ["label"])
            ])
          ])
        ])
      ]);
    };
  }
});
export {
  _sfc_main as default
};
//# sourceMappingURL=DownloadGitView-B3f7KHY3.js.map
