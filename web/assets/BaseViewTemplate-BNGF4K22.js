import { d as defineComponent, o as openBlock, f as createElementBlock, J as renderSlot, T as normalizeClass } from "./index-DjNHn37O.js";
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "BaseViewTemplate",
  props: {
    dark: { type: Boolean, default: false }
  },
  setup(__props) {
    const props = __props;
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", {
        class: normalizeClass(["font-sans w-screen h-screen flex items-center justify-center pointer-events-auto overflow-auto", [
          props.dark ? "text-neutral-300 bg-neutral-900 dark-theme" : "text-neutral-900 bg-neutral-300"
        ]])
      }, [
        renderSlot(_ctx.$slots, "default")
      ], 2);
    };
  }
});
export {
  _sfc_main as _
};
//# sourceMappingURL=BaseViewTemplate-BNGF4K22.js.map
