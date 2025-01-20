import { d as defineComponent, ad as ref, t as onMounted, bT as isElectron, bV as electronAPI, af as nextTick, o as openBlock, f as createElementBlock, i as withDirectives, v as vShow, m as createBaseVNode, M as renderSlot, V as normalizeClass } from "./index-QvfM__ze.js";
const _hoisted_1 = { class: "flex-grow w-full flex items-center justify-center overflow-auto" };
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "BaseViewTemplate",
  props: {
    dark: { type: Boolean, default: false }
  },
  setup(__props) {
    const props = __props;
    const darkTheme = {
      color: "rgba(0, 0, 0, 0)",
      symbolColor: "#d4d4d4"
    };
    const lightTheme = {
      color: "rgba(0, 0, 0, 0)",
      symbolColor: "#171717"
    };
    const topMenuRef = ref(null);
    const isNativeWindow = ref(false);
    onMounted(async () => {
      if (isElectron()) {
        const windowStyle = await electronAPI().Config.getWindowStyle();
        isNativeWindow.value = windowStyle === "custom";
        await nextTick();
        electronAPI().changeTheme({
          ...props.dark ? darkTheme : lightTheme,
          height: topMenuRef.value.getBoundingClientRect().height
        });
      }
    });
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", {
        class: normalizeClass(["font-sans w-screen h-screen flex flex-col pointer-events-auto", [
          props.dark ? "text-neutral-300 bg-neutral-900 dark-theme" : "text-neutral-900 bg-neutral-300"
        ]])
      }, [
        withDirectives(createBaseVNode("div", {
          ref_key: "topMenuRef",
          ref: topMenuRef,
          class: "app-drag w-full h-[var(--comfy-topbar-height)]"
        }, null, 512), [
          [vShow, isNativeWindow.value]
        ]),
        createBaseVNode("div", _hoisted_1, [
          renderSlot(_ctx.$slots, "default")
        ])
      ], 2);
    };
  }
});
export {
  _sfc_main as _
};
//# sourceMappingURL=BaseViewTemplate-BhQMaVFP.js.map
