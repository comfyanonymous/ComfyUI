var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });
import { d as defineComponent, r as ref, c6 as FilterMatchMode, ca as useExtensionStore, u as useSettingStore, o as onMounted, q as computed, g as openBlock, x as createBlock, y as withCtx, i as createVNode, c7 as SearchBox, z as unref, bT as script, A as createBaseVNode, h as createElementBlock, O as renderList, a6 as toDisplayString, aw as createTextVNode, N as Fragment, D as script$1, j as createCommentVNode, bV as script$3, c8 as _sfc_main$1 } from "./index-CoOvI8ZH.js";
import { s as script$2, a as script$4 } from "./index-DK6Kev7f.js";
import "./index-D4DWQPPQ.js";
const _hoisted_1 = { class: "flex justify-end" };
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "ExtensionPanel",
  setup(__props) {
    const filters = ref({
      global: { value: "", matchMode: FilterMatchMode.CONTAINS }
    });
    const extensionStore = useExtensionStore();
    const settingStore = useSettingStore();
    const editingEnabledExtensions = ref({});
    onMounted(() => {
      extensionStore.extensions.forEach((ext) => {
        editingEnabledExtensions.value[ext.name] = extensionStore.isExtensionEnabled(ext.name);
      });
    });
    const changedExtensions = computed(() => {
      return extensionStore.extensions.filter(
        (ext) => editingEnabledExtensions.value[ext.name] !== extensionStore.isExtensionEnabled(ext.name)
      );
    });
    const hasChanges = computed(() => {
      return changedExtensions.value.length > 0;
    });
    const updateExtensionStatus = /* @__PURE__ */ __name(() => {
      const editingDisabledExtensionNames = Object.entries(
        editingEnabledExtensions.value
      ).filter(([_, enabled]) => !enabled).map(([name]) => name);
      settingStore.set("Comfy.Extension.Disabled", [
        ...extensionStore.inactiveDisabledExtensionNames,
        ...editingDisabledExtensionNames
      ]);
    }, "updateExtensionStatus");
    const applyChanges = /* @__PURE__ */ __name(() => {
      window.location.reload();
    }, "applyChanges");
    return (_ctx, _cache) => {
      return openBlock(), createBlock(_sfc_main$1, {
        value: "Extension",
        class: "extension-panel"
      }, {
        header: withCtx(() => [
          createVNode(SearchBox, {
            modelValue: filters.value["global"].value,
            "onUpdate:modelValue": _cache[0] || (_cache[0] = ($event) => filters.value["global"].value = $event),
            placeholder: _ctx.$t("searchExtensions") + "..."
          }, null, 8, ["modelValue", "placeholder"]),
          hasChanges.value ? (openBlock(), createBlock(unref(script), {
            key: 0,
            severity: "info",
            "pt:text": "w-full"
          }, {
            default: withCtx(() => [
              createBaseVNode("ul", null, [
                (openBlock(true), createElementBlock(Fragment, null, renderList(changedExtensions.value, (ext) => {
                  return openBlock(), createElementBlock("li", {
                    key: ext.name
                  }, [
                    createBaseVNode("span", null, toDisplayString(unref(extensionStore).isExtensionEnabled(ext.name) ? "[-]" : "[+]"), 1),
                    createTextVNode(" " + toDisplayString(ext.name), 1)
                  ]);
                }), 128))
              ]),
              createBaseVNode("div", _hoisted_1, [
                createVNode(unref(script$1), {
                  label: _ctx.$t("reloadToApplyChanges"),
                  onClick: applyChanges,
                  outlined: "",
                  severity: "danger"
                }, null, 8, ["label"])
              ])
            ]),
            _: 1
          })) : createCommentVNode("", true)
        ]),
        default: withCtx(() => [
          createVNode(unref(script$4), {
            value: unref(extensionStore).extensions,
            stripedRows: "",
            size: "small",
            filters: filters.value
          }, {
            default: withCtx(() => [
              createVNode(unref(script$2), {
                field: "name",
                header: _ctx.$t("extensionName"),
                sortable: ""
              }, null, 8, ["header"]),
              createVNode(unref(script$2), { pt: {
                bodyCell: "flex items-center justify-end"
              } }, {
                body: withCtx((slotProps) => [
                  createVNode(unref(script$3), {
                    modelValue: editingEnabledExtensions.value[slotProps.data.name],
                    "onUpdate:modelValue": /* @__PURE__ */ __name(($event) => editingEnabledExtensions.value[slotProps.data.name] = $event, "onUpdate:modelValue"),
                    onChange: updateExtensionStatus
                  }, null, 8, ["modelValue", "onUpdate:modelValue"])
                ]),
                _: 1
              })
            ]),
            _: 1
          }, 8, ["value", "filters"])
        ]),
        _: 1
      });
    };
  }
});
export {
  _sfc_main as default
};
//# sourceMappingURL=ExtensionPanel-DsD42OtO.js.map
