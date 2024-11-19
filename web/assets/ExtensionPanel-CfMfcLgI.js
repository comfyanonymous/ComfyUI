var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });
import { d as defineComponent, c6 as useExtensionStore, u as useSettingStore, r as ref, o as onMounted, q as computed, g as openBlock, h as createElementBlock, i as createVNode, y as withCtx, z as unref, bT as script$1, A as createBaseVNode, x as createBlock, N as Fragment, O as renderList, a6 as toDisplayString, aw as createTextVNode, bR as script$3, j as createCommentVNode, D as script$4 } from "./index-B6dYHNhg.js";
import { s as script, a as script$2 } from "./index-CjwCGacA.js";
import "./index-MX9DEi8Q.js";
const _hoisted_1 = { class: "extension-panel" };
const _hoisted_2 = { class: "mt-4" };
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "ExtensionPanel",
  setup(__props) {
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
      return openBlock(), createElementBlock("div", _hoisted_1, [
        createVNode(unref(script$2), {
          value: unref(extensionStore).extensions,
          stripedRows: "",
          size: "small"
        }, {
          default: withCtx(() => [
            createVNode(unref(script), {
              field: "name",
              header: _ctx.$t("extensionName"),
              sortable: ""
            }, null, 8, ["header"]),
            createVNode(unref(script), { pt: {
              bodyCell: "flex items-center justify-end"
            } }, {
              body: withCtx((slotProps) => [
                createVNode(unref(script$1), {
                  modelValue: editingEnabledExtensions.value[slotProps.data.name],
                  "onUpdate:modelValue": /* @__PURE__ */ __name(($event) => editingEnabledExtensions.value[slotProps.data.name] = $event, "onUpdate:modelValue"),
                  onChange: updateExtensionStatus
                }, null, 8, ["modelValue", "onUpdate:modelValue"])
              ]),
              _: 1
            })
          ]),
          _: 1
        }, 8, ["value"]),
        createBaseVNode("div", _hoisted_2, [
          hasChanges.value ? (openBlock(), createBlock(unref(script$3), {
            key: 0,
            severity: "info"
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
              ])
            ]),
            _: 1
          })) : createCommentVNode("", true),
          createVNode(unref(script$4), {
            label: _ctx.$t("reloadToApplyChanges"),
            icon: "pi pi-refresh",
            onClick: applyChanges,
            disabled: !hasChanges.value,
            text: "",
            fluid: "",
            severity: "danger"
          }, null, 8, ["label", "disabled"])
        ])
      ]);
    };
  }
});
export {
  _sfc_main as default
};
//# sourceMappingURL=ExtensionPanel-CfMfcLgI.js.map
