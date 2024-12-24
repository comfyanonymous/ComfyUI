var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });
import { a as defineComponent, q as computed, f as openBlock, g as createElementBlock, P as Fragment, Q as renderList, h as createVNode, y as withCtx, ay as createTextVNode, a8 as toDisplayString, z as unref, aC as script, i as createCommentVNode, r as ref, ck as FilterMatchMode, O as useKeybindingStore, F as useCommandStore, I as useI18n, aS as normalizeI18nKey, aL as watchEffect, bn as useToast, t as resolveDirective, x as createBlock, cl as SearchBox, A as createBaseVNode, D as script$2, aq as script$4, br as withModifiers, bW as script$5, aI as script$6, v as withDirectives, cm as _sfc_main$2, R as pushScopeId, U as popScopeId, ce as KeyComboImpl, cn as KeybindingImpl, _ as _export_sfc } from "./index-DIU5yZe9.js";
import { s as script$1, a as script$3 } from "./index-D3u7l7ha.js";
import "./index-d698Brhb.js";
const _hoisted_1$1 = {
  key: 0,
  class: "px-2"
};
const _sfc_main$1 = /* @__PURE__ */ defineComponent({
  __name: "KeyComboDisplay",
  props: {
    keyCombo: {},
    isModified: { type: Boolean, default: false }
  },
  setup(__props) {
    const props = __props;
    const keySequences = computed(() => props.keyCombo.getKeySequences());
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("span", null, [
        (openBlock(true), createElementBlock(Fragment, null, renderList(keySequences.value, (sequence, index) => {
          return openBlock(), createElementBlock(Fragment, { key: index }, [
            createVNode(unref(script), {
              severity: _ctx.isModified ? "info" : "secondary"
            }, {
              default: withCtx(() => [
                createTextVNode(toDisplayString(sequence), 1)
              ]),
              _: 2
            }, 1032, ["severity"]),
            index < keySequences.value.length - 1 ? (openBlock(), createElementBlock("span", _hoisted_1$1, "+")) : createCommentVNode("", true)
          ], 64);
        }), 128))
      ]);
    };
  }
});
const _withScopeId = /* @__PURE__ */ __name((n) => (pushScopeId("data-v-c20ad403"), n = n(), popScopeId(), n), "_withScopeId");
const _hoisted_1 = { class: "actions invisible flex flex-row" };
const _hoisted_2 = ["title"];
const _hoisted_3 = { key: 1 };
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "KeybindingPanel",
  setup(__props) {
    const filters = ref({
      global: { value: "", matchMode: FilterMatchMode.CONTAINS }
    });
    const keybindingStore = useKeybindingStore();
    const commandStore = useCommandStore();
    const { t } = useI18n();
    const commandsData = computed(() => {
      return Object.values(commandStore.commands).map((command) => ({
        id: command.id,
        label: t(`commands.${normalizeI18nKey(command.id)}.label`, command.label),
        keybinding: keybindingStore.getKeybindingByCommandId(command.id)
      }));
    });
    const selectedCommandData = ref(null);
    const editDialogVisible = ref(false);
    const newBindingKeyCombo = ref(null);
    const currentEditingCommand = ref(null);
    const keybindingInput = ref(null);
    const existingKeybindingOnCombo = computed(() => {
      if (!currentEditingCommand.value) {
        return null;
      }
      if (currentEditingCommand.value.keybinding?.combo?.equals(
        newBindingKeyCombo.value
      )) {
        return null;
      }
      if (!newBindingKeyCombo.value) {
        return null;
      }
      return keybindingStore.getKeybinding(newBindingKeyCombo.value);
    });
    function editKeybinding(commandData) {
      currentEditingCommand.value = commandData;
      newBindingKeyCombo.value = commandData.keybinding ? commandData.keybinding.combo : null;
      editDialogVisible.value = true;
    }
    __name(editKeybinding, "editKeybinding");
    watchEffect(() => {
      if (editDialogVisible.value) {
        setTimeout(() => {
          keybindingInput.value?.$el?.focus();
        }, 300);
      }
    });
    function removeKeybinding(commandData) {
      if (commandData.keybinding) {
        keybindingStore.unsetKeybinding(commandData.keybinding);
        keybindingStore.persistUserKeybindings();
      }
    }
    __name(removeKeybinding, "removeKeybinding");
    function captureKeybinding(event) {
      const keyCombo = KeyComboImpl.fromEvent(event);
      newBindingKeyCombo.value = keyCombo;
    }
    __name(captureKeybinding, "captureKeybinding");
    function cancelEdit() {
      editDialogVisible.value = false;
      currentEditingCommand.value = null;
      newBindingKeyCombo.value = null;
    }
    __name(cancelEdit, "cancelEdit");
    function saveKeybinding() {
      if (currentEditingCommand.value && newBindingKeyCombo.value) {
        const updated = keybindingStore.updateKeybindingOnCommand(
          new KeybindingImpl({
            commandId: currentEditingCommand.value.id,
            combo: newBindingKeyCombo.value
          })
        );
        if (updated) {
          keybindingStore.persistUserKeybindings();
        }
      }
      cancelEdit();
    }
    __name(saveKeybinding, "saveKeybinding");
    const toast = useToast();
    async function resetKeybindings() {
      keybindingStore.resetKeybindings();
      await keybindingStore.persistUserKeybindings();
      toast.add({
        severity: "info",
        summary: "Info",
        detail: "Keybindings reset",
        life: 3e3
      });
    }
    __name(resetKeybindings, "resetKeybindings");
    return (_ctx, _cache) => {
      const _directive_tooltip = resolveDirective("tooltip");
      return openBlock(), createBlock(_sfc_main$2, {
        value: "Keybinding",
        class: "keybinding-panel"
      }, {
        header: withCtx(() => [
          createVNode(SearchBox, {
            modelValue: filters.value["global"].value,
            "onUpdate:modelValue": _cache[0] || (_cache[0] = ($event) => filters.value["global"].value = $event),
            placeholder: _ctx.$t("g.searchKeybindings") + "..."
          }, null, 8, ["modelValue", "placeholder"])
        ]),
        default: withCtx(() => [
          createVNode(unref(script$3), {
            value: commandsData.value,
            selection: selectedCommandData.value,
            "onUpdate:selection": _cache[1] || (_cache[1] = ($event) => selectedCommandData.value = $event),
            "global-filter-fields": ["id"],
            filters: filters.value,
            selectionMode: "single",
            stripedRows: "",
            pt: {
              header: "px-0"
            }
          }, {
            default: withCtx(() => [
              createVNode(unref(script$1), {
                field: "actions",
                header: ""
              }, {
                body: withCtx((slotProps) => [
                  createBaseVNode("div", _hoisted_1, [
                    createVNode(unref(script$2), {
                      icon: "pi pi-pencil",
                      class: "p-button-text",
                      onClick: /* @__PURE__ */ __name(($event) => editKeybinding(slotProps.data), "onClick")
                    }, null, 8, ["onClick"]),
                    createVNode(unref(script$2), {
                      icon: "pi pi-trash",
                      class: "p-button-text p-button-danger",
                      onClick: /* @__PURE__ */ __name(($event) => removeKeybinding(slotProps.data), "onClick"),
                      disabled: !slotProps.data.keybinding
                    }, null, 8, ["onClick", "disabled"])
                  ])
                ]),
                _: 1
              }),
              createVNode(unref(script$1), {
                field: "id",
                header: "Command ID",
                sortable: "",
                class: "max-w-64 2xl:max-w-full"
              }, {
                body: withCtx((slotProps) => [
                  createBaseVNode("div", {
                    class: "overflow-hidden text-ellipsis whitespace-nowrap",
                    title: slotProps.data.id
                  }, toDisplayString(slotProps.data.label), 9, _hoisted_2)
                ]),
                _: 1
              }),
              createVNode(unref(script$1), {
                field: "keybinding",
                header: "Keybinding"
              }, {
                body: withCtx((slotProps) => [
                  slotProps.data.keybinding ? (openBlock(), createBlock(_sfc_main$1, {
                    key: 0,
                    keyCombo: slotProps.data.keybinding.combo,
                    isModified: unref(keybindingStore).isCommandKeybindingModified(slotProps.data.id)
                  }, null, 8, ["keyCombo", "isModified"])) : (openBlock(), createElementBlock("span", _hoisted_3, "-"))
                ]),
                _: 1
              })
            ]),
            _: 1
          }, 8, ["value", "selection", "filters"]),
          createVNode(unref(script$6), {
            class: "min-w-96",
            visible: editDialogVisible.value,
            "onUpdate:visible": _cache[2] || (_cache[2] = ($event) => editDialogVisible.value = $event),
            modal: "",
            header: currentEditingCommand.value?.id,
            onHide: cancelEdit
          }, {
            footer: withCtx(() => [
              createVNode(unref(script$2), {
                label: "Save",
                icon: "pi pi-check",
                onClick: saveKeybinding,
                disabled: !!existingKeybindingOnCombo.value,
                autofocus: ""
              }, null, 8, ["disabled"])
            ]),
            default: withCtx(() => [
              createBaseVNode("div", null, [
                createVNode(unref(script$4), {
                  class: "mb-2 text-center",
                  ref_key: "keybindingInput",
                  ref: keybindingInput,
                  modelValue: newBindingKeyCombo.value?.toString() ?? "",
                  placeholder: "Press keys for new binding",
                  onKeydown: withModifiers(captureKeybinding, ["stop", "prevent"]),
                  autocomplete: "off",
                  fluid: "",
                  invalid: !!existingKeybindingOnCombo.value
                }, null, 8, ["modelValue", "invalid"]),
                existingKeybindingOnCombo.value ? (openBlock(), createBlock(unref(script$5), {
                  key: 0,
                  severity: "error"
                }, {
                  default: withCtx(() => [
                    createTextVNode(" Keybinding already exists on "),
                    createVNode(unref(script), {
                      severity: "secondary",
                      value: existingKeybindingOnCombo.value.commandId
                    }, null, 8, ["value"])
                  ]),
                  _: 1
                })) : createCommentVNode("", true)
              ])
            ]),
            _: 1
          }, 8, ["visible", "header"]),
          withDirectives(createVNode(unref(script$2), {
            class: "mt-4",
            label: _ctx.$t("g.reset"),
            icon: "pi pi-trash",
            severity: "danger",
            fluid: "",
            text: "",
            onClick: resetKeybindings
          }, null, 8, ["label"]), [
            [_directive_tooltip, _ctx.$t("g.resetKeybindingsTooltip")]
          ])
        ]),
        _: 1
      });
    };
  }
});
const KeybindingPanel = /* @__PURE__ */ _export_sfc(_sfc_main, [["__scopeId", "data-v-c20ad403"]]);
export {
  KeybindingPanel as default
};
//# sourceMappingURL=KeybindingPanel-DC2AxNNa.js.map
