var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });
import { d as defineComponent, a1 as useI18n, ab as ref, b_ as ProgressStatus, m as onMounted, o as openBlock, k as createBlock, M as withCtx, H as createBaseVNode, aE as createTextVNode, X as toDisplayString, j as unref, f as createElementBlock, I as createCommentVNode, N as createVNode, l as script, i as withDirectives, v as vShow, b$ as BaseTerminal, aL as pushScopeId, aM as popScopeId, c0 as electronAPI, _ as _export_sfc } from "./index-DjNHn37O.js";
import { _ as _sfc_main$1 } from "./BaseViewTemplate-BNGF4K22.js";
const _withScopeId = /* @__PURE__ */ __name((n) => (pushScopeId("data-v-42c1131d"), n = n(), popScopeId(), n), "_withScopeId");
const _hoisted_1 = { class: "text-2xl font-bold" };
const _hoisted_2 = { key: 0 };
const _hoisted_3 = {
  key: 0,
  class: "flex flex-col items-center gap-4"
};
const _hoisted_4 = { class: "flex items-center my-4 gap-2" };
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "ServerStartView",
  setup(__props) {
    const electron = electronAPI();
    const { t } = useI18n();
    const status = ref(ProgressStatus.INITIAL_STATE);
    const electronVersion = ref("");
    let xterm;
    const terminalVisible = ref(true);
    const updateProgress = /* @__PURE__ */ __name(({ status: newStatus }) => {
      status.value = newStatus;
      if (newStatus === ProgressStatus.ERROR) terminalVisible.value = false;
      else xterm?.clear();
    }, "updateProgress");
    const terminalCreated = /* @__PURE__ */ __name(({ terminal, useAutoSize }, root) => {
      xterm = terminal;
      useAutoSize(root, true, true);
      electron.onLogMessage((message) => {
        terminal.write(message);
      });
      terminal.options.cursorBlink = false;
      terminal.options.disableStdin = true;
      terminal.options.cursorInactiveStyle = "block";
    }, "terminalCreated");
    const reinstall = /* @__PURE__ */ __name(() => electron.reinstall(), "reinstall");
    const reportIssue = /* @__PURE__ */ __name(() => {
      window.open("https://forum.comfy.org/c/v1-feedback/", "_blank");
    }, "reportIssue");
    const openLogs = /* @__PURE__ */ __name(() => electron.openLogsFolder(), "openLogs");
    onMounted(async () => {
      electron.sendReady();
      electron.onProgressUpdate(updateProgress);
      electronVersion.value = await electron.getElectronVersion();
    });
    return (_ctx, _cache) => {
      return openBlock(), createBlock(_sfc_main$1, {
        dark: "",
        class: "flex-col"
      }, {
        default: withCtx(() => [
          createBaseVNode("h2", _hoisted_1, [
            createTextVNode(toDisplayString(unref(t)(`serverStart.process.${status.value}`)) + " ", 1),
            status.value === unref(ProgressStatus).ERROR ? (openBlock(), createElementBlock("span", _hoisted_2, " v" + toDisplayString(electronVersion.value), 1)) : createCommentVNode("", true)
          ]),
          status.value === unref(ProgressStatus).ERROR ? (openBlock(), createElementBlock("div", _hoisted_3, [
            createBaseVNode("div", _hoisted_4, [
              createVNode(unref(script), {
                icon: "pi pi-flag",
                severity: "secondary",
                label: unref(t)("serverStart.reportIssue"),
                onClick: reportIssue
              }, null, 8, ["label"]),
              createVNode(unref(script), {
                icon: "pi pi-file",
                severity: "secondary",
                label: unref(t)("serverStart.openLogs"),
                onClick: openLogs
              }, null, 8, ["label"]),
              createVNode(unref(script), {
                icon: "pi pi-refresh",
                label: unref(t)("serverStart.reinstall"),
                onClick: reinstall
              }, null, 8, ["label"])
            ]),
            !terminalVisible.value ? (openBlock(), createBlock(unref(script), {
              key: 0,
              icon: "pi pi-search",
              severity: "secondary",
              label: unref(t)("serverStart.showTerminal"),
              onClick: _cache[0] || (_cache[0] = ($event) => terminalVisible.value = true)
            }, null, 8, ["label"])) : createCommentVNode("", true)
          ])) : createCommentVNode("", true),
          withDirectives(createVNode(BaseTerminal, { onCreated: terminalCreated }, null, 512), [
            [vShow, terminalVisible.value]
          ])
        ]),
        _: 1
      });
    };
  }
});
const ServerStartView = /* @__PURE__ */ _export_sfc(_sfc_main, [["__scopeId", "data-v-42c1131d"]]);
export {
  ServerStartView as default
};
//# sourceMappingURL=ServerStartView-CIDTUh4x.js.map
