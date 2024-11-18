var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });
import { d as defineComponent, r as ref, o as onMounted, w as watch, I as onBeforeUnmount, g as openBlock, h as createElementBlock, i as createVNode, y as withCtx, A as createBaseVNode, a4 as toDisplayString, z as unref, bJ as script, bK as electronAPI } from "./index-bi78Y1IN.js";
import { o as oe, r as rr } from "./index-Ba5g1c58.js";
const _hoisted_1$1 = { class: "p-terminal rounded-none h-full w-full" };
const _hoisted_2$1 = { class: "px-4 whitespace-pre-wrap" };
const _sfc_main$1 = /* @__PURE__ */ defineComponent({
  __name: "LogTerminal",
  props: {
    fetchLogs: { type: Function },
    fetchInterval: {}
  },
  setup(__props) {
    const props = __props;
    const log = ref("");
    const scrollPanelRef = ref(null);
    const scrolledToBottom = ref(false);
    let intervalId = 0;
    onMounted(async () => {
      const element = scrollPanelRef.value?.$el;
      const scrollContainer = element?.querySelector(".p-scrollpanel-content");
      if (scrollContainer) {
        scrollContainer.addEventListener("scroll", () => {
          scrolledToBottom.value = scrollContainer.scrollTop + scrollContainer.clientHeight === scrollContainer.scrollHeight;
        });
      }
      const scrollToBottom = /* @__PURE__ */ __name(() => {
        if (scrollContainer) {
          scrollContainer.scrollTop = scrollContainer.scrollHeight;
        }
      }, "scrollToBottom");
      watch(log, () => {
        if (scrolledToBottom.value) {
          scrollToBottom();
        }
      });
      const fetchLogs = /* @__PURE__ */ __name(async () => {
        log.value = await props.fetchLogs();
      }, "fetchLogs");
      await fetchLogs();
      scrollToBottom();
      intervalId = window.setInterval(fetchLogs, props.fetchInterval);
    });
    onBeforeUnmount(() => {
      window.clearInterval(intervalId);
    });
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", _hoisted_1$1, [
        createVNode(unref(script), {
          class: "h-full w-full",
          ref_key: "scrollPanelRef",
          ref: scrollPanelRef
        }, {
          default: withCtx(() => [
            createBaseVNode("pre", _hoisted_2$1, toDisplayString(log.value), 1)
          ]),
          _: 1
        }, 512)
      ]);
    };
  }
});
const _hoisted_1 = { class: "font-sans flex flex-col justify-center items-center h-screen m-0 text-neutral-300 bg-neutral-900 dark-theme pointer-events-auto" };
const _hoisted_2 = { class: "text-2xl font-bold" };
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "ServerStartView",
  setup(__props) {
    const electron = electronAPI();
    const status = ref(oe.INITIAL_STATE);
    const logs = ref([]);
    const updateProgress = /* @__PURE__ */ __name(({ status: newStatus }) => {
      status.value = newStatus;
      logs.value = [];
    }, "updateProgress");
    const addLogMessage = /* @__PURE__ */ __name((message) => {
      logs.value = [...logs.value, message];
    }, "addLogMessage");
    const fetchLogs = /* @__PURE__ */ __name(async () => {
      return logs.value.join("\n");
    }, "fetchLogs");
    onMounted(() => {
      electron.sendReady();
      electron.onProgressUpdate(updateProgress);
      electron.onLogMessage((message) => {
        addLogMessage(message);
      });
    });
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", _hoisted_1, [
        createBaseVNode("h2", _hoisted_2, toDisplayString(unref(rr)[status.value]), 1),
        createVNode(_sfc_main$1, {
          "fetch-logs": fetchLogs,
          "fetch-interval": 500
        })
      ]);
    };
  }
});
export {
  _sfc_main as default
};
//# sourceMappingURL=ServerStartView-Bb00SCJ5.js.map
