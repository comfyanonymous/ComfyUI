var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });
import { d as defineComponent, bF as mergeModels, bz as useModel, o as openBlock, y as createBlock, z as withCtx, m as createBaseVNode, Z as normalizeClass, i as withDirectives, v as vShow, k as createVNode, j as unref, bN as script, l as script$1, c as computed, bO as PrimeIcons, bp as t, ah as script$2, bi as electronAPI, ad as defineStore, T as ref, bP as useTimeout, N as watch, f as createElementBlock, B as createCommentVNode, ak as createTextVNode, E as toDisplayString, aE as mergeProps, bQ as script$3, _ as _export_sfc, r as resolveDirective, bR as script$4, b3 as useToast, bS as useConfirm, F as Fragment, bu as script$5, D as renderList, bT as script$6, p as onMounted, bU as onUnmounted, a2 as script$7, as as isRef } from "./index-DIgj6hpb.js";
import { s as script$8 } from "./index-CE2hpomb.js";
import "./index-mHcHTYAB.js";
import { _ as _sfc_main$7 } from "./TerminalOutputDrawer-B_NZxAv8.js";
import { _ as _sfc_main$8 } from "./BaseViewTemplate-DaGOaycP.js";
import "./index-DdgRmRKP.js";
import "./index-D8hjiWMw.js";
import "./index-Czd3J-KM.js";
import "./index-C_ACzX5-.js";
import "./index-BKScv1mm.js";
const _sfc_main$6 = /* @__PURE__ */ defineComponent({
  __name: "RefreshButton",
  props: /* @__PURE__ */ mergeModels({
    outlined: { type: Boolean, default: true },
    disabled: { type: Boolean },
    severity: { default: "secondary" }
  }, {
    "modelValue": { type: Boolean, ...{ required: true } },
    "modelModifiers": {}
  }),
  emits: /* @__PURE__ */ mergeModels(["refresh"], ["update:modelValue"]),
  setup(__props) {
    const props = __props;
    const active = useModel(__props, "modelValue");
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(script$1), {
        class: "relative p-button-icon-only",
        outlined: props.outlined,
        severity: props.severity,
        disabled: active.value || props.disabled,
        onClick: _cache[0] || (_cache[0] = (event) => _ctx.$emit("refresh", event))
      }, {
        default: withCtx(() => [
          createBaseVNode("span", {
            class: normalizeClass(["p-button-icon pi pi-refresh transition-all", { "opacity-0": active.value }]),
            "data-pc-section": "icon"
          }, null, 2),
          _cache[1] || (_cache[1] = createBaseVNode("span", {
            class: "p-button-label",
            "data-pc-section": "label"
          }, " ", -1)),
          withDirectives(createVNode(unref(script), { class: "absolute w-1/2 h-1/2" }, null, 512), [
            [vShow, active.value]
          ])
        ]),
        _: 1
      }, 8, ["outlined", "severity", "disabled"]);
    };
  }
});
const _sfc_main$5 = /* @__PURE__ */ defineComponent({
  __name: "StatusTag",
  props: {
    error: { type: Boolean },
    refreshing: { type: Boolean }
  },
  setup(__props) {
    const props = __props;
    const icon = computed(() => {
      if (props.refreshing) return PrimeIcons.QUESTION;
      if (props.error) return PrimeIcons.TIMES;
      return PrimeIcons.CHECK;
    });
    const severity = computed(() => {
      if (props.refreshing) return "info";
      if (props.error) return "danger";
      return "success";
    });
    const value = computed(() => {
      if (props.refreshing) return t("maintenance.refreshing");
      if (props.error) return t("g.error");
      return t("maintenance.OK");
    });
    return (_ctx, _cache) => {
      return openBlock(), createBlock(unref(script$2), {
        icon: icon.value,
        severity: severity.value,
        value: value.value
      }, null, 8, ["icon", "severity", "value"]);
    };
  }
});
const electron = electronAPI();
const openUrl = /* @__PURE__ */ __name((url) => {
  window.open(url, "_blank");
  return true;
}, "openUrl");
const DESKTOP_MAINTENANCE_TASKS = [
  {
    id: "basePath",
    execute: /* @__PURE__ */ __name(async () => await electron.setBasePath(), "execute"),
    name: "Base path",
    shortDescription: "Change the application base path.",
    errorDescription: "Unable to open the base path.  Please select a new one.",
    description: "The base path is the default location where ComfyUI stores data. It is the location fo the python environment, and may also contain models, custom nodes, and other extensions.",
    isInstallationFix: true,
    button: {
      icon: PrimeIcons.QUESTION,
      text: "Select"
    }
  },
  {
    id: "git",
    headerImg: "assets/images/Git-Logo-White.svg",
    execute: /* @__PURE__ */ __name(() => openUrl("https://git-scm.com/downloads/"), "execute"),
    name: "Download git",
    shortDescription: "Open the git download page.",
    errorDescription: "Git is missing. Please download and install git, then restart ComfyUI Desktop.",
    description: "Git is required to download and manage custom nodes and other extensions. This task opens the download page in your default browser, where you can download the latest version of git. Once you have installed git, please restart ComfyUI Desktop.",
    button: {
      icon: PrimeIcons.EXTERNAL_LINK,
      text: "Download"
    }
  },
  {
    id: "vcRedist",
    execute: /* @__PURE__ */ __name(() => openUrl("https://aka.ms/vs/17/release/vc_redist.x64.exe"), "execute"),
    name: "Download VC++ Redist",
    shortDescription: "Download the latest VC++ Redistributable runtime.",
    description: "The Visual C++ runtime libraries are required to run ComfyUI. You will need to download and install this file.",
    button: {
      icon: PrimeIcons.EXTERNAL_LINK,
      text: "Download"
    }
  },
  {
    id: "reinstall",
    severity: "danger",
    requireConfirm: true,
    execute: /* @__PURE__ */ __name(async () => {
      await electron.reinstall();
      return true;
    }, "execute"),
    name: "Reinstall ComfyUI",
    shortDescription: "Deletes the desktop app config and load the welcome screen.",
    description: "Delete the desktop app config, restart the app, and load the installation screen.",
    confirmText: "Delete all saved config and reinstall?",
    button: {
      icon: PrimeIcons.EXCLAMATION_TRIANGLE,
      text: "Reinstall"
    }
  },
  {
    id: "pythonPackages",
    requireConfirm: true,
    execute: /* @__PURE__ */ __name(async () => {
      try {
        await electron.uv.installRequirements();
        return true;
      } catch (error) {
        return false;
      }
    }, "execute"),
    name: "Install python packages",
    shortDescription: "Installs the base python packages required to run ComfyUI.",
    errorDescription: "Python packages that are required to run ComfyUI are not installed.",
    description: "This will install the python packages required to run ComfyUI. This includes torch, torchvision, and other dependencies.",
    usesTerminal: true,
    isInstallationFix: true,
    button: {
      icon: PrimeIcons.DOWNLOAD,
      text: "Install"
    }
  },
  {
    id: "uv",
    execute: /* @__PURE__ */ __name(() => openUrl("https://docs.astral.sh/uv/getting-started/installation/"), "execute"),
    name: "uv executable",
    shortDescription: "uv installs and maintains the python environment.",
    description: "This will open the download page for Astral's uv tool. uv is used to install python and manage python packages.",
    button: {
      icon: "pi pi-asterisk",
      text: "Download"
    }
  },
  {
    id: "uvCache",
    severity: "danger",
    requireConfirm: true,
    execute: /* @__PURE__ */ __name(async () => await electron.uv.clearCache(), "execute"),
    name: "uv cache",
    shortDescription: "Remove the Astral uv cache of python packages.",
    description: "This will remove the uv cache directory and its contents. All downloaded python packages will need to be downloaded again.",
    confirmText: "Delete uv cache of python packages?",
    usesTerminal: true,
    isInstallationFix: true,
    button: {
      icon: PrimeIcons.TRASH,
      text: "Clear cache"
    }
  },
  {
    id: "venvDirectory",
    severity: "danger",
    requireConfirm: true,
    execute: /* @__PURE__ */ __name(async () => await electron.uv.resetVenv(), "execute"),
    name: "Reset virtual environment",
    shortDescription: "Remove and recreate the .venv directory. This removes all python packages.",
    description: "The python environment is where ComfyUI installs python and python packages. It is used to run the ComfyUI server.",
    confirmText: "Delete the .venv directory?",
    usesTerminal: true,
    isInstallationFix: true,
    button: {
      icon: PrimeIcons.FOLDER,
      text: "Recreate"
    }
  }
];
class MaintenanceTaskRunner {
  static {
    __name(this, "MaintenanceTaskRunner");
  }
  constructor(task) {
    this.task = task;
  }
  _state;
  /** The current state of the task. Setter also controls {@link resolved} as a side-effect. */
  get state() {
    return this._state;
  }
  /** Updates the task state and {@link resolved} status. */
  setState(value) {
    if (this._state === "error" && value === "OK") this.resolved = true;
    if (value === "error") this.resolved &&= false;
    this._state = value;
  }
  /** `true` if the task has been resolved (was `error`, now `OK`). This is a side-effect of the {@link state} setter. */
  resolved;
  /** Whether the task state is currently being refreshed. */
  refreshing;
  /** Whether the task is currently running. */
  executing;
  /** The error message that occurred when the task failed. */
  error;
  update(update) {
    const state = update[this.task.id];
    this.refreshing = state === void 0;
    if (state) this.setState(state);
  }
  finaliseUpdate(update) {
    this.refreshing = false;
    this.setState(update[this.task.id] ?? "skipped");
  }
  /** Wraps the execution of a maintenance task, updating state and rethrowing errors. */
  async execute(task) {
    try {
      this.executing = true;
      const success = await task.execute();
      if (!success) return false;
      this.error = void 0;
      return true;
    } catch (error) {
      this.error = error?.message;
      throw error;
    } finally {
      this.executing = false;
    }
  }
}
const useMaintenanceTaskStore = defineStore("maintenanceTask", () => {
  const electron2 = electronAPI();
  const isRefreshing = ref(false);
  const isRunningTerminalCommand = computed(
    () => tasks.value.filter((task) => task.usesTerminal).some((task) => getRunner(task)?.executing)
  );
  const isRunningInstallationFix = computed(
    () => tasks.value.filter((task) => task.isInstallationFix).some((task) => getRunner(task)?.executing)
  );
  const tasks = ref(DESKTOP_MAINTENANCE_TASKS);
  const taskRunners = ref(
    new Map(
      DESKTOP_MAINTENANCE_TASKS.map((x) => [x.id, new MaintenanceTaskRunner(x)])
    )
  );
  const anyErrors = computed(
    () => tasks.value.some((task) => getRunner(task).state === "error")
  );
  const getRunner = /* @__PURE__ */ __name((task) => taskRunners.value.get(task.id), "getRunner");
  const processUpdate = /* @__PURE__ */ __name((validationUpdate) => {
    const update = validationUpdate;
    isRefreshing.value = true;
    for (const task of tasks.value) {
      getRunner(task).update(update);
    }
    if (!update.inProgress && isRefreshing.value) {
      isRefreshing.value = false;
      for (const task of tasks.value) {
        getRunner(task).finaliseUpdate(update);
      }
    }
  }, "processUpdate");
  const clearResolved = /* @__PURE__ */ __name(() => {
    for (const task of tasks.value) {
      getRunner(task).resolved &&= false;
    }
  }, "clearResolved");
  const refreshDesktopTasks = /* @__PURE__ */ __name(async () => {
    isRefreshing.value = true;
    console.log("Refreshing desktop tasks");
    await electron2.Validation.validateInstallation(processUpdate);
  }, "refreshDesktopTasks");
  const execute = /* @__PURE__ */ __name(async (task) => {
    return getRunner(task).execute(task);
  }, "execute");
  return {
    tasks,
    isRefreshing,
    isRunningTerminalCommand,
    isRunningInstallationFix,
    execute,
    getRunner,
    processUpdate,
    clearResolved,
    /** True if any tasks are in an error state. */
    anyErrors,
    refreshDesktopTasks
  };
});
function useMinLoadingDurationRef(value, minDuration = 250) {
  const current = ref(value.value);
  const { ready, start } = useTimeout(minDuration, {
    controls: true,
    immediate: false
  });
  watch(value, (newValue) => {
    if (newValue && !current.value) start();
    current.value = newValue;
  });
  return computed(() => current.value || !ready.value);
}
__name(useMinLoadingDurationRef, "useMinLoadingDurationRef");
const _hoisted_1$3 = {
  key: 0,
  class: "pi pi-exclamation-triangle text-red-500 absolute m-2 top-0 -right-14 opacity-15",
  style: { "font-size": "10rem" }
};
const _hoisted_2$3 = ["src"];
const _hoisted_3$3 = { class: "flex gap-4 mt-1" };
const _hoisted_4$3 = {
  key: 0,
  class: "task-card-ok pi pi-check"
};
const _sfc_main$4 = /* @__PURE__ */ defineComponent({
  __name: "TaskCard",
  props: {
    task: {}
  },
  emits: ["execute"],
  setup(__props) {
    const taskStore = useMaintenanceTaskStore();
    const runner = computed(() => taskStore.getRunner(props.task));
    const props = __props;
    const description = computed(
      () => runner.value.state === "error" ? props.task.errorDescription ?? props.task.shortDescription : props.task.shortDescription
    );
    const reactiveLoading = computed(() => runner.value.refreshing);
    const reactiveExecuting = computed(() => runner.value.executing);
    const isLoading = useMinLoadingDurationRef(reactiveLoading, 250);
    const isExecuting = useMinLoadingDurationRef(reactiveExecuting, 250);
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", {
        class: normalizeClass(["task-div max-w-48 min-h-52 grid relative", { "opacity-75": unref(isLoading) }])
      }, [
        createVNode(unref(script$3), mergeProps({
          class: ["max-w-48 relative h-full overflow-hidden", { "opacity-65": runner.value.state !== "error" }]
        }, (({ onClick, ...rest }) => rest)(_ctx.$attrs)), {
          header: withCtx(() => [
            runner.value.state === "error" ? (openBlock(), createElementBlock("i", _hoisted_1$3)) : createCommentVNode("", true),
            _ctx.task.headerImg ? (openBlock(), createElementBlock("img", {
              key: 1,
              src: _ctx.task.headerImg,
              class: "object-contain w-full h-full opacity-25 pt-4 px-4"
            }, null, 8, _hoisted_2$3)) : createCommentVNode("", true)
          ]),
          title: withCtx(() => [
            createTextVNode(toDisplayString(_ctx.task.name), 1)
          ]),
          content: withCtx(() => [
            createTextVNode(toDisplayString(description.value), 1)
          ]),
          footer: withCtx(() => [
            createBaseVNode("div", _hoisted_3$3, [
              createVNode(unref(script$1), {
                icon: _ctx.task.button?.icon,
                label: _ctx.task.button?.text,
                class: "w-full",
                raised: "",
                "icon-pos": "right",
                onClick: _cache[0] || (_cache[0] = (event) => _ctx.$emit("execute", event)),
                loading: unref(isExecuting)
              }, null, 8, ["icon", "label", "loading"])
            ])
          ]),
          _: 1
        }, 16, ["class"]),
        !unref(isLoading) && runner.value.state === "OK" ? (openBlock(), createElementBlock("i", _hoisted_4$3)) : createCommentVNode("", true)
      ], 2);
    };
  }
});
const TaskCard = /* @__PURE__ */ _export_sfc(_sfc_main$4, [["__scopeId", "data-v-c3bd7658"]]);
const _sfc_main$3 = /* @__PURE__ */ defineComponent({
  __name: "TaskListStatusIcon",
  props: {
    state: {},
    loading: {}
  },
  setup(__props) {
    const tooltip = computed(() => {
      if (props.state === "error") {
        return t("g.error");
      } else if (props.state === "OK") {
        return t("maintenance.OK");
      } else {
        return t("maintenance.Skipped");
      }
    });
    const cssClasses = computed(() => {
      let classes;
      if (props.state === "error") {
        classes = `${PrimeIcons.EXCLAMATION_TRIANGLE} text-red-500`;
      } else if (props.state === "OK") {
        classes = `${PrimeIcons.CHECK} text-green-500`;
      } else {
        classes = PrimeIcons.MINUS;
      }
      return `text-3xl pi ${classes}`;
    });
    const props = __props;
    return (_ctx, _cache) => {
      const _directive_tooltip = resolveDirective("tooltip");
      return !_ctx.state || _ctx.loading ? (openBlock(), createBlock(unref(script), {
        key: 0,
        class: "h-8 w-8"
      })) : withDirectives((openBlock(), createElementBlock("i", {
        key: 1,
        class: normalizeClass(cssClasses.value)
      }, null, 2)), [
        [
          _directive_tooltip,
          { value: tooltip.value, showDelay: 250 },
          void 0,
          { top: true }
        ]
      ]);
    };
  }
});
const _hoisted_1$2 = { class: "text-center w-16" };
const _hoisted_2$2 = { class: "inline-block" };
const _hoisted_3$2 = { class: "whitespace-pre-line" };
const _hoisted_4$2 = { class: "text-right px-4" };
const _sfc_main$2 = /* @__PURE__ */ defineComponent({
  __name: "TaskListItem",
  props: {
    task: {}
  },
  emits: ["execute"],
  setup(__props) {
    const taskStore = useMaintenanceTaskStore();
    const runner = computed(() => taskStore.getRunner(props.task));
    const props = __props;
    const severity = computed(
      () => runner.value.state === "error" || runner.value.state === "warning" ? "primary" : "secondary"
    );
    const reactiveLoading = computed(() => runner.value.refreshing);
    const reactiveExecuting = computed(() => runner.value.executing);
    const isLoading = useMinLoadingDurationRef(reactiveLoading, 250);
    const isExecuting = useMinLoadingDurationRef(reactiveExecuting, 250);
    const infoPopover = ref();
    const toggle = /* @__PURE__ */ __name((event) => {
      infoPopover.value.toggle(event);
    }, "toggle");
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("tr", {
        class: normalizeClass(["border-neutral-700 border-solid border-y", {
          "opacity-50": runner.value.resolved,
          "opacity-75": unref(isLoading) && runner.value.resolved
        }])
      }, [
        createBaseVNode("td", _hoisted_1$2, [
          createVNode(_sfc_main$3, {
            state: runner.value.state,
            loading: unref(isLoading)
          }, null, 8, ["state", "loading"])
        ]),
        createBaseVNode("td", null, [
          createBaseVNode("p", _hoisted_2$2, toDisplayString(_ctx.task.name), 1),
          createVNode(unref(script$1), {
            class: "inline-block mx-2",
            type: "button",
            icon: unref(PrimeIcons).INFO_CIRCLE,
            severity: "secondary",
            text: true,
            onClick: toggle
          }, null, 8, ["icon"]),
          createVNode(unref(script$4), {
            ref_key: "infoPopover",
            ref: infoPopover,
            class: "block m-1 max-w-64 min-w-32"
          }, {
            default: withCtx(() => [
              createBaseVNode("span", _hoisted_3$2, toDisplayString(_ctx.task.description), 1)
            ]),
            _: 1
          }, 512)
        ]),
        createBaseVNode("td", _hoisted_4$2, [
          createVNode(unref(script$1), {
            icon: _ctx.task.button?.icon,
            label: _ctx.task.button?.text,
            severity: severity.value,
            "icon-pos": "right",
            onClick: _cache[0] || (_cache[0] = (event) => _ctx.$emit("execute", event)),
            loading: unref(isExecuting)
          }, null, 8, ["icon", "label", "severity", "loading"])
        ])
      ], 2);
    };
  }
});
const _hoisted_1$1 = { class: "my-4" };
const _hoisted_2$1 = { class: "text-neutral-400 w-full text-center" };
const _hoisted_3$1 = {
  key: 0,
  class: "w-full border-collapse border-hidden"
};
const _hoisted_4$1 = {
  key: 1,
  class: "flex flex-wrap justify-evenly gap-8 pad-y my-4"
};
const _sfc_main$1 = /* @__PURE__ */ defineComponent({
  __name: "TaskListPanel",
  props: {
    displayAsList: {},
    filter: {},
    isRefreshing: { type: Boolean }
  },
  setup(__props) {
    const toast = useToast();
    const confirm = useConfirm();
    const taskStore = useMaintenanceTaskStore();
    const props = __props;
    const executeTask = /* @__PURE__ */ __name(async (task) => {
      let message;
      try {
        if (await taskStore.execute(task) === true) return;
        message = t("maintenance.error.taskFailed");
      } catch (error) {
        message = error?.message;
      }
      toast.add({
        severity: "error",
        summary: t("maintenance.error.toastTitle"),
        detail: message ?? t("maintenance.error.defaultDescription"),
        life: 1e4
      });
    }, "executeTask");
    const confirmButton = /* @__PURE__ */ __name(async (event, task) => {
      if (!task.requireConfirm) {
        await executeTask(task);
        return;
      }
      confirm.require({
        target: event.currentTarget,
        message: task.confirmText ?? t("maintenance.confirmTitle"),
        icon: "pi pi-exclamation-circle",
        rejectProps: {
          label: t("g.cancel"),
          severity: "secondary",
          outlined: true
        },
        acceptProps: {
          label: task.button?.text ?? t("g.save"),
          severity: task.severity ?? "primary"
        },
        // TODO: Not awaited.
        accept: /* @__PURE__ */ __name(async () => {
          await executeTask(task);
        }, "accept")
      });
    }, "confirmButton");
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("section", _hoisted_1$1, [
        _ctx.filter.tasks.length === 0 ? (openBlock(), createElementBlock(Fragment, { key: 0 }, [
          createVNode(unref(script$5)),
          createBaseVNode("p", _hoisted_2$1, toDisplayString(_ctx.$t("maintenance.allOk")), 1)
        ], 64)) : (openBlock(), createElementBlock(Fragment, { key: 1 }, [
          _ctx.displayAsList === unref(PrimeIcons).LIST ? (openBlock(), createElementBlock("table", _hoisted_3$1, [
            (openBlock(true), createElementBlock(Fragment, null, renderList(_ctx.filter.tasks, (task) => {
              return openBlock(), createBlock(_sfc_main$2, {
                key: task.id,
                task,
                onExecute: /* @__PURE__ */ __name((event) => confirmButton(event, task), "onExecute")
              }, null, 8, ["task", "onExecute"]);
            }), 128))
          ])) : (openBlock(), createElementBlock("div", _hoisted_4$1, [
            (openBlock(true), createElementBlock(Fragment, null, renderList(_ctx.filter.tasks, (task) => {
              return openBlock(), createBlock(TaskCard, {
                key: task.id,
                task,
                onExecute: /* @__PURE__ */ __name((event) => confirmButton(event, task), "onExecute")
              }, null, 8, ["task", "onExecute"]);
            }), 128))
          ]))
        ], 64)),
        createVNode(unref(script$6))
      ]);
    };
  }
});
const _hoisted_1 = { class: "min-w-full min-h-full font-sans w-screen h-screen grid justify-around text-neutral-300 bg-neutral-900 dark-theme overflow-y-auto" };
const _hoisted_2 = { class: "max-w-screen-sm w-screen m-8 relative" };
const _hoisted_3 = { class: "backspan pi-wrench text-4xl font-bold" };
const _hoisted_4 = { class: "w-full flex flex-wrap gap-4 items-center" };
const _hoisted_5 = { class: "grow" };
const _hoisted_6 = { class: "flex gap-4 items-center" };
const _hoisted_7 = { class: "max-sm:hidden" };
const _hoisted_8 = { class: "flex justify-between gap-4 flex-row" };
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "MaintenanceView",
  setup(__props) {
    const electron2 = electronAPI();
    const toast = useToast();
    const taskStore = useMaintenanceTaskStore();
    const { clearResolved, processUpdate, refreshDesktopTasks } = taskStore;
    const terminalVisible = ref(false);
    const reactiveIsRefreshing = computed(() => taskStore.isRefreshing);
    const isRefreshing = useMinLoadingDurationRef(reactiveIsRefreshing, 250);
    const anyErrors = computed(() => taskStore.anyErrors);
    const displayAsList = ref(PrimeIcons.TH_LARGE);
    const errorFilter = computed(
      () => taskStore.tasks.filter((x) => {
        const { state, resolved } = taskStore.getRunner(x);
        return state === "error" || resolved;
      })
    );
    const filterOptions = ref([
      { icon: PrimeIcons.FILTER_FILL, value: "All", tasks: taskStore.tasks },
      { icon: PrimeIcons.EXCLAMATION_TRIANGLE, value: "Errors", tasks: errorFilter }
    ]);
    const filter = ref(filterOptions.value[0]);
    const completeValidation = /* @__PURE__ */ __name(async () => {
      const isValid = await electron2.Validation.complete();
      if (!isValid) {
        toast.add({
          severity: "error",
          summary: t("g.error"),
          detail: t("maintenance.error.cannotContinue"),
          life: 5e3
        });
      }
    }, "completeValidation");
    const toggleConsoleDrawer = /* @__PURE__ */ __name(() => {
      terminalVisible.value = !terminalVisible.value;
    }, "toggleConsoleDrawer");
    watch(
      () => taskStore.isRunningTerminalCommand,
      (value) => {
        terminalVisible.value = value;
      }
    );
    onMounted(async () => {
      electron2.Validation.onUpdate(processUpdate);
      const update = await electron2.Validation.getStatus();
      if (Object.values(update).some((x) => x === "error")) {
        filter.value = filterOptions.value[1];
      }
      processUpdate(update);
    });
    onUnmounted(() => electron2.Validation.dispose());
    return (_ctx, _cache) => {
      return openBlock(), createBlock(_sfc_main$8, { dark: "" }, {
        default: withCtx(() => [
          createBaseVNode("div", _hoisted_1, [
            createBaseVNode("div", _hoisted_2, [
              createBaseVNode("h1", _hoisted_3, toDisplayString(unref(t)("maintenance.title")), 1),
              createBaseVNode("div", _hoisted_4, [
                createBaseVNode("span", _hoisted_5, [
                  createTextVNode(toDisplayString(unref(t)("maintenance.status")) + ": ", 1),
                  createVNode(_sfc_main$5, {
                    refreshing: unref(isRefreshing),
                    error: anyErrors.value
                  }, null, 8, ["refreshing", "error"])
                ]),
                createBaseVNode("div", _hoisted_6, [
                  createVNode(unref(script$7), {
                    modelValue: displayAsList.value,
                    "onUpdate:modelValue": _cache[0] || (_cache[0] = ($event) => displayAsList.value = $event),
                    options: [unref(PrimeIcons).LIST, unref(PrimeIcons).TH_LARGE],
                    "allow-empty": false
                  }, {
                    option: withCtx((opts) => [
                      createBaseVNode("i", {
                        class: normalizeClass(opts.option)
                      }, null, 2)
                    ]),
                    _: 1
                  }, 8, ["modelValue", "options"]),
                  createVNode(unref(script$7), {
                    modelValue: filter.value,
                    "onUpdate:modelValue": _cache[1] || (_cache[1] = ($event) => filter.value = $event),
                    options: filterOptions.value,
                    "allow-empty": false,
                    optionLabel: "value",
                    dataKey: "value",
                    "area-labelledby": "custom",
                    onChange: unref(clearResolved)
                  }, {
                    option: withCtx((opts) => [
                      createBaseVNode("i", {
                        class: normalizeClass(opts.option.icon)
                      }, null, 2),
                      createBaseVNode("span", _hoisted_7, toDisplayString(opts.option.value), 1)
                    ]),
                    _: 1
                  }, 8, ["modelValue", "options", "onChange"]),
                  createVNode(_sfc_main$6, {
                    modelValue: unref(isRefreshing),
                    "onUpdate:modelValue": _cache[2] || (_cache[2] = ($event) => isRef(isRefreshing) ? isRefreshing.value = $event : null),
                    severity: "secondary",
                    onRefresh: unref(refreshDesktopTasks)
                  }, null, 8, ["modelValue", "onRefresh"])
                ])
              ]),
              createVNode(_sfc_main$1, {
                class: "border-neutral-700 border-solid border-x-0 border-y",
                filter: filter.value,
                displayAsList: displayAsList.value,
                isRefreshing: unref(isRefreshing)
              }, null, 8, ["filter", "displayAsList", "isRefreshing"]),
              createBaseVNode("div", _hoisted_8, [
                createVNode(unref(script$1), {
                  label: unref(t)("maintenance.consoleLogs"),
                  icon: "pi pi-desktop",
                  "icon-pos": "left",
                  severity: "secondary",
                  onClick: toggleConsoleDrawer
                }, null, 8, ["label"]),
                createVNode(unref(script$1), {
                  label: unref(t)("g.continue"),
                  icon: "pi pi-arrow-right",
                  "icon-pos": "left",
                  severity: anyErrors.value ? "secondary" : "primary",
                  onClick: _cache[3] || (_cache[3] = () => completeValidation()),
                  loading: unref(isRefreshing)
                }, null, 8, ["label", "severity", "loading"])
              ])
            ]),
            createVNode(_sfc_main$7, {
              modelValue: terminalVisible.value,
              "onUpdate:modelValue": _cache[4] || (_cache[4] = ($event) => terminalVisible.value = $event),
              header: unref(t)("g.terminal"),
              "default-message": unref(t)("maintenance.terminalDefaultMessage")
            }, null, 8, ["modelValue", "header", "default-message"]),
            createVNode(unref(script$8))
          ])
        ]),
        _: 1
      });
    };
  }
});
const MaintenanceView = /* @__PURE__ */ _export_sfc(_sfc_main, [["__scopeId", "data-v-1073ace1"]]);
export {
  MaintenanceView as default
};
//# sourceMappingURL=MaintenanceView-BFp9KqAr.js.map
