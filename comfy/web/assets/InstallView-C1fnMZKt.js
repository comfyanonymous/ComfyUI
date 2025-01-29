var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });
import { d as defineComponent, U as ref, bm as useModel, o as openBlock, f as createElementBlock, m as createBaseVNode, E as toDisplayString, k as createVNode, j as unref, bn as script, bh as script$1, ar as withModifiers, z as withCtx, ab as script$2, K as useI18n, c as computed, ai as normalizeClass, B as createCommentVNode, a4 as script$3, a7 as createTextVNode, b5 as electronAPI, _ as _export_sfc, p as onMounted, r as resolveDirective, bg as script$4, i as withDirectives, bo as script$5, bp as script$6, l as script$7, y as createBlock, bj as script$8, bq as MigrationItems, w as watchEffect, F as Fragment, D as renderList, br as script$9, be as useRouter, ag as toRaw } from "./index-BsGgXmrT.js";
import { s as script$a, a as script$b, b as script$c, c as script$d, d as script$e } from "./index-DC_-jkme.js";
import { _ as _sfc_main$5 } from "./BaseViewTemplate-DDUNNAbV.js";
const _hoisted_1$4 = { class: "flex flex-col gap-6 w-[600px]" };
const _hoisted_2$4 = { class: "flex flex-col gap-4" };
const _hoisted_3$4 = { class: "text-2xl font-semibold text-neutral-100" };
const _hoisted_4$4 = { class: "text-neutral-400 my-0" };
const _hoisted_5$3 = { class: "flex flex-col bg-neutral-800 p-4 rounded-lg" };
const _hoisted_6$3 = { class: "flex items-center gap-4" };
const _hoisted_7$3 = { class: "flex-1" };
const _hoisted_8$3 = { class: "text-lg font-medium text-neutral-100" };
const _hoisted_9$3 = { class: "text-sm text-neutral-400 mt-1" };
const _hoisted_10$3 = { class: "flex items-center gap-4" };
const _hoisted_11$3 = { class: "flex-1" };
const _hoisted_12$3 = { class: "text-lg font-medium text-neutral-100" };
const _hoisted_13$1 = { class: "text-sm text-neutral-400 mt-1" };
const _hoisted_14$1 = { class: "text-neutral-300" };
const _hoisted_15 = { class: "font-medium mb-2" };
const _hoisted_16 = { class: "list-disc pl-6 space-y-1" };
const _hoisted_17 = { class: "font-medium mt-4 mb-2" };
const _hoisted_18 = { class: "list-disc pl-6 space-y-1" };
const _hoisted_19 = { class: "mt-4" };
const _hoisted_20 = {
  href: "https://comfy.org/privacy",
  target: "_blank",
  class: "text-blue-400 hover:text-blue-300 underline"
};
const _sfc_main$4 = /* @__PURE__ */ defineComponent({
  __name: "DesktopSettingsConfiguration",
  props: {
    "autoUpdate": { type: Boolean, ...{ required: true } },
    "autoUpdateModifiers": {},
    "allowMetrics": { type: Boolean, ...{ required: true } },
    "allowMetricsModifiers": {}
  },
  emits: ["update:autoUpdate", "update:allowMetrics"],
  setup(__props) {
    const showDialog = ref(false);
    const autoUpdate = useModel(__props, "autoUpdate");
    const allowMetrics = useModel(__props, "allowMetrics");
    const showMetricsInfo = /* @__PURE__ */ __name(() => {
      showDialog.value = true;
    }, "showMetricsInfo");
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", _hoisted_1$4, [
        createBaseVNode("div", _hoisted_2$4, [
          createBaseVNode("h2", _hoisted_3$4, toDisplayString(_ctx.$t("install.desktopAppSettings")), 1),
          createBaseVNode("p", _hoisted_4$4, toDisplayString(_ctx.$t("install.desktopAppSettingsDescription")), 1)
        ]),
        createBaseVNode("div", _hoisted_5$3, [
          createBaseVNode("div", _hoisted_6$3, [
            createBaseVNode("div", _hoisted_7$3, [
              createBaseVNode("h3", _hoisted_8$3, toDisplayString(_ctx.$t("install.settings.autoUpdate")), 1),
              createBaseVNode("p", _hoisted_9$3, toDisplayString(_ctx.$t("install.settings.autoUpdateDescription")), 1)
            ]),
            createVNode(unref(script), {
              modelValue: autoUpdate.value,
              "onUpdate:modelValue": _cache[0] || (_cache[0] = ($event) => autoUpdate.value = $event)
            }, null, 8, ["modelValue"])
          ]),
          createVNode(unref(script$1)),
          createBaseVNode("div", _hoisted_10$3, [
            createBaseVNode("div", _hoisted_11$3, [
              createBaseVNode("h3", _hoisted_12$3, toDisplayString(_ctx.$t("install.settings.allowMetrics")), 1),
              createBaseVNode("p", _hoisted_13$1, toDisplayString(_ctx.$t("install.settings.allowMetricsDescription")), 1),
              createBaseVNode("a", {
                href: "#",
                class: "text-sm text-blue-400 hover:text-blue-300 mt-1 inline-block",
                onClick: withModifiers(showMetricsInfo, ["prevent"])
              }, toDisplayString(_ctx.$t("install.settings.learnMoreAboutData")), 1)
            ]),
            createVNode(unref(script), {
              modelValue: allowMetrics.value,
              "onUpdate:modelValue": _cache[1] || (_cache[1] = ($event) => allowMetrics.value = $event)
            }, null, 8, ["modelValue"])
          ])
        ]),
        createVNode(unref(script$2), {
          visible: showDialog.value,
          "onUpdate:visible": _cache[2] || (_cache[2] = ($event) => showDialog.value = $event),
          modal: "",
          header: _ctx.$t("install.settings.dataCollectionDialog.title")
        }, {
          default: withCtx(() => [
            createBaseVNode("div", _hoisted_14$1, [
              createBaseVNode("h4", _hoisted_15, toDisplayString(_ctx.$t("install.settings.dataCollectionDialog.whatWeCollect")), 1),
              createBaseVNode("ul", _hoisted_16, [
                createBaseVNode("li", null, toDisplayString(_ctx.$t("install.settings.dataCollectionDialog.collect.errorReports")), 1),
                createBaseVNode("li", null, toDisplayString(_ctx.$t("install.settings.dataCollectionDialog.collect.systemInfo")), 1),
                createBaseVNode("li", null, toDisplayString(_ctx.$t(
                  "install.settings.dataCollectionDialog.collect.userJourneyEvents"
                )), 1)
              ]),
              createBaseVNode("h4", _hoisted_17, toDisplayString(_ctx.$t("install.settings.dataCollectionDialog.whatWeDoNotCollect")), 1),
              createBaseVNode("ul", _hoisted_18, [
                createBaseVNode("li", null, toDisplayString(_ctx.$t(
                  "install.settings.dataCollectionDialog.doNotCollect.personalInformation"
                )), 1),
                createBaseVNode("li", null, toDisplayString(_ctx.$t(
                  "install.settings.dataCollectionDialog.doNotCollect.workflowContents"
                )), 1),
                createBaseVNode("li", null, toDisplayString(_ctx.$t(
                  "install.settings.dataCollectionDialog.doNotCollect.fileSystemInformation"
                )), 1),
                createBaseVNode("li", null, toDisplayString(_ctx.$t(
                  "install.settings.dataCollectionDialog.doNotCollect.customNodeConfigurations"
                )), 1)
              ]),
              createBaseVNode("div", _hoisted_19, [
                createBaseVNode("a", _hoisted_20, toDisplayString(_ctx.$t("install.settings.dataCollectionDialog.viewFullPolicy")), 1)
              ])
            ])
          ]),
          _: 1
        }, 8, ["visible", "header"])
      ]);
    };
  }
});
const _imports_0 = "" + new URL("images/nvidia-logo.svg", import.meta.url).href;
const _imports_1 = "" + new URL("images/apple-mps-logo.png", import.meta.url).href;
const _imports_2 = "" + new URL("images/manual-configuration.svg", import.meta.url).href;
const _hoisted_1$3 = { class: "flex flex-col gap-6 w-[600px] h-[30rem] select-none" };
const _hoisted_2$3 = { class: "grow flex flex-col gap-4 text-neutral-300" };
const _hoisted_3$3 = { class: "text-2xl font-semibold text-neutral-100" };
const _hoisted_4$3 = { class: "m-1 text-neutral-400" };
const _hoisted_5$2 = {
  key: 0,
  class: "m-1"
};
const _hoisted_6$2 = {
  key: 1,
  class: "m-1"
};
const _hoisted_7$2 = {
  key: 2,
  class: "text-neutral-300"
};
const _hoisted_8$2 = { class: "m-1" };
const _hoisted_9$2 = { key: 3 };
const _hoisted_10$2 = { class: "m-1" };
const _hoisted_11$2 = { class: "m-1" };
const _hoisted_12$2 = {
  for: "cpu-mode",
  class: "select-none"
};
const _sfc_main$3 = /* @__PURE__ */ defineComponent({
  __name: "GpuPicker",
  props: {
    "device": {
      required: true
    },
    "deviceModifiers": {}
  },
  emits: ["update:device"],
  setup(__props) {
    const { t } = useI18n();
    const cpuMode = computed({
      get: /* @__PURE__ */ __name(() => selected.value === "cpu", "get"),
      set: /* @__PURE__ */ __name((value) => {
        selected.value = value ? "cpu" : null;
      }, "set")
    });
    const selected = useModel(__props, "device");
    const electron = electronAPI();
    const platform = electron.getPlatform();
    const pickGpu = /* @__PURE__ */ __name((value) => {
      const newValue = selected.value === value ? null : value;
      selected.value = newValue;
    }, "pickGpu");
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", _hoisted_1$3, [
        createBaseVNode("div", _hoisted_2$3, [
          createBaseVNode("h2", _hoisted_3$3, toDisplayString(_ctx.$t("install.gpuSelection.selectGpu")), 1),
          createBaseVNode("p", _hoisted_4$3, toDisplayString(_ctx.$t("install.gpuSelection.selectGpuDescription")) + ": ", 1),
          createBaseVNode("div", {
            class: normalizeClass(["flex gap-2 text-center transition-opacity", { selected: selected.value }])
          }, [
            unref(platform) !== "darwin" ? (openBlock(), createElementBlock("div", {
              key: 0,
              class: normalizeClass(["gpu-button", { selected: selected.value === "nvidia" }]),
              role: "button",
              onClick: _cache[0] || (_cache[0] = ($event) => pickGpu("nvidia"))
            }, _cache[4] || (_cache[4] = [
              createBaseVNode("img", {
                class: "m-12",
                alt: "NVIDIA logo",
                width: "196",
                height: "32",
                src: _imports_0
              }, null, -1)
            ]), 2)) : createCommentVNode("", true),
            unref(platform) === "darwin" ? (openBlock(), createElementBlock("div", {
              key: 1,
              class: normalizeClass(["gpu-button", { selected: selected.value === "mps" }]),
              role: "button",
              onClick: _cache[1] || (_cache[1] = ($event) => pickGpu("mps"))
            }, _cache[5] || (_cache[5] = [
              createBaseVNode("img", {
                class: "rounded-lg hover-brighten",
                alt: "Apple Metal Performance Shaders Logo",
                width: "292",
                ratio: "",
                src: _imports_1
              }, null, -1)
            ]), 2)) : createCommentVNode("", true),
            createBaseVNode("div", {
              class: normalizeClass(["gpu-button", { selected: selected.value === "unsupported" }]),
              role: "button",
              onClick: _cache[2] || (_cache[2] = ($event) => pickGpu("unsupported"))
            }, _cache[6] || (_cache[6] = [
              createBaseVNode("img", {
                class: "m-12",
                alt: "Manual configuration",
                width: "196",
                src: _imports_2
              }, null, -1)
            ]), 2)
          ], 2),
          selected.value === "nvidia" ? (openBlock(), createElementBlock("p", _hoisted_5$2, [
            createVNode(unref(script$3), {
              icon: "pi pi-check",
              severity: "success",
              value: "CUDA"
            }),
            createTextVNode(" " + toDisplayString(_ctx.$t("install.gpuSelection.nvidiaDescription")), 1)
          ])) : createCommentVNode("", true),
          selected.value === "mps" ? (openBlock(), createElementBlock("p", _hoisted_6$2, [
            createVNode(unref(script$3), {
              icon: "pi pi-check",
              severity: "success",
              value: "MPS"
            }),
            createTextVNode(" " + toDisplayString(_ctx.$t("install.gpuSelection.mpsDescription")), 1)
          ])) : createCommentVNode("", true),
          selected.value === "unsupported" ? (openBlock(), createElementBlock("div", _hoisted_7$2, [
            createBaseVNode("p", _hoisted_8$2, [
              createVNode(unref(script$3), {
                icon: "pi pi-exclamation-triangle",
                severity: "warn",
                value: unref(t)("icon.exclamation-triangle")
              }, null, 8, ["value"]),
              createTextVNode(" " + toDisplayString(_ctx.$t("install.gpuSelection.customSkipsPython")), 1)
            ]),
            createBaseVNode("ul", null, [
              createBaseVNode("li", null, [
                createBaseVNode("strong", null, toDisplayString(_ctx.$t("install.gpuSelection.customComfyNeedsPython")), 1)
              ]),
              createBaseVNode("li", null, toDisplayString(_ctx.$t("install.gpuSelection.customManualVenv")), 1),
              createBaseVNode("li", null, toDisplayString(_ctx.$t("install.gpuSelection.customInstallRequirements")), 1),
              createBaseVNode("li", null, toDisplayString(_ctx.$t("install.gpuSelection.customMayNotWork")), 1)
            ])
          ])) : createCommentVNode("", true),
          selected.value === "cpu" ? (openBlock(), createElementBlock("div", _hoisted_9$2, [
            createBaseVNode("p", _hoisted_10$2, [
              createVNode(unref(script$3), {
                icon: "pi pi-exclamation-triangle",
                severity: "warn",
                value: unref(t)("icon.exclamation-triangle")
              }, null, 8, ["value"]),
              createTextVNode(" " + toDisplayString(_ctx.$t("install.gpuSelection.cpuModeDescription")), 1)
            ]),
            createBaseVNode("p", _hoisted_11$2, toDisplayString(_ctx.$t("install.gpuSelection.cpuModeDescription2")), 1)
          ])) : createCommentVNode("", true)
        ]),
        createBaseVNode("div", {
          class: normalizeClass(["transition-opacity flex gap-3 h-0", {
            "opacity-40": selected.value && selected.value !== "cpu"
          }])
        }, [
          createVNode(unref(script), {
            modelValue: cpuMode.value,
            "onUpdate:modelValue": _cache[3] || (_cache[3] = ($event) => cpuMode.value = $event),
            inputId: "cpu-mode",
            class: "-translate-y-40"
          }, null, 8, ["modelValue"]),
          createBaseVNode("label", _hoisted_12$2, toDisplayString(_ctx.$t("install.gpuSelection.enableCpuMode")), 1)
        ], 2)
      ]);
    };
  }
});
const GpuPicker = /* @__PURE__ */ _export_sfc(_sfc_main$3, [["__scopeId", "data-v-79125ff6"]]);
const _hoisted_1$2 = { class: "flex flex-col gap-6 w-[600px]" };
const _hoisted_2$2 = { class: "flex flex-col gap-4" };
const _hoisted_3$2 = { class: "text-2xl font-semibold text-neutral-100" };
const _hoisted_4$2 = { class: "text-neutral-400 my-0" };
const _hoisted_5$1 = { class: "flex gap-2" };
const _hoisted_6$1 = { class: "bg-neutral-800 p-4 rounded-lg" };
const _hoisted_7$1 = { class: "text-lg font-medium mt-0 mb-3 text-neutral-100" };
const _hoisted_8$1 = { class: "flex flex-col gap-2" };
const _hoisted_9$1 = { class: "flex items-center gap-2" };
const _hoisted_10$1 = { class: "text-neutral-200" };
const _hoisted_11$1 = { class: "pi pi-info-circle" };
const _hoisted_12$1 = { class: "flex items-center gap-2" };
const _hoisted_13 = { class: "text-neutral-200" };
const _hoisted_14 = { class: "pi pi-info-circle" };
const _sfc_main$2 = /* @__PURE__ */ defineComponent({
  __name: "InstallLocationPicker",
  props: {
    "installPath": { required: true },
    "installPathModifiers": {},
    "pathError": { required: true },
    "pathErrorModifiers": {}
  },
  emits: ["update:installPath", "update:pathError"],
  setup(__props) {
    const { t } = useI18n();
    const installPath = useModel(__props, "installPath");
    const pathError = useModel(__props, "pathError");
    const pathExists = ref(false);
    const appData = ref("");
    const appPath = ref("");
    const electron = electronAPI();
    onMounted(async () => {
      const paths = await electron.getSystemPaths();
      appData.value = paths.appData;
      appPath.value = paths.appPath;
      installPath.value = paths.defaultInstallPath;
      await validatePath(paths.defaultInstallPath);
    });
    const validatePath = /* @__PURE__ */ __name(async (path) => {
      try {
        pathError.value = "";
        pathExists.value = false;
        const validation = await electron.validateInstallPath(path);
        if (!validation.isValid) {
          const errors = [];
          if (validation.cannotWrite) errors.push(t("install.cannotWrite"));
          if (validation.freeSpace < validation.requiredSpace) {
            const requiredGB = validation.requiredSpace / 1024 / 1024 / 1024;
            errors.push(`${t("install.insufficientFreeSpace")}: ${requiredGB} GB`);
          }
          if (validation.parentMissing) errors.push(t("install.parentMissing"));
          if (validation.error)
            errors.push(`${t("install.unhandledError")}: ${validation.error}`);
          pathError.value = errors.join("\n");
        }
        if (validation.exists) pathExists.value = true;
      } catch (error) {
        pathError.value = t("install.pathValidationFailed");
      }
    }, "validatePath");
    const browsePath = /* @__PURE__ */ __name(async () => {
      try {
        const result = await electron.showDirectoryPicker();
        if (result) {
          installPath.value = result;
          await validatePath(result);
        }
      } catch (error) {
        pathError.value = t("install.failedToSelectDirectory");
      }
    }, "browsePath");
    return (_ctx, _cache) => {
      const _directive_tooltip = resolveDirective("tooltip");
      return openBlock(), createElementBlock("div", _hoisted_1$2, [
        createBaseVNode("div", _hoisted_2$2, [
          createBaseVNode("h2", _hoisted_3$2, toDisplayString(_ctx.$t("install.chooseInstallationLocation")), 1),
          createBaseVNode("p", _hoisted_4$2, toDisplayString(_ctx.$t("install.installLocationDescription")), 1),
          createBaseVNode("div", _hoisted_5$1, [
            createVNode(unref(script$6), { class: "flex-1" }, {
              default: withCtx(() => [
                createVNode(unref(script$4), {
                  modelValue: installPath.value,
                  "onUpdate:modelValue": [
                    _cache[0] || (_cache[0] = ($event) => installPath.value = $event),
                    validatePath
                  ],
                  class: normalizeClass(["w-full", { "p-invalid": pathError.value }])
                }, null, 8, ["modelValue", "class"]),
                withDirectives(createVNode(unref(script$5), { class: "pi pi-info-circle" }, null, 512), [
                  [_directive_tooltip, _ctx.$t("install.installLocationTooltip")]
                ])
              ]),
              _: 1
            }),
            createVNode(unref(script$7), {
              icon: "pi pi-folder",
              onClick: browsePath,
              class: "w-12"
            })
          ]),
          pathError.value ? (openBlock(), createBlock(unref(script$8), {
            key: 0,
            severity: "error",
            class: "whitespace-pre-line"
          }, {
            default: withCtx(() => [
              createTextVNode(toDisplayString(pathError.value), 1)
            ]),
            _: 1
          })) : createCommentVNode("", true),
          pathExists.value ? (openBlock(), createBlock(unref(script$8), {
            key: 1,
            severity: "warn"
          }, {
            default: withCtx(() => [
              createTextVNode(toDisplayString(_ctx.$t("install.pathExists")), 1)
            ]),
            _: 1
          })) : createCommentVNode("", true)
        ]),
        createBaseVNode("div", _hoisted_6$1, [
          createBaseVNode("h3", _hoisted_7$1, toDisplayString(_ctx.$t("install.systemLocations")), 1),
          createBaseVNode("div", _hoisted_8$1, [
            createBaseVNode("div", _hoisted_9$1, [
              _cache[1] || (_cache[1] = createBaseVNode("i", { class: "pi pi-folder text-neutral-400" }, null, -1)),
              _cache[2] || (_cache[2] = createBaseVNode("span", { class: "text-neutral-400" }, "App Data:", -1)),
              createBaseVNode("span", _hoisted_10$1, toDisplayString(appData.value), 1),
              withDirectives(createBaseVNode("span", _hoisted_11$1, null, 512), [
                [_directive_tooltip, _ctx.$t("install.appDataLocationTooltip")]
              ])
            ]),
            createBaseVNode("div", _hoisted_12$1, [
              _cache[3] || (_cache[3] = createBaseVNode("i", { class: "pi pi-desktop text-neutral-400" }, null, -1)),
              _cache[4] || (_cache[4] = createBaseVNode("span", { class: "text-neutral-400" }, "App Path:", -1)),
              createBaseVNode("span", _hoisted_13, toDisplayString(appPath.value), 1),
              withDirectives(createBaseVNode("span", _hoisted_14, null, 512), [
                [_directive_tooltip, _ctx.$t("install.appPathLocationTooltip")]
              ])
            ])
          ])
        ])
      ]);
    };
  }
});
const _hoisted_1$1 = { class: "flex flex-col gap-6 w-[600px]" };
const _hoisted_2$1 = { class: "flex flex-col gap-4" };
const _hoisted_3$1 = { class: "text-2xl font-semibold text-neutral-100" };
const _hoisted_4$1 = { class: "text-neutral-400 my-0" };
const _hoisted_5 = { class: "flex gap-2" };
const _hoisted_6 = {
  key: 0,
  class: "flex flex-col gap-4 bg-neutral-800 p-4 rounded-lg"
};
const _hoisted_7 = { class: "text-lg mt-0 font-medium text-neutral-100" };
const _hoisted_8 = { class: "flex flex-col gap-3" };
const _hoisted_9 = ["onClick"];
const _hoisted_10 = ["for"];
const _hoisted_11 = { class: "text-sm text-neutral-400 my-1" };
const _hoisted_12 = {
  key: 1,
  class: "text-neutral-400 italic"
};
const _sfc_main$1 = /* @__PURE__ */ defineComponent({
  __name: "MigrationPicker",
  props: {
    "sourcePath": { required: false },
    "sourcePathModifiers": {},
    "migrationItemIds": {
      required: false
    },
    "migrationItemIdsModifiers": {}
  },
  emits: ["update:sourcePath", "update:migrationItemIds"],
  setup(__props) {
    const { t } = useI18n();
    const electron = electronAPI();
    const sourcePath = useModel(__props, "sourcePath");
    const migrationItemIds = useModel(__props, "migrationItemIds");
    const migrationItems = ref(
      MigrationItems.map((item) => ({
        ...item,
        selected: true
      }))
    );
    const pathError = ref("");
    const isValidSource = computed(
      () => sourcePath.value !== "" && pathError.value === ""
    );
    const validateSource = /* @__PURE__ */ __name(async (sourcePath2) => {
      if (!sourcePath2) {
        pathError.value = "";
        return;
      }
      try {
        pathError.value = "";
        const validation = await electron.validateComfyUISource(sourcePath2);
        if (!validation.isValid) pathError.value = validation.error;
      } catch (error) {
        console.error(error);
        pathError.value = t("install.pathValidationFailed");
      }
    }, "validateSource");
    const browsePath = /* @__PURE__ */ __name(async () => {
      try {
        const result = await electron.showDirectoryPicker();
        if (result) {
          sourcePath.value = result;
          await validateSource(result);
        }
      } catch (error) {
        console.error(error);
        pathError.value = t("install.failedToSelectDirectory");
      }
    }, "browsePath");
    watchEffect(() => {
      migrationItemIds.value = migrationItems.value.filter((item) => item.selected).map((item) => item.id);
    });
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", _hoisted_1$1, [
        createBaseVNode("div", _hoisted_2$1, [
          createBaseVNode("h2", _hoisted_3$1, toDisplayString(_ctx.$t("install.migrateFromExistingInstallation")), 1),
          createBaseVNode("p", _hoisted_4$1, toDisplayString(_ctx.$t("install.migrationSourcePathDescription")), 1),
          createBaseVNode("div", _hoisted_5, [
            createVNode(unref(script$4), {
              modelValue: sourcePath.value,
              "onUpdate:modelValue": [
                _cache[0] || (_cache[0] = ($event) => sourcePath.value = $event),
                validateSource
              ],
              placeholder: "Select existing ComfyUI installation (optional)",
              class: normalizeClass(["flex-1", { "p-invalid": pathError.value }])
            }, null, 8, ["modelValue", "class"]),
            createVNode(unref(script$7), {
              icon: "pi pi-folder",
              onClick: browsePath,
              class: "w-12"
            })
          ]),
          pathError.value ? (openBlock(), createBlock(unref(script$8), {
            key: 0,
            severity: "error"
          }, {
            default: withCtx(() => [
              createTextVNode(toDisplayString(pathError.value), 1)
            ]),
            _: 1
          })) : createCommentVNode("", true)
        ]),
        isValidSource.value ? (openBlock(), createElementBlock("div", _hoisted_6, [
          createBaseVNode("h3", _hoisted_7, toDisplayString(_ctx.$t("install.selectItemsToMigrate")), 1),
          createBaseVNode("div", _hoisted_8, [
            (openBlock(true), createElementBlock(Fragment, null, renderList(migrationItems.value, (item) => {
              return openBlock(), createElementBlock("div", {
                key: item.id,
                class: "flex items-center gap-3 p-2 hover:bg-neutral-700 rounded",
                onClick: /* @__PURE__ */ __name(($event) => item.selected = !item.selected, "onClick")
              }, [
                createVNode(unref(script$9), {
                  modelValue: item.selected,
                  "onUpdate:modelValue": /* @__PURE__ */ __name(($event) => item.selected = $event, "onUpdate:modelValue"),
                  inputId: item.id,
                  binary: true,
                  onClick: _cache[1] || (_cache[1] = withModifiers(() => {
                  }, ["stop"]))
                }, null, 8, ["modelValue", "onUpdate:modelValue", "inputId"]),
                createBaseVNode("div", null, [
                  createBaseVNode("label", {
                    for: item.id,
                    class: "text-neutral-200 font-medium"
                  }, toDisplayString(item.label), 9, _hoisted_10),
                  createBaseVNode("p", _hoisted_11, toDisplayString(item.description), 1)
                ])
              ], 8, _hoisted_9);
            }), 128))
          ])
        ])) : (openBlock(), createElementBlock("div", _hoisted_12, toDisplayString(_ctx.$t("install.migrationOptional")), 1))
      ]);
    };
  }
});
const _hoisted_1 = { class: "flex pt-6 justify-end" };
const _hoisted_2 = { class: "flex pt-6 justify-between" };
const _hoisted_3 = { class: "flex pt-6 justify-between" };
const _hoisted_4 = { class: "flex pt-6 justify-between" };
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "InstallView",
  setup(__props) {
    const device = ref(null);
    const installPath = ref("");
    const pathError = ref("");
    const migrationSourcePath = ref("");
    const migrationItemIds = ref([]);
    const autoUpdate = ref(true);
    const allowMetrics = ref(true);
    const highestStep = ref(0);
    const handleStepChange = /* @__PURE__ */ __name((value) => {
      setHighestStep(value);
      electronAPI().Events.trackEvent("install_stepper_change", {
        step: value
      });
    }, "handleStepChange");
    const setHighestStep = /* @__PURE__ */ __name((value) => {
      const int = typeof value === "number" ? value : parseInt(value, 10);
      if (!isNaN(int) && int > highestStep.value) highestStep.value = int;
    }, "setHighestStep");
    const hasError = computed(() => pathError.value !== "");
    const noGpu = computed(() => typeof device.value !== "string");
    const electron = electronAPI();
    const router = useRouter();
    const install = /* @__PURE__ */ __name(() => {
      const options = {
        installPath: installPath.value,
        autoUpdate: autoUpdate.value,
        allowMetrics: allowMetrics.value,
        migrationSourcePath: migrationSourcePath.value,
        migrationItemIds: toRaw(migrationItemIds.value),
        device: device.value
      };
      electron.installComfyUI(options);
      const nextPage = options.device === "unsupported" ? "/manual-configuration" : "/server-start";
      router.push(nextPage);
    }, "install");
    onMounted(async () => {
      if (!electron) return;
      const detectedGpu = await electron.Config.getDetectedGpu();
      if (detectedGpu === "mps" || detectedGpu === "nvidia") {
        device.value = detectedGpu;
      }
      electronAPI().Events.trackEvent("install_stepper_change", {
        step: "0",
        gpu: detectedGpu
      });
    });
    return (_ctx, _cache) => {
      return openBlock(), createBlock(_sfc_main$5, { dark: "" }, {
        default: withCtx(() => [
          createVNode(unref(script$e), {
            class: "h-full p-8 2xl:p-16",
            value: "0",
            "onUpdate:value": handleStepChange
          }, {
            default: withCtx(() => [
              createVNode(unref(script$a), { class: "select-none" }, {
                default: withCtx(() => [
                  createVNode(unref(script$b), { value: "0" }, {
                    default: withCtx(() => [
                      createTextVNode(toDisplayString(_ctx.$t("install.gpu")), 1)
                    ]),
                    _: 1
                  }),
                  createVNode(unref(script$b), {
                    value: "1",
                    disabled: noGpu.value
                  }, {
                    default: withCtx(() => [
                      createTextVNode(toDisplayString(_ctx.$t("install.installLocation")), 1)
                    ]),
                    _: 1
                  }, 8, ["disabled"]),
                  createVNode(unref(script$b), {
                    value: "2",
                    disabled: noGpu.value || hasError.value || highestStep.value < 1
                  }, {
                    default: withCtx(() => [
                      createTextVNode(toDisplayString(_ctx.$t("install.migration")), 1)
                    ]),
                    _: 1
                  }, 8, ["disabled"]),
                  createVNode(unref(script$b), {
                    value: "3",
                    disabled: noGpu.value || hasError.value || highestStep.value < 2
                  }, {
                    default: withCtx(() => [
                      createTextVNode(toDisplayString(_ctx.$t("install.desktopSettings")), 1)
                    ]),
                    _: 1
                  }, 8, ["disabled"])
                ]),
                _: 1
              }),
              createVNode(unref(script$c), null, {
                default: withCtx(() => [
                  createVNode(unref(script$d), { value: "0" }, {
                    default: withCtx(({ activateCallback }) => [
                      createVNode(GpuPicker, {
                        device: device.value,
                        "onUpdate:device": _cache[0] || (_cache[0] = ($event) => device.value = $event)
                      }, null, 8, ["device"]),
                      createBaseVNode("div", _hoisted_1, [
                        createVNode(unref(script$7), {
                          label: _ctx.$t("g.next"),
                          icon: "pi pi-arrow-right",
                          iconPos: "right",
                          onClick: /* @__PURE__ */ __name(($event) => activateCallback("1"), "onClick"),
                          disabled: typeof device.value !== "string"
                        }, null, 8, ["label", "onClick", "disabled"])
                      ])
                    ]),
                    _: 1
                  }),
                  createVNode(unref(script$d), { value: "1" }, {
                    default: withCtx(({ activateCallback }) => [
                      createVNode(_sfc_main$2, {
                        installPath: installPath.value,
                        "onUpdate:installPath": _cache[1] || (_cache[1] = ($event) => installPath.value = $event),
                        pathError: pathError.value,
                        "onUpdate:pathError": _cache[2] || (_cache[2] = ($event) => pathError.value = $event)
                      }, null, 8, ["installPath", "pathError"]),
                      createBaseVNode("div", _hoisted_2, [
                        createVNode(unref(script$7), {
                          label: _ctx.$t("g.back"),
                          severity: "secondary",
                          icon: "pi pi-arrow-left",
                          onClick: /* @__PURE__ */ __name(($event) => activateCallback("0"), "onClick")
                        }, null, 8, ["label", "onClick"]),
                        createVNode(unref(script$7), {
                          label: _ctx.$t("g.next"),
                          icon: "pi pi-arrow-right",
                          iconPos: "right",
                          onClick: /* @__PURE__ */ __name(($event) => activateCallback("2"), "onClick"),
                          disabled: pathError.value !== ""
                        }, null, 8, ["label", "onClick", "disabled"])
                      ])
                    ]),
                    _: 1
                  }),
                  createVNode(unref(script$d), { value: "2" }, {
                    default: withCtx(({ activateCallback }) => [
                      createVNode(_sfc_main$1, {
                        sourcePath: migrationSourcePath.value,
                        "onUpdate:sourcePath": _cache[3] || (_cache[3] = ($event) => migrationSourcePath.value = $event),
                        migrationItemIds: migrationItemIds.value,
                        "onUpdate:migrationItemIds": _cache[4] || (_cache[4] = ($event) => migrationItemIds.value = $event)
                      }, null, 8, ["sourcePath", "migrationItemIds"]),
                      createBaseVNode("div", _hoisted_3, [
                        createVNode(unref(script$7), {
                          label: _ctx.$t("g.back"),
                          severity: "secondary",
                          icon: "pi pi-arrow-left",
                          onClick: /* @__PURE__ */ __name(($event) => activateCallback("1"), "onClick")
                        }, null, 8, ["label", "onClick"]),
                        createVNode(unref(script$7), {
                          label: _ctx.$t("g.next"),
                          icon: "pi pi-arrow-right",
                          iconPos: "right",
                          onClick: /* @__PURE__ */ __name(($event) => activateCallback("3"), "onClick")
                        }, null, 8, ["label", "onClick"])
                      ])
                    ]),
                    _: 1
                  }),
                  createVNode(unref(script$d), { value: "3" }, {
                    default: withCtx(({ activateCallback }) => [
                      createVNode(_sfc_main$4, {
                        autoUpdate: autoUpdate.value,
                        "onUpdate:autoUpdate": _cache[5] || (_cache[5] = ($event) => autoUpdate.value = $event),
                        allowMetrics: allowMetrics.value,
                        "onUpdate:allowMetrics": _cache[6] || (_cache[6] = ($event) => allowMetrics.value = $event)
                      }, null, 8, ["autoUpdate", "allowMetrics"]),
                      createBaseVNode("div", _hoisted_4, [
                        createVNode(unref(script$7), {
                          label: _ctx.$t("g.back"),
                          severity: "secondary",
                          icon: "pi pi-arrow-left",
                          onClick: /* @__PURE__ */ __name(($event) => activateCallback("2"), "onClick")
                        }, null, 8, ["label", "onClick"]),
                        createVNode(unref(script$7), {
                          label: _ctx.$t("g.install"),
                          icon: "pi pi-check",
                          iconPos: "right",
                          disabled: hasError.value,
                          onClick: _cache[7] || (_cache[7] = ($event) => install())
                        }, null, 8, ["label", "disabled"])
                      ])
                    ]),
                    _: 1
                  })
                ]),
                _: 1
              })
            ]),
            _: 1
          })
        ]),
        _: 1
      });
    };
  }
});
const InstallView = /* @__PURE__ */ _export_sfc(_sfc_main, [["__scopeId", "data-v-0a97b0ae"]]);
export {
  InstallView as default
};
//# sourceMappingURL=InstallView-C1fnMZKt.js.map
