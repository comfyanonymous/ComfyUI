var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });
import { a$ as useKeybindingStore, a2 as useCommandStore, a as useSettingStore, cq as KeyComboImpl, cr as KeybindingImpl } from "./index-DjNHn37O.js";
const CORE_KEYBINDINGS = [
  {
    combo: {
      ctrl: true,
      key: "Enter"
    },
    commandId: "Comfy.QueuePrompt"
  },
  {
    combo: {
      ctrl: true,
      shift: true,
      key: "Enter"
    },
    commandId: "Comfy.QueuePromptFront"
  },
  {
    combo: {
      ctrl: true,
      alt: true,
      key: "Enter"
    },
    commandId: "Comfy.Interrupt"
  },
  {
    combo: {
      key: "r"
    },
    commandId: "Comfy.RefreshNodeDefinitions"
  },
  {
    combo: {
      key: "q"
    },
    commandId: "Workspace.ToggleSidebarTab.queue"
  },
  {
    combo: {
      key: "w"
    },
    commandId: "Workspace.ToggleSidebarTab.workflows"
  },
  {
    combo: {
      key: "n"
    },
    commandId: "Workspace.ToggleSidebarTab.node-library"
  },
  {
    combo: {
      key: "m"
    },
    commandId: "Workspace.ToggleSidebarTab.model-library"
  },
  {
    combo: {
      key: "s",
      ctrl: true
    },
    commandId: "Comfy.SaveWorkflow"
  },
  {
    combo: {
      key: "o",
      ctrl: true
    },
    commandId: "Comfy.OpenWorkflow"
  },
  {
    combo: {
      key: "Backspace"
    },
    commandId: "Comfy.ClearWorkflow"
  },
  {
    combo: {
      key: "g",
      ctrl: true
    },
    commandId: "Comfy.Graph.GroupSelectedNodes"
  },
  {
    combo: {
      key: ",",
      ctrl: true
    },
    commandId: "Comfy.ShowSettingsDialog"
  },
  // For '=' both holding shift and not holding shift
  {
    combo: {
      key: "=",
      alt: true
    },
    commandId: "Comfy.Canvas.ZoomIn",
    targetSelector: "#graph-canvas"
  },
  {
    combo: {
      key: "+",
      alt: true,
      shift: true
    },
    commandId: "Comfy.Canvas.ZoomIn",
    targetSelector: "#graph-canvas"
  },
  // For number pad '+'
  {
    combo: {
      key: "+",
      alt: true
    },
    commandId: "Comfy.Canvas.ZoomIn",
    targetSelector: "#graph-canvas"
  },
  {
    combo: {
      key: "-",
      alt: true
    },
    commandId: "Comfy.Canvas.ZoomOut",
    targetSelector: "#graph-canvas"
  },
  {
    combo: {
      key: "."
    },
    commandId: "Comfy.Canvas.FitView",
    targetSelector: "#graph-canvas"
  },
  {
    combo: {
      key: "p"
    },
    commandId: "Comfy.Canvas.ToggleSelected.Pin",
    targetSelector: "#graph-canvas"
  },
  {
    combo: {
      key: "c",
      alt: true
    },
    commandId: "Comfy.Canvas.ToggleSelectedNodes.Collapse",
    targetSelector: "#graph-canvas"
  },
  {
    combo: {
      key: "b",
      ctrl: true
    },
    commandId: "Comfy.Canvas.ToggleSelectedNodes.Bypass",
    targetSelector: "#graph-canvas"
  },
  {
    combo: {
      key: "m",
      ctrl: true
    },
    commandId: "Comfy.Canvas.ToggleSelectedNodes.Mute",
    targetSelector: "#graph-canvas"
  },
  {
    combo: {
      key: "`",
      ctrl: true
    },
    commandId: "Workspace.ToggleBottomPanelTab.logs-terminal"
  },
  {
    combo: {
      key: "f"
    },
    commandId: "Workspace.ToggleFocusMode"
  }
];
const useKeybindingService = /* @__PURE__ */ __name(() => {
  const keybindingStore = useKeybindingStore();
  const commandStore = useCommandStore();
  const settingStore = useSettingStore();
  const keybindHandler = /* @__PURE__ */ __name(async function(event) {
    const keyCombo = KeyComboImpl.fromEvent(event);
    if (keyCombo.isModifier) {
      return;
    }
    const target = event.composedPath()[0];
    if (!keyCombo.hasModifier && (target.tagName === "TEXTAREA" || target.tagName === "INPUT" || target.tagName === "SPAN" && target.classList.contains("property_value"))) {
      return;
    }
    const keybinding = keybindingStore.getKeybinding(keyCombo);
    if (keybinding && keybinding.targetSelector !== "#graph-canvas") {
      event.preventDefault();
      await commandStore.execute(keybinding.commandId);
      return;
    }
    if (event.ctrlKey || event.altKey || event.metaKey) {
      return;
    }
    if (event.key === "Escape") {
      const modals = document.querySelectorAll(".comfy-modal");
      for (const modal of modals) {
        const modalDisplay = window.getComputedStyle(modal).getPropertyValue("display");
        if (modalDisplay !== "none") {
          modal.style.display = "none";
          break;
        }
      }
      for (const d of document.querySelectorAll("dialog")) d.close();
    }
  }, "keybindHandler");
  const registerCoreKeybindings = /* @__PURE__ */ __name(() => {
    for (const keybinding of CORE_KEYBINDINGS) {
      keybindingStore.addDefaultKeybinding(new KeybindingImpl(keybinding));
    }
  }, "registerCoreKeybindings");
  function registerUserKeybindings() {
    const unsetBindings = settingStore.get("Comfy.Keybinding.UnsetBindings");
    for (const keybinding of unsetBindings) {
      keybindingStore.unsetKeybinding(new KeybindingImpl(keybinding));
    }
    const newBindings = settingStore.get("Comfy.Keybinding.NewBindings");
    for (const keybinding of newBindings) {
      keybindingStore.addUserKeybinding(new KeybindingImpl(keybinding));
    }
  }
  __name(registerUserKeybindings, "registerUserKeybindings");
  async function persistUserKeybindings() {
    await settingStore.set(
      "Comfy.Keybinding.NewBindings",
      Object.values(keybindingStore.getUserKeybindings())
    );
    await settingStore.set(
      "Comfy.Keybinding.UnsetBindings",
      Object.values(keybindingStore.getUserUnsetKeybindings())
    );
  }
  __name(persistUserKeybindings, "persistUserKeybindings");
  return {
    keybindHandler,
    registerCoreKeybindings,
    registerUserKeybindings,
    persistUserKeybindings
  };
}, "useKeybindingService");
export {
  useKeybindingService as u
};
//# sourceMappingURL=keybindingService-Bx7YdkXn.js.map
