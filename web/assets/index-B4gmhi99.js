const o = {
  LOADING_PROGRESS: "loading-progress",
  IS_PACKAGED: "is-packaged",
  RENDERER_READY: "renderer-ready",
  RESTART_APP: "restart-app",
  REINSTALL: "reinstall",
  LOG_MESSAGE: "log-message",
  OPEN_DIALOG: "open-dialog",
  DOWNLOAD_PROGRESS: "download-progress",
  START_DOWNLOAD: "start-download",
  PAUSE_DOWNLOAD: "pause-download",
  RESUME_DOWNLOAD: "resume-download",
  CANCEL_DOWNLOAD: "cancel-download",
  DELETE_MODEL: "delete-model",
  GET_ALL_DOWNLOADS: "get-all-downloads",
  GET_ELECTRON_VERSION: "get-electron-version",
  SEND_ERROR_TO_SENTRY: "send-error-to-sentry",
  GET_BASE_PATH: "get-base-path",
  GET_MODEL_CONFIG_PATH: "get-model-config-path",
  OPEN_PATH: "open-path",
  OPEN_LOGS_PATH: "open-logs-path",
  OPEN_DEV_TOOLS: "open-dev-tools",
  IS_FIRST_TIME_SETUP: "is-first-time-setup",
  GET_SYSTEM_PATHS: "get-system-paths",
  VALIDATE_INSTALL_PATH: "validate-install-path",
  VALIDATE_COMFYUI_SOURCE: "validate-comfyui-source",
  SHOW_DIRECTORY_PICKER: "show-directory-picker",
  INSTALL_COMFYUI: "install-comfyui"
};
var t = /* @__PURE__ */ ((e) => (e.INITIAL_STATE = "initial-state", e.PYTHON_SETUP = "python-setup", e.STARTING_SERVER = "starting-server", e.READY = "ready", e.ERROR = "error", e.ERROR_INSTALL_PATH = "error-install-path", e))(t || {});
const s = {
  "initial-state": "Loading...",
  "python-setup": "Setting up Python Environment...",
  "starting-server": "Starting ComfyUI server...",
  ready: "Finishing...",
  error: "Was not able to start ComfyUI. Please check the logs for more details. You can open it from the Help menu. Please report issues to: https://forum.comfy.org",
  "error-install-path": "Installation path does not exist. Please reset the installation location."
}, a = "electronAPI", n = "https://942cadba58d247c9cab96f45221aa813@o4507954455314432.ingest.us.sentry.io/4508007940685824", r = [
  {
    id: "user_files",
    label: "User Files",
    description: "Settings and user-created workflows"
  },
  {
    id: "models",
    label: "Models",
    description: "Reference model files from existing ComfyUI installations. (No copy)"
  }
  // TODO: Decide whether we want to auto-migrate custom nodes, and install their dependencies.
  // huchenlei: This is a very essential thing for migration experience.
  // {
  //   id: 'custom_nodes',
  //   label: 'Custom Nodes',
  //   description: 'Reference custom node files from existing ComfyUI installations. (No copy)',
  // },
];
export {
  r,
  s,
  t
};
//# sourceMappingURL=index-B4gmhi99.js.map
