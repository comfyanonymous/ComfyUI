const IPC_CHANNELS = {
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
  TERMINAL_WRITE: "execute-terminal-command",
  TERMINAL_RESIZE: "resize-terminal",
  TERMINAL_RESTORE: "restore-terminal",
  TERMINAL_ON_OUTPUT: "terminal-output",
  IS_FIRST_TIME_SETUP: "is-first-time-setup",
  GET_SYSTEM_PATHS: "get-system-paths",
  VALIDATE_INSTALL_PATH: "validate-install-path",
  VALIDATE_COMFYUI_SOURCE: "validate-comfyui-source",
  SHOW_DIRECTORY_PICKER: "show-directory-picker",
  INSTALL_COMFYUI: "install-comfyui"
};
var ProgressStatus = /* @__PURE__ */ ((ProgressStatus2) => {
  ProgressStatus2["INITIAL_STATE"] = "initial-state";
  ProgressStatus2["PYTHON_SETUP"] = "python-setup";
  ProgressStatus2["STARTING_SERVER"] = "starting-server";
  ProgressStatus2["READY"] = "ready";
  ProgressStatus2["ERROR"] = "error";
  return ProgressStatus2;
})(ProgressStatus || {});
const ProgressMessages = {
  [
    "initial-state"
    /* INITIAL_STATE */
  ]: "Loading...",
  [
    "python-setup"
    /* PYTHON_SETUP */
  ]: "Setting up Python Environment...",
  [
    "starting-server"
    /* STARTING_SERVER */
  ]: "Starting ComfyUI server...",
  [
    "ready"
    /* READY */
  ]: "Finishing...",
  [
    "error"
    /* ERROR */
  ]: "Was not able to start ComfyUI. Please check the logs for more details. You can open it from the Help menu. Please report issues to: https://forum.comfy.org"
};
const ELECTRON_BRIDGE_API = "electronAPI";
const SENTRY_URL_ENDPOINT = "https://942cadba58d247c9cab96f45221aa813@o4507954455314432.ingest.us.sentry.io/4508007940685824";
const MigrationItems = [
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
const DEFAULT_SERVER_ARGS = {
  /** The host to use for the ComfyUI server. */
  host: "127.0.0.1",
  /** The port to use for the ComfyUI server. */
  port: 8e3,
  // Extra arguments to pass to the ComfyUI server.
  extraServerArgs: {}
};
var DownloadStatus = /* @__PURE__ */ ((DownloadStatus2) => {
  DownloadStatus2["PENDING"] = "pending";
  DownloadStatus2["IN_PROGRESS"] = "in_progress";
  DownloadStatus2["COMPLETED"] = "completed";
  DownloadStatus2["PAUSED"] = "paused";
  DownloadStatus2["ERROR"] = "error";
  DownloadStatus2["CANCELLED"] = "cancelled";
  return DownloadStatus2;
})(DownloadStatus || {});
export {
  MigrationItems as M,
  ProgressStatus as P
};
//# sourceMappingURL=index-BppSBmxJ.js.map
