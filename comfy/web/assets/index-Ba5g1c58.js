var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });
import { bU as getAugmentedNamespace, bV as getDefaultExportFromCjs } from "./index-bi78Y1IN.js";
const __viteBrowserExternal = {};
const __viteBrowserExternal$1 = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  default: __viteBrowserExternal
}, Symbol.toStringTag, { value: "Module" }));
const require$$1 = /* @__PURE__ */ getAugmentedNamespace(__viteBrowserExternal$1);
var define_process_env_default$1 = {};
const fs = require$$1;
const path = require$$1;
const pathFile = path.join(__dirname, "path.txt");
function getElectronPath() {
  let executablePath;
  if (fs.existsSync(pathFile)) {
    executablePath = fs.readFileSync(pathFile, "utf-8");
  }
  if (define_process_env_default$1.ELECTRON_OVERRIDE_DIST_PATH) {
    return path.join(define_process_env_default$1.ELECTRON_OVERRIDE_DIST_PATH, executablePath || "electron");
  }
  if (executablePath) {
    return path.join(__dirname, "dist", executablePath);
  } else {
    throw new Error("Electron failed to install correctly, please delete node_modules/electron and try installing again");
  }
}
__name(getElectronPath, "getElectronPath");
var electron = getElectronPath();
const se = /* @__PURE__ */ getDefaultExportFromCjs(electron);
var define_process_env_default = {};
var re = Object.defineProperty;
var ne = /* @__PURE__ */ __name((s, e, t) => e in s ? re(s, e, { enumerable: true, configurable: true, writable: true, value: t }) : s[e] = t, "ne");
var l = /* @__PURE__ */ __name((s, e, t) => ne(s, typeof e != "symbol" ? e + "" : e, t), "l");
const E = {
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
var oe = /* @__PURE__ */ ((s) => (s.INITIAL_STATE = "initial-state", s.PYTHON_SETUP = "python-setup", s.STARTING_SERVER = "starting-server", s.READY = "ready", s.ERROR = "error", s.ERROR_INSTALL_PATH = "error-install-path", s))(oe || {});
const rr = {
  "initial-state": "Loading...",
  "python-setup": "Setting up Python Environment...",
  "starting-server": "Starting ComfyUI server...",
  ready: "Finishing...",
  error: "Was not able to start ComfyUI. Please check the logs for more details. You can open it from the Help menu. Please report issues to: https://forum.comfy.org",
  "error-install-path": "Installation path does not exist. Please reset the installation location."
}, nr = "electronAPI", sr = "https://942cadba58d247c9cab96f45221aa813@o4507954455314432.ingest.us.sentry.io/4508007940685824", or = [
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
], ie = {}, ae = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  default: ie
}, Symbol.toStringTag, { value: "Module" }));
function ce(s) {
  return s && s.__esModule && Object.prototype.hasOwnProperty.call(s, "default") ? s.default : s;
}
__name(ce, "ce");
function le(s) {
  if (s.__esModule) return s;
  var e = s.default;
  if (typeof e == "function") {
    var t = /* @__PURE__ */ __name(function r() {
      return this instanceof r ? Reflect.construct(e, arguments, this.constructor) : e.apply(this, arguments);
    }, "r");
    t.prototype = e.prototype;
  } else t = {};
  return Object.defineProperty(t, "__esModule", { value: true }), Object.keys(s).forEach(function(r) {
    var n = Object.getOwnPropertyDescriptor(s, r);
    Object.defineProperty(t, r, n.get ? n : {
      enumerable: true,
      get: /* @__PURE__ */ __name(function() {
        return s[r];
      }, "get")
    });
  }), t;
}
__name(le, "le");
const u = /* @__PURE__ */ le(ae), H = u, b = u;
var pe = {
  findAndReadPackageJson: he,
  tryReadJsonAt: P
};
function he() {
  return P(de()) || P(fe()) || P(process.resourcesPath, "app.asar") || P(process.resourcesPath, "app") || P(process.cwd()) || { name: void 0, version: void 0 };
}
__name(he, "he");
function P(...s) {
  if (s[0])
    try {
      const e = b.join(...s), t = ue("package.json", e);
      if (!t)
        return;
      const r = JSON.parse(H.readFileSync(t, "utf8")), n = (r == null ? void 0 : r.productName) || (r == null ? void 0 : r.name);
      return !n || n.toLowerCase() === "electron" ? void 0 : n ? { name: n, version: r == null ? void 0 : r.version } : void 0;
    } catch {
      return;
    }
}
__name(P, "P");
function ue(s, e) {
  let t = e;
  for (; ; ) {
    const r = b.parse(t), n = r.root, o = r.dir;
    if (H.existsSync(b.join(t, s)))
      return b.resolve(b.join(t, s));
    if (t === n)
      return null;
    t = o;
  }
}
__name(ue, "ue");
function fe() {
  const s = process.argv.filter((t) => t.indexOf("--user-data-dir=") === 0);
  return s.length === 0 || typeof s[0] != "string" ? null : s[0].replace("--user-data-dir=", "");
}
__name(fe, "fe");
function de() {
  var s;
  try {
    return (s = require.main) == null ? void 0 : s.filename;
  } catch {
    return;
  }
}
__name(de, "de");
const ge = u, S = u, O = u, me = pe;
let ye = class {
  static {
    __name(this, "ye");
  }
  constructor() {
    l(this, "appName");
    l(this, "appPackageJson");
    l(this, "platform", process.platform);
  }
  getAppLogPath(e = this.getAppName()) {
    return this.platform === "darwin" ? O.join(this.getSystemPathHome(), "Library/Logs", e) : O.join(this.getAppUserDataPath(e), "logs");
  }
  getAppName() {
    var t;
    const e = this.appName || ((t = this.getAppPackageJson()) == null ? void 0 : t.name);
    if (!e)
      throw new Error(
        "electron-log can't determine the app name. It tried these methods:\n1. Use `electron.app.name`\n2. Use productName or name from the nearest package.json`\nYou can also set it through log.transports.file.setAppName()"
      );
    return e;
  }
  /**
   * @private
   * @returns {undefined}
   */
  getAppPackageJson() {
    return typeof this.appPackageJson != "object" && (this.appPackageJson = me.findAndReadPackageJson()), this.appPackageJson;
  }
  getAppUserDataPath(e = this.getAppName()) {
    return e ? O.join(this.getSystemPathAppData(), e) : void 0;
  }
  getAppVersion() {
    var e;
    return (e = this.getAppPackageJson()) == null ? void 0 : e.version;
  }
  getElectronLogPath() {
    return this.getAppLogPath();
  }
  getMacOsVersion() {
    const e = Number(S.release().split(".")[0]);
    return e <= 19 ? `10.${e - 4}` : e - 9;
  }
  /**
   * @protected
   * @returns {string}
   */
  getOsVersion() {
    let e = S.type().replace("_", " "), t = S.release();
    return e === "Darwin" && (e = "macOS", t = this.getMacOsVersion()), `${e} ${t}`;
  }
  /**
   * @return {PathVariables}
   */
  getPathVariables() {
    const e = this.getAppName(), t = this.getAppVersion(), r = this;
    return {
      appData: this.getSystemPathAppData(),
      appName: e,
      appVersion: t,
      get electronDefaultDir() {
        return r.getElectronLogPath();
      },
      home: this.getSystemPathHome(),
      libraryDefaultDir: this.getAppLogPath(e),
      libraryTemplate: this.getAppLogPath("{appName}"),
      temp: this.getSystemPathTemp(),
      userData: this.getAppUserDataPath(e)
    };
  }
  getSystemPathAppData() {
    const e = this.getSystemPathHome();
    switch (this.platform) {
      case "darwin":
        return O.join(e, "Library/Application Support");
      case "win32":
        return define_process_env_default.APPDATA || O.join(e, "AppData/Roaming");
      default:
        return define_process_env_default.XDG_CONFIG_HOME || O.join(e, ".config");
    }
  }
  getSystemPathHome() {
    var e;
    return ((e = S.homedir) == null ? void 0 : e.call(S)) || define_process_env_default.HOME;
  }
  getSystemPathTemp() {
    return S.tmpdir();
  }
  getVersions() {
    return {
      app: `${this.getAppName()} ${this.getAppVersion()}`,
      electron: void 0,
      os: this.getOsVersion()
    };
  }
  isDev() {
    return define_process_env_default.ELECTRON_IS_DEV === "1";
  }
  isElectron() {
    return !!process.versions.electron;
  }
  onAppEvent(e, t) {
  }
  onAppReady(e) {
    e();
  }
  onEveryWebContentsEvent(e, t) {
  }
  /**
   * Listen to async messages sent from opposite process
   * @param {string} channel
   * @param {function} listener
   */
  onIpc(e, t) {
  }
  onIpcInvoke(e, t) {
  }
  /**
   * @param {string} url
   * @param {Function} [logFunction]
   */
  openUrl(e, t = console.error) {
    const n = { darwin: "open", win32: "start", linux: "xdg-open" }[process.platform] || "xdg-open";
    ge.exec(`${n} ${e}`, {}, (o) => {
      o && t(o);
    });
  }
  setAppName(e) {
    this.appName = e;
  }
  setPlatform(e) {
    this.platform = e;
  }
  setPreloadFileForSessions({
    filePath: e,
    // eslint-disable-line no-unused-vars
    includeFutureSession: t = true,
    // eslint-disable-line no-unused-vars
    getSessions: r = /* @__PURE__ */ __name(() => [], "r")
    // eslint-disable-line no-unused-vars
  }) {
  }
  /**
   * Sent a message to opposite process
   * @param {string} channel
   * @param {any} message
   */
  sendIpc(e, t) {
  }
  showErrorBox(e, t) {
  }
};
var ve = ye;
const Ee = u, Se = ve;
let we = class extends Se {
  static {
    __name(this, "we");
  }
  /**
   * @param {object} options
   * @param {typeof Electron} [options.electron]
   */
  constructor({ electron: t } = {}) {
    super();
    l(this, "electron");
    this.electron = t;
  }
  getAppName() {
    var r, n;
    let t;
    try {
      t = this.appName || ((r = this.electron.app) == null ? void 0 : r.name) || ((n = this.electron.app) == null ? void 0 : n.getName());
    } catch {
    }
    return t || super.getAppName();
  }
  getAppUserDataPath(t) {
    return this.getPath("userData") || super.getAppUserDataPath(t);
  }
  getAppVersion() {
    var r;
    let t;
    try {
      t = (r = this.electron.app) == null ? void 0 : r.getVersion();
    } catch {
    }
    return t || super.getAppVersion();
  }
  getElectronLogPath() {
    return this.getPath("logs") || super.getElectronLogPath();
  }
  /**
   * @private
   * @param {any} name
   * @returns {string|undefined}
   */
  getPath(t) {
    var r;
    try {
      return (r = this.electron.app) == null ? void 0 : r.getPath(t);
    } catch {
      return;
    }
  }
  getVersions() {
    return {
      app: `${this.getAppName()} ${this.getAppVersion()}`,
      electron: `Electron ${process.versions.electron}`,
      os: this.getOsVersion()
    };
  }
  getSystemPathAppData() {
    return this.getPath("appData") || super.getSystemPathAppData();
  }
  isDev() {
    var t;
    return ((t = this.electron.app) == null ? void 0 : t.isPackaged) !== void 0 ? !this.electron.app.isPackaged : typeof process.execPath == "string" ? Ee.basename(process.execPath).toLowerCase().startsWith("electron") : super.isDev();
  }
  onAppEvent(t, r) {
    var n;
    return (n = this.electron.app) == null || n.on(t, r), () => {
      var o;
      (o = this.electron.app) == null || o.off(t, r);
    };
  }
  onAppReady(t) {
    var r, n, o;
    (r = this.electron.app) != null && r.isReady() ? t() : (n = this.electron.app) != null && n.once ? (o = this.electron.app) == null || o.once("ready", t) : t();
  }
  onEveryWebContentsEvent(t, r) {
    var o, i, a;
    return (i = (o = this.electron.webContents) == null ? void 0 : o.getAllWebContents()) == null || i.forEach((h) => {
      h.on(t, r);
    }), (a = this.electron.app) == null || a.on("web-contents-created", n), () => {
      var h, c;
      (h = this.electron.webContents) == null || h.getAllWebContents().forEach((p) => {
        p.off(t, r);
      }), (c = this.electron.app) == null || c.off("web-contents-created", n);
    };
    function n(h, c) {
      c.on(t, r);
    }
    __name(n, "n");
  }
  /**
   * Listen to async messages sent from opposite process
   * @param {string} channel
   * @param {function} listener
   */
  onIpc(t, r) {
    var n;
    (n = this.electron.ipcMain) == null || n.on(t, r);
  }
  onIpcInvoke(t, r) {
    var n, o;
    (o = (n = this.electron.ipcMain) == null ? void 0 : n.handle) == null || o.call(n, t, r);
  }
  /**
   * @param {string} url
   * @param {Function} [logFunction]
   */
  openUrl(t, r = console.error) {
    var n;
    (n = this.electron.shell) == null || n.openExternal(t).catch(r);
  }
  setPreloadFileForSessions({
    filePath: t,
    includeFutureSession: r = true,
    getSessions: n = /* @__PURE__ */ __name(() => {
      var o;
      return [(o = this.electron.session) == null ? void 0 : o.defaultSession];
    }, "n")
  }) {
    for (const i of n().filter(Boolean))
      o(i);
    r && this.onAppEvent("session-created", (i) => {
      o(i);
    });
    function o(i) {
      i.setPreloads([...i.getPreloads(), t]);
    }
    __name(o, "o");
  }
  /**
   * Sent a message to opposite process
   * @param {string} channel
   * @param {any} message
   */
  sendIpc(t, r) {
    var n, o;
    (o = (n = this.electron.BrowserWindow) == null ? void 0 : n.getAllWindows()) == null || o.forEach((i) => {
      var a;
      ((a = i.webContents) == null ? void 0 : a.isDestroyed()) === false && i.webContents.send(t, r);
    });
  }
  showErrorBox(t, r) {
    var n;
    (n = this.electron.dialog) == null || n.showErrorBox(t, r);
  }
};
var Oe = we, J = { exports: {} };
(function(s) {
  let e = {};
  try {
    e = require("electron");
  } catch {
  }
  e.ipcRenderer && t(e), s.exports = t;
  function t({ contextBridge: r, ipcRenderer: n }) {
    if (!n)
      return;
    n.on("__ELECTRON_LOG_IPC__", (i, a) => {
      window.postMessage({ cmd: "message", ...a });
    }), n.invoke("__ELECTRON_LOG__", { cmd: "getOptions" }).catch((i) => console.error(new Error(
      `electron-log isn't initialized in the main process. Please call log.initialize() before. ${i.message}`
    )));
    const o = {
      sendToMain(i) {
        try {
          n.send("__ELECTRON_LOG__", i);
        } catch (a) {
          console.error("electronLog.sendToMain ", a, "data:", i), n.send("__ELECTRON_LOG__", {
            cmd: "errorHandler",
            error: { message: a == null ? void 0 : a.message, stack: a == null ? void 0 : a.stack },
            errorName: "sendToMain"
          });
        }
      },
      log(...i) {
        o.sendToMain({ data: i, level: "info" });
      }
    };
    for (const i of ["error", "warn", "info", "verbose", "debug", "silly"])
      o[i] = (...a) => o.sendToMain({
        data: a,
        level: i
      });
    if (r && process.contextIsolated)
      try {
        r.exposeInMainWorld("__electronLog", o);
      } catch {
      }
    typeof window == "object" ? window.__electronLog = o : __electronLog = o;
  }
  __name(t, "t");
})(J);
var Pe = J.exports;
const x = u, Ae = u, C = u, be = Pe;
var Le = {
  initialize({
    externalApi: s,
    getSessions: e,
    includeFutureSession: t,
    logger: r,
    preload: n = true,
    spyRendererConsole: o = false
  }) {
    s.onAppReady(() => {
      try {
        n && De({
          externalApi: s,
          getSessions: e,
          includeFutureSession: t,
          preloadOption: n
        }), o && _e({ externalApi: s, logger: r });
      } catch (i) {
        r.warn(i);
      }
    });
  }
};
function De({
  externalApi: s,
  getSessions: e,
  includeFutureSession: t,
  preloadOption: r
}) {
  let n = typeof r == "string" ? r : void 0;
  try {
    n = C.resolve(
      __dirname,
      "../renderer/electron-log-preload.js"
    );
  } catch {
  }
  if (!n || !x.existsSync(n)) {
    n = C.join(
      s.getAppUserDataPath() || Ae.tmpdir(),
      "electron-log-preload.js"
    );
    const o = `
      try {
        (${be.toString()})(require('electron'));
      } catch(e) {
        console.error(e);
      }
    `;
    x.writeFileSync(n, o, "utf8");
  }
  s.setPreloadFileForSessions({
    filePath: n,
    includeFutureSession: t,
    getSessions: e
  });
}
__name(De, "De");
function _e({ externalApi: s, logger: e }) {
  const t = ["verbose", "info", "warning", "error"];
  s.onEveryWebContentsEvent(
    "console-message",
    (r, n, o) => {
      e.processMessage({
        data: [o],
        level: t[n],
        variables: { processType: "renderer" }
      });
    }
  );
}
__name(_e, "_e");
var Ne = $e;
function $e(s) {
  return Object.defineProperties(e, {
    defaultLabel: { value: "", writable: true },
    labelPadding: { value: true, writable: true },
    maxLabelLength: { value: 0, writable: true },
    labelLength: {
      get() {
        switch (typeof e.labelPadding) {
          case "boolean":
            return e.labelPadding ? e.maxLabelLength : 0;
          case "number":
            return e.labelPadding;
          default:
            return 0;
        }
      }
    }
  });
  function e(t) {
    e.maxLabelLength = Math.max(e.maxLabelLength, t.length);
    const r = {};
    for (const n of [...s.levels, "log"])
      r[n] = (...o) => s.logData(o, { level: n, scope: t });
    return r;
  }
  __name(e, "e");
}
__name($e, "$e");
const Te = Ne;
var v;
let Fe = (v = class {
  static {
    __name(this, "v");
  }
  constructor({
    allowUnknownLevel: e = false,
    dependencies: t = {},
    errorHandler: r,
    eventLogger: n,
    initializeFn: o,
    isDev: i = false,
    levels: a = ["error", "warn", "info", "verbose", "debug", "silly"],
    logId: h,
    transportFactories: c = {},
    variables: p
  } = {}) {
    l(this, "dependencies", {});
    l(this, "errorHandler", null);
    l(this, "eventLogger", null);
    l(this, "functions", {});
    l(this, "hooks", []);
    l(this, "isDev", false);
    l(this, "levels", null);
    l(this, "logId", null);
    l(this, "scope", null);
    l(this, "transports", {});
    l(this, "variables", {});
    this.addLevel = this.addLevel.bind(this), this.create = this.create.bind(this), this.initialize = this.initialize.bind(this), this.logData = this.logData.bind(this), this.processMessage = this.processMessage.bind(this), this.allowUnknownLevel = e, this.dependencies = t, this.initializeFn = o, this.isDev = i, this.levels = a, this.logId = h, this.transportFactories = c, this.variables = p || {}, this.scope = Te(this);
    for (const f of this.levels)
      this.addLevel(f, false);
    this.log = this.info, this.functions.log = this.log, this.errorHandler = r, r == null || r.setOptions({ ...t, logFn: this.error }), this.eventLogger = n, n == null || n.setOptions({ ...t, logger: this });
    for (const [f, g] of Object.entries(c))
      this.transports[f] = g(this, t);
    v.instances[h] = this;
  }
  static getInstance({ logId: e }) {
    return this.instances[e] || this.instances.default;
  }
  addLevel(e, t = this.levels.length) {
    t !== false && this.levels.splice(t, 0, e), this[e] = (...r) => this.logData(r, { level: e }), this.functions[e] = this[e];
  }
  catchErrors(e) {
    return this.processMessage(
      {
        data: ["log.catchErrors is deprecated. Use log.errorHandler instead"],
        level: "warn"
      },
      { transports: ["console"] }
    ), this.errorHandler.startCatching(e);
  }
  create(e) {
    return typeof e == "string" && (e = { logId: e }), new v({
      dependencies: this.dependencies,
      errorHandler: this.errorHandler,
      initializeFn: this.initializeFn,
      isDev: this.isDev,
      transportFactories: this.transportFactories,
      variables: { ...this.variables },
      ...e
    });
  }
  compareLevels(e, t, r = this.levels) {
    const n = r.indexOf(e), o = r.indexOf(t);
    return o === -1 || n === -1 ? true : o <= n;
  }
  initialize(e = {}) {
    this.initializeFn({ logger: this, ...this.dependencies, ...e });
  }
  logData(e, t = {}) {
    this.processMessage({ data: e, ...t });
  }
  processMessage(e, { transports: t = this.transports } = {}) {
    if (e.cmd === "errorHandler") {
      this.errorHandler.handle(e.error, {
        errorName: e.errorName,
        processType: "renderer",
        showDialog: !!e.showDialog
      });
      return;
    }
    let r = e.level;
    this.allowUnknownLevel || (r = this.levels.includes(e.level) ? e.level : "info");
    const n = {
      date: /* @__PURE__ */ new Date(),
      ...e,
      level: r,
      variables: {
        ...this.variables,
        ...e.variables
      }
    };
    for (const [o, i] of this.transportEntries(t))
      if (!(typeof i != "function" || i.level === false) && this.compareLevels(i.level, e.level))
        try {
          const a = this.hooks.reduce((h, c) => h && c(h, i, o), n);
          a && i({ ...a, data: [...a.data] });
        } catch (a) {
          this.processInternalErrorFn(a);
        }
  }
  processInternalErrorFn(e) {
  }
  transportEntries(e = this.transports) {
    return (Array.isArray(e) ? e : Object.entries(e)).map((r) => {
      switch (typeof r) {
        case "string":
          return this.transports[r] ? [r, this.transports[r]] : null;
        case "function":
          return [r.name, r];
        default:
          return Array.isArray(r) ? r : null;
      }
    }).filter(Boolean);
  }
}, l(v, "instances", {}), v);
var Re = Fe;
let Ie = class {
  static {
    __name(this, "Ie");
  }
  constructor({
    externalApi: e,
    logFn: t = void 0,
    onError: r = void 0,
    showDialog: n = void 0
  } = {}) {
    l(this, "externalApi");
    l(this, "isActive", false);
    l(this, "logFn");
    l(this, "onError");
    l(this, "showDialog", true);
    this.createIssue = this.createIssue.bind(this), this.handleError = this.handleError.bind(this), this.handleRejection = this.handleRejection.bind(this), this.setOptions({ externalApi: e, logFn: t, onError: r, showDialog: n }), this.startCatching = this.startCatching.bind(this), this.stopCatching = this.stopCatching.bind(this);
  }
  handle(e, {
    logFn: t = this.logFn,
    onError: r = this.onError,
    processType: n = "browser",
    showDialog: o = this.showDialog,
    errorName: i = ""
  } = {}) {
    var a;
    e = xe(e);
    try {
      if (typeof r == "function") {
        const h = ((a = this.externalApi) == null ? void 0 : a.getVersions()) || {}, c = this.createIssue;
        if (r({
          createIssue: c,
          error: e,
          errorName: i,
          processType: n,
          versions: h
        }) === false)
          return;
      }
      i ? t(i, e) : t(e), o && !i.includes("rejection") && this.externalApi && this.externalApi.showErrorBox(
        `A JavaScript error occurred in the ${n} process`,
        e.stack
      );
    } catch {
      console.error(e);
    }
  }
  setOptions({ externalApi: e, logFn: t, onError: r, showDialog: n }) {
    typeof e == "object" && (this.externalApi = e), typeof t == "function" && (this.logFn = t), typeof r == "function" && (this.onError = r), typeof n == "boolean" && (this.showDialog = n);
  }
  startCatching({ onError: e, showDialog: t } = {}) {
    this.isActive || (this.isActive = true, this.setOptions({ onError: e, showDialog: t }), process.on("uncaughtException", this.handleError), process.on("unhandledRejection", this.handleRejection));
  }
  stopCatching() {
    this.isActive = false, process.removeListener("uncaughtException", this.handleError), process.removeListener("unhandledRejection", this.handleRejection);
  }
  createIssue(e, t) {
    var r;
    (r = this.externalApi) == null || r.openUrl(
      `${e}?${new URLSearchParams(t).toString()}`
    );
  }
  handleError(e) {
    this.handle(e, { errorName: "Unhandled" });
  }
  handleRejection(e) {
    const t = e instanceof Error ? e : new Error(JSON.stringify(e));
    this.handle(t, { errorName: "Unhandled rejection" });
  }
};
function xe(s) {
  if (s instanceof Error)
    return s;
  if (s && typeof s == "object") {
    if (s.message)
      return Object.assign(new Error(s.message), s);
    try {
      return new Error(JSON.stringify(s));
    } catch (e) {
      return new Error(`Couldn't normalize error ${String(s)}: ${e}`);
    }
  }
  return new Error(`Can't normalize error ${String(s)}`);
}
__name(xe, "xe");
var Ce = Ie;
let je = class {
  static {
    __name(this, "je");
  }
  constructor(e = {}) {
    l(this, "disposers", []);
    l(this, "format", "{eventSource}#{eventName}:");
    l(this, "formatters", {
      app: {
        "certificate-error": /* @__PURE__ */ __name(({ args: e2 }) => this.arrayToObject(e2.slice(1, 4), [
          "url",
          "error",
          "certificate"
        ]), "certificate-error"),
        "child-process-gone": /* @__PURE__ */ __name(({ args: e2 }) => e2.length === 1 ? e2[0] : e2, "child-process-gone"),
        "render-process-gone": /* @__PURE__ */ __name(({ args: [e2, t] }) => t && typeof t == "object" ? { ...t, ...this.getWebContentsDetails(e2) } : [], "render-process-gone")
      },
      webContents: {
        "console-message": /* @__PURE__ */ __name(({ args: [e2, t, r, n] }) => {
          if (!(e2 < 3))
            return { message: t, source: `${n}:${r}` };
        }, "console-message"),
        "did-fail-load": /* @__PURE__ */ __name(({ args: e2 }) => this.arrayToObject(e2, [
          "errorCode",
          "errorDescription",
          "validatedURL",
          "isMainFrame",
          "frameProcessId",
          "frameRoutingId"
        ]), "did-fail-load"),
        "did-fail-provisional-load": /* @__PURE__ */ __name(({ args: e2 }) => this.arrayToObject(e2, [
          "errorCode",
          "errorDescription",
          "validatedURL",
          "isMainFrame",
          "frameProcessId",
          "frameRoutingId"
        ]), "did-fail-provisional-load"),
        "plugin-crashed": /* @__PURE__ */ __name(({ args: e2 }) => this.arrayToObject(e2, ["name", "version"]), "plugin-crashed"),
        "preload-error": /* @__PURE__ */ __name(({ args: e2 }) => this.arrayToObject(e2, ["preloadPath", "error"]), "preload-error")
      }
    });
    l(this, "events", {
      app: {
        "certificate-error": true,
        "child-process-gone": true,
        "render-process-gone": true
      },
      webContents: {
        // 'console-message': true,
        "did-fail-load": true,
        "did-fail-provisional-load": true,
        "plugin-crashed": true,
        "preload-error": true,
        unresponsive: true
      }
    });
    l(this, "externalApi");
    l(this, "level", "error");
    l(this, "scope", "");
    this.setOptions(e);
  }
  setOptions({
    events: e,
    externalApi: t,
    level: r,
    logger: n,
    format: o,
    formatters: i,
    scope: a
  }) {
    typeof e == "object" && (this.events = e), typeof t == "object" && (this.externalApi = t), typeof r == "string" && (this.level = r), typeof n == "object" && (this.logger = n), (typeof o == "string" || typeof o == "function") && (this.format = o), typeof i == "object" && (this.formatters = i), typeof a == "string" && (this.scope = a);
  }
  startLogging(e = {}) {
    this.setOptions(e), this.disposeListeners();
    for (const t of this.getEventNames(this.events.app))
      this.disposers.push(
        this.externalApi.onAppEvent(t, (...r) => {
          this.handleEvent({ eventSource: "app", eventName: t, handlerArgs: r });
        })
      );
    for (const t of this.getEventNames(this.events.webContents))
      this.disposers.push(
        this.externalApi.onEveryWebContentsEvent(
          t,
          (...r) => {
            this.handleEvent(
              { eventSource: "webContents", eventName: t, handlerArgs: r }
            );
          }
        )
      );
  }
  stopLogging() {
    this.disposeListeners();
  }
  arrayToObject(e, t) {
    const r = {};
    return t.forEach((n, o) => {
      r[n] = e[o];
    }), e.length > t.length && (r.unknownArgs = e.slice(t.length)), r;
  }
  disposeListeners() {
    this.disposers.forEach((e) => e()), this.disposers = [];
  }
  formatEventLog({ eventName: e, eventSource: t, handlerArgs: r }) {
    var p;
    const [n, ...o] = r;
    if (typeof this.format == "function")
      return this.format({ args: o, event: n, eventName: e, eventSource: t });
    const i = (p = this.formatters[t]) == null ? void 0 : p[e];
    let a = o;
    if (typeof i == "function" && (a = i({ args: o, event: n, eventName: e, eventSource: t })), !a)
      return;
    const h = {};
    return Array.isArray(a) ? h.args = a : typeof a == "object" && Object.assign(h, a), t === "webContents" && Object.assign(h, this.getWebContentsDetails(n == null ? void 0 : n.sender)), [this.format.replace("{eventSource}", t === "app" ? "App" : "WebContents").replace("{eventName}", e), h];
  }
  getEventNames(e) {
    return !e || typeof e != "object" ? [] : Object.entries(e).filter(([t, r]) => r).map(([t]) => t);
  }
  getWebContentsDetails(e) {
    if (!(e != null && e.loadURL))
      return {};
    try {
      return {
        webContents: {
          id: e.id,
          url: e.getURL()
        }
      };
    } catch {
      return {};
    }
  }
  handleEvent({ eventName: e, eventSource: t, handlerArgs: r }) {
    var o;
    const n = this.formatEventLog({ eventName: e, eventSource: t, handlerArgs: r });
    if (n) {
      const i = this.scope ? this.logger.scope(this.scope) : this.logger;
      (o = i == null ? void 0 : i[this.level]) == null || o.call(i, ...n);
    }
  }
};
var Me = je, L = { transform: We };
function We({
  logger: s,
  message: e,
  transport: t,
  initialData: r = (e == null ? void 0 : e.data) || [],
  transforms: n = t == null ? void 0 : t.transforms
}) {
  return n.reduce((o, i) => typeof i == "function" ? i({ data: o, logger: s, message: e, transport: t }) : o, r);
}
__name(We, "We");
const { transform: ze } = L;
var Y = {
  concatFirstStringElements: Ue,
  formatScope: j,
  formatText: W,
  formatVariables: M,
  timeZoneFromOffset: q,
  format({ message: s, logger: e, transport: t, data: r = s == null ? void 0 : s.data }) {
    switch (typeof t.format) {
      case "string":
        return ze({
          message: s,
          logger: e,
          transforms: [M, j, W],
          transport: t,
          initialData: [t.format, ...r]
        });
      case "function":
        return t.format({
          data: r,
          level: (s == null ? void 0 : s.level) || "info",
          logger: e,
          message: s,
          transport: t
        });
      default:
        return r;
    }
  }
};
function Ue({ data: s }) {
  return typeof s[0] != "string" || typeof s[1] != "string" || s[0].match(/%[1cdfiOos]/) ? s : [`${s[0]} ${s[1]}`, ...s.slice(2)];
}
__name(Ue, "Ue");
function q(s) {
  const e = Math.abs(s), t = s >= 0 ? "-" : "+", r = Math.floor(e / 60).toString().padStart(2, "0"), n = (e % 60).toString().padStart(2, "0");
  return `${t}${r}:${n}`;
}
__name(q, "q");
function j({ data: s, logger: e, message: t }) {
  const { defaultLabel: r, labelLength: n } = (e == null ? void 0 : e.scope) || {}, o = s[0];
  let i = t.scope;
  i || (i = r);
  let a;
  return i === "" ? a = n > 0 ? "".padEnd(n + 3) : "" : typeof i == "string" ? a = ` (${i})`.padEnd(n + 3) : a = "", s[0] = o.replace("{scope}", a), s;
}
__name(j, "j");
function M({ data: s, message: e }) {
  let t = s[0];
  if (typeof t != "string")
    return s;
  t = t.replace("{level}]", `${e.level}]`.padEnd(6, " "));
  const r = e.date || /* @__PURE__ */ new Date();
  return s[0] = t.replace(/\{(\w+)}/g, (n, o) => {
    var i;
    switch (o) {
      case "level":
        return e.level || "info";
      case "logId":
        return e.logId;
      case "y":
        return r.getFullYear().toString(10);
      case "m":
        return (r.getMonth() + 1).toString(10).padStart(2, "0");
      case "d":
        return r.getDate().toString(10).padStart(2, "0");
      case "h":
        return r.getHours().toString(10).padStart(2, "0");
      case "i":
        return r.getMinutes().toString(10).padStart(2, "0");
      case "s":
        return r.getSeconds().toString(10).padStart(2, "0");
      case "ms":
        return r.getMilliseconds().toString(10).padStart(3, "0");
      case "z":
        return q(r.getTimezoneOffset());
      case "iso":
        return r.toISOString();
      default:
        return ((i = e.variables) == null ? void 0 : i[o]) || n;
    }
  }).trim(), s;
}
__name(M, "M");
function W({ data: s }) {
  const e = s[0];
  if (typeof e != "string")
    return s;
  if (e.lastIndexOf("{text}") === e.length - 6)
    return s[0] = e.replace(/\s?{text}/, ""), s[0] === "" && s.shift(), s;
  const r = e.split("{text}");
  let n = [];
  return r[0] !== "" && n.push(r[0]), n = n.concat(s.slice(1)), r[1] !== "" && n.push(r[1]), n;
}
__name(W, "W");
var Q = { exports: {} };
(function(s) {
  const e = u;
  s.exports = {
    serialize: r,
    maxDepth({ data: n, transport: o, depth: i = (o == null ? void 0 : o.depth) ?? 6 }) {
      if (!n)
        return n;
      if (i < 1)
        return Array.isArray(n) ? "[array]" : typeof n == "object" && n ? "[object]" : n;
      if (Array.isArray(n))
        return n.map((h) => s.exports.maxDepth({
          data: h,
          depth: i - 1
        }));
      if (typeof n != "object" || n && typeof n.toISOString == "function")
        return n;
      if (n === null)
        return null;
      if (n instanceof Error)
        return n;
      const a = {};
      for (const h in n)
        Object.prototype.hasOwnProperty.call(n, h) && (a[h] = s.exports.maxDepth({
          data: n[h],
          depth: i - 1
        }));
      return a;
    },
    toJSON({ data: n }) {
      return JSON.parse(JSON.stringify(n, t()));
    },
    toString({ data: n, transport: o }) {
      const i = (o == null ? void 0 : o.inspectOptions) || {}, a = n.map((h) => {
        if (h !== void 0)
          try {
            const c = JSON.stringify(h, t(), "  ");
            return c === void 0 ? void 0 : JSON.parse(c);
          } catch {
            return h;
          }
      });
      return e.formatWithOptions(i, ...a);
    }
  };
  function t(n = {}) {
    const o = /* @__PURE__ */ new WeakSet();
    return function(i, a) {
      if (typeof a == "object" && a !== null) {
        if (o.has(a))
          return;
        o.add(a);
      }
      return r(i, a, n);
    };
  }
  __name(t, "t");
  function r(n, o, i = {}) {
    const a = (i == null ? void 0 : i.serializeMapAndSet) !== false;
    return o instanceof Error ? o.stack : o && (typeof o == "function" ? `[function] ${o.toString()}` : o instanceof Date ? o.toISOString() : a && o instanceof Map && Object.fromEntries ? Object.fromEntries(o) : a && o instanceof Set && Array.from ? Array.from(o) : o);
  }
  __name(r, "r");
})(Q);
var _ = Q.exports, T = {
  transformStyles: $,
  applyAnsiStyles({ data: s }) {
    return $(s, ke, Ve);
  },
  removeStyles({ data: s }) {
    return $(s, () => "");
  }
};
const K = {
  unset: "\x1B[0m",
  black: "\x1B[30m",
  red: "\x1B[31m",
  green: "\x1B[32m",
  yellow: "\x1B[33m",
  blue: "\x1B[34m",
  magenta: "\x1B[35m",
  cyan: "\x1B[36m",
  white: "\x1B[37m"
};
function ke(s) {
  const e = s.replace(/color:\s*(\w+).*/, "$1").toLowerCase();
  return K[e] || "";
}
__name(ke, "ke");
function Ve(s) {
  return s + K.unset;
}
__name(Ve, "Ve");
function $(s, e, t) {
  const r = {};
  return s.reduce((n, o, i, a) => {
    if (r[i])
      return n;
    if (typeof o == "string") {
      let h = i, c = false;
      o = o.replace(/%[1cdfiOos]/g, (p) => {
        if (h += 1, p !== "%c")
          return p;
        const f = a[h];
        return typeof f == "string" ? (r[h] = true, c = true, e(f, o)) : p;
      }), c && t && (o = t(o));
    }
    return n.push(o), n;
  }, []);
}
__name($, "$");
const { concatFirstStringElements: Be, format: Ge } = Y, { maxDepth: He, toJSON: Je } = _, { applyAnsiStyles: Ye, removeStyles: qe } = T, { transform: Qe } = L, z = {
  error: console.error,
  warn: console.warn,
  info: console.info,
  verbose: console.info,
  debug: console.debug,
  silly: console.debug,
  log: console.log
};
var Ke = X;
const Xe = process.platform === "win32" ? ">" : "›", F = `%c{h}:{i}:{s}.{ms}{scope}%c ${Xe} {text}`;
Object.assign(X, {
  DEFAULT_FORMAT: F
});
function X(s) {
  return Object.assign(e, {
    format: F,
    level: "silly",
    transforms: [
      Ze,
      Ge,
      tt,
      Be,
      He,
      Je
    ],
    useStyles: define_process_env_default.FORCE_STYLES,
    writeFn({ message: t }) {
      (z[t.level] || z.info)(...t.data);
    }
  });
  function e(t) {
    const r = Qe({ logger: s, message: t, transport: e });
    e.writeFn({
      message: { ...t, data: r }
    });
  }
  __name(e, "e");
}
__name(X, "X");
function Ze({ data: s, message: e, transport: t }) {
  return t.format !== F ? s : [`color:${rt(e.level)}`, "color:unset", ...s];
}
__name(Ze, "Ze");
function et(s, e) {
  if (typeof s == "boolean")
    return s;
  const r = e === "error" || e === "warn" ? process.stderr : process.stdout;
  return r && r.isTTY;
}
__name(et, "et");
function tt(s) {
  const { message: e, transport: t } = s;
  return (et(t.useStyles, e.level) ? Ye : qe)(s);
}
__name(tt, "tt");
function rt(s) {
  const e = { error: "red", warn: "yellow", info: "cyan", default: "unset" };
  return e[s] || e.default;
}
__name(rt, "rt");
const nt = u, y = u, U = u;
let st = class extends nt {
  static {
    __name(this, "st");
  }
  constructor({
    path: t,
    writeOptions: r = { encoding: "utf8", flag: "a", mode: 438 },
    writeAsync: n = false
  }) {
    super();
    l(this, "asyncWriteQueue", []);
    l(this, "bytesWritten", 0);
    l(this, "hasActiveAsyncWriting", false);
    l(this, "path", null);
    l(this, "initialSize");
    l(this, "writeOptions", null);
    l(this, "writeAsync", false);
    this.path = t, this.writeOptions = r, this.writeAsync = n;
  }
  get size() {
    return this.getSize();
  }
  clear() {
    try {
      return y.writeFileSync(this.path, "", {
        mode: this.writeOptions.mode,
        flag: "w"
      }), this.reset(), true;
    } catch (t) {
      return t.code === "ENOENT" ? true : (this.emit("error", t, this), false);
    }
  }
  crop(t) {
    try {
      const r = ot(this.path, t || 4096);
      this.clear(), this.writeLine(`[log cropped]${U.EOL}${r}`);
    } catch (r) {
      this.emit(
        "error",
        new Error(`Couldn't crop file ${this.path}. ${r.message}`),
        this
      );
    }
  }
  getSize() {
    if (this.initialSize === void 0)
      try {
        const t = y.statSync(this.path);
        this.initialSize = t.size;
      } catch {
        this.initialSize = 0;
      }
    return this.initialSize + this.bytesWritten;
  }
  increaseBytesWrittenCounter(t) {
    this.bytesWritten += Buffer.byteLength(t, this.writeOptions.encoding);
  }
  isNull() {
    return false;
  }
  nextAsyncWrite() {
    const t = this;
    if (this.hasActiveAsyncWriting || this.asyncWriteQueue.length === 0)
      return;
    const r = this.asyncWriteQueue.join("");
    this.asyncWriteQueue = [], this.hasActiveAsyncWriting = true, y.writeFile(this.path, r, this.writeOptions, (n) => {
      t.hasActiveAsyncWriting = false, n ? t.emit(
        "error",
        new Error(`Couldn't write to ${t.path}. ${n.message}`),
        this
      ) : t.increaseBytesWrittenCounter(r), t.nextAsyncWrite();
    });
  }
  reset() {
    this.initialSize = void 0, this.bytesWritten = 0;
  }
  toString() {
    return this.path;
  }
  writeLine(t) {
    if (t += U.EOL, this.writeAsync) {
      this.asyncWriteQueue.push(t), this.nextAsyncWrite();
      return;
    }
    try {
      y.writeFileSync(this.path, t, this.writeOptions), this.increaseBytesWrittenCounter(t);
    } catch (r) {
      this.emit(
        "error",
        new Error(`Couldn't write to ${this.path}. ${r.message}`),
        this
      );
    }
  }
};
var Z = st;
function ot(s, e) {
  const t = Buffer.alloc(e), r = y.statSync(s), n = Math.min(r.size, e), o = Math.max(0, r.size - e), i = y.openSync(s, "r"), a = y.readSync(i, t, 0, n, o);
  return y.closeSync(i), t.toString("utf8", 0, a);
}
__name(ot, "ot");
const it = Z;
let at = class extends it {
  static {
    __name(this, "at");
  }
  clear() {
  }
  crop() {
  }
  getSize() {
    return 0;
  }
  isNull() {
    return true;
  }
  writeLine() {
  }
};
var ct = at;
const lt = u, k = u, V = u, pt = Z, ht = ct;
let ut = class extends lt {
  static {
    __name(this, "ut");
  }
  constructor() {
    super();
    l(this, "store", {});
    this.emitError = this.emitError.bind(this);
  }
  /**
   * Provide a File object corresponding to the filePath
   * @param {string} filePath
   * @param {WriteOptions} [writeOptions]
   * @param {boolean} [writeAsync]
   * @return {File}
   */
  provide({ filePath: t, writeOptions: r = {}, writeAsync: n = false }) {
    let o;
    try {
      if (t = V.resolve(t), this.store[t])
        return this.store[t];
      o = this.createFile({ filePath: t, writeOptions: r, writeAsync: n });
    } catch (i) {
      o = new ht({ path: t }), this.emitError(i, o);
    }
    return o.on("error", this.emitError), this.store[t] = o, o;
  }
  /**
   * @param {string} filePath
   * @param {WriteOptions} writeOptions
   * @param {boolean} async
   * @return {File}
   * @private
   */
  createFile({ filePath: t, writeOptions: r, writeAsync: n }) {
    return this.testFileWriting({ filePath: t, writeOptions: r }), new pt({ path: t, writeOptions: r, writeAsync: n });
  }
  /**
   * @param {Error} error
   * @param {File} file
   * @private
   */
  emitError(t, r) {
    this.emit("error", t, r);
  }
  /**
   * @param {string} filePath
   * @param {WriteOptions} writeOptions
   * @private
   */
  testFileWriting({ filePath: t, writeOptions: r }) {
    k.mkdirSync(V.dirname(t), { recursive: true }), k.writeFileSync(t, "", { flag: "a", mode: r.mode });
  }
};
var ft = ut;
const D = u, dt = u, A = u, gt = ft, { transform: mt } = L, { removeStyles: yt } = T, {
  format: vt,
  concatFirstStringElements: Et
} = Y, { toString: St } = _;
var wt = Pt;
const Ot = new gt();
function Pt(s, { registry: e = Ot, externalApi: t } = {}) {
  let r;
  return e.listenerCount("error") < 1 && e.on("error", (c, p) => {
    i(`Can't write to ${p}`, c);
  }), Object.assign(n, {
    fileName: At(s.variables.processType),
    format: "[{y}-{m}-{d} {h}:{i}:{s}.{ms}] [{level}]{scope} {text}",
    getFile: a,
    inspectOptions: { depth: 5 },
    level: "silly",
    maxSize: 1024 ** 2,
    readAllLogs: h,
    sync: true,
    transforms: [yt, vt, Et, St],
    writeOptions: { flag: "a", mode: 438, encoding: "utf8" },
    archiveLogFn(c) {
      const p = c.toString(), f = A.parse(p);
      try {
        D.renameSync(p, A.join(f.dir, `${f.name}.old${f.ext}`));
      } catch (g) {
        i("Could not rotate log", g);
        const te = Math.round(n.maxSize / 4);
        c.crop(Math.min(te, 256 * 1024));
      }
    },
    resolvePathFn(c) {
      return A.join(c.libraryDefaultDir, c.fileName);
    },
    setAppName(c) {
      s.dependencies.externalApi.setAppName(c);
    }
  });
  function n(c) {
    const p = a(c);
    n.maxSize > 0 && p.size > n.maxSize && (n.archiveLogFn(p), p.reset());
    const g = mt({ logger: s, message: c, transport: n });
    p.writeLine(g);
  }
  __name(n, "n");
  function o() {
    r || (r = Object.create(
      Object.prototype,
      {
        ...Object.getOwnPropertyDescriptors(
          t.getPathVariables()
        ),
        fileName: {
          get() {
            return n.fileName;
          },
          enumerable: true
        }
      }
    ), typeof n.archiveLog == "function" && (n.archiveLogFn = n.archiveLog, i("archiveLog is deprecated. Use archiveLogFn instead")), typeof n.resolvePath == "function" && (n.resolvePathFn = n.resolvePath, i("resolvePath is deprecated. Use resolvePathFn instead")));
  }
  __name(o, "o");
  function i(c, p = null, f = "error") {
    const g = [`electron-log.transports.file: ${c}`];
    p && g.push(p), s.transports.console({ data: g, date: /* @__PURE__ */ new Date(), level: f });
  }
  __name(i, "i");
  function a(c) {
    o();
    const p = n.resolvePathFn(r, c);
    return e.provide({
      filePath: p,
      writeAsync: !n.sync,
      writeOptions: n.writeOptions
    });
  }
  __name(a, "a");
  function h({ fileFilter: c = /* @__PURE__ */ __name((p) => p.endsWith(".log"), "c") } = {}) {
    o();
    const p = A.dirname(n.resolvePathFn(r));
    return D.existsSync(p) ? D.readdirSync(p).map((f) => A.join(p, f)).filter(c).map((f) => {
      try {
        return {
          path: f,
          lines: D.readFileSync(f, "utf8").split(dt.EOL)
        };
      } catch {
        return null;
      }
    }).filter(Boolean) : [];
  }
  __name(h, "h");
}
__name(Pt, "Pt");
function At(s = process.type) {
  switch (s) {
    case "renderer":
      return "renderer.log";
    case "worker":
      return "worker.log";
    default:
      return "main.log";
  }
}
__name(At, "At");
const { maxDepth: bt, toJSON: Lt } = _, { transform: Dt } = L;
var _t = Nt;
function Nt(s, { externalApi: e }) {
  return Object.assign(t, {
    depth: 3,
    eventId: "__ELECTRON_LOG_IPC__",
    level: s.isDev ? "silly" : false,
    transforms: [Lt, bt]
  }), e != null && e.isElectron() ? t : void 0;
  function t(r) {
    var n;
    ((n = r == null ? void 0 : r.variables) == null ? void 0 : n.processType) !== "renderer" && (e == null || e.sendIpc(t.eventId, {
      ...r,
      data: Dt({ logger: s, message: r, transport: t })
    }));
  }
  __name(t, "t");
}
__name(Nt, "Nt");
const $t = u, Tt = u, { transform: Ft } = L, { removeStyles: Rt } = T, { toJSON: It, maxDepth: xt } = _;
var Ct = jt;
function jt(s) {
  return Object.assign(e, {
    client: { name: "electron-application" },
    depth: 6,
    level: false,
    requestOptions: {},
    transforms: [Rt, It, xt],
    makeBodyFn({ message: t }) {
      return JSON.stringify({
        client: e.client,
        data: t.data,
        date: t.date.getTime(),
        level: t.level,
        scope: t.scope,
        variables: t.variables
      });
    },
    processErrorFn({ error: t }) {
      s.processMessage(
        {
          data: [`electron-log: can't POST ${e.url}`, t],
          level: "warn"
        },
        { transports: ["console", "file"] }
      );
    },
    sendRequestFn({ serverUrl: t, requestOptions: r, body: n }) {
      const i = (t.startsWith("https:") ? Tt : $t).request(t, {
        method: "POST",
        ...r,
        headers: {
          "Content-Type": "application/json",
          "Content-Length": n.length,
          ...r.headers
        }
      });
      return i.write(n), i.end(), i;
    }
  });
  function e(t) {
    if (!e.url)
      return;
    const r = e.makeBodyFn({
      logger: s,
      message: { ...t, data: Ft({ logger: s, message: t, transport: e }) },
      transport: e
    }), n = e.sendRequestFn({
      serverUrl: e.url,
      requestOptions: e.requestOptions,
      body: Buffer.from(r, "utf8")
    });
    n.on("error", (o) => e.processErrorFn({
      error: o,
      logger: s,
      message: t,
      request: n,
      transport: e
    }));
  }
  __name(e, "e");
}
__name(jt, "jt");
const B = Re, Mt = Ce, Wt = Me, zt = Ke, Ut = wt, kt = _t, Vt = Ct;
var Bt = Gt;
function Gt({ dependencies: s, initializeFn: e }) {
  var r;
  const t = new B({
    dependencies: s,
    errorHandler: new Mt(),
    eventLogger: new Wt(),
    initializeFn: e,
    isDev: (r = s.externalApi) == null ? void 0 : r.isDev(),
    logId: "default",
    transportFactories: {
      console: zt,
      file: Ut,
      ipc: kt,
      remote: Vt
    },
    variables: {
      processType: "main"
    }
  });
  return t.default = t, t.Logger = B, t.processInternalErrorFn = (n) => {
    t.transports.console.writeFn({
      message: {
        data: ["Unhandled electron-log error", n],
        level: "error"
      }
    });
  }, t;
}
__name(Gt, "Gt");
const Ht = se, Jt = Oe, { initialize: Yt } = Le, qt = Bt, R = new Jt({ electron: Ht }), N = qt({
  dependencies: { externalApi: R },
  initializeFn: Yt
});
var Qt = N;
R.onIpc("__ELECTRON_LOG__", (s, e) => {
  e.scope && N.Logger.getInstance(e).scope(e.scope);
  const t = new Date(e.date);
  ee({
    ...e,
    date: t.getTime() ? t : /* @__PURE__ */ new Date()
  });
});
R.onIpcInvoke("__ELECTRON_LOG__", (s, { cmd: e = "", logId: t }) => {
  switch (e) {
    case "getOptions":
      return {
        levels: N.Logger.getInstance({ logId: t }).levels,
        logId: t
      };
    default:
      return ee({ data: [`Unknown cmd '${e}'`], level: "error" }), {};
  }
});
function ee(s) {
  var e;
  (e = N.Logger.getInstance(s)) == null || e.processMessage(s);
}
__name(ee, "ee");
const Kt = Qt;
var Xt = Kt;
const d = /* @__PURE__ */ ce(Xt);
var Zt = /* @__PURE__ */ ((s) => (s.PENDING = "pending", s.IN_PROGRESS = "in_progress", s.COMPLETED = "completed", s.PAUSED = "paused", s.ERROR = "error", s.CANCELLED = "cancelled", s))(Zt || {});
const m = class m2 {
  static {
    __name(this, "m2");
  }
  constructor(e, t) {
    l(this, "downloads");
    l(this, "mainWindow");
    l(this, "modelsDirectory");
    this.downloads = /* @__PURE__ */ new Map(), this.mainWindow = e, this.modelsDirectory = t, electron.session.defaultSession.on("will-download", (r, n, o) => {
      const i = n.getURLChain()[0];
      d.info("Will-download event ", i);
      const a = this.downloads.get(i);
      a && (this.reportProgress({
        url: i,
        filename: a.filename,
        savePath: a.savePath,
        progress: 0,
        status: "pending"
        /* PENDING */
      }), n.setSavePath(a.tempPath), a.item = n, d.info(`Setting save path to ${n.getSavePath()}`), n.on("updated", (h, c) => {
        if (c === "interrupted")
          d.info("Download is interrupted but can be resumed");
        else if (c === "progressing") {
          const p = n.getReceivedBytes() / n.getTotalBytes();
          n.isPaused() ? (d.info("Download is paused"), this.reportProgress({
            url: i,
            progress: p,
            filename: a.filename,
            savePath: a.savePath,
            status: "paused"
            /* PAUSED */
          })) : this.reportProgress({
            url: i,
            progress: p,
            filename: a.filename,
            savePath: a.savePath,
            status: "in_progress"
            /* IN_PROGRESS */
          });
        }
      }), n.once("done", (h, c) => {
        if (c === "completed") {
          try {
            (void 0)(a.tempPath, a.savePath), d.info(`Successfully renamed ${a.tempPath} to ${a.savePath}`);
          } catch (p) {
            d.error(`Failed to rename downloaded file: ${p}. Deleting temp file.`), (void 0)(a.tempPath);
          }
          this.reportProgress({
            url: i,
            filename: a.filename,
            savePath: a.savePath,
            progress: 1,
            status: "completed"
            /* COMPLETED */
          }), this.downloads.delete(i);
        } else {
          d.info(`Download failed: ${c}`);
          const p = n.getReceivedBytes() / n.getTotalBytes();
          this.reportProgress({
            url: i,
            filename: a.filename,
            progress: p,
            status: "error",
            savePath: a.savePath
          });
        }
      }));
    });
  }
  startDownload(e, t, r) {
    const n = this.getLocalSavePath(r, t);
    if (!this.isPathInModelsDirectory(n))
      return d.error(`Save path ${n} is not in models directory ${this.modelsDirectory}`), this.reportProgress({
        url: e,
        savePath: t,
        filename: r,
        progress: 0,
        status: "error",
        message: "Save path is not in models directory"
      }), false;
    const o = this.validateSafetensorsFile(e, r);
    if (!o.isValid)
      return d.error(o.error), this.reportProgress({
        url: e,
        savePath: t,
        filename: r,
        progress: 0,
        status: "error",
        message: o.error
      }), false;
    (void 0)(n);
    const i = this.downloads.get(e);
    if (i)
      return d.info("Download already exists"), i.item && i.item.isPaused() && this.resumeDownload(e), true;
    d.info(`Starting download ${e} to ${n}`);
    const a = this.getTempPath(r, t);
    return this.downloads.set(e, { url: e, savePath: n, tempPath: a, filename: r, item: null }), electron.session.defaultSession.downloadURL(e), true;
  }
  cancelDownload(e) {
    const t = this.downloads.get(e);
    t && t.item && (d.info("Cancelling download"), t.item.cancel(), this.downloads.delete(e));
  }
  pauseDownload(e) {
    const t = this.downloads.get(e);
    t && t.item && (d.info("Pausing download"), t.item.pause());
  }
  resumeDownload(e) {
    const t = this.downloads.get(e);
    t && (t.item && t.item.canResume() ? (d.info("Resuming download"), t.item.resume()) : this.startDownload(t.url, t.savePath, t.filename));
  }
  deleteModel(e, t) {
    const r = this.getLocalSavePath(e, t);
    if (!this.isPathInModelsDirectory(r))
      return d.error(`Save path ${r} is not in models directory ${this.modelsDirectory}`), false;
    const n = this.getTempPath(e, t);
    try {
      (void 0)(r);
    } catch (o) {
      d.error(`Failed to delete file ${r}: ${o}`);
    }
    try {
      (void 0)(n);
    } catch (o) {
      d.error(`Failed to delete file ${n}: ${o}`);
    }
    return true;
  }
  getAllDownloads() {
    return Array.from(this.downloads.values()).filter((e) => e.item !== null).map((e) => {
      var t, r, n, o;
      return {
        url: e.url,
        filename: e.filename,
        tempPath: e.tempPath,
        state: this.convertDownloadState((t = e.item) == null ? void 0 : t.getState()),
        receivedBytes: ((r = e.item) == null ? void 0 : r.getReceivedBytes()) || 0,
        totalBytes: ((n = e.item) == null ? void 0 : n.getTotalBytes()) || 0,
        isPaused: ((o = e.item) == null ? void 0 : o.isPaused()) || false
      };
    });
  }
  convertDownloadState(e) {
    switch (e) {
      case "progressing":
        return "in_progress";
      case "completed":
        return "completed";
      case "cancelled":
        return "cancelled";
      case "interrupted":
        return "error";
      default:
        return "error";
    }
  }
  getTempPath(e, t) {
    return (void 0)(this.modelsDirectory, t, `Unconfirmed ${e}.tmp`);
  }
  // Only allow .safetensors files to be downloaded.
  validateSafetensorsFile(e, t) {
    try {
      return !new URL(e).pathname.toLowerCase().endsWith(".safetensors") && !t.toLowerCase().endsWith(".safetensors") ? {
        isValid: false,
        error: "Invalid file type: must be a .safetensors file"
      } : { isValid: true };
    } catch (r) {
      return {
        isValid: false,
        error: `Invalid URL format: ${r}`
      };
    }
  }
  getLocalSavePath(e, t) {
    return (void 0)(this.modelsDirectory, t, e);
  }
  isPathInModelsDirectory(e) {
    const t = (void 0)(e), r = (void 0)(this.modelsDirectory);
    return t.startsWith(r);
  }
  reportProgress({
    url: e,
    progress: t,
    status: r,
    savePath: n,
    filename: o,
    message: i = ""
  }) {
    d.info(`Download progress [${o}]: ${t}, status: ${r}, message: ${i}`), this.mainWindow.send(E.DOWNLOAD_PROGRESS, {
      url: e,
      progress: t,
      status: r,
      message: i,
      savePath: n,
      filename: o
    });
  }
  static getInstance(e, t) {
    return m2.instance || (m2.instance = new m2(e, t), m2.instance.registerIpcHandlers()), m2.instance;
  }
  registerIpcHandlers() {
    electron.ipcMain.handle(
      E.START_DOWNLOAD,
      (e, { url: t, path: r, filename: n }) => this.startDownload(t, r, n)
    ), electron.ipcMain.handle(E.PAUSE_DOWNLOAD, (e, t) => this.pauseDownload(t)), electron.ipcMain.handle(E.RESUME_DOWNLOAD, (e, t) => this.resumeDownload(t)), electron.ipcMain.handle(E.CANCEL_DOWNLOAD, (e, t) => this.cancelDownload(t)), electron.ipcMain.handle(E.GET_ALL_DOWNLOADS, (e) => this.getAllDownloads()), electron.ipcMain.handle(E.DELETE_MODEL, (e, { filename: t, path: r }) => this.deleteModel(t, r));
  }
};
l(m, "instance");
let G = m;
export {
  or as a,
  oe as o,
  rr as r
};
//# sourceMappingURL=index-Ba5g1c58.js.map
