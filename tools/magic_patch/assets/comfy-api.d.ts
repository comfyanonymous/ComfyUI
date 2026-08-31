/**
 * The published ComfyUI custom-node API — the complete surface.
 *
 * Generated from src/platform/nodeApi. If a member is not here it does not
 * exist: do not call it, and punt as api-gap naming what is missing.
 * Reached from a converted pack as:
 *
 *   import { comfy } from '/comfy/api/v2.js'
 */

// ─── backendHandle.ts ────────────────────────────────────────────

export interface BackendHandle {
  /**
   * Absolute URL for a backend route, honouring however the host is served —
   * a base path, a different port, a proxy.
   */
  url(route: string): string
  /**
   * Absolute URL for a file the host serves, rather than an API route.
   *
   * Distinct from `url()` because that one addresses the API and prepends
   * `/api`, so a static path built through it produced `/api/extensions/…`,
   * which 404s.
   *
   * This is for a path the caller already knows absolutely. It is *not* the
   * way a pack should reach its own neighbouring files: the host serves those
   * from `/extensions/<install-dir>/`, and that directory name is chosen when
   * the pack is installed and can be renamed, so it is not knowable from
   * source. `new URL('x.css', import.meta.url)` resolves against the module's
   * real location and stays correct. One pack ships two spellings of its own
   * directory with an `onerror` fallback between them, which is what guessing
   * costs.
   */
  assetUrl(route: string): string
  /**
   * Identifies this frontend connection to a pack's own backend route.
   * Undefined until the backend establishes the connection; do not persist it.
   */
  sessionId(): string | undefined
  /**
   * Fires when {@link sessionId} becomes a different value.
   *
   * A pack that keys ephemeral server-side work by session — a scratch
   * directory, a warmed model, a subscription — needs to know its old key is
   * dead. The id changes on the first connection and again whenever the socket
   * reconnects under a new identity, and the work filed under the previous one
   * is no longer addressable.
   *
   * The session is not the user, the workflow or the node. It does not survive
   * a reload, and storing it in any of those is how a pack ends up reading
   * another tab's scratch state.
   */
  onSessionChanged(
    listener: (sessionId: string | undefined) => void
  ): Unsubscribe
  /**
   * Subscribes to a backend message. The name is whatever the backend emits;
   * `detail` is its payload, unparsed.
   */
  on(event: string, listener: (detail: unknown) => void): Unsubscribe
  /**
   * Calls a backend route with the host's own credentials attached.
   *
   * `url()` only builds a string, so a pack calling `fetch()` on it sends an
   * unauthenticated request — fine on a local install, a 401 on a hosted one.
   * Packs ship their own Python routes and were reaching for `api.fetchApi`
   * precisely to inherit the session; this is that, and nothing more.
   *
   * The route is API-relative and must start with `/`, as `url()` requires.
   */
  fetch(route: string, init?: RequestInit): Promise<Response>
}

// ─── chromeContributions.ts ──────────────────────────────────────

export interface BadgeContribution {
  /** Namespaced, e.g. `Crystools.monitor`. Registering the same id twice throws. */
  readonly id: string
  readonly text: string
  readonly label?: string
  readonly variant?: 'info' | 'warning' | 'error'
  /** An iconify or PrimeIcons class, e.g. `pi-chart-bar`. */
  readonly icon?: string
  readonly tooltip?: string
}

/** What a pack keeps after contributing something to the chrome. */
export interface ChromeItemHandle<T> {
  /** Changes what is shown. Only the fields given are replaced. */
  update(changes: Partial<Omit<T, 'id'>>): void
  remove(): void
}

export interface ButtonContribution {
  readonly id: string
  readonly icon: string
  readonly label?: string
  readonly tooltip?: string
  /**
   * The click. The event is passed because packs branch on modifiers — one
   * opens its panel in a sized window on shift-click — and without it that
   * behaviour has nothing to read.
   */
  run(event: MouseEvent): void
}

// ─── boundedFiles.ts ─────────────────────────────────────────────

export interface FilePickOptions {
  readonly extensions?: readonly string[]
  readonly mimeTypes?: readonly string[]
  /** Maximum accepted file size. The host-wide ceiling is 16 MiB. */
  readonly maxBytes: number
}

export interface FilePickManyOptions extends FilePickOptions {
  /** Maximum number of selected files. The host-wide ceiling is 50. */
  readonly maxFiles: number
  /** Maximum aggregate payload. The host-wide ceiling is 256 MiB. */
  readonly maxTotalBytes: number
}

export interface PickedFileData {
  /** Basename only; no host path is exposed. */
  readonly name: string
  readonly type: string
  readonly bytes: Uint8Array
}

export interface FileDownloadOptions {
  /** Safe basename only. */
  readonly name: string
  readonly mimeType: string
  /** At most 16 MiB. */
  readonly bytes: Uint8Array
}

export interface FilesHandle {
  /** Opens one explicit host file picker; cancellation resolves undefined. */
  pick(options: FilePickOptions): Promise<PickedFileData | undefined>
  /** Opens one bounded multi-file picker; cancellation resolves an empty list. */
  pickMany(options: FilePickManyOptions): Promise<PickedFileData[]>
  /** Asks the host to download one bounded in-memory file. */
  download(options: FileDownloadOptions): Promise<void>
}

// ─── cryptoHandle.ts ─────────────────────────────────────────────

export interface AesCbcEncryptOptions {
  readonly key: Uint8Array
  readonly iv: Uint8Array
  readonly plaintext: Uint8Array
}

export interface AesCbcDecryptOptions {
  readonly key: Uint8Array
  readonly iv: Uint8Array
  readonly ciphertext: Uint8Array
}

export interface HmacSha256Options {
  readonly key: Uint8Array
  readonly data: Uint8Array
}

export interface VerifyHmacSha256Options extends HmacSha256Options {
  readonly signature: Uint8Array
}

/** Fixed canonical primitives; no caller-selected algorithms or retained keys. */
export interface CryptoHandle {
  aesCbcEncrypt(options: AesCbcEncryptOptions): Promise<Uint8Array>
  aesCbcDecrypt(options: AesCbcDecryptOptions): Promise<Uint8Array>
  hmacSha256(options: HmacSha256Options): Promise<Uint8Array>
  verifyHmacSha256(options: VerifyHmacSha256Options): Promise<boolean>
}

// ─── integrationsHandle.ts ───────────────────────────────────────

export interface OllamaListModelsOptions {
  /** Exact loopback Ollama origin or an `ollama://name` admin profile. */
  readonly endpoint: string
}

export interface OllamaIntegrationHandle {
  listModels(options: OllamaListModelsOptions): Promise<string[]>
}

/** Vendor pass-throughs have a weaker stability promise than generic APIs. */
export interface IntegrationsHandle {
  readonly ollama: OllamaIntegrationHandle
}

// ─── closedProxy.ts ──────────────────────────────────────────────

/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface PropSpec<TTarget> {
  get(target: TTarget): unknown
  set?(target: TTarget, value: unknown): void
  /** Appended to the error when a pack assigns to a read-only property. */
  readonlyHint?: string
}

export interface HandleSpec<TTarget> {
  /** Used in errors and `Symbol.toStringTag`, e.g. 'node'. */
  readonly kind: string
  readonly props: Readonly<Record<string, PropSpec<TTarget>>>
  readonly methods?: Readonly<
    Record<string, (target: TTarget, ...args: never[]) => unknown>
  >
  /**
   * Methods that also need the handle's own id.
   *
   * A widget target is just the widget: it holds no reference back to its
   * node, by design, so a method that has to name a sibling cannot find one
   * from the target alone. Separate from `methods` so the common signature
   * stays two arguments.
   */
  readonly idMethods?: Readonly<
    Record<string, (target: TTarget, id: string, ...args: never[]) => unknown>
  >
  /**
   * Props that remain readable after deletion. Identity only — an id or type is
   * still useful for logging and cleanup once the entity is gone.
   */
  readonly identityProps?: readonly string[]
}

/** Present on every handle. Never throws, even when the entity is gone. */
export interface HandleCommon {
  readonly isDeleted: boolean
}

export interface HandleToken {
  readonly kind: string
  readonly id: string
}

// ─── comfyApi.ts ─────────────────────────────────────────────────

export interface Comfy {
  /**
   * `major.minor`. Prefer `supports()` over comparing this — a capability
   * survives being backported or reordered across minors; a version comparison
   * does not.
   */
  readonly version: string
  /** Breaking-change generation. Incremented only when something is removed. */
  readonly major: number
  /**
   * Cheap, never throws. The supported way to branch.
   *
   * Answers whether this host can do something, under the grant it is running
   * with. It is not a permission request: asking does not obtain authority, and
   * a pack never enumerates capabilities to be allowed to run.
   */
  supports(capability: string): boolean
  /** Asserts a capability, with an actionable error naming it. */
  require(capability: string): void
  /** Every capability this host provides. */
  capabilities(): readonly string[]
  /**
   * Pins to a specific major.
   *
   * A major stays available until it is announced for removal and withdrawn
   * through the normal phased deprecation process, so a pack written against
   * one keeps working across that period rather than breaking on a release.
   */
  forMajor(major: number): Comfy

  /**
   * True when two handles refer to the same entity, whatever major, API
   * instance or graph scope produced them.
   *
   * `===` is only reliable for handles from the same instance, the same major
   * AND the same scope. Scope is the one most likely to catch a pack out: a
   * node reached through `comfy.graph` while it is on screen and the same node
   * reached through `graph.subgraphs()` or through a document-scoped
   * `onNodeChanged` come from different handle caches, so they are equal here
   * and not equal under `===`. Use this whenever a handle may have come from
   * another pack, from an event, or from a graph other than the visible one.
   */
  sameEntity(a: unknown, b: unknown): boolean

  /**
   * Re-resolves a handle from any major or instance into one of this instance's
   * own. Returns `undefined` if it is not a handle, or its entity is gone.
   */
  adopt(handle: unknown): NodeHandle | undefined

  readonly graph: GraphHandle
  /** Node definitions, and the replacement for `beforeRegisterNodeDef`. */
  readonly defs: DefRegistry
  /** Declaring, reading and writing pack settings. */
  readonly settings: SettingsHandle
  /**
   * Per-user persistent storage for documents the pack's users author —
   * templates, presets, saved prompts. Server-side, so it follows the user
   * between machines.
   */
  readonly storage: StorageHandle
  /** Bounded, host-sampled hardware metrics. */
  readonly system: SystemHandle
  /** The sanctioned slice of app chrome — sidebar tabs. */
  readonly ui: UiHandle
  /** Host-owned facilities shared by widget implementations. */
  readonly widgets: WidgetsHandle
  /** Bounded declarative locale catalogs rendered by host-native i18n. */
  readonly localization: LocalizationHandle
  /** Commands, their keybindings, and notifications. */
  readonly commands: CommandsHandle
  /** Backend URLs and messages, including a pack's own events. */
  readonly backend: BackendHandle
  /** Loading a parsed workflow into a new active document. */
  readonly workflow: WorkflowHandle
  /** Explicit, bounded host file selection and download. */
  readonly files: FilesHandle
  /** Fixed host cryptographic primitives available to opaque-origin workers. */
  readonly crypto: CryptoHandle
  /** Bounded vendor-specific facilities. */
  readonly integrations: IntegrationsHandle
  /**
   * The editor is already mid-gesture — dragging a link, resizing a node,
   * dragging a widget. A pack running its own pointer gesture must stand down
   * while this is true.
   */
  isInteracting(): boolean
  /**
   * Observes nodes being moved, under either renderer.
   *
   * For building an editing gesture — swap, insert-on-link, shake-to-detach.
   * A pack that moves nodes itself will see its own writes, so guard re-entry.
   */
  onNodeMoved(listener: (event: NodeMoveEvent) => void): Unsubscribe
  /**
   * A drag finished; every node it moved.
   *
   * Where an editing gesture commits — swap the pair, insert into the link
   * under the cursor. **Nodes 2.0 only**: the legacy canvas renderer publishes
   * no drag lifecycle, so this never fires under it.
   */
  onNodeDragEnd(listener: (nodes: readonly NodeHandle[]) => void): Unsubscribe
  /**
   * The view panned, zoomed or was resized.
   *
   * For keeping something anchored to a node in sync — ask
   * `node.getScreenRect()` again when this fires. Carries no payload: where a
   * node is belongs to the node, and the transform belongs to the renderer.
   */
  onViewportChanged(listener: () => void): Unsubscribe
  /**
   * A node changed — its mode, title, colour or shape.
   *
   * For observing nodes the pack does not own. rgthree's relay polls every
   * 500ms and installs a `defineProperty` trap on `mode` because nothing
   * reports it; this is that signal.
   *
   * One stream rather than a subscription per node, deliberately: node
   * identity does not survive undo, reload or re-entering a subgraph, so
   * anything keyed by the object stops firing silently, and keying by id
   * instead never gets collected. Filter by `event.node.id`.
   *
   * Only fields the host tracks are reported. Position is not among them — it
   * changes per frame during a drag and is served by {@link onNodeMoved}.
   *
   * Reports the graph on screen unless `scope: 'document'` asks for the root
   * graph and every subgraph definition as well. A pack that computes from
   * other nodes wants `'document'`: a relay in a subgraph the user has
   * navigated away from otherwise stops recomputing while still asserting its
   * last answer. Each event names the graph it came from, and resolves its node
   * there — ids repeat across definitions, so `event.node.id` alone is not a
   * key.
   */
  onNodeChanged(
    listener: (event: NodeChangeEvent) => void,
    options?: NodeChangeOptions
  ): Unsubscribe
  /**
   * The application has finished starting: canvas, settings and graph all
   * exist, and node definitions are registered.
   *
   * This is `registerExtension({ setup })`. A pack's module body is the `init`
   * half — it runs before definitions register — so anything that needs the
   * running app belongs here. Registering after the app has already started is
   * fine; the listener is called on the next microtask rather than dropped,
   * which is what makes this safe for a pack loaded lazily.
   *
   * Do not poll for the DOM instead. Several packs shipped a `waitForElements`
   * loop to paper over the missing hook, and a poll that outlives its target
   * is a leak that only shows up on someone else's machine.
   */
  onReady(listener: () => void): Unsubscribe
  /** Starting a run, and knowing when one starts. */
  queue: QueueHandle
  /**
   * The node the backend is executing, or `undefined` between runs.
   *
   * Packs tracked this from the raw `executing` message to badge the running
   * node or follow it with the view.
   */
  executingNode(): NodeHandle | undefined
  /** Resolves a backend execution id, including a nested subgraph path. */
  executionNode(id: string): NodeHandle | undefined
  /** Fires when {@link executingNode} changes, including to nothing. */
  onExecutingNodeChanged(
    listener: (node: NodeHandle | undefined) => void
  ): Unsubscribe
  /**
   * A workflow finished loading, and the graph is the new one.
   *
   * This is `afterConfigureGraph`. Unlike {@link onReady} it fires again for
   * every workflow the user opens, which is what a pack re-attaching itself to
   * the document needs — `onReady` fires once and misses every later open.
   *
   * It also fires for undo, redo and a reload of the same document, because a
   * pack rebuilding state from the graph needs those too. The handle says
   * which of them happened: an id equal to the one from last time means this
   * document was rebuilt, not replaced. `undefined` when the host cannot name
   * a document, as when raw workflow data is loaded with no file behind it.
   */
  onWorkflowLoaded(
    listener: (document: DocumentHandle | undefined) => void
  ): Unsubscribe
  /**
   * A document's editing session began.
   *
   * Where per-document state belongs. Fires for a tab opened in the
   * background too, so a pack that allocates here and releases in
   * {@link onDocumentClosed} stays balanced however the user moves around.
   */
  onDocumentOpened(listener: (document: DocumentHandle) => void): Unsubscribe
  /**
   * A document became the one on screen.
   *
   * Distinct from opening: the user returning to a tab activates a document
   * that was already open, and its state is still valid. Anything tied to
   * *being visible* — a panel, a canvas overlay — belongs here.
   */
  onDocumentActivated(listener: (document: DocumentHandle) => void): Unsubscribe
  /**
   * A document stopped being the one on screen, but is still open.
   *
   * Fires before the next document is activated, so a pack moving something
   * between them never sees two claiming the screen at once.
   */
  onDocumentDeactivated(
    listener: (document: DocumentHandle) => void
  ): Unsubscribe
  /**
   * A document's editing session ended, however it ended — the user closing
   * the tab, a temporary workflow being deleted, or the host discarding a
   * background tab whose file changed on disk.
   *
   * Release everything keyed to it. The handle already reports `isDeleted`,
   * and carries the id so a pack can find what it stored; it will not describe
   * the document, because there is no longer one to describe.
   */
  onDocumentClosed(listener: (document: DocumentHandle) => void): Unsubscribe
}

// ─── commandsHandle.ts ───────────────────────────────────────────

/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface KeyCombo {
  readonly key: string
  readonly ctrl?: boolean
  readonly alt?: boolean
  readonly shift?: boolean
  readonly meta?: boolean
}

/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface CommandDef {
  /** Namespaced, e.g. `MyPack.doTheThing`. Shared with core and every pack. */
  readonly id: string
  /**
   * A function when the label depends on state — a toggle that reads "Follow
   * execution" and then "Stop following execution". It is read each time the
   * label is shown, so it must return quickly.
   */
  readonly label: string | (() => string)
  readonly run: () => void | Promise<void>
  /** Bound as a default, so a user's own binding still wins. */
  readonly keybinding?: KeyCombo
  /**
   * Where the keybinding applies. Defaults to anywhere in the application.
   *
   * `'canvas'` limits it to the graph, so it will not fire while the user is
   * typing in a node's text widget or any other field. The host already
   * withholds combos a text input owns — every bare arrow, Ctrl+Left/Right,
   * Ctrl+A/C/V/X/Z — but a pack binding something it does not, say Ctrl+Up,
   * would otherwise fire mid-sentence.
   */
  readonly scope?: 'canvas'
}

/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface NotifyDef {
  readonly severity?: 'success' | 'info' | 'warn' | 'error'
  readonly summary: string
  readonly detail?: string
  /** Milliseconds. Omit for the host's default. */
  readonly life?: number
}

export interface CommandsHandle {
  register(def: CommandDef): void
  notify(def: NotifyDef): void
  /**
   * Runs a command the host or another pack registered, by id.
   *
   * Packs reached into internals to do what a command already does — opening
   * the mask editor was `ComfyApp.copyToClipspace` plus `clipspace_return_node`
   * plus invoking `Comfy.MaskEditor.OpenMaskEditor` by hand. Commands are the
   * sanctioned action layer, so a pack can ask for the behaviour without the
   * host having to publish the machinery behind it.
   *
   * Rejects if no such command is registered — a pack naming a command that
   * has been renamed should hear about it rather than silently do nothing.
   */
  run(id: string): Promise<void>
  /** Whether a command exists, for a pack that offers an entry conditionally. */
  has(id: string): boolean
}

// ─── defsRegistry.ts ─────────────────────────────────────────────

/**
 * The read view of a node definition. Frozen and inert, like every read here.
 *
 * @knipIgnoreUnusedButUsedByCustomNodes
 */
export interface NodeDef {
  readonly type: string
  readonly title: string
  readonly category: string
  readonly description: string
  readonly inputs: readonly Readonly<{
    name: string
    type: string
    /** The translated caption core renders for this input, when it differs. */
    localizedName?: string
    /** The declared choices for a COMBO input, in backend order. */
    values?: readonly (string | number)[]
    /**
     * The input's declaration dict, verbatim from the backend.
     *
     * Same passthrough reasoning as `ExecutionResult.raw`: a pack declares its
     * own keys on its own Python input spec and reads them back here to drive
     * frontend behaviour, so discarding unrecognised keys breaks the pack
     * against its own data. Carries `default`, `min`, `max` and the like too.
     */
    options: Readonly<Record<string, unknown>>
  }>[]
  readonly outputs: readonly Readonly<{
    name: string
    type: string
    tooltip?: string
  }>[]
  readonly isOutputNode: boolean
  /**
   * The node's `hidden` input declarations, verbatim.
   *
   * Deliberately not merged into {@link inputs}: a hidden input is not a slot,
   * and listing it as one would put a connectable input on the node for
   * something the server fills in.
   *
   * Packs ship their own data here and read it back — easy-use and
   * tinyterraNodes both carry an XY-plot axis catalogue as
   * `input.hidden.plot_dict[0]`, on their own key, from their own Python spec.
   * That is the same passthrough reasoning `inputs[].options` already rests on,
   * and dropping it broke both packs against their own data.
   *
   * These are declarations, not values. `PROMPT`, `UNIQUE_ID` and
   * `EXTRA_PNGINFO` appear here as the type markers the node asked for; the
   * server substitutes the real thing at execution time and it never passes
   * through here.
   */
  readonly hidden: Readonly<Record<string, unknown>>
  /** Which pack supplied it, when the backend reports one. */
  readonly source: string | undefined
}

/**
 * Node output as it arrives from the backend.
 *
 * `raw` carries everything else verbatim — ADR 0007's passthrough schema
 * guarantees custom output keys survive, so a pack reading a bespoke key keeps
 * working.
 *
 * @knipIgnoreUnusedButUsedByCustomNodes
 */
export interface ExecutionResult {
  readonly images: readonly Readonly<Record<string, unknown>>[]
  readonly text: readonly string[]
  readonly raw: Readonly<Record<string, unknown>>
}

/**
 * A preview frame the backend produced while this node was running.
 *
 * Per node rather than per channel, deliberately. Packs currently subscribe to
 * `b_preview_with_metadata` *and* `b_preview`, track the executing node id in a
 * module global to correlate the second one, and probe
 * `serverSupportsFeature('supports_preview_metadata')` to decide which to
 * trust — all to answer "is this frame mine?". Answering it once here removes
 * the global, and with it the mis-attribution when two nodes preview at once.
 */
export interface PreviewFrame {
  readonly blob: Blob
  /** Object URL for the blob, revoked when the next frame arrives. */
  readonly url: string
}

/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface ConnectionChangeEvent {
  readonly side: 'input' | 'output'
  readonly index: number
  readonly connected: boolean
  /**
   * The node at the other end, or `undefined` on a disconnect.
   *
   * Packs read `link_info.origin_id` to decide what the new neighbour means —
   * retype a slot to match it, adopt its label. Knowing only that *something*
   * connected forced a re-walk of the whole graph to find out what.
   */
  readonly peerNodeId?: string
  /** The slot index at the other end, or `undefined` on a disconnect. */
  readonly peerIndex?: number
}

export interface NodeDefBuilder {
  /** Current state of the definition, after any earlier extensions ran. */
  readonly def: NodeDef

  setTitle(title: string): void
  setCategory(category: string): void
  /**
   * Declares that this node type never reaches the backend.
   *
   * `defs.define` takes `execution: 'frontend'` for a type the pack owns, but
   * packs also mark *backend-registered* types frontend-only — a tools or
   * control node that exists to drive other nodes and must not appear in the
   * prompt. Without this they reach for `node.isVirtualNode`, and dropping that
   * line puts a new node into `graphToPrompt`, which is a wire-format break.
   *
   * Supply `resolve` when the node carries a value through to something else;
   * omit it and the node is simply left out. See `resolution.ts` — `resolve` is
   * pure over a read-only view and must not mutate the graph.
   */
  setExecution(execution: 'backend' | 'frontend', resolve?: Resolver): void
  /**
   * Declares what this node feeds into *other* nodes' unconnected inputs.
   *
   * The counterpart of `setExecution`'s `resolve`, which answers only "what
   * feeds my own outputs" and is never called for a node with none. Broadcast
   * packs are the reverse: they name inputs on nodes that are not themselves,
   * and discover those edges rather than declaring them.
   *
   * Available here and not only on `defs.define` because the types that
   * broadcast are registered by the pack's Python, and `defs.define` refuses a
   * type that already exists — which left `supply` unreachable for every pack
   * that actually needed it.
   *
   * Not gated on `setExecution('frontend')`: feeding somebody else and being
   * skipped by the prompt builder are separate questions, and a node may
   * legitimately both execute and broadcast.
   */
  setSupply(supply: Supplier): void
  addWidget(def: WidgetDef): void
  hideWidget(name: string): void

  // Behaviour hooks, ordered by measured usage across the 1,265 packs.
  /**
   * Fires once the node exists *and is addressable* — after it joins a graph.
   *
   * Deliberately not litegraph's `onNodeCreated`, which runs inside
   * `createNode()` before the node has an id, a graph, or store registration.
   * A handle is id-backed, so at that moment there is nothing to hand back, and
   * widget writes would land on an unregistered node and be lost on insert.
   */
  onCreated(callback: (node: NodeHandle, event: NodeCreatedEvent) => void): void // 943 packs
  onExecuted(
    callback: (node: NodeHandle, result: ExecutionResult) => void
  ): void // 497 packs
  onConfigured(
    callback: (node: NodeHandle, data: Record<string, unknown>) => void
  ): void // 429 packs
  onConnectionsChanged(
    callback: (node: NodeHandle, event: ConnectionChangeEvent) => void
  ): void // 223 packs
  onRemoved(callback: (node: NodeHandle) => void): void // 158 packs
  /**
   * The node was resized, by the user or by a layout pass.
   *
   * Packs hung a `ResizeObserver` on their mounted element to notice this,
   * which fires for the element rather than the node and misses a resize that
   * does not change the element.
   */
  onResized(callback: (node: NodeHandle, size: Size) => void): void
  /**
   * The pointer entered or left the node.
   *
   * Packs read `canvas.node_over` or set `node.mouseOver` to rebuild a list
   * the moment the pointer arrives, or to decide which node a tooltip belongs
   * to. Both are canvas internals, and the canvas is what Nodes 2.0 replaces.
   */
  onHover(callback: (node: NodeHandle, hovering: boolean) => void): void
  /**
   * The node was double-clicked.
   *
   * Deliberately carries no coordinates. Hit-testing a pointer against
   * node-local geometry is a pack drawing its own front end; the published
   * answer is `widgets.mount` and ordinary DOM events on the element you own.
   */
  onDoubleClick(callback: (node: NodeHandle) => void): void
  /**
   * Whether this node can accept the current browser drag.
   *
   * The event is the browser's data-transfer surface, not a renderer object.
   * Returning `true` makes both node renderers present and route the drop.
   */
  onDragOver(
    callback: (node: NodeHandle, event: DragEvent) => boolean | void
  ): void
  /** Handles a drop the node accepted. Returning `true` claims it. */
  onDrop(
    callback: (
      node: NodeHandle,
      event: DragEvent
    ) => boolean | void | Promise<boolean | void>
  ): void
  /**
   * A property the user edited in the node's properties panel.
   *
   * Packs used `onPropertyChanged` to keep a hand-entered value sane — rgthree
   * clamps a seed's `randomMax` as it is typed. litegraph's own callback can
   * only veto, reverting to the previous value, which throws the user's input
   * away rather than correcting it. `setValue` replaces it instead, and writes
   * without going back through `setProperty`, so a clamp cannot recurse.
   */
  onPropertyChanged(
    callback: (node: NodeHandle, event: PropertyChangeEvent) => void
  ): void
  /** Preview frames for this node, already correlated. */
  onPreview(callback: (node: NodeHandle, frame: PreviewFrame) => void): void
  /**
   * Contributes the pack's own state to the saved node.
   *
   * The returned object is merged into the serialized node, and comes back
   * through `onConfigured`. Only keys the pack owns: core fields are not
   * writable from here, because a pack must not be able to change what the
   * workflow means.
   */
  onSerialize(callback: (node: NodeHandle) => Record<string, unknown>): void
  /**
   * Vetoes or permits an incoming connection *before* it is wired.
   *
   * Distinct from `onConnectionsChanged`, which fires after the fact — packs
   * use the pre-hook to refuse an incompatible link or relabel a slot while
   * the type is still known. Returning `false` refuses.
   */
  onBeforeConnect(
    callback: (node: NodeHandle, event: BeforeConnectEvent) => boolean | void
  ): void
  /**
   * The user dropped a link on a node's body and the host found no single slot
   * that fits. Wire it yourself and return `true`; return nothing to let the
   * host report the drop unplaceable.
   *
   * For a node whose one slot carries a bundle of values — a context, a pipe —
   * and which wants to unpack it into several of the peer's slots at once. Both
   * ends of the drag are asked, the one the user aimed at first, because the
   * node with the knowledge is the drop target in one direction and the drag's
   * origin in the other.
   *
   * The published alternative to replacing `connectByType` on the prototype,
   * which is how packs did this: that changes link routing for every node in
   * the document, so one pack's convenience became every other pack's
   * behaviour.
   */
  onUnplacedLink(
    callback: (node: NodeHandle, event: UnplacedLinkEvent) => boolean | void
  ): void
  /** Adds an entry to this node type's context menu. */
  addMenuItem(item: NodeMenuItem): void
}

export interface NodeCreatedEvent {
  /**
   * The node arrived carrying saved state — pasted, duplicated, or loaded from
   * a workflow — rather than being made fresh.
   *
   * Read as "was `configure` called on it before it joined the graph", which is
   * what actually distinguishes the cases. Packs overrode `clone()` to reset
   * state a copy should not inherit — a duplicated node keeping the dynamic
   * slots that were fed by the original's upstream, a duplicated reroute born
   * hard-typed and refusing every other type — and `clone()` runs before the
   * node has an id, so there is nothing to hand a pack there.
   */
  readonly restored: boolean
  /**
   * The whole graph was being loaded, so {@link restored} means "came from the
   * saved file" rather than "came from the clipboard".
   *
   * The distinction is the point: a pasted node should drop slots it cannot
   * still be fed through, and a loaded one must keep every one of them or the
   * workflow opens wrong.
   */
  readonly loading: boolean
}

export interface UnplacedLinkEvent {
  /** Which of this node's slots the link would land on. */
  readonly side: 'input' | 'output'
  /** The node at the other end of the drag. */
  readonly peerNodeId: string
  /** The slot on the peer the drag started from. */
  readonly peerIndex: number
  readonly type: string
  /**
   * The user held the modifier that means "overwrite what is already wired".
   *
   * Published because packs read a global keyboard service of their own to get
   * it, and which modifier means this is the host's to decide.
   */
  readonly replaceExisting: boolean
}

/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface BeforeConnectEvent {
  readonly side: 'input' | 'output'
  readonly index: number
  /** The node at the other end, when one is known. */
  readonly peerNodeId: string | undefined
  /** The slot at the other end, when one is known. */
  readonly peerIndex: number | undefined
  readonly peerType: string | undefined
}

/** One entry inside a menu item's submenu. */
/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface NodeSubMenuItem {
  readonly label: string
  run(node: NodeHandle): void
}

/**
 * One entry of ComfyUI's node palette: the title bar, the body, and the shade
 * a group of that colour is filled with.
 *
 * @knipIgnoreUnusedButUsedByCustomNodes
 */
export interface NodeColor {
  readonly color: string
  readonly bgColor: string
  readonly groupColor: string
}

/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface NodeMenuItem {
  /**
   * A function when the text depends on the node — packs label entries with
   * the current state ("Unmute 3 nodes"), which a string fixed at
   * registration cannot express.
   */
  readonly label: string | ((node: NodeHandle) => string)
  /**
   * Shown only when this returns true. Without it a pack that wants an entry
   * to appear conditionally has to either show it always or not at all —
   * efficiency-nodes hides its seed submenu when the feature is off, and
   * flattening that to a permanent entry is a worse lie than omitting it.
   */
  when?(node: NodeHandle): boolean
  /** Omit when the item only opens a submenu. */
  run?(node: NodeHandle): void
  /**
   * Turns the entry into a submenu. One level deep, deliberately: every
   * measured pack uses exactly one, and nesting further is a menu design
   * problem rather than an API one.
   *
   * A function when the children depend on the node's current state, which is
   * the common case rather than the exotic one: efficiency-nodes' LoRA Stacker
   * declares fifty `lora_name_N` widgets and lists only the two or three a
   * user has filled. A fixed array would put fifty rows in that menu, which is
   * a different menu, so the alternative to this was omitting the feature.
   */
  readonly items?:
    | readonly NodeSubMenuItem[]
    | ((node: NodeHandle) => readonly NodeSubMenuItem[])
  /**
   * Sort position among this node's pack-added entries. Lower first; entries
   * without one keep registration order, which is module-load order and so
   * depends on import sequence rather than intent.
   */
  readonly order?: number
}

/**
 * Which definitions an extension applies to.
 *
 * Indexed rather than run-and-return: this predicate is almost always the guard
 * clause the pack already had at the top of its hook.
 */
export type DefSelector =
  | string
  | readonly string[]
  | RegExp
  /**
   * A predicate over the definition, for a guard the other forms cannot
   * express — "any node taking a VAE input", which is a shape rather than a
   * name.
   *
   * Deliberately last, and deliberately discouraged. The declarative forms
   * exist because a name check can be indexed, while a predicate has to run for
   * every registered type; with thousands of types that is the boot cost this
   * API set out to remove. Use it only when the guard genuinely reads a def's
   * inputs or outputs.
   */
  | ((def: NodeDef) => boolean)
  /**
   * A `RegExp` category covers the prefix filter 53 packs open their hook with
   * (`nodeData.category.startsWith('KJNodes')` → `{ category: /^KJNodes/ }`).
   */
  | { readonly category: string | RegExp }

/**
 * A node type the pack owns, declared rather than subclassed.
 *
 * 86 packs (18.2% of installs) do this today with `extends LGraphNode` +
 * `LiteGraph.registerNodeType`, which is OOP entity modelling — the thing ADR
 * 0008 rules out. Here the definition is plain data; the class behind it is an
 * internal detail of this layer, never the pack's.
 *
 * @knipIgnoreUnusedButUsedByCustomNodes
 */
export interface NodeDefinition {
  readonly type: string
  readonly title?: string
  readonly category?: string
  readonly description?: string
  readonly inputs?: readonly { name: string; type: string }[]
  readonly outputs?: readonly {
    name: string
    type: string
    shape?: SlotShape
  }[]
  readonly widgets?: readonly WidgetDef[]
  /**
   * `'frontend'` nodes never reach the backend: they are resolved away at
   * prompt time by the resolution system, or simply omitted.
   */
  readonly execution?: 'backend' | 'frontend'
  /**
   * Answers what each output resolves to, purely, over a read-only view.
   * See `resolution.ts` — this replaces `applyToGraph`, which mutated the
   * live graph mid-serialize.
   */
  readonly resolve?: Resolver
  /**
   * What this node feeds into *other* nodes' unconnected inputs.
   *
   * The broadcast direction: `resolve` cannot express it, because the nodes
   * being fed are not this one and the edges are discovered rather than
   * declared.
   */
  readonly supply?: Supplier

  onCreated?(node: NodeHandle, event: NodeCreatedEvent): void
  onExecuted?(node: NodeHandle, result: ExecutionResult): void
  onConfigured?(node: NodeHandle, data: Record<string, unknown>): void
  onConnectionsChanged?(node: NodeHandle, event: ConnectionChangeEvent): void
  onPropertyChanged?(node: NodeHandle, event: PropertyChangeEvent): void
  onDragOver?(node: NodeHandle, event: DragEvent): boolean | void
  onDrop?(
    node: NodeHandle,
    event: DragEvent
  ): boolean | void | Promise<boolean | void>
  onRemoved?(node: NodeHandle): void
  onSerialize?(node: NodeHandle): Record<string, unknown>
}

export interface DefRegistry {
  /**
   * Declares how an input *type* is presented — the replacement for
   * `getCustomWidgets`.
   *
   * Not decoration: the host decides widget-vs-socket purely by whether a type
   * is registered, so an unregistered one turns the input into a socket and
   * drops its value from `widgets_values`. See `widgetTypes.ts`.
   */
  defineWidgetType(type: string, def: WidgetTypeDef): Unsubscribe
  /**
   * Registers a node type the pack owns. Returns a handle that unregisters
   * it — which `LiteGraph.registerNodeType` never offered.
   */
  define(definition: NodeDefinition): Unsubscribe
  get(type: string): NodeDef | undefined
  all(): readonly NodeDef[]
  has(type: string): boolean
  extend(
    selector: DefSelector,
    apply: (builder: NodeDefBuilder) => void
  ): Unsubscribe
  /**
   * Asks the host to reload node definitions from the backend.
   *
   * Combo inputs whose values the backend supplies — model lists, LoRA names,
   * sampler names — are captured when definitions load, so a pack that adds a
   * file server-side leaves every open picker showing the old list. This is
   * `app.refreshComboInNodes()`, which packs called after saving a model
   * preview or writing a new file.
   *
   * Refreshing is not free: it refetches every definition. Call it after a
   * change the user made, not on a timer.
   */
  /**
   * The colour links and slots of a type are drawn in.
   *
   * A pack matching the theme in its own DOM — a legend, a chip, a preview —
   * read `LGraphCanvas.link_type_colors` for this. Reading a design token to
   * match is the opposite of drawing your own front end, so it is published;
   * the table itself is not.
   */
  typeColor(type: string): string
  /**
   * The colours behind a name in ComfyUI's node palette — `red`, `pale_blue` —
   * or `undefined` for a name it does not define.
   *
   * Same reasoning as {@link typeColor}, and the same limit: the resolver is
   * published, the table is not. What makes this a design token rather than a
   * renderer internal is that the names are the user's own vocabulary. They
   * pick "green" from a menu; nothing records the word, only the hex it stood
   * for. So a pack offering "mute every red group" cannot match what the user
   * chose without being told which hex "red" meant, and two packs did it by
   * reading `LGraphCanvas.node_colors` directly.
   *
   * Colours move with the palette, names do not. Resolve on use; do not cache
   * the result and do not persist it in a workflow.
   */
  nodeColor(name: string): NodeColor | undefined
  /**
   * Tests an output type against an input type using the host's connection
   * rules, including wildcards and comma-delimited unions.
   */
  isTypeCompatible(outputType: string, inputType: string): boolean
  /**
   * Declares the colour for a data type this pack introduces.
   *
   * Packs shipping their own types — `PIPE_LINE`, `LORA_STACK`, `XYPLOT` —
   * wrote straight into `LGraphCanvas.link_type_colors` so their links were
   * not all grey.
   *
   * Refuses a type the host already colours. That write is global: one pack
   * recolouring `IMAGE` restyles every graph for every other pack and the
   * user has no way to see who did it. Colouring a type you brought is
   * additive; colouring one you did not is not yours to decide.
   */
  setTypeColor(type: string, color: string): Unsubscribe
  refresh(): Promise<void>
  /**
   * Node definitions were reloaded — by this pack, another pack, or the user.
   *
   * The listening half of `refresh()`, and what the `refreshComboInNodes`
   * extension hook gave packs. A pack holding its own cached copy of a combo's
   * values — a model list it filters, a picker it built — needs to rebuild it
   * when the list changes underneath, and the pack that caused the change is
   * usually not this one.
   */
  onRefreshed(listener: () => void): Unsubscribe
}

/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface PropertyChangeEvent {
  readonly name: string
  readonly value: unknown
  readonly previous: unknown
  /** Replaces what is stored. Last writer wins if several packs respond. */
  setValue(value: unknown): void
  /** Discards the edit, restoring `previous`. */
  reject(): void
}

// ─── documentHandle.ts ───────────────────────────────────────────

export interface DocumentHandle extends HandleCommon {
  /**
   * Identity of this editing session. Stable for as long as the document is
   * open — including across undo, redo and tab switches — and never reused.
   *
   * Not the id inside the workflow JSON, which travels with the file, so two
   * opens of it and any copy made outside the app all share one value. Not the
   * path either, which is a storage address and changes on rename. Do not
   * persist this: it means nothing in the next page load.
   */
  readonly id: string
  /** Display name, without the directory or extension. */
  readonly name: string | undefined
  /**
   * Storage path, for addressing the file. Undefined for a document with no
   * file behind it yet. Changes when the user renames, so key pack state on
   * {@link id} instead.
   */
  readonly path: string | undefined
  /** Whether there are edits the user has not saved. */
  readonly isModified: boolean
  /**
   * True once this editing session has ended.
   *
   * A handle is a snapshot of a session, and a pack may hold one across a tab
   * close or a background unload. Check before acting on stored state rather
   * than trusting a captured handle, exactly as for a node or a widget.
   */
  readonly isDeleted: boolean
}

/** What the host must supply to describe one open document. */
export interface DocumentSource {
  readonly sessionId: string | null
  readonly filename?: string
  readonly path?: string
  readonly isModified?: boolean
  /** Whether this is the document the editor is showing. */
  readonly isActive?: boolean
}

/**
 * Every document currently open, including background tabs.
 *
 * One reader rather than one per question: a handle has to answer for a
 * document that is open but not on screen, and a lookup that only knew the
 * active one would report every background tab as closed.
 */
export type DocumentReader = () => readonly DocumentSource[]

// ─── documentLifecycle.ts ────────────────────────────────────────

/**
 * The transitions a document makes.
 *
 * `opened` and `closed` bracket the session's existence; `activated` and
 * `deactivated` bracket its time on screen. A document opened in the
 * background is `opened` without being `activated`, which is why they are
 * separate: a pack that allocates on `opened` and releases on `closed` stays
 * balanced no matter how the user moves between tabs.

 */
export type DocumentPhase = 'opened' | 'activated' | 'deactivated' | 'closed'

// ─── graphHandle.ts ──────────────────────────────────────────────

/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface NodeInit {
  title?: string
  position?: { x: number; y: number }
}

/**
 * How far {@link GraphHandle.queryNodes} looks.
 *
 * `'visible'` is the graph on screen and the default, matching `nodes()`.
 * `'root-and-subgraphs'` is the root graph and every subgraph *definition* —
 * the same set `onNodeChanged`'s `'document'` scope reports over. A subgraph
 * placed three times contributes its nodes once, which is what a pack acting
 * on "each of my nodes" means.
 *
 * @knipIgnoreUnusedButUsedByCustomNodes
 */
export type NodeQueryScope = 'visible' | 'root-and-subgraphs'

/**
 * Which nodes {@link GraphHandle.queryNodes} should return.
 *
 * Every field narrows; omitting all of them returns the whole scope. They
 * compose as AND, because the cases packs actually hand-rolled — "my nodes,
 * anywhere in the document", "everything in this group" — are intersections.
 *
 * @knipIgnoreUnusedButUsedByCustomNodes
 */
export interface NodeQuery {
  readonly scope?: NodeQueryScope
  /**
   * Node type. A string matches exactly, an array matches any of them, and a
   * regular expression matches by pattern — which is how a pack asks for its
   * own nodes without listing every type it ships.
   */
  readonly type?: string | RegExp | readonly string[]
  /**
   * Restrict to nodes in the graph the user is looking at.
   *
   * Only meaningful under `'root-and-subgraphs'`: it is the difference between
   * "every node in the document" and "the ones the user can currently see".
   * This is *not* a viewport test — a node scrolled off the edge of a graph
   * the user is in is still rendered by this definition. Culling belongs to
   * the renderer and differs between the two of them.
   */
  readonly rendered?: boolean
  /** Restrict to nodes the group currently contains. */
  readonly group?: GroupHandle
}

export interface GraphHandle {
  readonly id: string
  node(id: string): NodeHandle | undefined
  nodes(): readonly NodeHandle[]
  nodesOfType(type: string): readonly NodeHandle[]
  /**
   * One flat query over graph-scoped nodes.
   *
   * `nodes()` and `nodesOfType()` address the graph on screen, so a pack that
   * wanted "every node of mine in this document" had to walk `root()` and each
   * `subgraphs()` entry itself and concatenate the results — and the ones that
   * did not simply stopped working the moment a user nested anything.
   *
   * Handles come from the scope that owns each node, so a node reached here
   * under `'root-and-subgraphs'` is not `===` the one `graph.node()` returns
   * for it. That is the same scope rule `sameEntity()` exists for; compare
   * with `comfy.sameEntity()` rather than `===`.
   */
  queryNodes(query?: NodeQuery): readonly NodeHandle[]
  add(type: string, init?: NodeInit): NodeHandle
  remove(id: string): boolean
  links(): readonly LinkInfo[]
  /**
   * The supply edges prompt execution would use in this graph right now.
   *
   * Re-runs the registered pure suppliers and the host's priority arbitration,
   * returning graph-local ids suitable for {@link OutputSlotHandle.connectTo}.
   * Exact priority ties are absent, just as they are from the prompt. The
   * frozen snapshot never mutates the graph.
   */
  resolvedSupplies(): readonly ResolvedSupply[]
  /**
   * The nodes the user currently has selected.
   *
   * 15 packs read `canvas.selected_nodes` or `selectedItems` for this — a
   * canvas internal, and the canvas is exactly what Nodes 2.0 replaces.
   * Selection is a property of the document, so it is asked of the graph.
   */
  selection(): readonly NodeHandle[]
  /**
   * Replaces the selection with these nodes. An empty list clears it.
   *
   * A node a pack just created is the usual case — `LGraphCanvas.add`'s
   * `options.select` put it straight under the user's cursor, and without this
   * the node appears but the user has to find and click it.
   *
   * `add: true` extends the selection instead of replacing it.
   */
  select(nodes: readonly NodeHandle[], options?: { add?: boolean }): void
  /**
   * Pans the view so a node sits in the middle of it.
   *
   * Packs wrote `canvas.ds.offset` themselves to do this, which bakes in the
   * renderer's transform and the device pixel ratio. Does not change zoom.
   */
  centerOn(node: NodeHandle): void
  /**
   * The groups on the canvas, in draw order.
   *
   * Packs read `graph._groups` to build a group muter, a group runner, or a
   * navigator. A group is a rectangle plus a title: which nodes it holds is
   * derived from what it overlaps, which is why `nodes()` is a method and not
   * a stored list.
   */
  groups(): readonly GroupHandle[]
  /**
   * Scales the view. 1 is unzoomed.
   *
   * Packs saved a zoom level alongside a node to restore a view; without this
   * a bookmark could pan but the number it stored was inert. Clamped to what
   * the canvas allows, so a stored extreme cannot strand the user.
   */
  setZoom(scale: number): void
  /**
   * Where the pointer is, in graph space — the coordinates {@link nodeAt} and
   * {@link NodeHandle.setPosition} use.
   *
   * A pack adding a node from a menu put it under the cursor. Without this the
   * node lands at the graph origin, which on any panned view is off screen.
   *
   * `undefined` when there is no canvas to measure against.
   */
  pointerPosition(): Point | undefined
  /**
   * The document's root graph, even while the user is viewing a subgraph.
   * Undefined before a document exists.
   */
  root(): GraphScopeHandle | undefined
  /**
   * The subgraph definitions in the document, each scoped to its own nodes.
   *
   * `nodes()` and `node()` address the graph on screen only, so a pack that
   * must reach every node — refreshing its own nodes after a run, walking a
   * chain — misses anything nested.
   *
   * Access is *through* the subgraph rather than a flattened list. Ids are
   * allocated from the root graph's counter, so they do not collide among
   * nodes created in one session — but a subgraph loaded from a file brings
   * its authored ids, and `configure` raises that counter without renumbering
   * anything. Two independently authored subgraphs can therefore carry the
   * same id. Resolving inside the owning graph is correct either way, and does
   * not rest on an invariant litegraph does not promise.
   *
   * These are definitions, not instances. A subgraph placed three times has
   * one entry, and its nodes appear once — which is what a pack acting on
   * "each of my nodes" wants.
   */
  subgraphs(): readonly GraphScopeHandle[]
  /**
   * Runs several mutations as one undo step.
   *
   * Without it, a pack that adds three nodes and wires them leaves the user
   * pressing undo four times to get back. `graph.beforeChange()` /
   * `afterChange()` did this by counting nesting depth.
   *
   * A scope rather than a pair of calls: the counter only captures when it
   * returns to zero, so one throw between a manual `before` and `after` stops
   * undo capturing anything at all, for the rest of the session, with nothing
   * to show why. The scope closes on the way out either way.
   *
   * Synchronous on purpose. Holding the group open across an `await` would
   * fold whatever the user did while waiting into the pack's undo step.
   */
  batch<T>(mutations: () => T): T
  /**
   * The topmost node at a point in graph space, if any.
   *
   * Packs building a gesture were walking every node and re-deriving its
   * rectangle from renderer constants. The graph already knows, and its answer
   * respects z-order, collapsed nodes and the active renderer's layout.
   *
   * Answers against the *rendered* layout, which is the only sensible reading
   * of "what is under this point" — and is why it is not refreshed per call: a
   * gesture asks this on every pointer move, and remeasuring every node each
   * time would be the expensive mistake. Before the first frame it finds
   * nothing.
   */
  nodeAt(point: { x: number; y: number }): NodeHandle | undefined
  /**
   * A copy of a node, carrying its widget values and properties, added to the
   * graph without links.
   *
   * `add(type)` only makes a fresh node of a type, so a pack duplicating a
   * configured node — a prompt box the user has filled in — had no way to keep
   * what it contained. Links are deliberately not copied: a duplicate wired
   * into the same places is a different operation, and the caller can connect
   * it themselves.
   *
   * `undefined` if the node is gone, or if its type is not registered — the
   * copy is built through the registry, so there is nothing to build from.
   * Widget values carry over only for a type that serializes them, which every
   * backend-registered type does.
   */
  duplicate(
    id: string,
    position?: { x: number; y: number }
  ): NodeHandle | undefined
  /**
   * Rebuilds a node, optionally as another type, keeping what the user set and
   * every link that still fits. Replacing with the same type repairs a node
   * whose registered definition changed without discarding its state.
   * `undefined` if the node is gone; throws if the type is not registered.
   *
   * This is a real feature four packs ship — "Convert to Context Big", "Swap to
   * KSampler (Efficient)" — and all four hand-rolled it out of `graph.links`,
   * `getNodeById` and `LiteGraph.createNode`, which is most of what this
   * migration exists to delete. All four also got it wrong: one drops every
   * widget value and hardcodes "slot 0 only", the other recurses through
   * requestAnimationFrame forever on an inverted comparison and leaves a
   * separate undo step for the add, each connection, and the remove.
   *
   * Position, custom title, colour, mode, declared properties and widget values
   * carry over by name. Size is the larger of what the user set and what the new
   * type needs, so a node that grew more slots is not clipped. Links are re-made
   * by slot name, falling back to the same index; type checking is the ordinary
   * connection rule, so a link that no longer fits is dropped and warned about
   * rather than forced. The whole swap is one undo step.
   */
  replace(id: string, type: string): NodeHandle | undefined
  /**
   * Changes when the graph does: nodes added, removed or reconfigured, links
   * connected or disconnected, slots and subgraph inputs/outputs altered, and
   * the node flags a reader can see — collapsed, pinned, advanced.
   *
   * Hold one and compare it later to learn whether anything moved since. That
   * is the whole contract: an opaque token, not a count. Do not subtract two
   * of them, do not expect it to start anywhere in particular, and do not
   * expect consecutive changes to differ by one. Coalesced edits are free to
   * advance it once, and `batch()` exists precisely so they can.
   *
   * A widget value committed by the user or through
   * `WidgetHandle.setValue()` advances it through the same host protocol. Data
   * a pack keeps outside graph and widget state does not; a canvas widget
   * holding such data has `redraw()`.
   */
  readonly version: number
  /** Diagnostics: live handle-cache slots across all kinds. */
  readonly cacheSize: number
}

/**
 * A subgraph definition, scoped to its own contents.
 *
 * Deliberately narrower than {@link GraphHandle}: adding, selecting, centring
 * and zooming all address what the user is looking at, and a subgraph
 * definition is not that. This is for reading and reaching nodes.
 */
/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface GraphScopeHandle {
  /** Stable across every instance of this subgraph. */
  readonly id: string
  readonly name: string | undefined
  nodes(): readonly NodeHandle[]
  node(nodeId: string): NodeHandle | undefined
  /**
   * The groups drawn inside this subgraph.
   *
   * A group muter or runner that skipped these reported nothing for a
   * subgraph's contents while appearing to work.
   */
  groups(): readonly GroupHandle[]
  /** The supply edges prompt execution would use inside this graph. */
  resolvedSupplies(): readonly ResolvedSupply[]
}

// ─── groupHandle.ts ──────────────────────────────────────────────

export interface GroupHandle {
  readonly id: string
  getTitle(): string
  setTitle(title: string): void
  /** Colour as the renderer holds it, or undefined for the default. */
  getColor(): string | undefined
  setColor(color: string): void
  /**
   * The nodes the group currently contains, recomputed on each call.
   *
   * Packs muted or queued "the group", which always meant its nodes. Do not
   * cache the result: a drag changes it with no event.
   */
  nodes(): readonly NodeHandle[]
  /** The group's rectangle in graph space, title bar included. */
  getBounds(): Bounds
  /** Pans the view so this group is in the middle of it. Zoom is unchanged. */
  centerOn(): void
}

// ─── interaction.ts ──────────────────────────────────────────────

export interface NodeMoveEvent {
  readonly node: NodeHandle
  readonly position: { readonly x: number; readonly y: number }
}

/**
 * Where movement comes from, supplied by the renderer.
 *
 * `platform/` cannot import `renderer/`, and the layout store lives there. This
 * is the same seam `registerBadgeRowsProvider` uses so litegraph never reaches
 * into the store: the upper layer pushes the source down at boot.
 */
export type NodeMoveSource = (
  onMove: (nodeId: string, position: { x: number; y: number }) => void
) => Unsubscribe

/** Reports a completed drag with the ids of every node it moved. */
export type NodeDragEndSource = (
  onDragEnd: (nodeIds: readonly string[]) => void
) => Unsubscribe

// ─── nodeChanges.ts ──────────────────────────────────────────────

/** A field the host tracks and reports. Not every property is one. */
export type TrackedProperty =
  | 'title'
  | 'mode'
  | 'color'
  | 'bgcolor'
  | 'shape'
  | 'showAdvanced'

/**
 * Which graphs a listener hears from.
 *
 * `'visible'` is the default and the graph on screen, following the user into
 * and out of subgraphs — what a pack decorating what the user is looking at
 * wants.
 *
 * `'document'` is the root graph and every subgraph definition. A pack that
 * *computes* from other nodes needs it: rgthree's relay derives a group's mute
 * state from its inputs, and inside a subgraph the user had navigated away from
 * it stopped recomputing while still asserting its last answer — so a group
 * stayed muted against its inputs, intermittently, and healed on navigation.
 */
export type NodeChangeScope = 'visible' | 'document'

export interface NodeChangeOptions {
  scope?: NodeChangeScope
}

export interface NodeChangeEvent {
  /** The node that changed. It may belong to another pack, or to none. */
  readonly node: NodeHandle
  /**
   * The graph the change happened in — the root graph's id, or a subgraph
   * definition's. Node ids are unique only within a graph, so a pack keeping
   * its own records under `'document'` must key on both.
   */
  readonly graphId: string
  /**
   * The editing session the change happened in, or `undefined` when the host
   * cannot name one.
   *
   * `graphId` is restored from the saved workflow and round-trips through
   * `serialize()`, so it identifies the graph on disk, not the document open
   * in front of the user — two opens of one file report the same value. A pack
   * holding records across a document swap needs this to know they are stale.
   */
  readonly documentId: string | undefined
  readonly property: TrackedProperty
  readonly from: unknown
  readonly to: unknown
}

// ─── nodeHandle.ts ───────────────────────────────────────────────

/** @knipIgnoreUnusedButUsedByCustomNodes */
export type NodeMode = 'always' | 'never' | 'bypass' | 'on-event' | 'on-trigger'

/** @knipIgnoreUnusedButUsedByCustomNodes */
export type NodeShape = 'default' | 'box' | 'round' | 'circle' | 'card'

export interface BadgeDef {
  readonly text: string
  /** Text colour. Defaults to core's badge foreground. */
  readonly color?: string
  /** Background colour. Defaults to core's badge background. */
  readonly bgColor?: string
  /**
   * Makes the badge clickable.
   *
   * Two conversions declined to turn a button into a badge because a badge
   * that looks pressable and does nothing is worse than the thing it replaced.
   */
  onClick?(): void
}

export interface Point {
  readonly x: number
  readonly y: number
}

export interface Size {
  readonly width: number
  readonly height: number
}

/** A rectangle in graph space. */
export interface Bounds {
  readonly x: number
  readonly y: number
  readonly width: number
  readonly height: number
}

export interface NodeSnapshot {
  readonly id: string
  readonly type: string
  readonly title: string
  readonly mode: NodeMode
  readonly collapsed: boolean
  readonly pinned: boolean
  readonly color: string | undefined
  readonly bgColor: string | undefined
  readonly shape: NodeShape
  readonly position: Point
  readonly size: Size
}

/**
 * Shapes follow `src/types/extensionV2.ts`, the agreed extension contract:
 * accessor methods rather than properties, so a read can be a store query and
 * a write can dispatch a command.
 */
export interface SizeConstraints {
  minWidth?: number
  minHeight?: number
  maxWidth?: number
  maxHeight?: number
  /** Grow to fit content rather than holding a fixed height. */
  autoHeight?: boolean
}

export interface NodeHandle extends HandleCommon {
  readonly id: string
  readonly type: string
  readonly comfyClass: string

  getTitle(): string
  setTitle(title: string): void
  getMode(): NodeMode
  setMode(mode: NodeMode): void
  isCollapsed(): boolean
  setCollapsed(collapsed: boolean): void
  isPinned(): boolean
  setPinned(pinned: boolean): void
  getColor(): string | undefined
  setColor(color: string | undefined): void
  getBgColor(): string | undefined
  setBgColor(color: string | undefined): void
  getShape(): NodeShape
  setShape(shape: NodeShape): void
  getProperty<T = unknown>(key: string): T | undefined
  getProperties(): Readonly<Record<string, unknown>>
  setProperty(key: string, value: WidgetValue): void
  /**
   * Whether this node emits `widgets_values` when the workflow is serialized.
   *
   * Writable because packs vary it per node type, and the value is part of the
   * wire format — a conversion that could not set it would change what the
   * saved workflow contains.
   */
  isSerializingWidgets(): boolean
  setSerializeWidgets(serialize: boolean): void

  getPosition(): Point
  setPosition(pos: Point): void
  getSize(): Size
  /** Changes size through the host's resize protocol, including `onResized`. */
  setSize(size: Size): void
  /**
   * The node's rectangle in graph space, title bar included.
   *
   * `getPosition()` is the body's top-left, so packs building a gesture were
   * reconstructing this by subtracting a title height read off the renderer —
   * which is only right for the default layout, and wrong for a collapsed node
   * or under a different renderer. Ask the renderer instead of re-deriving it.
   */
  getBounds(): Bounds
  /**
   * Where a slot sits, in graph space.
   *
   * The renderer's own answer, so it stays correct for collapsed nodes,
   * widget-backed inputs and layouts that are not the default vertical stack —
   * all cases the `(index + 0.7) * slotHeight` reconstruction gets wrong.
   *
   * `undefined` if there is no slot at that index.
   */
  getSlotPosition(side: 'input' | 'output', index: number): Point | undefined
  /**
   * Where the node currently sits on screen, in client coordinates.
   *
   * For anchoring a floating panel to a node. Packs did this by reading the
   * viewport's pan and zoom and doing the arithmetic themselves, which is both
   * the renderer's business and wrong the moment the transform changes shape.
   *
   * The answer already accounts for zoom, so a pack needing to convert a pixel
   * drag into graph units can divide by `width / getBounds().width` rather than
   * asking for the scale factor.
   *
   * `undefined` when nothing is on screen to measure against.
   */
  getScreenRect(): Bounds | undefined
  /**
   * URLs of the images this node produced when it last executed.
   *
   * Packs read `node.imgs` — the loaded `HTMLImageElement`s core hangs on the
   * node — to walk upstream for the nearest ancestor holding a composite, or
   * to scan the selection for something to feed an editor. `onExecuted` does
   * not answer that: it is per node type, so it never sees another pack's
   * outputs, and it only fires at the moment of execution.
   *
   * URLs rather than elements, deliberately. The loaded element is the
   * renderer's, and its lifetime is the renderer's; a pack that wants pixels
   * can load the URL itself and own the result. This also covers previews,
   * which are what the node is showing when a run is still in flight.
   *
   * Empty when the node has not produced images.
   */
  getOutputImages(): readonly string[]
  /**
   * Which of {@link getOutputImages} the user is looking at, or `undefined`
   * when they have neither selected nor hovered one.
   *
   * A pack copying "the image" or saving one as a model's preview meant the
   * one under the cursor, not the first of the batch. `undefined` is why this
   * is not simply `0`: an entry that acts on a guess writes the wrong file to
   * the server, silently.
   */
  getDisplayedImageIndex(): number | undefined
  /**
   * The id of the graph holding this node — the root graph's id, or a
   * subgraph's.
   *
   * A pack keeping its own records against nodes needs it: node ids are unique
   * per graph, so a key built from the id alone collides once subgraphs are
   * involved. Pair it with `comfy.graph.subgraphs()` to get back to the node.
   */
  readonly graphId: string | undefined
  /**
   * Puts a small label on the node's title bar. Returns a handle that removes
   * it again.
   *
   * Packs draw a status, a count, a cost, a model name. They did it by
   * overriding `onDrawForeground` and painting into the canvas context, which
   * only works under the legacy renderer and puts the pack in the business of
   * laying out text. `badges` is core's own extension point and both renderers
   * draw it.
   *
   * Pass a function for a label that changes: it is called each time the node
   * is drawn, so return quickly and do not build strings you could cache.
   */
  addBadge(badge: BadgeDef | (() => BadgeDef)): Unsubscribe
  /**
   * Declares how the node may be sized, instead of re-asserting it per frame.
   *
   * 39 packs recompute size inside a draw or resize callback, which is both a
   * per-frame cost and a fight with the layout. `autoHeight` is usually the
   * real intent: the pack mounted something of unknown height and wants the
   * node to fit it.
   */
  setSizeConstraints(constraints: SizeConstraints): void
  getSizeConstraints(): Readonly<SizeConstraints>

  readonly inputs: SlotCollection<InputSlotHandle>
  readonly outputs: SlotCollection<OutputSlotHandle>
  readonly widgets: WidgetCollection
  snapshot(): Readonly<NodeSnapshot> | undefined
  remove(): void
}

/** Per-node collections, supplied by the graph layer that owns their caches. */
export interface NodeCollections {
  inputs(nodeId: string): SlotCollection<InputSlotHandle>
  outputs(nodeId: string): SlotCollection<OutputSlotHandle>
  widgets(nodeId: string): WidgetCollection
}

// ─── queueHandle.ts ──────────────────────────────────────────────

/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface RunOptions {
  /**
   * Run only these nodes and whatever feeds them, instead of the whole
   * workflow. Empty is rejected rather than treated as "everything": a filter
   * that matched nothing must not silently run the entire graph.
   */
  nodes?: readonly NodeHandle[]
  /** How many times to run. Defaults to 1. */
  batch?: number
}

/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface RunSubmittedEvent {
  /** Ids the backend accepted, in submission order. */
  readonly promptIds: readonly string[]
  /** The accepted prompts and how many backend nodes each will execute. */
  readonly submissions?: readonly RunSubmission[]
  /** How many submissions the backend refused. */
  readonly rejected: number
}

/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface RunSubmission {
  readonly promptId: string
  readonly nodeCount: number
}

/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface RunRejectionError {
  readonly type: string
  readonly message: string
  readonly details: string
  readonly inputName?: string
}

/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface RunRejectedNode {
  readonly nodeId: string
  readonly nodeType: string
  readonly errors: readonly RunRejectionError[]
}

/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface RunRejectedEvent {
  readonly status?: number
  readonly error: RunRejectionError
  readonly nodeErrors: readonly RunRejectedNode[]
}

/** @knipIgnoreUnusedButUsedByCustomNodes */
export type AutoQueueMode = 'disabled' | 'change' | 'instant'

export interface QueueHandle {
  /**
   * Queues the current workflow, exactly as pressing Run does.
   *
   * Resolves once the prompt has been submitted — not when it finishes
   * executing. `false` means another queue call was already in flight and this
   * one was folded into it.
   */
  run(options?: RunOptions): Promise<boolean>
  /**
   * A run is about to be submitted.
   *
   * This is `beforeQueuing`. For a last write before the prompt is built —
   * syncing a value the pack keeps outside the widget. Keep it synchronous:
   * the prompt build does not wait, so work started here can lose the race.
   */
  /**
   * Return a function to have it run when the attempt is over — whether the
   * run started, was refused, or threw.
   *
   * For a pack that changes the graph to build the prompt and must put it back:
   * unmute a branch, let the prompt be built, re-mute it. Pairing it with the
   * setup rather than publishing a second top-level event is deliberate — you
   * cannot receive the cleanup without having run the setup, and there is no
   * second "after" member to confuse with {@link onAfterRun}, which means
   * something different and narrower.
   */
  onBeforeRun(listener: () => (() => void) | void): Unsubscribe
  /**
   * A run was submitted. This is `afterQueued` — for advancing state that
   * should differ on the next run.
   *
   * The event names what the backend accepted, so a pack can tie its own
   * progress tracking to the run it started rather than guessing that the next
   * execution message belongs to it. Each submission includes the exact count
   * of executable backend nodes without exposing the built prompt. `rejected`
   * is how many submissions the backend refused: `onBeforeRun` fires either
   * way, so without this a pack cannot tell a run that started from one that
   * never did.
   */
  onAfterRun(listener: (event: RunSubmittedEvent) => void): Unsubscribe
  /**
   * The backend refused a submitted prompt before execution began.
   *
   * This exposes prompt and per-node validation details without coupling a
   * pack to host notifications. It does not fire for transport failures or an
   * error raised after execution starts.
   */
  onRejected(listener: (event: RunRejectedEvent) => void): Unsubscribe
  /**
   * How many runs are waiting, including the one executing.
   *
   * Packs tracked this from the backend's own `status` message to re-implement
   * `app.ui.lastQueueSize` — deciding whether a button says Run or Cancel,
   * whether an auto-runner should submit again.
   */
  pending(): number
  /** Fires whenever {@link pending} changes, with the new count. */
  onPendingChanged(listener: (pending: number) => void): Unsubscribe
  /**
   * Cancels the run in progress. The rest of the queue is untouched.
   *
   * Packs wrapped `api.interrupt` both to call it and to notice one — a node
   * waiting on the user needs to stop waiting when the run is cancelled.
   * {@link onInterrupted} is that second half.
   */
  interrupt(): Promise<void>
  /** Execution was interrupted, by this pack, another, or the user. */
  onInterrupted(listener: () => void): Unsubscribe
  /** The user-facing automatic queue mode. Both internal instant states read as `instant`. */
  autoQueueMode(): AutoQueueMode
  /** Changes automatic queuing. `instant` arms continuous execution. */
  setAutoQueueMode(mode: AutoQueueMode): void
  /** The batch count the host's own Run action will use. */
  batchCount(): number
  /** Changes the host Run action's batch count. */
  setBatchCount(count: number): void
  /**
   * Turns off automatic queuing without cancelling the current run.
   *
   * A conditional workflow can use this before interrupting itself so the
   * stopped iteration does not immediately start again.
   */
  disableAutoQueue(): void
  /**
   * Holds a run until a check finishes, and can cancel it.
   *
   * {@link onBeforeRun} only observes: it is a notification, and the prompt
   * build does not wait. Packs that needed to *stop* a run — confirm an
   * incoming prompt, validate a field, warn about a cost — wrapped
   * `app.queuePrompt` to do it, which is the surface being retired.
   *
   * Return `false` to cancel. Every guard runs, and any one `false` cancels;
   * the user is not asked twice.
   *
   * A guard that never settles would make the application unrunnable, so one
   * that takes longer than a few seconds is abandoned and the run proceeds. Do
   * not put a dialog with no timeout behind this.
   */
  guard(check: () => boolean | Promise<boolean>): Unsubscribe
}

// ─── resolution.ts ───────────────────────────────────────────────

/**
 * "Whatever feeds this input." The only way one resolution names another.
 *
 * @knipIgnoreUnusedButUsedByCustomNodes
 */
export interface InputRef {
  readonly nodeId: string
  readonly input: number
}

export type OutputResolution =
  | { readonly omit: true }
  | { readonly forwardTo: InputRef }
  | { readonly literal: WidgetValue }

/**
 * What a resolver may see. Reads only — there is nothing here that writes.
 *
 * @knipIgnoreUnusedButUsedByCustomNodes
 */
export interface ResolvedNodeView {
  readonly id: string
  readonly type: string
  /**
   * The node's own properties, frozen.
   *
   * A broadcaster keeps its per-node opt-in here — cg-use-everywhere reads
   * `properties.ue_properties` to decide what it may feed. Candidate inputs
   * already carry `nodeProperties`, so without this a supplier could read
   * every node's configuration except its own.
   */
  readonly properties: Readonly<Record<string, unknown>>
  /** The groups this node sits inside — the other half of "my group". */
  readonly groups: readonly GroupMembership[]
  /** Muted, bypassed or normal, as `LGraphEventMode`. */
  readonly mode: number
  readonly color: string | undefined
  /**
   * This node's own inputs.
   *
   * `unconnectedInputs()` already describes every *other* node's slots, and a
   * supplier needs the same of its own: "send whatever is plugged into me to
   * every unconnected input of the same type" cannot be written without
   * knowing what type is plugged in. Without it a supplier is type-blind and
   * would feed a CLIP into a MODEL slot in silence.
   *
   * `type` is the slot's declared type; `connectedType` is what actually
   * arrives, resolved through reroutes, and is undefined when nothing is
   * connected.
   */
  readonly inputs: readonly OwnInput[]
  /** This node's own outputs, in slot order. */
  readonly outputs: readonly OwnOutput[]
  widgetValue(name: string): WidgetValue | undefined
  input(ref: string | number): InputRef | undefined
}

/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface ResolveView {
  readonly self: ResolvedNodeView
  nodesOfType(type: string): readonly ResolvedNodeView[]
}

/**
 * May answer asynchronously: a sandboxed pack's resolver runs in a worker, so
 * its answer can only arrive as a promise. The prompt path awaits it; the
 * synchronous entry points (`input.resolvedSource()`, `resolvedSupplies()`)
 * treat a promise as unresolved and say so — see `resolution.async.test.ts`.
 */
export type Resolver = (
  view: ResolveView
) =>
  | Record<string, OutputResolution>
  | Promise<Record<string, OutputResolution>>

/** Where an output ends up after every frontend node in the chain resolves. */
export type ResolvedSource =
  | {
      readonly kind: 'output'
      readonly nodeId: string
      readonly output: number
    }
  | { readonly kind: 'literal'; readonly value: WidgetValue }
  | { readonly kind: 'omitted'; readonly reason: string }

/** An input in the graph that no link feeds. */
/** One of a node's own inputs, as its supplier sees it. */
/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface OwnInput {
  readonly index: number
  readonly name: string
  /** What the user sees — `label`, else `localized_name`, else `name`. */
  readonly label: string
  readonly type: string
  readonly connected: boolean
  /** The type actually arriving, or undefined when nothing is connected. */
  readonly connectedType: string | undefined
  /** The node feeding this input, if any. */
  readonly sourceNodeId: string | undefined
}

/** One of a node's own outputs, as its supplier sees it. */
/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface OwnOutput {
  readonly index: number
  readonly name: string
  /** What the user sees — `label`, else `localized_name`, else `name`. */
  readonly label: string
  readonly type: string
}

/** A group a node sits inside. */
/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface GroupMembership {
  readonly id: string
  readonly title: string
}

export interface UnconnectedInput {
  readonly nodeId: string
  readonly nodeType: string
  readonly input: number
  readonly name: string
  readonly type: string
  /**
   * What the user actually sees on the slot — `label`, else `localized_name`,
   * else `name`. Broadcast packs match against this, not `name`, and the two
   * differ in every non-English locale.
   */
  readonly label: string
  /** The socket form of a widget rather than a plain input. */
  readonly isWidgetInput: boolean
  /** The owning node, for matching by title, mode, colour, or opt-in flags. */
  readonly nodeTitle: string
  readonly nodeMode: number
  readonly nodeColor: string | undefined
  /**
   * The groups the owning node sits inside, innermost first.
   *
   * Broadcast packs restrict by group — "only nodes in my group", "only nodes
   * outside it", "only groups whose title matches this regex". Membership is
   * geometric and recomputed here, so it matches what the user sees rather
   * than anything stored.
   */
  readonly nodeGroups: readonly GroupMembership[]
  /**
   * The owning node's properties, frozen.
   *
   * Broadcast packs keep their per-node opt-in here — which inputs a user has
   * allowed to be fed. Without it a supplier can only match by type and would
   * feed every unconnected input of that type, which is the silent
   * wrong-broadcast failure this view exists to prevent.
   */
  readonly nodeProperties: Readonly<Record<string, unknown>>
}

/**
 * An edge a node supplies into somebody else's unconnected input.
 *
 * `from` is the supplier's own output index, or a literal. It is deliberately
 * not an arbitrary node reference: a node may only offer what it itself has,
 * so one pack cannot rewire two other nodes to each other.
 */
export interface SuppliedEdge {
  readonly to: InputRef
  /**
   * Which claim wins when several suppliers name the same input. Higher wins;
   * defaults to 0.
   *
   * **Equal claims feed nothing.** Two suppliers that both say "highest
   * priority" for one input have no correct answer, and picking either makes
   * the prompt depend on node order — so the input is left unfed and the
   * conflict logged. That is what the broadcast pack this exists for does, and
   * it is the only choice that cannot silently produce a different image.
   */
  readonly priority?: number
  readonly from:
    | { readonly output: number }
    | { readonly literal: WidgetValue }
    /**
     * Whatever feeds this node's own input `k` — for a node that rebroadcasts
     * its upstream rather than producing a value.
     *
     * The broadcast nodes this exists for have inputs and **no outputs**, so
     * `{ output: n }` cannot describe them: it would name a slot the backend
     * never declared and force it to execute a node that produces nothing.
     * Resolved exactly as `Resolver`'s `forwardTo`, so it chains through
     * reroutes for free.
     */
    | { readonly forwardInput: number }
}

export interface SupplyView {
  readonly self: ResolvedNodeView
  nodesOfType(type: string): readonly ResolvedNodeView[]
  /**
   * Every unfed input in the graph — what a broadcaster matches against by
   * type, by name, or by its own regex.
   */
  unconnectedInputs(): readonly UnconnectedInput[]
}

/**
 * Answers "what do I feed", the mirror of `Resolver`'s "what feeds me".
 *
 * `Resolver` is demand-side: it is asked about the resolver's own outputs, and
 * is never called for a node with none. cg-use-everywhere broadcasts a value
 * into every matching unconnected input in the graph, which that shape cannot
 * express at all — the nodes being fed are not the resolver, and the edges are
 * discovered rather than declared. Hence a second, supply-side pass.
 *
 */
/** May answer asynchronously, under the same rules as {@link Resolver}. */
export type Supplier = (
  view: SupplyView
) => readonly SuppliedEdge[] | Promise<readonly SuppliedEdge[]>

/**
 * One winning supply after priority arbitration and source resolution.
 */
export interface ResolvedSupply {
  /** The node whose supplier offered this edge. */
  readonly supplierNodeId: string
  /** The unconnected input the supplier won. */
  readonly to: InputRef
  /** The final source the prompt builder will use. */
  readonly from: ResolvedSource
}

// ─── settingsHandle.ts ───────────────────────────────────────────

/** @knipIgnoreUnusedButUsedByCustomNodes */
export type SettingValue = string | number | boolean | readonly string[]

/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface SettingDef {
  /**
   * Namespaced, by convention `<Pack>.<name>` — it shares one space with core
   * and every other pack, and it is what the value is stored under forever.
   */
  readonly id: string
  readonly name: string
  /**
   * Which control the panel shows. Every one of these is declarative — the
   * host renders it.
   *
   * A pack-supplied renderer is deliberately absent. Core's own setting type
   * accepts a function that is handed the value and a setter and returns an
   * element; publishing that would put packs in charge of the settings
   * panel's markup, which is the thing that cannot then be restyled. Packs
   * that needed a colour or a file were falling back to a text field the user
   * pasted into, so the gap was the missing *types*, not a missing slot.
   */
  readonly type:
    | 'boolean'
    | 'number'
    | 'slider'
    | 'knob'
    | 'combo'
    | 'radio'
    | 'text'
    | 'password'
    | 'color'
    | 'image'
    | 'url'
  readonly defaultValue: SettingValue
  readonly tooltip?: string
  /** Panel grouping. Defaults to the id split on dots. */
  readonly category?: readonly string[]
  /**
   * Choices for `combo` and `radio`.
   *
   * A bare string is both the stored value and the label. Use the pair form
   * when they differ — several packs store a semantic number and show words
   * for it (`0` = off, `1` = selected, `2` = all), and comparing those
   * numerically is the whole point. Flattening them to strings silently
   * re-types every user's saved choice.
   */
  readonly options?: readonly SettingOption[]
  /**
   * Bounds for `number` and `slider`. Without these a slider has no range to
   * draw and packs fall back to a plain text box.
   */
  readonly attrs?: SettingAttrs
  readonly onChange?: (value: SettingValue, previous?: SettingValue) => void
}

/** @knipIgnoreUnusedButUsedByCustomNodes */
export type SettingOption =
  | string
  | { readonly value: string | number; readonly label: string }

/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface SettingAttrs {
  readonly min?: number
  readonly max?: number
  readonly step?: number
}

export interface SettingsHandle {
  /**
   * Registers a setting. Call once, at extension load: a value already stored
   * for this id survives, so re-declaring cannot reset a user's choice.
   */
  declare(def: SettingDef): void
  get<T extends SettingValue = SettingValue>(id: string): T | undefined
  set(id: string, value: SettingValue): Promise<void>
  /**
   * Watches a setting, including one the pack did not declare.
   *
   * `declare`'s own `onChange` only fires for settings the pack owns, so a
   * pack that needs to react to a *core* preference — colour palette, link
   * render mode, locale — had nothing to observe and polled or ignored it.
   *
   * Fires on change only, not on registration. Returns a function that stops
   * watching; call it from wherever the pack tears down.
   */
  onChange<T extends SettingValue = SettingValue>(
    id: string,
    listener: (value: T | undefined, previous: T | undefined) => void
  ): Unsubscribe
}

// ─── slotHandle.ts ───────────────────────────────────────────────

export interface LinkInfo {
  readonly id: string
  readonly sourceNodeId: string
  readonly sourceSlotId: SlotId
  readonly targetNodeId: string
  readonly targetSlotId: SlotId
  readonly type: string
  /** Position at snapshot time. Do not store across mutations. */
  readonly sourceIndex: number
  readonly targetIndex: number
}

/**
 * Fields a pack may change on an existing slot.
 *
 * Applied atomically as one command, so a retype-plus-rename is a single undo
 * step rather than two. Retyping deliberately **keeps existing links**: dynamic
 * retyping (`*` -> `MODEL`) is the whole point for `SetNode`-style packs, and
 * silently dropping connections is the failure mode this API exists to end.
 *
 * @knipIgnoreUnusedButUsedByCustomNodes
 */
/**
 * A slot's type, which may be a union.
 *
 * An array spells "this slot accepts any of these" — rgthree's
 * `addInput('input', ['IMAGE', 'LATENT', 'MASK'])` is the shipped example, so
 * packs do write it even though litegraph's own `ISlotType` says
 * `number | string`.
 *
 * Both forms are accepted and stored as the comma string, because that is what
 * litegraph compares against: it normalises with `String(type).split(',')`, so
 * `['IMAGE','LATENT','MASK']` and `'IMAGE,LATENT,MASK'` are the same slot to
 * every connection check. The saved workflow therefore holds the string where
 * the original held an array — a byte difference with no behavioural one, and
 * the same call already taken for slot `shape`.
 *
 * Reads stay `string` for the same reason.
 * @knipIgnoreUnusedButUsedByCustomNodes
 */
export type SlotType = string | string[]

/** @knipIgnoreUnusedButUsedByCustomNodes */
export type SlotDirection = 'none' | 'up' | 'down' | 'left' | 'right' | 'center'

/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface SlotPosition {
  readonly x: number
  readonly y: number
}

/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface SlotPatch {
  name?: string
  label?: string | undefined
  /** The backend-provided translated caption. Null clears it. */
  localizedName?: string | null
  type?: SlotType
  /** Slot centre relative to the node body. Null restores automatic layout. */
  position?: SlotPosition | null
  /** Direction in which links leave the slot. Null restores the default. */
  direction?: SlotDirection | null
  /**
   * The dot's colour when connected and when not.
   *
   * Not decoration, despite appearances: both sit on `INodeSlot` and
   * `ISerialisableNodeInput` omits only `boundingRect`, `widget` and `link`,
   * so they are written into the saved workflow. A pack that coloured its
   * slots and then stopped saves different bytes than it used to.
   *
   * `null` clears one back to the renderer's default.
   */
  color?: string | null
  colorWhenUnconnected?: string | null
  /**
   * Sits on the same `INodeSlot` as the colours above and is omitted by the
   * same `Omit`, so the argument made for them holds verbatim: a pack that
   * shaped its slots and then stopped saves different bytes than it used to.
   *
   * `'default'` clears it back to the renderer's own choice.
   */
  shape?: SlotShape
}

/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface InputSlotPatch extends SlotPatch {
  /** Retargets the widget this input is the socket form of. Null clears it. */
  widget?: string | null
  /** Replaces the input declaration used by connected Primitive nodes. */
  widgetConfig?: InputWidgetConfig
}

/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface InputWidgetConfig {
  /** Backend input type, or the choices for a COMBO input. */
  readonly type: string | readonly (string | number)[]
  readonly options?: Readonly<Record<string, unknown>>
}

/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface SlotSnapshot {
  readonly id: SlotId
  readonly index: number
  readonly name: string
  readonly type: string
  readonly label: string | undefined
  readonly localizedName: string | undefined
  readonly position: SlotPosition | undefined
  readonly direction: SlotDirection | undefined
  readonly shape: SlotShape
  readonly isConnected: boolean
}

/** @knipIgnoreUnusedButUsedByCustomNodes */
export type ResolvedInputSource =
  | {
      readonly kind: 'output'
      readonly graphId: string
      readonly nodeId: string
      readonly outputIndex: number
    }
  | { readonly kind: 'literal'; readonly value: WidgetValue }
  | { readonly kind: 'omitted'; readonly reason: string }

export interface InputSlotHandle {
  readonly id: SlotId
  /** Volatile — shifts when other slots are added or removed. */
  readonly index: number
  readonly name: string
  readonly type: string
  readonly label: string | undefined
  readonly isConnected: boolean
  /** The type arriving through the link, including across a subgraph input. */
  readonly connectedType: string | undefined
  /** Whether this input is the socket form of a widget. */
  readonly isWidgetInput: boolean
  /** The declaration a connected Primitive node renders. */
  widgetConfig(): Readonly<InputWidgetConfig> | undefined
  /** Intersects this input's declaration with another compatible one. */
  mergeWidgetConfig(
    config: InputWidgetConfig
  ): Readonly<InputWidgetConfig> | undefined
  link(): LinkInfo | undefined
  source(): { nodeId: string; outputIndex: number } | undefined
  /**
   * What ultimately feeds this input after frontend nodes resolve.
   *
   * `source()` reports the physical link, which is right for editing topology.
   * This reports the executable source through reroutes, Get/Set nodes and any
   * other frontend node declared with `defs.define({ resolve })`. Resolution is
   * read-only and leaves the graph untouched.
   */
  resolvedSource(): ResolvedInputSource | undefined
  disconnect(): boolean
  modify(patch: InputSlotPatch): void
  /** Replaces `{...input}`, which now yields nothing useful. */
  snapshot(): Readonly<SlotSnapshot>
}

export interface OutputSlotHandle {
  readonly id: SlotId
  readonly index: number
  readonly name: string
  readonly type: string
  readonly label: string | undefined
  readonly isConnected: boolean
  /** Frozen snapshot — safe to iterate while disconnecting. */
  links(): readonly LinkInfo[]
  targets(): readonly { nodeId: string; inputIndex: number }[]
  connectTo(targetNodeId: string, input: SlotRef): LinkInfo | undefined
  disconnect(targetNodeId?: string): boolean
  modify(patch: SlotPatch): void
  /**
   * Moves every link on this output to another output of the same node,
   * **preserving link ids**.
   *
   * Disconnect-and-reconnect is not equivalent: it allocates new ids, so the
   * serialized workflow changes. Packs that re-home their own outputs during a
   * migration depend on identity being kept.
   *
   * Slot types are **not** re-validated. The real-world sequence moves links
   * off an output and then retypes it, so enforcing compatibility mid-move
   * would reject exactly the case this exists for.
   */
  moveLinksTo(target: SlotRef): readonly LinkInfo[]
  snapshot(): Readonly<SlotSnapshot>
}

export interface SlotCollection<THandle> {
  readonly length: number
  get(ref: SlotRef): THandle | undefined
  byId(id: SlotId): THandle | undefined
  byName(name: string): THandle | undefined
  /** Explicit positional access. */
  at(index: number): THandle | undefined
  all(): readonly THandle[]
  ids(): readonly SlotId[]
  names(): readonly string[]
  /**
   * Adds a slot. 18 packs grow their inputs as the last one fills — the
   * "Multi" combiner pattern — which needed `node.addInput` until now.
   *
   * `shape` is not decoration: it is written into the saved workflow, so a
   * slot added without the one its pack used to set serialises differently
   * from one the pack itself wrote. `'optional'` is the hollow circle
   * ComfyUI draws for an input that need not be connected.
   */
  add(name: string, type: SlotType, options?: SlotOptions): THandle
  /**
   * Removes a slot by reference. Any link into it is dropped, as it would be
   * on the legacy path.
   */
  remove(ref: SlotRef): boolean
  /**
   * Puts the slots in the given order. `names` must be a permutation of the
   * current ones.
   *
   * Every link into or out of this node is re-pointed as part of the move, in
   * one batch, so link ids — and therefore the saved workflow's `links` array
   * — are unchanged. That is the whole reason this exists rather than being
   * left to packs: a link stores its endpoint as a slot *index*, so a pack
   * permuting the array itself silently re-points every connection, and the
   * damage only shows when the workflow is next run.
   *
   * The slot *order* is serialized, so this changes the saved file by design —
   * it is how a pack keeps its dynamic inputs matching what the backend
   * declares.
   */
  reorder(names: readonly string[]): void
  [Symbol.iterator](): Iterator<THandle>
}

/**
 * How a slot is drawn, which ComfyUI overloads to mean how it behaves.
 *
 * Named rather than numbered: packs wrote `{ shape: 7 }`, and 7 is meaningless
 * without litegraph's RenderShape enum in front of you.
 */
export type SlotShape = 'default' | 'optional' | 'list' | 'directional'

/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface SlotOptions {
  /**
   * `'optional'` is the hollow circle for an input that need not be connected,
   * `'list'` the grid ComfyUI draws for an output that yields many values, and
   * `'directional'` the arrow a pack uses for a slot that only ever feeds one
   * particular kind of node.
   */
  shape?: SlotShape
  localizedName?: string
  position?: SlotPosition
  direction?: SlotDirection
  /**
   * Names the widget this slot is the socket form of — the "convert widget to
   * input" shape.
   *
   * Not decoration either: a slot carrying it serialises as
   * `{ widget: { name } }` where a plain socket serialises as `{ pos }`, and
   * the widget keeps its place in `widgets_values`. A dynamic input added
   * without it changes the saved file.
   */
  widget?: string
  /** The declaration a connected Primitive node should render. */
  widgetConfig?: InputWidgetConfig
}

// ─── slotRef.ts ──────────────────────────────────────────────────

export type SlotId = string & { readonly __brand: 'SlotId' }

/**
 * A slot reference: a string (id or name), or an explicit `{ index }`.
 *
 * A bare `number` is deliberately not accepted so positional access is visible
 * at the call site and greppable:
 *
 *     output.connectTo(node, 'image')       // by name — preferred
 *     output.connectTo(node, { index: 0 })  // by position — explicit
 */
export type SlotRef = SlotId | string | { readonly index: number }

export interface ResolveOptions {
  /**
   * Whether the backend supplies slot names yet. While false, a canonical
   * integer string resolves positionally, so `'0'` addresses slot 0 and call
   * sites need no rewrite once names arrive.
   *
   * Retire this together with the release that ships names — until then a pack
   * passing `'2'` meaning a name would silently bind slot 2.
   */
  readonly namedSlotsAvailable: boolean
}

// ─── storageHandle.ts ────────────────────────────────────────────

/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface StorageUsage {
  /** Total bytes stored under the namespace. */
  readonly usedBytes: number
  /** How many entries make up {@link usedBytes}. */
  readonly entryCount: number
  /**
   * The ceiling this host enforces, or `undefined` where it enforces none.
   *
   * Undefined is the honest answer for a local install with the user's own
   * disk behind it, and it is deliberately not reported as `Infinity`: a pack
   * dividing by it to draw a gauge would get a meaningless bar rather than the
   * chance to skip drawing one. Do not treat a present number as a promise
   * that a write below it succeeds — another namespace shares the same store.
   */
  readonly quotaBytes?: number
}

export interface StorageHandle {
  /**
   * Names stored under a namespace, which must be one this pack owns.
   *
   * Empty when nothing has been stored yet — absence is not an error.
   */
  list(namespace: string): Promise<readonly string[]>
  /** The stored text, or `undefined` if there is none. */
  get(name: string): Promise<string | undefined>
  set(name: string, value: string): Promise<void>
  remove(name: string): Promise<void>
  /**
   * What a namespace currently occupies.
   *
   * For a pack that stores things a user accumulates — presets, captions,
   * saved prompts — so it can show what it is holding and offer to prune it,
   * rather than growing without bound until someone else's write fails.
   */
  usage(namespace: string): Promise<StorageUsage>
}

// ─── systemHandle.ts ─────────────────────────────────────────────

/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface SystemMonitorCpu {
  readonly utilization_percent: number | null
}

/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface SystemMonitorMemory {
  readonly total: number
  readonly available: number
}

/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface SystemMonitorVolume extends SystemMonitorMemory {
  readonly id: string
  readonly label: string
}

/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface SystemMonitorAccelerator {
  readonly id: string
  readonly name: string
  readonly memory_total: number
  readonly memory_available: number
  readonly utilization_percent: number | null
  readonly temperature_c: number | null
}

/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface SystemMonitorSnapshot {
  readonly cpu: SystemMonitorCpu
  readonly memory: SystemMonitorMemory
  readonly volumes: readonly SystemMonitorVolume[]
  readonly accelerators: readonly SystemMonitorAccelerator[]
}

export interface SystemHandle {
  /**
   * Returns one host-sampled hardware snapshot. Volume ids are opaque and
   * unsupported utilization or temperature sensors are null.
   */
  monitor(): Promise<SystemMonitorSnapshot>
}

// ─── uiHandle.ts ─────────────────────────────────────────────────

/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface SidebarTabBase {
  /**
   * Unique across every pack, so namespace it — `'mtb.assets'`, not
   * `'assets'`. Registering an id twice throws rather than silently replacing
   * the other pack's tab.
   */
  readonly id: string
  readonly title: string
  /**
   * An iconify class, e.g. `'icon-[lucide--activity]'`. Omit for no icon.
   */
  readonly icon?: string
  readonly tooltip?: string
}

/**
 * A tab the pack draws into a container itself.
 *
 * Framework-agnostic, and the only form available to a pack that ships
 * hand-written ES modules with no build step — which is most of them.
 */
/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface MountedSidebarTab extends SidebarTabBase {
  /**
   * Fills the tab's panel. Called each time the tab becomes visible, so treat
   * it as mount rather than as one-time setup, and put teardown in `destroy`.
   */
  render(container: HTMLElement): void
  /** Releases what `render` retained — listeners, timers, observers. */
  destroy?(): void
}

/**
 * A tab that is a Vue component, mounted and torn down by the host.
 *
 * The preferred form where a pack can build. It keeps reactivity, scoped
 * styles and `onUnmounted`, and the host mounts and unmounts it.
 *
 * Per ADR 0005 the pack bundles its own Vue (~30KB gzipped) — there is no
 * import map, so `import { defineComponent } from 'vue'` resolves at the
 * pack's build time, not ours. That is a second Vue instance on the page,
 * which the ADR weighed and accepted; nothing is shared across the boundary,
 * so the two runtimes never touch.
 */
/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface VueSidebarTab extends SidebarTabBase {
  readonly component: VueComponent
}

/** @knipIgnoreUnusedButUsedByCustomNodes */
export type SidebarTabDef = MountedSidebarTab | VueSidebarTab

/** A Vue component bundled by the pack. */
/** @knipIgnoreUnusedButUsedByCustomNodes */
export type VueComponent = object

/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface DialogBase {
  /**
   * Unique across every pack, so namespace it. The host prefixes it with
   * `extension-`, which keeps packs out of the internal dialog keyspace.
   */
  readonly key: string
  readonly title?: string
}

/** A bounded keyboard event captured while a mounted dialog owns focus. */
/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface DialogKeyEvent {
  readonly key: string
  readonly code: string
  readonly repeat: boolean
  readonly altKey: boolean
  readonly ctrlKey: boolean
  readonly metaKey: boolean
  readonly shiftKey: boolean
  /** True for an input, textarea, select, or editable content target. */
  readonly editableTarget: boolean
}

/** A dialog the pack draws into a container itself. */
/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface MountedDialog extends DialogBase {
  render(container: HTMLElement): void
  /** Receives dialog-scoped key events even before a child takes focus. */
  onKeyDown?(event: DialogKeyEvent): void | Promise<void>
  destroy?(): void
}

/** A dialog that is a Vue component, mounted and torn down by the host. */
/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface VueDialog extends DialogBase {
  readonly component: VueComponent
  readonly props?: Readonly<Record<string, unknown>>
}

/** @knipIgnoreUnusedButUsedByCustomNodes */
export type DialogDef = MountedDialog | VueDialog

/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface DialogHandle {
  close(): void
}

export interface UiHandle {
  /**
   * Adds a tab to the sidebar. Returns a function that removes it again.
   */
  addSidebarTab(def: SidebarTabDef): Unsubscribe
  /**
   * Shows a small readout in the top bar — a status, a count, a live metric.
   *
   * Replaces `app.menu.settingsGroup` and inserting an element next to
   * `.comfy-settings-btn`. Declarative on purpose: the pack says what to show
   * and the host renders it, in house style and at whatever size the viewport
   * allows. Nothing here takes an element, a class or a style, which is what
   * keeps the chrome ours to restyle.
   *
   * Returns a handle rather than an unsubscribe: for a value that changes,
   * call `update({ text })`. A closure would not work — the host renders when
   * reactive state changes and cannot see a plain function, so the readout
   * would show its first value forever.
   */
  addTopBarBadge(badge: BadgeContribution): ChromeItemHandle<BadgeContribution>
  /**
   * Adds a button to the action bar. `run` is called on click.
   *
   * For a pack that also wants a keyboard shortcut or a palette entry,
   * register a command and call it from `run`, rather than duplicating the
   * behaviour in both places.
   */
  addActionBarButton(
    button: ButtonContribution
  ): ChromeItemHandle<ButtonContribution>
  /**
   * Opens a modal dialog. Returns a handle that closes it again.
   *
   * Replaces `app.ui.dialog` and the `new app.ui.dialog.constructor()` idiom.
   * Several conversions hand-rolled a native `<dialog>` or borrowed core's
   * `.comfy-modal` class names instead — the latter couples a pack to markup
   * we rename freely, so both are worth retiring.
   */
  showDialog(def: DialogDef): DialogHandle
  /**
   * Shows a menu where the user clicked.
   *
   * `b.addMenuItem` is the node's own context menu — a different menu, on a
   * different target, opened by the host. This is for a menu a pack raises
   * itself: a lora row's Move Up / Remove, a chip that picks an output type.
   * Four files hand-rolled it by constructing the renderer's menu class
   * directly, which pins them to a renderer we intend to replace.
   *
   * Positioned from the event so the menu lands under the pointer, which is the
   * only placement that reads as a context menu. Arrow keys traverse nested
   * items, Enter or Tab selects one, and Escape closes the menu.
   */
  showMenu(def: MenuDef): MenuHandle
  /**
   * Asks the user for a value. Resolves `undefined` if they cancel.
   *
   * Packs called `canvas.prompt(...)`, which draws a small field at the cursor
   * — clicking a lora's strength to type a new one. That field belongs to the
   * legacy canvas and the host itself no longer uses it; this is the prompt the
   * host does use, so a pack keeps the capability and loses only the placement.
   */
  prompt(def: PromptDef): Promise<string | undefined>
}

/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface PromptDef {
  /** What is being asked for — "Strength", "Label". */
  readonly label: string
  readonly value?: string
  readonly placeholder?: string
}

/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface MenuItemDef {
  readonly label: string
  /** Shown but not selectable. */
  readonly disabled?: boolean
  /** A nested menu. Mutually exclusive with {@link run}. */
  readonly submenu?: readonly MenuItemDef[]
  run?(): void
}

/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface MenuDef {
  readonly items: readonly MenuItemDef[]
  /** Shown above the items. */
  readonly title?: string
  /** The event that asked for the menu; it decides where the menu appears. */
  readonly event: MouseEvent
}

/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface MenuHandle {
  close(): void
}

// ─── widgetHandle.ts ─────────────────────────────────────────────

// `null` is included because core's own `WidgetValue` has it and
// `addWidget('button', name, null, cb)` produced exactly that. Omitting it made
// a null value inexpressible through the published API, so a converted button's
// `widgets_values` entry changed and the saved workflow differed.
export type WidgetValue = string | number | boolean | object | undefined | null

/** Options understood by core or by a widget type declared by the pack. */
/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface WidgetOptions {
  readonly [key: string]: unknown
  readonly on?: string
  readonly off?: string
  readonly max?: number
  readonly min?: number
  readonly precision?: number
  readonly read_only?: boolean
  readonly step?: number
  readonly step2?: number
  readonly multiline?: boolean
  readonly property?: string
  readonly socketless?: boolean
  readonly canvasOnly?: boolean
  readonly hideInPanel?: boolean
  readonly nodeType?: string
  readonly serialize?: boolean
  readonly values?: unknown
  readonly iconClass?: string
  readonly disabled?: boolean
  readonly useGrouping?: boolean
  readonly placeholder?: string
  readonly showThumbnails?: boolean
  readonly showItemNavigators?: boolean
  readonly hidden?: boolean
}

/**
 * Shapes follow `src/types/extensionV2.ts`, the agreed extension contract.
 *
 * Accessor methods rather than properties, so a read can be a store query and
 * a write can dispatch a command.
 */
export interface WidgetHandle extends HandleCommon {
  readonly name: string
  readonly widgetType: string

  getValue<T = WidgetValue>(): T
  /**
   * Commits a value exactly as a user edit does: the value is written, a
   * widget bound to a node property syncs it, the widget's callback chain and
   * the node's `onWidgetChanged` run, and `graph.version` advances. This
   * replaces the manual pair `widget.value = x; widget.callback?.(x)` — and
   * the bare write too, because a write the rest of the system cannot see was
   * never a feature, it was litegraph defaulting to inconsistency.
   *
   * Writing the current value again is a no-op, which is also what ends a
   * cycle of handlers writing to each other. `on('change')` fires once per
   * commit; `on('activate')` does not fire, because activate reports a user's
   * act.
   */
  setValue(value: WidgetValue): void

  /**
   * The widgets core attached to this one — a seed's `control_after_generate`,
   * a bounding box's components.
   *
   * `setHidden` already cascades through these, so hiding needs no call here.
   * What does is reading one: a pack asks a seed's control widget whether it
   * says `fixed` or `randomize` to know what the node will do next.
   */
  linked(): readonly WidgetHandle[]
  /**
   * Replaces the controls attached to this widget.
   *
   * Core uses this relationship for compound inputs: hiding a seed also hides
   * its `control_after_generate` picker. Packs build the same compound control
   * when they add a random-seed button or an index policy, and assigning
   * `linkedWidgets` directly was the only way to make conversion-to-input hide
   * the whole unit.
   *
   * Every name must identify another widget on this node. Pass an empty array
   * to clear the relationship.
   */
  setLinked(names: readonly string[]): void

  isHidden(): boolean
  /**
   * Replaces the `type = 'converted-widget'` hack. Value is retained.
   *
   * Cascades to the widgets core attached to this one — a seed's
   * `control_after_generate`, a bounding box's components. The legacy
   * `hideWidget` helper this replaces recursed through `linkedWidgets`, and
   * packs that lost the cascade were left with an orphaned control widget
   * floating where its owner used to be.
   */
  setHidden(hidden: boolean): void
  getOptions(): Readonly<WidgetOptions> | undefined
  setOption(key: string, value: unknown): void
  setLabel(label: string): void

  isDisabled(): boolean
  setDisabled(disabled: boolean): void
  isSerialized(): boolean
  /** The height the host most recently allocated, or undefined before layout. */
  getHeight(): number | undefined
  /**
   * Pins the widget's height in graph units, instead of letting it share
   * whatever space the node has spare.
   *
   * The node divides free height between every widget that does not state one,
   * so a node carrying two mounted strips gave each half the node however
   * small they were meant to be. `MountDef.height` does not do this — it sets
   * the container's CSS height *inside* an allocation the renderer already
   * chose, which is why a fixed strip still drifted.
   *
   * Replaces re-assigning `node.computeSize`, which is what packs did and
   * which is not published. Omit it for a panel meant to fill the node: the
   * growable path is the one that fills.
   */
  setHeight(px: number): void

  /**
   * Replaces capture-and-chain on `widget.callback`, which 1,000+ sites do and
   * which silently drops an earlier pack's listener whenever one forgets to
   * call through. Listeners here are additive and independent.
   */
  on(
    event: 'change',
    listener: (value: WidgetValue, oldValue: WidgetValue) => void
  ): Unsubscribe
  on(event: 'removed', listener: () => void): Unsubscribe
  /**
   * The widget was activated — a button click, or a value committed.
   *
   * Buttons carry no value, so `change` can never fire for one and a button
   * created through this API would otherwise be inert. Prefer `change` when you
   * care about the value; use this when you care that the user acted — a
   * programmatic `setValue` never fires it.
   */
  on(event: 'activate', listener: (value: WidgetValue) => void): Unsubscribe
  /**
   * Contributes behavior to a host-owned multiline text editor without exposing
   * its DOM. The event reports the live value and caret on each input,
   * selection change, or wheel gesture; its write method preserves both the
   * widget commit protocol and the requested selection.
   */
  on(
    event: 'textInteraction',
    listener: (event: WidgetTextInteractionEvent) => void
  ): Unsubscribe
  /**
   * The value is about to be written out, and may be replaced for this
   * destination only.
   *
   * This is what `widget.serializeValue` did, and the reason it is back: a
   * static `serialize` flag can only *suppress* a value, and a whole class of
   * packs needs to *supply* a different one. rgthree's Seed keeps the sentinel
   * `-1` in the saved workflow and sends the rolled seed; pysssss' PresetText
   * expands `@name` into the queued prompt while the user keeps seeing the
   * reference; Impact Pack embeds image data the canvas never shows.
   *
   * `context` says which destination is being built, because those packs want
   * to change one and not the other:
   *
   * - `'workflow'` — the file the user saves.
   * - `'prompt'` — the queued API payload the backend executes.
   * - `'embedded'` — the copy of the workflow that travels with that prompt
   *   and is written into the output image. Distinct from `'workflow'`
   *   because a pack may want the image to reproduce the run while the saved
   *   file keeps its sentinel: rgthree's Seed saves `-1` but embeds the seed
   *   it actually rolled, so dragging the PNG back in reproduces it.
   *
   * A handler that ignores `context` changes all three.
   *
   * Calling `setSerializedValue` replaces the value for this write only; the
   * widget itself is untouched, so the user still sees what they typed. Last
   * handler to call it wins.
   */
  on(
    event: 'beforeSerialize',
    listener: (event: WidgetSerializeEvent) => void
  ): Unsubscribe
}

/** Where a value is being written, and the chance to change it. */
export interface WidgetSerializeEvent {
  readonly context: 'workflow' | 'prompt' | 'embedded'
  /** What would be written if no handler intervened. */
  readonly value: WidgetValue
  setSerializedValue(value: WidgetValue): void
}

export type Unsubscribe = () => void

/**
 * A widget whose body the pack renders itself.
 *
 * @knipIgnoreUnusedButUsedByCustomNodes
 */
export interface MountDef {
  readonly name: string
  /**
   * Fills the mounted container. Called once, with an element already attached
   * to the node.
   *
   * `value` holds meaningful serialized state only when `defaultValue` was
   * given. A decorative mount receives the same accessor for one render shape,
   * but should not use it as storage.
   */
  render(container: HTMLElement, value: MountedValue): void
  /** Releases anything `render` retained — listeners, timers, observers. */
  destroy?(): void
  /** Reserved height in graph units. Omit to size to content. */
  readonly height?: number
  /** Set false to keep the element rendered at low zoom. Defaults to true. */
  readonly hideOnZoom?: boolean
  readonly hidden?: boolean
  /**
   * Whether the value is written into the saved workflow.
   *
   * Defaults to `true` when `defaultValue` makes this a value-holding control,
   * and to `false` for a decorative mount.
   */
  readonly serialize?: boolean
  /**
   * Whether the value is sent in the API prompt. Defaults to `serialize`.
   *
   * These are two different flags in litegraph — `widget.serialize` gates the
   * saved workflow, `options.serialize` gates the prompt — and collapsing them
   * into one boolean made two states unsayable. "Saved but not sent" is the
   * one packs need: it is exactly what the legacy
   * `addDOMWidget(…, { serialize: false })` did, and a readout that a node
   * fills in from its own execution result belongs in the workflow but has no
   * business appearing as an input on the next queue.
   *
   * Set it apart from `serialize` only when the two genuinely differ.
   */
  readonly sendToPrompt?: boolean
  /**
   * Makes this a value-holding widget rather than decoration.
   *
   * Without it a mount is a drawing: it can occupy a `widgets_values` slot but
   * has nothing to put in it, so a colour picker or a text box converted onto
   * `mount` kept its position and silently lost what the user typed. Supplying
   * a default gives the widget a real cell, reachable through `render`'s second
   * argument.
   */
  readonly defaultValue?: MountedData
}

/** What a mounted control can hold. @knipIgnoreUnusedButUsedByCustomNodes */
export type MountedData = string | number | boolean | object | null

/**
 * Reading and writing a mounted widget's value.
 *
 * @knipIgnoreUnusedButUsedByCustomNodes
 */
export interface MountedValue {
  get(): MountedData
  set(value: MountedData): void
  /** Notified when the value changed elsewhere — a workflow load. */
  onChange(listener: (value: MountedData) => void): Unsubscribe
}

/**
 * A pointer event on the widget's own canvas, in the same units `draw` uses.
 *
 * @knipIgnoreUnusedButUsedByCustomNodes
 */
export interface CanvasPointerEvent {
  /** Distance from the canvas's left edge, in CSS pixels. */
  readonly x: number
  /** Distance from its top edge, in CSS pixels. */
  readonly y: number
  /** The DOM event, for modifier keys, `button`, and `preventDefault()`. */
  readonly event: PointerEvent
}

/**
 * The colours a pack should draw its own controls in.
 *
 * Published because we told packs to draw. A widget that hardcodes its palette
 * looks wrong the moment the user switches theme, and the alternative — reading
 * `LiteGraph.WIDGET_BGCOLOR` and friends — is a renderer constant we intend to
 * delete. These are the design system's own tokens, resolved from the widget's
 * computed style, so they follow the theme without the pack knowing which one
 * is active.
 *
 * Named by intent rather than by token, because the token names will churn and
 * a pack should not have to follow. Re-read on every draw, so a theme switch
 * needs nothing from the pack.
 *
 * @knipIgnoreUnusedButUsedByCustomNodes
 */
export interface CanvasTheme {
  /** A control's background. */
  readonly surface: string
  /** The same under the pointer. */
  readonly surfaceHovered: string
  /** A control's outline. */
  readonly border: string
  /** A label. */
  readonly text: string
  /** A value, a unit, anything the label outranks. */
  readonly textSecondary: string
}

/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface CanvasDef {
  readonly name: string
  /** Reserved height in pixels. Omit to size to the node's width. */
  readonly height?: number
  draw(
    context: CanvasRenderingContext2D,
    size: readonly [number, number],
    theme: CanvasTheme,
    value: MountedValue | undefined
  ): void
  /**
   * The pointer went down on this widget.
   *
   * Coordinates are relative to the canvas and in the same units `draw`
   * receives, so a hit test written against the drawing works unchanged —
   * which is the point. A pack that drew its own controls keeps both the
   * drawing and the hit testing; only the surface changes, from the host's
   * canvas to its own.
   *
   * The primary button is taken: it stops here rather than also reaching the
   * node, or adjusting a slider would drag the node underneath it. Middle and
   * right are left alone, so panning and the context menu still work over the
   * widget.
   *
   * The pointer is captured for the gesture, so a drag that leaves the widget
   * still reports moves and the release.
   */
  onPointerDown?(event: CanvasPointerEvent): void
  /** Moves during a drag, and hover when no button is down. */
  onPointerMove?(event: CanvasPointerEvent): void
  onPointerUp?(event: CanvasPointerEvent): void
  /**
   * The secondary button went down on this widget.
   *
   * Right-click is left alone by {@link onPointerDown} so the node's own
   * context menu keeps working over a widget, which is right by default and
   * wrong for a widget that has its own menu — a lora row wants Move Up, Move
   * Down, Remove. Declaring this claims the gesture: the browser menu is
   * suppressed and the node's does not open.
   */
  onContextMenu?(event: CanvasPointerEvent): void
  /**
   * Makes the surface hold a value rather than only draw one.
   *
   * Without it a drawn control that stores something has to be two widgets — a
   * hidden value widget and a surface — and two widgets cannot occupy the one
   * position the original had. That is not a tidiness point: `serialize` writes
   * at each widget's own index and leaves a hole where a non-serializing widget
   * sits, so the pair has to be ordered value-first to keep the saved array
   * intact, and a pack that gets that wrong writes a null into every workflow
   * the node has ever appeared in. It moved rgthree's Power Puter chip row
   * below its code box.
   *
   * `draw` receives the current value as its fourth argument.
   */
  readonly defaultValue?: MountedData
  /** Whether the value reaches the saved workflow. See {@link MountDef.serialize}. */
  readonly serialize?: boolean
  /** Whether the value reaches the API prompt. See {@link MountDef.sendToPrompt}. */
  readonly sendToPrompt?: boolean
}

/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface CanvasHandle {
  readonly widget: WidgetHandle
  /** Redraws now. Call when the data behind the drawing changed. */
  redraw(): void
}

/** Everything needed to create a widget. */
export interface WidgetDef {
  readonly type: string
  readonly name: string
  readonly value?: WidgetValue
  readonly options?: WidgetOptions
  /** Display-only widgets — replaces the readOnly/opacity DOM fiddling. */
  readonly disabled?: boolean
  readonly hidden?: boolean
  /**
   * Whether the value is written into the saved workflow.
   *
   * Replaces `widget.serializeValue = async () => {}`, the idiom packs use to
   * keep a derived readout out of `widgets_values`. Orthogonal to `hidden`.
   */
  readonly serialize?: boolean
}

export interface WidgetCollection {
  readonly length: number
  get(name: string): WidgetHandle | undefined
  at(index: number): WidgetHandle | undefined
  all(): readonly WidgetHandle[]
  names(): readonly string[]
  /**
   * Replaces splice/assign reordering. `names` must be a permutation of the
   * current names — a partial list throws rather than silently dropping
   * widgets, which is how the array-splice idiom lost them.
   */
  reorder(names: readonly string[]): void
  move(name: string, toIndex: number): void
  /**
   * Creates a widget on this node.
   *
   * The counterpart to `remove` — packs that rebuild a readout widget do
   * remove-then-create, and without this only half the operation has a
   * destination, which makes the conversion cosmetic.
   */
  add(def: WidgetDef): WidgetHandle
  /**
   * Mounts an element on the node and hands it to the pack to fill.
   *
   * The replacement for `addDOMWidget`, and the destination for hand-painted
   * canvas controls. Across kjnodes' canvas editors the drawing is rectangles,
   * images, straight lines and text — all DOM primitives — but a pack that
   * wants to keep its existing `ctx` code can append a `<canvas>` to the
   * container and carry it over unchanged.
   *
   * The gain is not the drawing, it is the input: these editors hand-roll
   * hit-testing against bounding boxes because canvas gives them nothing to
   * attach a listener to. Mounted in the DOM, pointer events land on the
   * element and most of that code goes away.
   */
  mount(def: MountDef): WidgetHandle
  /**
   * A per-node drawing surface, and the destination for `onDrawForeground`.
   *
   * Works under both renderers without the pack knowing which it is on: the
   * canvas is a DOM element, which the legacy renderer positions over the
   * graph canvas and Nodes 2.0 renders directly. That is the whole reason it
   * is a mounted element rather than a hook into the graph's own context —
   * drawing into the shared context is what ties a pack to the old renderer.
   *
   * `draw` is called on mount, on resize, and whenever `redraw()` is called.
   */
  canvas(def: CanvasDef): CanvasHandle
  remove(name: string): boolean
  [Symbol.iterator](): Iterator<WidgetHandle>
}

export interface ComboPreviewRegistration {
  /** Namespaced registration id. */
  readonly id: string
  /** Managed model catalogues searched in order. */
  readonly modelCategories: readonly (
    | 'loras'
    | 'checkpoints'
    | 'unet'
    | 'diffusion_models'
  )[]
  /** Model filename suffixes that activate this policy. */
  readonly extensions: readonly (
    | 'safetensors'
    | 'sft'
    | 'pt'
    | 'ckpt'
    | 'gguf'
  )[]
  /** Host-owned adjacent-preview lookup policy. */
  readonly candidatePolicy: 'adjacent-model-preview-v1'
  /** Preview media types the host may display. */
  readonly media: readonly (
    | 'image/png'
    | 'image/webp'
    | 'image/jpeg'
    | 'video/mp4'
    | 'video/webm'
  )[]
}

export interface ComboPreviewAssignment {
  /** Managed model catalogue containing `modelValue`. */
  readonly category: 'loras' | 'checkpoints' | 'unet' | 'diffusion_models'
  /** Logical model filename from the managed combo; never a host path. */
  readonly modelValue: string
  /** Graph node whose host-owned output image is used as the preview. */
  readonly sourceNodeId: string
  /** Exact image in that node's current host-owned output list. */
  readonly imageIndex: number
  readonly policy: 'adjacent-model-preview-v1'
}

export interface WidgetsHandle {
  /**
   * Adds a declarative preview policy to host-owned combo option menus.
   * The host resolves managed assets and renders the hover surface; the pack
   * receives neither filesystem paths nor media URLs.
   */
  registerComboPreview(definition: ComboPreviewRegistration): Unsubscribe
  /**
   * Re-encodes one managed graph output as an adjacent managed-model preview.
   * The host resolves both resources; the pack receives no path or image bytes.
   */
  assignComboPreview(assignment: ComboPreviewAssignment): Promise<void>
}

export type LocalizationMessage =
  | string
  | null
  | { readonly [key: string]: LocalizationMessage }

export interface LocalizationCatalog {
  /** Native vue-i18n-shaped messages such as main/nodeDefs/nodeCategories. */
  readonly messages: Readonly<Record<string, LocalizationMessage>>
  /** Exact-source fallback translations used only at host-owned render points. */
  readonly phrases?: Readonly<Record<string, string>>
}

export interface LocalizationHandle {
  /**
   * Contributes one bounded catalog for a host-supported locale. The host
   * owns merging, rendering, precedence, and cleanup; no DOM access is given.
   */
  registerCatalog(locale: string, catalog: LocalizationCatalog): Unsubscribe
}

// ─── widgetTextInteraction.ts ────────────────────────────────────

/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface WidgetTextSelection {
  readonly start: number
  readonly end: number
}

/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface WidgetTextEventBase {
  readonly value: string
  readonly selection: WidgetTextSelection
  /** Positions a host menu at the text editor without exposing its element. */
  readonly menuEvent: MouseEvent
  /** Commits through the widget protocol and optionally restores the caret. */
  setValue(value: string, selection?: WidgetTextSelection): void
  focus(): void
}

/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface WidgetTextInputEvent extends WidgetTextEventBase {
  readonly kind: 'input' | 'selection'
}

/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface WidgetTextWheelEvent extends WidgetTextEventBase {
  readonly kind: 'wheel'
  readonly deltaY: number
  readonly ctrlKey: boolean
  /** Claims the wheel gesture so the canvas does not pan or zoom. */
  preventDefault(): void
}

/** @knipIgnoreUnusedButUsedByCustomNodes */
export interface WidgetTextKeyEvent extends WidgetTextEventBase {
  readonly kind: 'keydown'
  readonly key: string
  readonly ctrlKey: boolean
  readonly altKey: boolean
  readonly shiftKey: boolean
  readonly metaKey: boolean
  readonly repeat: boolean
  preventDefault(): void
  stopPropagation(): void
}

/**
 * An interaction with a host-owned multiline text editor.
 *
 * This is the renderer-independent replacement for reaching through
 * `widget.inputEl`: packs can inspect the live caret, offer a menu through
 * `menuEvent`, replace text, and implement selection-based wheel edits without
 * receiving the host's element or markup.
 */
export type WidgetTextInteractionEvent =
  | WidgetTextInputEvent
  | WidgetTextWheelEvent
  | WidgetTextKeyEvent

// ─── widgetTypes.ts ──────────────────────────────────────────────

/** What a pack-declared widget can hold. */
export type WidgetTypeData = string | number | boolean | object | null

/**
 * Reading and writing the widget's value, for the renderer to bind to.
 *
 * @knipIgnoreUnusedButUsedByCustomNodes
 */
export interface WidgetTypeValue {
  get(): WidgetTypeData
  set(value: WidgetTypeData): void
  /** Notified when the value changes for any other reason — a workflow load. */
  onChange(listener: (value: WidgetTypeData) => void): Unsubscribe
}

export interface WidgetTypeContext {
  /** A frozen snapshot of the input declaration's current options. */
  getOptions(): Readonly<Record<string, unknown>>
  /**
   * Runs while the widget's owning node belongs to a graph.
   *
   * Widget constructors run before a node has an id or graph, so a node handle
   * cannot be supplied directly to `render`. The listener runs after the node
   * joins a graph and tears down when it leaves.
   */
  onNodeReady(listener: (node: NodeHandle) => Unsubscribe | void): Unsubscribe
}

export interface WidgetTypeDef {
  /** Used when the definition supplies none. */
  readonly defaultValue?: WidgetTypeData
  /** Height in pixels. Omit to size to content. */
  readonly height?: number
  /** Smallest width the control needs, in pixels. */
  readonly minWidth?: number
  /** Smallest height the control needs, in pixels. */
  readonly minHeight?: number
  /**
   * Whether the value is saved and sent. Defaults to `true`: this widget holds
   * a real input value, unlike a mounted decoration.
   */
  readonly serialize?: boolean
  /**
   * Fills the container. Return a teardown if the control owns listeners,
   * timers or observers.
   *
   * `name` is the input being rendered — controls commonly label themselves
   * with it, which a type-level renderer otherwise has no way to know.
   */
  render(
    container: HTMLElement,
    value: WidgetTypeValue,
    name: string,
    context: WidgetTypeContext
  ): Unsubscribe | void
}

// ─── workflowHandle.ts ───────────────────────────────────────────

/** Parsed ComfyUI workflow JSON. */
export type WorkflowData = Readonly<Record<string, unknown>>

export interface WorkflowImportContext {
  readonly name: string
  readonly type: string
}

export type WorkflowImportResult =
  | { readonly workflow: WorkflowData | string }
  | { readonly prompt: Readonly<Record<string, unknown>> | string }

export interface WorkflowImporter {
  /** Namespaced and unique within the pack. */
  readonly id: string
  readonly mimeTypes?: readonly string[]
  readonly extensions?: readonly string[]
  /** Per-file limit; the host-wide ceiling is 16 MiB. */
  readonly maxBytes: number
  enabled?(): boolean | Promise<boolean>
  parse(
    bytes: Uint8Array,
    context: WorkflowImportContext
  ):
    | WorkflowImportResult
    | null
    | undefined
    | Promise<WorkflowImportResult | null | undefined>
}

export interface WorkflowHandle {
  /** Replaces the active document with parsed ComfyUI workflow JSON. */
  open(data: WorkflowData): Promise<void>
  /** Returns the current saved-format workflow, bounded to 8 MiB. */
  snapshot(): Promise<WorkflowData>
  /** Registers a bounded worker-side parser for host-opened or dropped files. */
  registerImporter(importer: WorkflowImporter): Unsubscribe
  /** Expands the active document's `%date:...%` and `%Node.widget%` tokens. */
  applyTextReplacements(value: string): string
  /**
   * The active document's identity: a process-local id minted fresh each time
   * a workflow finishes loading — including a second load of the same file,
   * which gets a different id from the first. `undefined` before the first
   * workflow has loaded this page load.
   *
   * Distinct from the workflow's own saved identity (its file path, or the
   * `id` written into the workflow JSON): that one is meant to survive a
   * reload and compare equal across sessions. This one is the opposite by
   * design — it exists so a pack can tell "the document I was looking at got
   * replaced" from "the document I was looking at got edited", which
   * comparing graph contents cannot do, since editing IS mutating the graph
   * contents of the very document that is still current.
   *
   * Equivalent to `current()?.id`, and kept because reading the id is the
   * common case and does not need a handle.
   */
  documentId(): string | undefined
  /**
   * The document on screen, or `undefined` before one is open.
   *
   * A handle rather than the bare id when a pack needs to know what it is
   * looking at — the name to label its own UI, whether there are unsaved
   * edits, and whether a document it stored state for is still open.
   *
   * Read-only: opening has its own explicit call, and saving, closing and
   * renaming belong to the user.
   */
  current(): DocumentHandle | undefined
}
