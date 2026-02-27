/**
 * V2 Workflow Persistence Type Definitions
 *
 * Two-layer state system:
 * - sessionStorage: Per-tab pointers (tiny, scoped by clientId)
 * - localStorage: Persistent drafts (per-workspace, per-draft keys)
 */

/**
 * Metadata for a single draft entry stored in the index.
 * The actual workflow data is stored separately in a Draft payload key.
 */
export interface DraftEntryMeta {
  /** Workflow path (e.g., "workflows/Untitled.json") */
  path: string
  /** Display name of the workflow */
  name: string
  /** Whether this is an unsaved temporary workflow */
  isTemporary: boolean
  /** Last update timestamp (ms since epoch) */
  updatedAt: number
}

/**
 * Draft index stored in localStorage.
 * Contains LRU order and metadata for all drafts in a workspace.
 *
 * Key: `Comfy.Workflow.DraftIndex.v2:${workspaceId}`
 */
export interface DraftIndexV2 {
  /** Schema version */
  v: 2
  /** Last update timestamp */
  updatedAt: number
  /** LRU order: oldest → newest (draftKey array) */
  order: string[]
  /** Metadata keyed by draftKey (hash of path) */
  entries: Record<string, DraftEntryMeta>
}

/**
 * Individual draft payload stored in localStorage.
 *
 * Key: `Comfy.Workflow.Draft.v2:${workspaceId}:${draftKey}`
 */
export interface DraftPayloadV2 {
  /** Serialized workflow JSON */
  data: string
  /** Last update timestamp */
  updatedAt: number
}

/**
 * Pointer stored in sessionStorage to track active workflow per tab.
 * Includes workspaceId for validation on read.
 *
 * Key: `Comfy.Workflow.ActivePath:${clientId}`
 */
export interface ActivePathPointer {
  /** Workspace ID for validation */
  workspaceId: string
  /** Path to the active workflow */
  path: string
}

/**
 * Pointer stored in sessionStorage to track open workflow tabs.
 * Includes workspaceId for validation on read.
 *
 * Key: `Comfy.Workflow.OpenPaths:${clientId}`
 */
export interface OpenPathsPointer {
  /** Workspace ID for validation */
  workspaceId: string
  /** Ordered list of open workflow paths */
  paths: string[]
  /** Index of the active workflow in paths array */
  activeIndex: number
}

/** Maximum number of drafts to keep per workspace */
export const MAX_DRAFTS = 32

export const PERSIST_DEBOUNCE_MS = 512
