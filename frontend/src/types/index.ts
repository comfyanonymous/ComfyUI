import type {
  DeviceStats,
  EmbeddingsResponse,
  ExtensionsResponse,
  LogEntry,
  LogsRawResponse,
  NodeError,
  PromptResponse,
  Settings,
  SystemStats,
  TerminalSize,
  User,
  UserData,
  UserDataFullInfo
} from '@/schemas/apiSchema'
import type { ComfyApp } from '@/scripts/app'

import type {
  BottomPanelExtension,
  CommandManager,
  ExtensionManager,
  SidebarTabExtension,
  ToastManager,
  ToastMessageOptions
} from './extensionTypes'

export type { ComfyExtension } from './comfy'
export type { ComfyApi } from '@/scripts/api'
export type { ComfyApp } from '@/scripts/app'
export type { ComfyNodeDef } from '@/schemas/nodeDefSchema'
export type { InputSpec } from '@/schemas/nodeDefSchema'
export type {
  NodeLocatorId,
  NodeExecutionId,
  isNodeLocatorId,
  isNodeExecutionId,
  parseNodeLocatorId,
  createNodeLocatorId,
  parseNodeExecutionId,
  createNodeExecutionId
} from './nodeIdentification'
export type { DOMWidget, DOMWidgetOptions } from '@/scripts/domWidget'
export type {
  EmbeddingsResponse,
  ExtensionsResponse,
  PromptResponse,
  NodeError,
  Settings,
  DeviceStats,
  SystemStats,
  User,
  UserData,
  UserDataFullInfo,
  TerminalSize,
  LogEntry,
  LogsRawResponse
}
export type {
  SidebarTabExtension,
  BottomPanelExtension,
  ToastManager,
  ExtensionManager,
  CommandManager,
  ToastMessageOptions
}

interface CapturedMessages {
  clientFeatureFlags: { type: string; data: Record<string, unknown> } | null
  serverFeatureFlags: Record<string, unknown> | null
}

interface AppReadiness {
  featureFlagsReceived: boolean
  apiInitialized: boolean
  appInitialized: boolean
}

declare global {
  interface Window {
    /** For use by extensions and in the browser console. Where possible, import `app` from '@/scripts/app' instead. */
    app?: ComfyApp

    /** For use by extensions and in the browser console. Where possible, import `app` and access via `app.graph` instead. */
    graph?: unknown

    /** For use in tests to capture WebSocket messages */
    __capturedMessages?: CapturedMessages

    /** For use in tests to track app initialization state */
    __appReadiness?: AppReadiness
  }
}
