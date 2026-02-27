/**
 * V2 Storage I/O - localStorage read/write with error handling.
 *
 * Handles quota management, orphan cleanup, and graceful degradation.
 */

import type {
  ActivePathPointer,
  DraftIndexV2,
  DraftPayloadV2,
  OpenPathsPointer
} from './draftTypes'
import { StorageKeys } from './storageKeys'

/** Flag indicating if storage is available */
let storageAvailable = true

export function isStorageAvailable(): boolean {
  return storageAvailable
}

export function markStorageUnavailable(): void {
  storageAvailable = false
}

function isQuotaExceeded(error: unknown): boolean {
  return (
    error instanceof DOMException &&
    (error.name === 'QuotaExceededError' ||
      error.name === 'NS_ERROR_DOM_QUOTA_REACHED' ||
      error.code === 22 ||
      error.code === 1014)
  )
}

function isValidIndex(value: unknown): value is DraftIndexV2 {
  if (typeof value !== 'object' || value === null) return false
  const obj = value as Record<string, unknown>
  return (
    obj.v === 2 &&
    typeof obj.updatedAt === 'number' &&
    Array.isArray(obj.order) &&
    typeof obj.entries === 'object' &&
    obj.entries !== null
  )
}

/**
 * Reads and parses the draft index from localStorage.
 */
export function readIndex(workspaceId: string): DraftIndexV2 | null {
  if (!storageAvailable) return null

  try {
    const key = StorageKeys.draftIndex(workspaceId)
    const json = localStorage.getItem(key)
    if (!json) return null

    const parsed = JSON.parse(json)
    if (!isValidIndex(parsed)) return null

    return parsed
  } catch {
    return null
  }
}

/**
 * Writes the draft index to localStorage.
 */
export function writeIndex(workspaceId: string, index: DraftIndexV2): boolean {
  if (!storageAvailable) return false

  try {
    const key = StorageKeys.draftIndex(workspaceId)
    localStorage.setItem(key, JSON.stringify(index))
    return true
  } catch (error) {
    if (isQuotaExceeded(error)) return false
    throw error
  }
}

/**
 * Reads a draft payload from localStorage.
 */
export function readPayload(
  workspaceId: string,
  draftKey: string
): DraftPayloadV2 | null {
  if (!storageAvailable) return null

  try {
    const key = `${StorageKeys.prefixes.draftPayload}${workspaceId}:${draftKey}`
    const json = localStorage.getItem(key)
    if (!json) return null

    return JSON.parse(json) as DraftPayloadV2
  } catch {
    return null
  }
}

/**
 * Writes a draft payload to localStorage.
 */
export function writePayload(
  workspaceId: string,
  draftKey: string,
  payload: DraftPayloadV2
): boolean {
  if (!storageAvailable) return false

  try {
    const key = `${StorageKeys.prefixes.draftPayload}${workspaceId}:${draftKey}`
    localStorage.setItem(key, JSON.stringify(payload))
    return true
  } catch (error) {
    if (isQuotaExceeded(error)) return false
    throw error
  }
}

/**
 * Deletes a draft payload from localStorage.
 */
export function deletePayload(workspaceId: string, draftKey: string): void {
  try {
    const key = `${StorageKeys.prefixes.draftPayload}${workspaceId}:${draftKey}`
    localStorage.removeItem(key)
  } catch {
    // Ignore errors during deletion
  }
}

/**
 * Deletes multiple draft payloads from localStorage.
 */
export function deletePayloads(workspaceId: string, draftKeys: string[]): void {
  for (const draftKey of draftKeys) {
    deletePayload(workspaceId, draftKey)
  }
}

/**
 * Gets all draft payload keys for a workspace from localStorage.
 */
export function getPayloadKeys(workspaceId: string): string[] {
  if (!storageAvailable) return []

  const prefix = `${StorageKeys.prefixes.draftPayload}${workspaceId}:`
  const keys: string[] = []

  try {
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i)
      if (key?.startsWith(prefix)) {
        keys.push(key.slice(prefix.length))
      }
    }
  } catch {
    return []
  }

  return keys
}

/**
 * Deletes orphan payloads that are not in the index.
 */
export function deleteOrphanPayloads(
  workspaceId: string,
  indexKeys: Set<string>
): number {
  const payloadKeys = getPayloadKeys(workspaceId)
  let deleted = 0

  for (const key of payloadKeys) {
    if (!indexKeys.has(key)) {
      deletePayload(workspaceId, key)
      deleted++
    }
  }

  return deleted
}

/**
 * Searches sessionStorage for a pointer matching the target workspaceId
 * when the exact clientId key has no entry (e.g. clientId changed after reload).
 * Migrates the found pointer to the new clientId key.
 */
function findAndMigratePointer<T extends { workspaceId: string }>(
  newKey: string,
  prefix: string,
  targetWorkspaceId: string
): T | null {
  for (let i = 0; i < sessionStorage.length; i++) {
    const storageKey = sessionStorage.key(i)
    if (!storageKey?.startsWith(prefix) || storageKey === newKey) continue

    const json = sessionStorage.getItem(storageKey)
    if (!json) continue

    try {
      const pointer = JSON.parse(json) as T
      if (pointer.workspaceId === targetWorkspaceId) {
        sessionStorage.setItem(newKey, json)
        sessionStorage.removeItem(storageKey)
        return pointer
      }
    } catch {
      continue
    }
  }
  return null
}

/**
 * Reads a session pointer by clientId with workspace-based fallback.
 * Validates workspace on exact match and removes stale cross-workspace pointers.
 * If no valid entry exists, searches for any pointer matching the target
 * workspaceId and migrates it to the new key.
 */
function readSessionPointer<T extends { workspaceId: string }>(
  key: string,
  prefix: string,
  targetWorkspaceId?: string
): T | null {
  try {
    const json = sessionStorage.getItem(key)
    if (json) {
      const pointer = JSON.parse(json) as T
      if (targetWorkspaceId && pointer.workspaceId !== targetWorkspaceId) {
        sessionStorage.removeItem(key)
      } else {
        return pointer
      }
    }

    if (targetWorkspaceId) {
      return findAndMigratePointer<T>(key, prefix, targetWorkspaceId)
    }

    return null
  } catch {
    return null
  }
}

/**
 * Reads the active path pointer from sessionStorage.
 * Falls back to workspace-based search when clientId changes after reload.
 */
export function readActivePath(
  clientId: string,
  targetWorkspaceId?: string
): ActivePathPointer | null {
  return readSessionPointer<ActivePathPointer>(
    StorageKeys.activePath(clientId),
    StorageKeys.prefixes.activePath,
    targetWorkspaceId
  )
}

/**
 * Writes the active path pointer to sessionStorage.
 */
export function writeActivePath(
  clientId: string,
  pointer: ActivePathPointer
): void {
  try {
    const key = StorageKeys.activePath(clientId)
    sessionStorage.setItem(key, JSON.stringify(pointer))
  } catch {
    // Best effort - ignore errors
  }
}

/**
 * Reads the open paths pointer from sessionStorage.
 * Falls back to workspace-based search when clientId changes after reload.
 */
export function readOpenPaths(
  clientId: string,
  targetWorkspaceId?: string
): OpenPathsPointer | null {
  return readSessionPointer<OpenPathsPointer>(
    StorageKeys.openPaths(clientId),
    StorageKeys.prefixes.openPaths,
    targetWorkspaceId
  )
}

/**
 * Writes the open paths pointer to sessionStorage.
 */
export function writeOpenPaths(
  clientId: string,
  pointer: OpenPathsPointer
): void {
  try {
    const key = StorageKeys.openPaths(clientId)
    sessionStorage.setItem(key, JSON.stringify(pointer))
  } catch {
    // Best effort - ignore errors
  }
}

/**
 * Clears all V2 workflow persistence data from storage.
 * Used during signout to prevent data leakage.
 */
export function clearAllV2Storage(): void {
  if (!storageAvailable) return

  const prefixes = [
    StorageKeys.prefixes.draftIndex,
    StorageKeys.prefixes.draftPayload
  ]

  try {
    for (let i = localStorage.length - 1; i >= 0; i--) {
      const key = localStorage.key(i)
      if (key && prefixes.some((prefix) => key.startsWith(prefix))) {
        try {
          localStorage.removeItem(key)
        } catch {
          // Ignore
        }
      }
    }
  } catch {
    // Ignore
  }

  const sessionPrefixes = [
    StorageKeys.prefixes.activePath,
    StorageKeys.prefixes.openPaths
  ]

  try {
    for (let i = sessionStorage.length - 1; i >= 0; i--) {
      const key = sessionStorage.key(i)
      if (key && sessionPrefixes.some((prefix) => key.startsWith(prefix))) {
        try {
          sessionStorage.removeItem(key)
        } catch {
          // Ignore
        }
      }
    }
  } catch {
    // Ignore
  }
}
