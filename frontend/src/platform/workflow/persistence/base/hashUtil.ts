/**
 * FNV-1a hash function for creating short, deterministic keys from strings.
 *
 * Used to create 8-character hex keys from workflow paths for localStorage keys.
 * FNV-1a is chosen for its simplicity, speed, and good distribution properties.
 *
 * @param str - The string to hash (typically a workflow path)
 * @returns A 32-bit unsigned integer hash
 */
export function fnv1a(str: string): number {
  let hash = 2166136261
  for (let i = 0; i < str.length; i++) {
    hash ^= str.charCodeAt(i)
    hash = Math.imul(hash, 16777619)
  }
  return hash >>> 0
}

/**
 * Creates an 8-character hex key from a workflow path using FNV-1a hash.
 *
 * @param path - The workflow path (e.g., "workflows/My Workflow.json")
 * @returns An 8-character hex string (e.g., "a1b2c3d4")
 *
 * @example
 * hashPath("workflows/Untitled.json") // "1a2b3c4d"
 */
export function hashPath(path: string): string {
  return fnv1a(path).toString(16).padStart(8, '0')
}
