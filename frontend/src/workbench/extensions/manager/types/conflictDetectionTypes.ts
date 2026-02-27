/**
 * Type definitions for the conflict detection system.
 * These types are used to detect compatibility issues between Node Packs and the system environment.
 *
 * This file extends and uses types from comfyRegistryTypes.ts to maintain consistency
 * with the Registry API schema.
 */
import type { components } from '@/types/comfyRegistryTypes'

// Re-export core types from Registry API
export type Node = components['schemas']['Node']

/**
 * Conflict types that can be detected in the system
 * @enum {string}
 */
export type ConflictType =
  | 'comfyui_version' // ComfyUI version mismatch
  | 'frontend_version' // Frontend version mismatch
  | 'import_failed'
  | 'os' // Operating system incompatibility
  | 'accelerator' // GPU/accelerator incompatibility
  | 'banned' // Banned package
  | 'pending' // Security verification pending

/**
 * Node Pack requirements from Registry API
 * Extends Node type with additional installation and compatibility metadata
 */
export interface NodeRequirements extends Node {
  installed_version: string
  is_enabled: boolean
  is_banned: boolean
  is_pending: boolean
  // Aliases for backwards compatibility with existing code
  version_status?: string
}

/**
 * Current system environment information
 */
export interface SystemEnvironment {
  // Version information
  comfyui_version?: string
  frontend_version?: string
  // Platform information
  os?: string
  // GPU/accelerator information
  accelerator?: string
}

/**
 * Individual conflict detection result for a package
 */
export interface ConflictDetectionResult {
  package_id: string
  package_name: string
  has_conflict: boolean
  conflicts: ConflictDetail[]
  is_compatible: boolean
}

/**
 * Detailed information about a specific conflict
 */
export interface ConflictDetail {
  type: ConflictType
  current_value: string
  required_value: string
}

/**
 * Response payload from conflict detection API
 */
export interface ConflictDetectionResponse {
  success: boolean
  error_message?: string
  results: ConflictDetectionResult[]
  detected_system_environment?: Partial<SystemEnvironment>
}

/**
 * Detailed information about a Python import failure
 */
interface ImportFailureDetail {
  error?: string
  traceback?: string
}

/**
 * Map of package IDs to their import failure information
 */
export type ImportFailureMap = Record<string, ImportFailureDetail | null>
