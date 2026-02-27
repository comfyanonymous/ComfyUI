/**
 * Layout System Constants
 *
 * Centralized configuration values for the layout system.
 * These values control spatial indexing, performance, and behavior.
 */
import { LayoutSource } from '@/renderer/core/layout/types'

/**
 * QuadTree configuration for spatial indexing
 */
export const QUADTREE_CONFIG = {
  /** Default bounds for the QuadTree - covers a large canvas area */
  DEFAULT_BOUNDS: {
    x: -10000,
    y: -10000,
    width: 20000,
    height: 20000
  },
  /** Maximum tree depth to prevent excessive subdivision */
  MAX_DEPTH: 6,
  /** Maximum items per node before subdivision */
  MAX_ITEMS_PER_NODE: 4
} as const

/**
 * Performance and optimization settings
 */
export const PERFORMANCE_CONFIG = {
  /** RAF-based change detection interval (roughly 60fps) */
  CHANGE_DETECTION_INTERVAL: 16,
  /** Spatial query cache TTL in milliseconds */
  SPATIAL_CACHE_TTL: 1000,
  /** Maximum cache size for spatial queries */
  SPATIAL_CACHE_MAX_SIZE: 100,
  /** Batch update delay in milliseconds */
  BATCH_UPDATE_DELAY: 4
} as const

/**
 * Actor and source identifiers
 */
export const ACTOR_CONFIG = {
  /** Prefix for auto-generated actor IDs */
  USER_PREFIX: 'user-',
  /** Length of random suffix for actor IDs */
  ID_LENGTH: 9,
  /** Default source when not specified */
  DEFAULT_SOURCE: LayoutSource.External
} as const
