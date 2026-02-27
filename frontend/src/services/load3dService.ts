/**
 * Load3D Service - provides access to Load3D instances
 *
 * This service uses lazy imports to avoid pulling THREE.js into the main bundle.
 * The nodeToLoad3dMap is accessed lazily - it will only be available after
 * the load3d extension has been loaded.
 */
import { toRaw } from 'vue'

import type Load3d from '@/extensions/core/load3d/Load3d'
import type { LGraphNode } from '@/lib/litegraph/src/litegraph'
import type { NodeId } from '@/platform/workflow/validation/schemas/workflowSchema'
import type { Object3D } from 'three'

// Type for the useLoad3dViewer composable function
// Using explicit type to avoid import() type annotations (lint rule)
type UseLoad3dViewerFn = (node?: LGraphNode) => {
  initializeViewer: (containerRef: HTMLElement, source: Load3d) => Promise<void>
  initializeStandaloneViewer: (
    containerRef: HTMLElement,
    modelUrl: string
  ) => Promise<void>
  cleanup: () => void
  handleResize: () => void
  handleMouseEnter: () => void
  handleMouseLeave: () => void
  applyChanges: () => Promise<boolean>
  restoreInitialState: () => void
  refreshViewport: () => void
  exportModel: (format: string) => Promise<void>
  handleBackgroundImageUpdate: (file: File | null) => Promise<void>
  handleModelDrop: (file: File) => Promise<void>
  handleSeek: (progress: number) => void
  needApplyChanges: { value: boolean }
}

// Type for SkeletonUtils module
type SkeletonUtilsModule = { clone: (source: Object3D) => Object3D }

// Cache for lazy-loaded modules
let cachedNodeToLoad3dMap: Map<LGraphNode, Load3d> | null = null
let cachedUseLoad3dViewer: UseLoad3dViewerFn | null = null
let cachedSkeletonUtils: SkeletonUtilsModule | null = null

// Sync accessor - returns null if module not yet loaded
function getNodeToLoad3dMapSync(): Map<LGraphNode, Load3d> | null {
  return cachedNodeToLoad3dMap
}

// Async loader for nodeToLoad3dMap - also caches for sync access
async function loadNodeToLoad3dMap(): Promise<Map<LGraphNode, Load3d>> {
  if (!cachedNodeToLoad3dMap) {
    const module = await import('@/composables/useLoad3d')
    cachedNodeToLoad3dMap = module.nodeToLoad3dMap
  }
  return cachedNodeToLoad3dMap
}

async function loadUseLoad3dViewer() {
  if (!cachedUseLoad3dViewer) {
    const module = await import('@/composables/useLoad3dViewer')
    cachedUseLoad3dViewer = module.useLoad3dViewer
  }
  return cachedUseLoad3dViewer
}

async function loadSkeletonUtils() {
  if (!cachedSkeletonUtils) {
    cachedSkeletonUtils = await import('three/examples/jsm/utils/SkeletonUtils')
  }
  return cachedSkeletonUtils
}

// Type definitions for Load3D node
interface SceneConfig {
  backgroundImage?: string
}

interface Load3DNode extends LGraphNode {
  syncLoad3dConfig?: () => void
}

const viewerInstances = new Map<NodeId, ReturnType<UseLoad3dViewerFn>>()

export class Load3dService {
  private static instance: Load3dService

  private constructor() {}

  static getInstance(): Load3dService {
    if (!Load3dService.instance) {
      Load3dService.instance = new Load3dService()
    }
    return Load3dService.instance
  }

  /**
   * Get Load3d instance for a node (synchronous).
   * Returns null if the load3d module hasn't been loaded yet.
   */
  getLoad3d(node: LGraphNode): Load3d | null {
    const rawNode = toRaw(node)
    const map = getNodeToLoad3dMapSync()
    if (!map) return null
    return map.get(rawNode) || null
  }

  /**
   * Get Load3d instance for a node (async, loads module if needed).
   */
  async getLoad3dAsync(node: LGraphNode): Promise<Load3d | null> {
    const rawNode = toRaw(node)
    const map = await loadNodeToLoad3dMap()
    return map.get(rawNode) || null
  }

  getNodeByLoad3d(load3d: Load3d): LGraphNode | null {
    const map = getNodeToLoad3dMapSync()
    if (!map) return null
    for (const [node, instance] of map) {
      if (instance === load3d) {
        return node
      }
    }
    return null
  }

  removeLoad3d(node: LGraphNode) {
    const rawNode = toRaw(node)
    const map = getNodeToLoad3dMapSync()
    if (!map) return

    const instance = map.get(rawNode)

    if (instance) {
      instance.remove()
      map.delete(rawNode)
    }
  }

  clear() {
    const map = getNodeToLoad3dMapSync()
    if (!map) return
    for (const [node] of map) {
      this.removeLoad3d(node)
    }
  }

  /**
   * Get or create viewer (async, loads module if needed).
   * Use this for initial viewer creation.
   */
  async getOrCreateViewer(node: LGraphNode) {
    if (!viewerInstances.has(node.id)) {
      const useLoad3dViewer = await loadUseLoad3dViewer()
      viewerInstances.set(node.id, useLoad3dViewer(node))
    }

    return viewerInstances.get(node.id)
  }

  /**
   * Get or create viewer (sync version).
   * Only works after useLoad3dViewer has been loaded.
   * Returns null if module not yet loaded - use async version instead.
   */
  getOrCreateViewerSync<T extends UseLoad3dViewerFn>(
    node: LGraphNode,
    useLoad3dViewer: T
  ): ReturnType<T> {
    if (!viewerInstances.has(node.id)) {
      viewerInstances.set(node.id, useLoad3dViewer(node))
    }

    return viewerInstances.get(node.id) as ReturnType<T>
  }

  removeViewer(node: LGraphNode) {
    const viewer = viewerInstances.get(node.id)

    if (viewer) {
      viewer.cleanup()
    }

    viewerInstances.delete(node.id)
  }

  async copyLoad3dState(source: Load3d, target: Load3d) {
    const sourceModel = source.modelManager.currentModel

    if (sourceModel) {
      // Remove existing model from target scene before adding new one
      const existingModel = target.getModelManager().currentModel
      if (existingModel) {
        target.getSceneManager().scene.remove(existingModel)
      }

      if (source.isSplatModel()) {
        const originalURL = source.modelManager.originalURL
        if (originalURL) {
          await target.loadModel(originalURL)
        }
      } else {
        // Use SkeletonUtils.clone for proper skeletal animation support
        const SkeletonUtils = await loadSkeletonUtils()
        const modelClone = SkeletonUtils.clone(sourceModel)

        target.getModelManager().currentModel = modelClone
        target.getSceneManager().scene.add(modelClone)

        const sourceOriginalModel = source.getModelManager().originalModel

        if (sourceOriginalModel) {
          target.getModelManager().originalModel = sourceOriginalModel
        }

        target.getModelManager().materialMode =
          source.getModelManager().materialMode

        target.getModelManager().currentUpDirection =
          source.getModelManager().currentUpDirection

        target.setMaterialMode(source.getModelManager().materialMode)
        target.setUpDirection(source.getModelManager().currentUpDirection)

        if (source.getModelManager().appliedTexture) {
          target.getModelManager().appliedTexture =
            source.getModelManager().appliedTexture
        }

        // Copy animation state
        if (source.hasAnimations()) {
          target.animationManager.setupModelAnimations(
            modelClone,
            sourceOriginalModel
          )
        }
      }
    }

    const sourceCameraType = source.getCurrentCameraType()
    const sourceCameraState = source.getCameraState()

    target.toggleCamera(sourceCameraType)
    target.setCameraState(sourceCameraState)

    target.setBackgroundColor(source.getSceneManager().currentBackgroundColor)

    target.toggleGrid(source.getSceneManager().gridHelper.visible)

    const sourceBackgroundInfo = source
      .getSceneManager()
      .getCurrentBackgroundInfo()
    if (sourceBackgroundInfo.type === 'image') {
      const sourceNode = this.getNodeByLoad3d(source)
      const sceneConfig = sourceNode?.properties?.['Scene Config'] as
        | SceneConfig
        | undefined
      const backgroundPath = sceneConfig?.backgroundImage
      if (backgroundPath) {
        await target.setBackgroundImage(backgroundPath)
      }
    } else {
      await target.setBackgroundImage('')
    }

    target.setLightIntensity(
      source.getLightingManager().lights[1]?.intensity || 1
    )

    if (sourceCameraType === 'perspective') {
      target.setFOV(source.getCameraManager().perspectiveCamera.fov)
    }
  }

  handleViewportRefresh(load3d: Load3d | null) {
    if (!load3d) return

    load3d.handleResize()

    const currentType = load3d.getCurrentCameraType()

    load3d.toggleCamera(
      currentType === 'perspective' ? 'orthographic' : 'perspective'
    )
    load3d.toggleCamera(currentType)

    load3d.getControlsManager().controls.update()
  }

  async handleViewerClose(node: LGraphNode) {
    const viewer = await useLoad3dService().getOrCreateViewer(node)
    if (!viewer) return

    if (viewer.needApplyChanges.value) {
      await viewer.applyChanges()

      // Sync configuration back to the node's UI
      const load3DNode = node as Load3DNode
      if (load3DNode.syncLoad3dConfig) {
        load3DNode.syncLoad3dConfig()
      }
    }

    useLoad3dService().removeViewer(node)
  }
}

export const useLoad3dService = () => {
  return Load3dService.getInstance()
}
