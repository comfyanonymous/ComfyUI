import type { MaybeRef } from 'vue'

import { toRef } from '@vueuse/core'
import { nextTick, ref, toRaw, watch } from 'vue'

import Load3d from '@/extensions/core/load3d/Load3d'
import Load3dUtils from '@/extensions/core/load3d/Load3dUtils'
import type {
  AnimationItem,
  CameraConfig,
  CameraState,
  CameraType,
  EventCallback,
  LightConfig,
  MaterialMode,
  ModelConfig,
  SceneConfig,
  UpDirection
} from '@/extensions/core/load3d/interfaces'
import { t } from '@/i18n'
import type { LGraphNode } from '@/lib/litegraph/src/LGraphNode'
import { LiteGraph } from '@/lib/litegraph/src/litegraph'
import { useToastStore } from '@/platform/updates/common/toastStore'
import { api } from '@/scripts/api'
import { app } from '@/scripts/app'
import { useLoad3dService } from '@/services/load3dService'

type Load3dReadyCallback = (load3d: Load3d) => void
export const nodeToLoad3dMap = new Map<LGraphNode, Load3d>()
const pendingCallbacks = new Map<LGraphNode, Load3dReadyCallback[]>()

export const useLoad3d = (nodeOrRef: MaybeRef<LGraphNode | null>) => {
  const nodeRef = toRef(nodeOrRef)
  let load3d: Load3d | null = null

  const sceneConfig = ref<SceneConfig>({
    showGrid: true,
    backgroundColor: '#000000',
    backgroundImage: '',
    backgroundRenderMode: 'tiled'
  })

  const modelConfig = ref<ModelConfig>({
    upDirection: 'original',
    materialMode: 'original',
    showSkeleton: false
  })

  const hasSkeleton = ref(false)

  const cameraConfig = ref<CameraConfig>({
    cameraType: 'perspective',
    fov: 75
  })

  const lightConfig = ref<LightConfig>({
    intensity: 5
  })

  const isRecording = ref(false)
  const hasRecording = ref(false)
  const recordingDuration = ref(0)

  const animations = ref<AnimationItem[]>([])
  const playing = ref(false)
  const selectedSpeed = ref(1)
  const selectedAnimation = ref(0)
  const animationProgress = ref(0)
  const animationDuration = ref(0)
  const loading = ref(false)
  const loadingMessage = ref('')
  const isPreview = ref(false)
  const isSplatModel = ref(false)
  const isPlyModel = ref(false)

  const initializeLoad3d = async (containerRef: HTMLElement) => {
    const rawNode = toRaw(nodeRef.value)
    if (!containerRef || !rawNode) return

    const node = rawNode as LGraphNode

    try {
      const widthWidget = node.widgets?.find((w) => w.name === 'width')
      const heightWidget = node.widgets?.find((w) => w.name === 'height')

      if (!(widthWidget && heightWidget)) {
        isPreview.value = true
      }

      load3d = new Load3d(containerRef, {
        width: widthWidget?.value as number | undefined,
        height: heightWidget?.value as number | undefined,
        // Provide dynamic dimension getter for reactive updates
        getDimensions:
          widthWidget && heightWidget
            ? () => ({
                width: widthWidget.value as number,
                height: heightWidget.value as number
              })
            : undefined,
        onContextMenu: (event) => {
          const menuOptions = app.canvas.getNodeMenuOptions(node)
          new LiteGraph.ContextMenu(menuOptions, {
            event,
            title: node.type,
            extra: node
          })
        }
      })

      await restoreConfigurationsFromNode(node)

      node.onMouseEnter = function () {
        load3d?.refreshViewport()

        load3d?.updateStatusMouseOnNode(true)
      }

      node.onMouseLeave = function () {
        load3d?.updateStatusMouseOnNode(false)
      }

      node.onResize = function () {
        load3d?.handleResize()
      }

      node.onDrawBackground = function () {
        if (load3d) {
          load3d.renderer.domElement.hidden = this.flags.collapsed ?? false
        }
      }

      node.onRemoved = function () {
        useLoad3dService().removeLoad3d(node)
        pendingCallbacks.delete(node)
      }

      nodeToLoad3dMap.set(node, load3d)

      const callbacks = pendingCallbacks.get(node)

      if (callbacks && load3d) {
        callbacks.forEach((callback) => {
          if (load3d) {
            callback(load3d)
          }
        })
        pendingCallbacks.delete(node)
      }

      handleEvents('add')
    } catch (error) {
      console.error('Error initializing Load3d:', error)
      useToastStore().addAlert(t('toastMessages.failedToInitializeLoad3d'))
    }
  }

  const restoreConfigurationsFromNode = async (node: LGraphNode) => {
    if (!load3d) return

    // Restore configs - watchers will handle applying them to the Three.js scene
    const savedSceneConfig = node.properties['Scene Config'] as SceneConfig
    if (savedSceneConfig) {
      sceneConfig.value = {
        ...sceneConfig.value,
        ...savedSceneConfig,
        backgroundRenderMode: savedSceneConfig.backgroundRenderMode || 'tiled'
      }
    }

    const savedModelConfig = node.properties['Model Config'] as ModelConfig
    if (savedModelConfig) {
      modelConfig.value = savedModelConfig
    }

    const savedCameraConfig = node.properties['Camera Config'] as CameraConfig
    const cameraStateToRestore = savedCameraConfig?.state

    if (savedCameraConfig) {
      cameraConfig.value = savedCameraConfig
    }

    const savedLightConfig = node.properties['Light Config'] as LightConfig
    if (savedLightConfig) {
      lightConfig.value = savedLightConfig
    }

    const modelWidget = node.widgets?.find((w) => w.name === 'model_file')
    if (modelWidget?.value) {
      const modelUrl = getModelUrl(modelWidget.value as string)
      if (modelUrl) {
        loading.value = true
        loadingMessage.value = t('load3d.reloadingModel')
        try {
          await load3d.loadModel(modelUrl)

          if (cameraStateToRestore) {
            await nextTick()
            load3d.setCameraState(cameraStateToRestore)
          }
        } catch (error) {
          console.error('Failed to reload model:', error)
          useToastStore().addAlert(t('toastMessages.failedToLoadModel'))
        } finally {
          loading.value = false
          loadingMessage.value = ''
        }
      }
    } else if (cameraStateToRestore) {
      load3d.setCameraState(cameraStateToRestore)
    }
  }

  const getModelUrl = (modelPath: string): string | null => {
    if (!modelPath) return null

    try {
      if (modelPath.startsWith('http')) {
        return modelPath
      }

      const trimmed = modelPath.trim()
      const hasOutputSuffix = trimmed.endsWith('[output]')
      const cleanPath = hasOutputSuffix
        ? trimmed.replace(/\s*\[output\]$/, '')
        : trimmed
      const type = hasOutputSuffix || isPreview.value ? 'output' : 'input'

      const [subfolder, filename] = Load3dUtils.splitFilePath(cleanPath)
      return api.apiURL(Load3dUtils.getResourceURL(subfolder, filename, type))
    } catch (error) {
      console.error('Failed to construct model URL:', error)
      return null
    }
  }

  const waitForLoad3d = (callback: Load3dReadyCallback) => {
    const rawNode = toRaw(nodeRef.value)
    if (!rawNode) return

    const node = rawNode as LGraphNode
    const existingInstance = nodeToLoad3dMap.get(node)

    if (existingInstance) {
      callback(existingInstance)

      return
    }

    if (!pendingCallbacks.has(node)) {
      pendingCallbacks.set(node, [])
    }

    pendingCallbacks.get(node)!.push(callback)
  }

  watch(
    sceneConfig,
    async (newValue) => {
      if (load3d && nodeRef.value) {
        nodeRef.value.properties['Scene Config'] = newValue
        load3d.toggleGrid(newValue.showGrid)
        load3d.setBackgroundColor(newValue.backgroundColor)

        await load3d.setBackgroundImage(newValue.backgroundImage || '')

        if (newValue.backgroundRenderMode) {
          load3d.setBackgroundRenderMode(newValue.backgroundRenderMode)
        }
      }
    },
    { deep: true }
  )

  watch(
    modelConfig,
    (newValue) => {
      if (load3d && nodeRef.value) {
        nodeRef.value.properties['Model Config'] = newValue
        load3d.setUpDirection(newValue.upDirection)
        load3d.setMaterialMode(newValue.materialMode)
        load3d.setShowSkeleton(newValue.showSkeleton)
      }
    },
    { deep: true }
  )

  watch(
    cameraConfig,
    (newValue) => {
      if (load3d && nodeRef.value) {
        nodeRef.value.properties['Camera Config'] = newValue
        load3d.toggleCamera(newValue.cameraType)
        load3d.setFOV(newValue.fov)
      }
    },
    { deep: true }
  )

  watch(
    lightConfig,
    (newValue) => {
      if (load3d && nodeRef.value) {
        nodeRef.value.properties['Light Config'] = newValue
        load3d.setLightIntensity(newValue.intensity)
      }
    },
    { deep: true }
  )

  watch(playing, (newValue) => {
    if (load3d) {
      load3d.toggleAnimation(newValue)
    }
  })

  watch(selectedSpeed, (newValue) => {
    if (load3d && newValue) {
      load3d.setAnimationSpeed(newValue)
    }
  })

  watch(selectedAnimation, (newValue) => {
    if (load3d && newValue !== undefined) {
      load3d.updateSelectedAnimation(newValue)
    }
  })

  const handleMouseEnter = () => {
    load3d?.updateStatusMouseOnScene(true)
  }

  const handleMouseLeave = () => {
    load3d?.updateStatusMouseOnScene(false)
  }

  const handleStartRecording = async () => {
    if (load3d) {
      await load3d.startRecording()
      isRecording.value = true
    }
  }

  const handleStopRecording = () => {
    if (load3d) {
      load3d.stopRecording()
      isRecording.value = false
      recordingDuration.value = load3d.getRecordingDuration()
      hasRecording.value = recordingDuration.value > 0
    }
  }

  const handleExportRecording = () => {
    if (load3d) {
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-')
      const filename = `${timestamp}-scene-recording.mp4`
      load3d.exportRecording(filename)
    }
  }

  const handleClearRecording = () => {
    if (load3d) {
      load3d.clearRecording()
      hasRecording.value = false
      recordingDuration.value = 0
    }
  }

  const handleSeek = (progress: number) => {
    if (load3d && animationDuration.value > 0) {
      const time = (progress / 100) * animationDuration.value
      load3d.setAnimationTime(time)
    }
  }

  const handleBackgroundImageUpdate = async (file: File | null) => {
    if (!file) {
      sceneConfig.value.backgroundImage = ''
      await load3d?.setBackgroundImage('')
      return
    }

    const resourceFolder =
      (nodeRef.value?.properties['Resource Folder'] as string) || ''

    const subfolder = resourceFolder.trim()
      ? `3d/${resourceFolder.trim()}`
      : '3d'

    const uploadedPath = await Load3dUtils.uploadFile(file, subfolder)
    sceneConfig.value.backgroundImage = uploadedPath
    await load3d?.setBackgroundImage(uploadedPath)
  }

  const handleExportModel = async (format: string) => {
    if (!load3d) {
      useToastStore().addAlert(t('toastMessages.no3dSceneToExport'))
      return
    }

    try {
      await load3d.exportModel(format)
    } catch (error) {
      console.error('Error exporting model:', error)
      useToastStore().addAlert(
        t('toastMessages.failedToExportModel', {
          format: format.toUpperCase()
        })
      )
    }
  }

  const handleModelDrop = async (file: File) => {
    if (!load3d) {
      useToastStore().addAlert(t('toastMessages.no3dScene'))
      return
    }

    const node = toRaw(nodeRef.value)
    if (!node) return

    try {
      const resourceFolder =
        (node.properties['Resource Folder'] as string) || ''

      const subfolder = resourceFolder.trim()
        ? `3d/${resourceFolder.trim()}`
        : '3d'

      loading.value = true
      loadingMessage.value = t('load3d.uploadingModel')

      const uploadedPath = await Load3dUtils.uploadFile(file, subfolder)

      if (!uploadedPath) {
        useToastStore().addAlert(t('toastMessages.fileUploadFailed'))
        return
      }

      const modelUrl = api.apiURL(
        Load3dUtils.getResourceURL(
          ...Load3dUtils.splitFilePath(uploadedPath),
          isPreview.value ? 'output' : 'input'
        )
      )

      loadingMessage.value = t('load3d.loadingModel')
      await load3d.loadModel(modelUrl)

      const modelWidget = node.widgets?.find((w) => w.name === 'model_file')

      if (modelWidget) {
        const options = modelWidget.options as { values?: string[] } | undefined
        if (options?.values && !options.values.includes(uploadedPath)) {
          options.values.push(uploadedPath)
        }
        modelWidget.value = uploadedPath
      }
    } catch (error) {
      console.error('Model drop failed:', error)
      useToastStore().addAlert(t('toastMessages.failedToLoadModel'))
    } finally {
      loading.value = false
      loadingMessage.value = ''
    }
  }

  const eventConfig = {
    materialModeChange: (value: string) => {
      modelConfig.value.materialMode = value as MaterialMode
    },
    backgroundColorChange: (value: string) => {
      sceneConfig.value.backgroundColor = value
    },
    backgroundRenderModeChange: (value: string) => {
      sceneConfig.value.backgroundRenderMode = value as 'tiled' | 'panorama'
    },
    lightIntensityChange: (value: number) => {
      lightConfig.value.intensity = value
    },
    fovChange: (value: number) => {
      cameraConfig.value.fov = value
    },
    cameraTypeChange: (value: string) => {
      cameraConfig.value.cameraType = value as CameraType
    },
    showGridChange: (value: boolean) => {
      sceneConfig.value.showGrid = value
    },
    upDirectionChange: (value: string) => {
      modelConfig.value.upDirection = value as UpDirection
    },
    backgroundImageChange: (value: string) => {
      sceneConfig.value.backgroundImage = value
    },
    backgroundImageLoadingStart: () => {
      loadingMessage.value = t('load3d.loadingBackgroundImage')
      loading.value = true
    },
    backgroundImageLoadingEnd: () => {
      loadingMessage.value = ''
      loading.value = false
    },
    modelLoadingStart: () => {
      loadingMessage.value = t('load3d.loadingModel')
      loading.value = true
    },
    modelLoadingEnd: () => {
      loadingMessage.value = ''
      loading.value = false
      isSplatModel.value = load3d?.isSplatModel() ?? false
      isPlyModel.value = load3d?.isPlyModel() ?? false
      hasSkeleton.value = load3d?.hasSkeleton() ?? false
      // Reset skeleton visibility when loading new model
      modelConfig.value.showSkeleton = false

      if (load3d) {
        const node = nodeRef.value

        const modelWidget = node?.widgets?.find(
          (w) => w.name === 'model_file' || w.name === 'image'
        )
        const value = modelWidget?.value
        if (typeof value === 'string') {
          void Load3dUtils.generateThumbnailIfNeeded(
            load3d,
            value,
            isPreview.value ? 'output' : 'input'
          )
        }
      }
    },
    skeletonVisibilityChange: (value: boolean) => {
      modelConfig.value.showSkeleton = value
    },
    exportLoadingStart: (message: string) => {
      loadingMessage.value = message || t('load3d.exportingModel')
      loading.value = true
    },
    exportLoadingEnd: () => {
      loadingMessage.value = ''
      loading.value = false
    },
    recordingStatusChange: (value: boolean) => {
      isRecording.value = value

      if (!value && load3d) {
        recordingDuration.value = load3d.getRecordingDuration()
        hasRecording.value = recordingDuration.value > 0
      }
    },
    animationListChange: (newValue: AnimationItem[]) => {
      animations.value = newValue
    },
    animationProgressChange: (data: {
      progress: number
      currentTime: number
      duration: number
    }) => {
      animationProgress.value = data.progress
      animationDuration.value = data.duration
    },
    cameraChanged: (cameraState: CameraState) => {
      const rawNode = toRaw(nodeRef.value)
      if (rawNode) {
        const node = rawNode as LGraphNode
        if (!node.properties) node.properties = {}
        const cameraConfigProp = node.properties['Camera Config']

        if (cameraConfigProp) {
          ;(cameraConfigProp as CameraConfig).state = cameraState
        } else {
          node.properties['Camera Config'] = {
            cameraType: cameraConfig.value.cameraType,
            fov: cameraConfig.value.fov,
            state: cameraState
          }
        }
      }
    }
  } as const

  const handleEvents = (action: 'add' | 'remove') => {
    Object.entries(eventConfig).forEach(([event, handler]) => {
      const method = `${action}EventListener` as const
      load3d?.[method](event, handler as EventCallback)
    })
  }

  const cleanup = () => {
    handleEvents('remove')

    const rawNode = toRaw(nodeRef.value)
    if (!rawNode) return

    const node = rawNode as LGraphNode
    if (nodeToLoad3dMap.get(node) === load3d) {
      nodeToLoad3dMap.delete(node)
    }

    load3d?.remove()
    load3d = null
  }

  return {
    // state
    load3d,
    sceneConfig,
    modelConfig,
    cameraConfig,
    lightConfig,
    isRecording,
    isPreview,
    isSplatModel,
    isPlyModel,
    hasSkeleton,
    hasRecording,
    recordingDuration,
    animations,
    playing,
    selectedSpeed,
    selectedAnimation,
    animationProgress,
    animationDuration,
    loading,
    loadingMessage,

    // Methods
    initializeLoad3d,
    waitForLoad3d,
    handleMouseEnter,
    handleMouseLeave,
    handleStartRecording,
    handleStopRecording,
    handleExportRecording,
    handleClearRecording,
    handleSeek,
    handleBackgroundImageUpdate,
    handleExportModel,
    handleModelDrop,
    cleanup
  }
}
