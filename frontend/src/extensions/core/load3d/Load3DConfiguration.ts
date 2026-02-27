import Load3d from '@/extensions/core/load3d/Load3d'
import Load3dUtils from '@/extensions/core/load3d/Load3dUtils'
import type {
  CameraConfig,
  CameraState,
  LightConfig,
  ModelConfig,
  SceneConfig
} from '@/extensions/core/load3d/interfaces'
import type { Dictionary } from '@/lib/litegraph/src/interfaces'
import type { NodeProperty } from '@/lib/litegraph/src/LGraphNode'
import type { IBaseWidget } from '@/lib/litegraph/src/types/widgets'
import { useSettingStore } from '@/platform/settings/settingStore'
import { api } from '@/scripts/api'

type Load3DConfigurationSettings = {
  loadFolder: string
  modelWidget: IBaseWidget
  cameraState?: CameraState
  width?: IBaseWidget
  height?: IBaseWidget
  bgImagePath?: string
}

class Load3DConfiguration {
  constructor(
    private load3d: Load3d,
    private properties?: Dictionary<NodeProperty | undefined>
  ) {}

  configureForSaveMesh(loadFolder: 'input' | 'output', filePath: string) {
    this.setupModelHandlingForSaveMesh(filePath, loadFolder)
    this.setupDefaultProperties()
  }

  configure(setting: Load3DConfigurationSettings) {
    this.setupModelHandling(
      setting.modelWidget,
      setting.loadFolder,
      setting.cameraState
    )
    this.setupTargetSize(setting.width, setting.height)
    this.setupDefaultProperties(setting.bgImagePath)
  }

  private setupTargetSize(width?: IBaseWidget, height?: IBaseWidget) {
    if (width && height) {
      this.load3d.setTargetSize(width.value as number, height.value as number)

      width.callback = (value: number) => {
        this.load3d.setTargetSize(value, height.value as number)
      }

      height.callback = (value: number) => {
        this.load3d.setTargetSize(width.value as number, value)
      }
    }
  }

  private setupModelHandlingForSaveMesh(filePath: string, loadFolder: string) {
    const onModelWidgetUpdate = this.createModelUpdateHandler(loadFolder)

    if (filePath) {
      onModelWidgetUpdate(filePath)
    }
  }

  private setupModelHandling(
    modelWidget: IBaseWidget,
    loadFolder: string,
    cameraState?: CameraState
  ) {
    const onModelWidgetUpdate = this.createModelUpdateHandler(
      loadFolder,
      cameraState
    )
    if (modelWidget.value) {
      onModelWidgetUpdate(modelWidget.value)
    }

    const originalCallback = modelWidget.callback

    let currentValue = modelWidget.value
    Object.defineProperty(modelWidget, 'value', {
      get() {
        return currentValue
      },
      set(newValue) {
        currentValue = newValue
        if (modelWidget.callback && newValue !== undefined && newValue !== '') {
          modelWidget.callback(newValue)
        }
      },
      enumerable: true,
      configurable: true
    })

    modelWidget.callback = (value: string | number | boolean | object) => {
      onModelWidgetUpdate(value)

      if (originalCallback) {
        originalCallback(value)
      }
    }
  }

  private setupDefaultProperties(bgImagePath?: string) {
    const sceneConfig = this.loadSceneConfig()
    this.applySceneConfig(sceneConfig, bgImagePath)

    const cameraConfig = this.loadCameraConfig()
    this.applyCameraConfig(cameraConfig)

    const lightConfig = this.loadLightConfig()
    this.applyLightConfig(lightConfig)
  }

  private loadSceneConfig(): SceneConfig {
    if (this.properties && 'Scene Config' in this.properties) {
      return this.properties['Scene Config'] as SceneConfig
    }

    return {
      showGrid: useSettingStore().get('Comfy.Load3D.ShowGrid'),
      backgroundColor:
        '#' + useSettingStore().get('Comfy.Load3D.BackgroundColor'),
      backgroundImage: ''
    } as SceneConfig
  }

  private loadCameraConfig(): CameraConfig {
    if (this.properties && 'Camera Config' in this.properties) {
      return this.properties['Camera Config'] as CameraConfig
    }

    return {
      cameraType: useSettingStore().get('Comfy.Load3D.CameraType'),
      fov: 35
    } as CameraConfig
  }

  private loadLightConfig(): LightConfig {
    if (this.properties && 'Light Config' in this.properties) {
      return this.properties['Light Config'] as LightConfig
    }

    return {
      intensity: useSettingStore().get('Comfy.Load3D.LightIntensity')
    } as LightConfig
  }

  private loadModelConfig(): ModelConfig {
    if (this.properties && 'Model Config' in this.properties) {
      return this.properties['Model Config'] as ModelConfig
    }

    return {
      upDirection: 'original',
      materialMode: 'original',
      showSkeleton: false
    }
  }

  private applySceneConfig(config: SceneConfig, bgImagePath?: string) {
    this.load3d.toggleGrid(config.showGrid)
    this.load3d.setBackgroundColor(config.backgroundColor)
    if (config.backgroundImage) {
      if (bgImagePath && bgImagePath != config.backgroundImage) {
        return
      }

      void this.load3d.setBackgroundImage(config.backgroundImage)

      if (config.backgroundRenderMode) {
        this.load3d.setBackgroundRenderMode(config.backgroundRenderMode)
      }
    }
  }

  private applyCameraConfig(config: CameraConfig) {
    this.load3d.toggleCamera(config.cameraType)
    this.load3d.setFOV(config.fov)

    if (config.state) {
      this.load3d.setCameraState(config.state)
    }
  }

  private applyLightConfig(config: LightConfig) {
    this.load3d.setLightIntensity(config.intensity)
  }

  private applyModelConfig(config: ModelConfig) {
    this.load3d.setUpDirection(config.upDirection)
    this.load3d.setMaterialMode(config.materialMode)
  }

  private createModelUpdateHandler(
    loadFolder: string,
    cameraState?: CameraState
  ) {
    let isFirstLoad = true
    return async (value: string | number | boolean | object) => {
      if (!value) return

      const filename = value as string

      this.setResourceFolder(filename)

      const modelUrl = api.apiURL(
        Load3dUtils.getResourceURL(
          ...Load3dUtils.splitFilePath(filename),
          loadFolder
        )
      )

      await this.load3d.loadModel(modelUrl, filename)

      const modelConfig = this.loadModelConfig()
      this.applyModelConfig(modelConfig)

      if (isFirstLoad && cameraState) {
        try {
          this.load3d.setCameraState(cameraState)
        } catch (error) {
          console.warn('Failed to restore camera state:', error)
        }
        isFirstLoad = false
      }
    }
  }

  private setResourceFolder(filename: string): void {
    const pathParts = filename.split('/').filter((part) => part.trim())

    if (pathParts.length <= 2) {
      return
    }

    const subfolderParts = pathParts.slice(1, -1)
    const subfolder = subfolderParts.join('/')

    if (subfolder && this.properties) {
      this.properties['Resource Folder'] = subfolder
    }
  }
}

export default Load3DConfiguration
