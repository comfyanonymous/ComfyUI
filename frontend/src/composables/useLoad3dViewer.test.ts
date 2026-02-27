import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick } from 'vue'

import { useLoad3dViewer } from '@/composables/useLoad3dViewer'
import Load3d from '@/extensions/core/load3d/Load3d'
import Load3dUtils from '@/extensions/core/load3d/Load3dUtils'
import type { LGraph } from '@/lib/litegraph/src/LGraph'
import type { LGraphNode } from '@/lib/litegraph/src/LGraphNode'
import { useToastStore } from '@/platform/updates/common/toastStore'
import { useLoad3dService } from '@/services/load3dService'
import { createMockLGraphNode } from '@/utils/__tests__/litegraphTestUtils'

vi.mock('@/services/load3dService', () => ({
  useLoad3dService: vi.fn()
}))

vi.mock('@/platform/updates/common/toastStore', () => ({
  useToastStore: vi.fn()
}))

vi.mock('@/extensions/core/load3d/Load3dUtils', () => ({
  default: {
    uploadFile: vi.fn()
  }
}))

vi.mock('@/i18n', () => ({
  t: vi.fn((key) => key)
}))

vi.mock('@/extensions/core/load3d/Load3d', () => ({
  default: vi.fn()
}))

function createMockSceneManager(): Load3d['sceneManager'] {
  const mock: Partial<Load3d['sceneManager']> = {
    scene: {} as Load3d['sceneManager']['scene'],
    backgroundScene: {} as Load3d['sceneManager']['backgroundScene'],
    backgroundCamera: {} as Load3d['sceneManager']['backgroundCamera'],
    currentBackgroundColor: '#282828',
    gridHelper: { visible: true } as Load3d['sceneManager']['gridHelper'],
    getCurrentBackgroundInfo: vi.fn().mockReturnValue({
      type: 'color',
      value: '#282828'
    })
  }
  return mock as Load3d['sceneManager']
}

describe('useLoad3dViewer', () => {
  let mockLoad3d: Partial<Load3d>
  let mockSourceLoad3d: Partial<Load3d>
  let mockLoad3dService: ReturnType<typeof useLoad3dService>
  let mockToastStore: ReturnType<typeof useToastStore>
  let mockNode: LGraphNode

  beforeEach(() => {
    vi.clearAllMocks()

    mockNode = createMockLGraphNode({
      properties: {
        'Scene Config': {
          backgroundColor: '#282828',
          showGrid: true,
          backgroundImage: '',
          backgroundRenderMode: 'tiled'
        },
        'Camera Config': {
          cameraType: 'perspective',
          fov: 75
        },
        'Light Config': {
          intensity: 1
        },
        'Model Config': {
          upDirection: 'original',
          materialMode: 'original'
        },
        'Resource Folder': ''
      },
      graph: {
        setDirtyCanvas: vi.fn()
      } as Partial<LGraph> as LGraph,
      widgets: []
    })

    mockLoad3d = {
      setBackgroundColor: vi.fn(),
      toggleGrid: vi.fn(),
      toggleCamera: vi.fn(),
      setFOV: vi.fn(),
      setLightIntensity: vi.fn(),
      setBackgroundImage: vi.fn().mockResolvedValue(undefined),
      setUpDirection: vi.fn(),
      setMaterialMode: vi.fn(),
      exportModel: vi.fn().mockResolvedValue(undefined),
      handleResize: vi.fn(),
      updateStatusMouseOnViewer: vi.fn(),
      getCameraState: vi.fn().mockReturnValue({
        position: { x: 0, y: 0, z: 0 },
        target: { x: 0, y: 0, z: 0 },
        zoom: 1,
        cameraType: 'perspective'
      }),
      forceRender: vi.fn(),
      remove: vi.fn(),
      setTargetSize: vi.fn()
    }

    mockSourceLoad3d = {
      getCurrentCameraType: vi.fn().mockReturnValue('perspective'),
      getCameraState: vi.fn().mockReturnValue({
        position: { x: 1, y: 1, z: 1 },
        target: { x: 0, y: 0, z: 0 },
        zoom: 1,
        cameraType: 'perspective'
      }),
      sceneManager: createMockSceneManager(),
      lightingManager: {
        lights: [null, { intensity: 1 }]
      } as Load3d['lightingManager'],
      cameraManager: {
        perspectiveCamera: { fov: 75 }
      } as Load3d['cameraManager'],
      modelManager: {
        currentUpDirection: 'original',
        materialMode: 'original'
      } as Load3d['modelManager'],
      setBackgroundImage: vi.fn().mockResolvedValue(undefined),
      setBackgroundRenderMode: vi.fn(),
      forceRender: vi.fn()
    }

    vi.mocked(Load3d).mockImplementation(function () {
      Object.assign(this, mockLoad3d)
    })

    mockLoad3dService = {
      copyLoad3dState: vi.fn().mockResolvedValue(undefined),
      handleViewportRefresh: vi.fn(),
      getLoad3d: vi.fn().mockReturnValue(mockSourceLoad3d)
    } as Partial<ReturnType<typeof useLoad3dService>> as ReturnType<
      typeof useLoad3dService
    >
    vi.mocked(useLoad3dService).mockReturnValue(mockLoad3dService)

    mockToastStore = {
      addAlert: vi.fn()
    } as Partial<ReturnType<typeof useToastStore>> as ReturnType<
      typeof useToastStore
    >
    vi.mocked(useToastStore).mockReturnValue(mockToastStore)
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  describe('initialization', () => {
    it('should initialize with default values', () => {
      const viewer = useLoad3dViewer(mockNode)

      expect(viewer.backgroundColor.value).toBe('')
      expect(viewer.showGrid.value).toBe(true)
      expect(viewer.cameraType.value).toBe('perspective')
      expect(viewer.fov.value).toBe(75)
      expect(viewer.lightIntensity.value).toBe(1)
      expect(viewer.backgroundImage.value).toBe('')
      expect(viewer.hasBackgroundImage.value).toBe(false)
      expect(viewer.upDirection.value).toBe('original')
      expect(viewer.materialMode.value).toBe('original')
    })

    it('should initialize viewer with source Load3d state', async () => {
      const viewer = useLoad3dViewer(mockNode)
      const containerRef = document.createElement('div')

      await viewer.initializeViewer(containerRef, mockSourceLoad3d as Load3d)

      expect(Load3d).toHaveBeenCalledWith(containerRef, {
        width: undefined,
        height: undefined,
        getDimensions: undefined,
        isViewerMode: false
      })

      expect(mockLoad3dService.copyLoad3dState).toHaveBeenCalledWith(
        mockSourceLoad3d,
        mockLoad3d
      )

      expect(viewer.cameraType.value).toBe('perspective')
      expect(viewer.backgroundColor.value).toBe('#282828')
      expect(viewer.showGrid.value).toBe(true)
      expect(viewer.lightIntensity.value).toBe(1)
      expect(viewer.fov.value).toBe(75)
      expect(viewer.upDirection.value).toBe('original')
      expect(viewer.materialMode.value).toBe('original')
    })

    it('should handle background image during initialization', async () => {
      vi.mocked(
        mockSourceLoad3d.sceneManager!.getCurrentBackgroundInfo
      ).mockReturnValue({
        type: 'image',
        value: ''
      })
      ;(
        mockNode.properties!['Scene Config'] as Record<string, unknown>
      ).backgroundImage = 'test-image.jpg'

      const viewer = useLoad3dViewer(mockNode)
      const containerRef = document.createElement('div')

      await viewer.initializeViewer(containerRef, mockSourceLoad3d as Load3d)

      expect(viewer.backgroundImage.value).toBe('test-image.jpg')
      expect(viewer.hasBackgroundImage.value).toBe(true)
    })

    it('should handle initialization errors', async () => {
      vi.mocked(Load3d).mockImplementationOnce(function () {
        throw new Error('Load3d creation failed')
      })

      const viewer = useLoad3dViewer(mockNode)
      const containerRef = document.createElement('div')

      await viewer.initializeViewer(containerRef, mockSourceLoad3d as Load3d)

      expect(mockToastStore.addAlert).toHaveBeenCalledWith(
        'toastMessages.failedToInitializeLoad3dViewer'
      )
    })
  })

  describe('state watchers', () => {
    it('should update background color when state changes', async () => {
      const viewer = useLoad3dViewer(mockNode)
      const containerRef = document.createElement('div')

      await viewer.initializeViewer(containerRef, mockSourceLoad3d as Load3d)

      viewer.backgroundColor.value = '#ff0000'
      await nextTick()

      expect(mockLoad3d.setBackgroundColor).toHaveBeenCalledWith('#ff0000')
    })

    it('should update grid visibility when state changes', async () => {
      const viewer = useLoad3dViewer(mockNode)
      const containerRef = document.createElement('div')

      await viewer.initializeViewer(containerRef, mockSourceLoad3d as Load3d)

      viewer.showGrid.value = false
      await nextTick()

      expect(mockLoad3d.toggleGrid).toHaveBeenCalledWith(false)
    })

    it('should update camera type when state changes', async () => {
      const viewer = useLoad3dViewer(mockNode)
      const containerRef = document.createElement('div')

      await viewer.initializeViewer(containerRef, mockSourceLoad3d as Load3d)

      viewer.cameraType.value = 'orthographic'
      await nextTick()

      expect(mockLoad3d.toggleCamera).toHaveBeenCalledWith('orthographic')
    })

    it('should update FOV when state changes', async () => {
      const viewer = useLoad3dViewer(mockNode)
      const containerRef = document.createElement('div')

      await viewer.initializeViewer(containerRef, mockSourceLoad3d as Load3d)

      viewer.fov.value = 90
      await nextTick()

      expect(mockLoad3d.setFOV).toHaveBeenCalledWith(90)
    })

    it('should update light intensity when state changes', async () => {
      const viewer = useLoad3dViewer(mockNode)
      const containerRef = document.createElement('div')

      await viewer.initializeViewer(containerRef, mockSourceLoad3d as Load3d)

      viewer.lightIntensity.value = 2
      await nextTick()

      expect(mockLoad3d.setLightIntensity).toHaveBeenCalledWith(2)
    })

    it('should update background image when state changes', async () => {
      const viewer = useLoad3dViewer(mockNode)
      const containerRef = document.createElement('div')

      await viewer.initializeViewer(containerRef, mockSourceLoad3d as Load3d)

      viewer.backgroundImage.value = 'new-bg.jpg'
      await nextTick()

      expect(mockLoad3d.setBackgroundImage).toHaveBeenCalledWith('new-bg.jpg')
      expect(viewer.hasBackgroundImage.value).toBe(true)
    })

    it('should update up direction when state changes', async () => {
      const viewer = useLoad3dViewer(mockNode)
      const containerRef = document.createElement('div')

      await viewer.initializeViewer(containerRef, mockSourceLoad3d as Load3d)

      viewer.upDirection.value = '+y'
      await nextTick()

      expect(mockLoad3d.setUpDirection).toHaveBeenCalledWith('+y')
    })

    it('should update material mode when state changes', async () => {
      const viewer = useLoad3dViewer(mockNode)
      const containerRef = document.createElement('div')

      await viewer.initializeViewer(containerRef, mockSourceLoad3d as Load3d)

      viewer.materialMode.value = 'wireframe'
      await nextTick()

      expect(mockLoad3d.setMaterialMode).toHaveBeenCalledWith('wireframe')
    })

    it('should handle watcher errors gracefully', async () => {
      vi.mocked(mockLoad3d.setBackgroundColor!).mockImplementationOnce(
        function () {
          throw new Error('Color update failed')
        }
      )

      const viewer = useLoad3dViewer(mockNode)
      const containerRef = document.createElement('div')

      await viewer.initializeViewer(containerRef, mockSourceLoad3d as Load3d)

      viewer.backgroundColor.value = '#ff0000'
      await nextTick()

      expect(mockToastStore.addAlert).toHaveBeenCalledWith(
        'toastMessages.failedToUpdateBackgroundColor'
      )
    })
  })

  describe('exportModel', () => {
    it('should export model successfully', async () => {
      const viewer = useLoad3dViewer(mockNode)
      const containerRef = document.createElement('div')

      await viewer.initializeViewer(containerRef, mockSourceLoad3d as Load3d)

      await viewer.exportModel('glb')

      expect(mockLoad3d.exportModel).toHaveBeenCalledWith('glb')
    })

    it('should handle export errors', async () => {
      vi.mocked(mockLoad3d.exportModel!).mockRejectedValueOnce(
        new Error('Export failed')
      )

      const viewer = useLoad3dViewer(mockNode)
      const containerRef = document.createElement('div')

      await viewer.initializeViewer(containerRef, mockSourceLoad3d as Load3d)

      await viewer.exportModel('glb')

      expect(mockToastStore.addAlert).toHaveBeenCalledWith(
        'toastMessages.failedToExportModel'
      )
    })

    it('should not export when load3d is not initialized', async () => {
      const viewer = useLoad3dViewer(mockNode)

      await viewer.exportModel('glb')

      expect(mockLoad3d.exportModel).not.toHaveBeenCalled()
    })
  })

  describe('UI interaction methods', () => {
    it('should handle resize', async () => {
      const viewer = useLoad3dViewer(mockNode)
      const containerRef = document.createElement('div')

      await viewer.initializeViewer(containerRef, mockSourceLoad3d as Load3d)

      viewer.handleResize()

      expect(mockLoad3d.handleResize).toHaveBeenCalled()
    })

    it('should handle mouse enter', async () => {
      const viewer = useLoad3dViewer(mockNode)
      const containerRef = document.createElement('div')

      await viewer.initializeViewer(containerRef, mockSourceLoad3d as Load3d)

      viewer.handleMouseEnter()

      expect(mockLoad3d.updateStatusMouseOnViewer).toHaveBeenCalledWith(true)
    })

    it('should handle mouse leave', async () => {
      const viewer = useLoad3dViewer(mockNode)
      const containerRef = document.createElement('div')

      await viewer.initializeViewer(containerRef, mockSourceLoad3d as Load3d)

      viewer.handleMouseLeave()

      expect(mockLoad3d.updateStatusMouseOnViewer).toHaveBeenCalledWith(false)
    })
  })

  describe('restoreInitialState', () => {
    it('should restore all properties to initial values', async () => {
      const viewer = useLoad3dViewer(mockNode)
      const containerRef = document.createElement('div')

      await viewer.initializeViewer(containerRef, mockSourceLoad3d as Load3d)
      ;(
        mockNode.properties!['Scene Config'] as Record<string, unknown>
      ).backgroundColor = '#ff0000'
      ;(
        mockNode.properties!['Scene Config'] as Record<string, unknown>
      ).showGrid = false

      viewer.restoreInitialState()

      expect(
        (mockNode.properties!['Scene Config'] as Record<string, unknown>)
          .backgroundColor
      ).toBe('#282828')
      expect(
        (mockNode.properties!['Scene Config'] as Record<string, unknown>)
          .showGrid
      ).toBe(true)
      expect(
        (mockNode.properties!['Camera Config'] as Record<string, unknown>)
          .cameraType
      ).toBe('perspective')
      expect(
        (mockNode.properties!['Camera Config'] as Record<string, unknown>).fov
      ).toBe(75)
      expect(
        (mockNode.properties!['Light Config'] as Record<string, unknown>)
          .intensity
      ).toBe(1)
    })
  })

  describe('applyChanges', () => {
    it('should apply all changes to source and node', async () => {
      const viewer = useLoad3dViewer(mockNode)
      const containerRef = document.createElement('div')

      await viewer.initializeViewer(containerRef, mockSourceLoad3d as Load3d)

      viewer.backgroundColor.value = '#ff0000'
      viewer.showGrid.value = false

      const result = await viewer.applyChanges()

      expect(result).toBe(true)
      expect(
        (mockNode.properties!['Scene Config'] as Record<string, unknown>)
          .backgroundColor
      ).toBe('#ff0000')
      expect(
        (mockNode.properties!['Scene Config'] as Record<string, unknown>)
          .showGrid
      ).toBe(false)
      expect(mockLoad3dService.copyLoad3dState).toHaveBeenCalledWith(
        mockLoad3d,
        mockSourceLoad3d
      )
      expect(mockSourceLoad3d.forceRender).toHaveBeenCalled()
      expect(mockNode.graph!.setDirtyCanvas).toHaveBeenCalledWith(true, true)
    })

    it('should handle background image during apply', async () => {
      const viewer = useLoad3dViewer(mockNode)
      const containerRef = document.createElement('div')

      await viewer.initializeViewer(containerRef, mockSourceLoad3d as Load3d)

      viewer.backgroundImage.value = 'new-bg.jpg'

      await viewer.applyChanges()

      expect(mockSourceLoad3d.setBackgroundImage).toHaveBeenCalledWith(
        'new-bg.jpg'
      )
    })

    it('should return false when no load3d instances', async () => {
      const viewer = useLoad3dViewer(mockNode)

      const result = await viewer.applyChanges()

      expect(result).toBe(false)
    })
  })

  describe('refreshViewport', () => {
    it('should refresh viewport', async () => {
      const viewer = useLoad3dViewer(mockNode)
      const containerRef = document.createElement('div')

      await viewer.initializeViewer(containerRef, mockSourceLoad3d as Load3d)

      viewer.refreshViewport()

      expect(mockLoad3dService.handleViewportRefresh).toHaveBeenCalledWith(
        mockLoad3d
      )
    })
  })

  describe('handleBackgroundImageUpdate', () => {
    it('should upload and set background image', async () => {
      vi.mocked(Load3dUtils.uploadFile).mockResolvedValue('uploaded-image.jpg')

      const viewer = useLoad3dViewer(mockNode)
      const containerRef = document.createElement('div')

      await viewer.initializeViewer(containerRef, mockSourceLoad3d as Load3d)

      const file = new File([''], 'test.jpg', { type: 'image/jpeg' })
      await viewer.handleBackgroundImageUpdate(file)

      expect(Load3dUtils.uploadFile).toHaveBeenCalledWith(file, '3d')
      expect(viewer.backgroundImage.value).toBe('uploaded-image.jpg')
      expect(viewer.hasBackgroundImage.value).toBe(true)
    })

    it('should use resource folder for upload', async () => {
      mockNode.properties['Resource Folder'] = 'subfolder'
      vi.mocked(Load3dUtils.uploadFile).mockResolvedValue('uploaded-image.jpg')

      const viewer = useLoad3dViewer(mockNode)
      const containerRef = document.createElement('div')

      await viewer.initializeViewer(containerRef, mockSourceLoad3d as Load3d)

      const file = new File([''], 'test.jpg', { type: 'image/jpeg' })
      await viewer.handleBackgroundImageUpdate(file)

      expect(Load3dUtils.uploadFile).toHaveBeenCalledWith(file, '3d/subfolder')
    })

    it('should clear background image when file is null', async () => {
      const viewer = useLoad3dViewer(mockNode)
      const containerRef = document.createElement('div')

      await viewer.initializeViewer(containerRef, mockSourceLoad3d as Load3d)

      viewer.backgroundImage.value = 'existing.jpg'
      viewer.hasBackgroundImage.value = true

      await viewer.handleBackgroundImageUpdate(null)

      expect(viewer.backgroundImage.value).toBe('')
      expect(viewer.hasBackgroundImage.value).toBe(false)
    })

    it('should handle upload errors', async () => {
      vi.mocked(Load3dUtils.uploadFile).mockRejectedValueOnce(
        new Error('Upload failed')
      )

      const viewer = useLoad3dViewer(mockNode)
      const containerRef = document.createElement('div')

      await viewer.initializeViewer(containerRef, mockSourceLoad3d as Load3d)

      const file = new File([''], 'test.jpg', { type: 'image/jpeg' })
      await viewer.handleBackgroundImageUpdate(file)

      expect(mockToastStore.addAlert).toHaveBeenCalledWith(
        'toastMessages.failedToUploadBackgroundImage'
      )
    })
  })

  describe('cleanup', () => {
    it('should clean up resources', async () => {
      const viewer = useLoad3dViewer(mockNode)
      const containerRef = document.createElement('div')

      await viewer.initializeViewer(containerRef, mockSourceLoad3d as Load3d)

      viewer.cleanup()

      expect(mockLoad3d.remove).toHaveBeenCalled()
    })

    it('should handle cleanup when not initialized', () => {
      const viewer = useLoad3dViewer(mockNode)

      expect(() => viewer.cleanup()).not.toThrow()
    })
  })

  describe('edge cases', () => {
    it('should handle missing container ref', async () => {
      const viewer = useLoad3dViewer(mockNode)

      await viewer.initializeViewer(null!, mockSourceLoad3d as Load3d)

      expect(Load3d).not.toHaveBeenCalled()
    })

    it('should handle orthographic camera', async () => {
      vi.mocked(mockSourceLoad3d.getCurrentCameraType!).mockReturnValue(
        'orthographic'
      )
      mockSourceLoad3d.cameraManager = {
        perspectiveCamera: { fov: 75 }
      } as Partial<Load3d['cameraManager']> as Load3d['cameraManager']
      delete (mockNode.properties!['Camera Config'] as Record<string, unknown>)
        .cameraType

      const viewer = useLoad3dViewer(mockNode)
      const containerRef = document.createElement('div')

      await viewer.initializeViewer(containerRef, mockSourceLoad3d as Load3d)

      expect(viewer.cameraType.value).toBe('orthographic')
    })

    it('should handle missing lights', async () => {
      mockSourceLoad3d.lightingManager!.lights = []

      const viewer = useLoad3dViewer(mockNode)
      const containerRef = document.createElement('div')

      await viewer.initializeViewer(containerRef, mockSourceLoad3d as Load3d)

      expect(viewer.lightIntensity.value).toBe(1) // Default value
    })
  })
})
