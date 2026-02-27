import * as THREE from 'three'

import { AnimationManager } from './AnimationManager'
import { CameraManager } from './CameraManager'
import { ControlsManager } from './ControlsManager'
import { EventManager } from './EventManager'
import { LightingManager } from './LightingManager'
import { LoaderManager } from './LoaderManager'
import { ModelExporter } from './ModelExporter'
import { RecordingManager } from './RecordingManager'
import { SceneManager } from './SceneManager'
import { SceneModelManager } from './SceneModelManager'
import { ViewHelperManager } from './ViewHelperManager'
import {
  type CameraState,
  type CaptureResult,
  type EventCallback,
  type Load3DOptions,
  type MaterialMode,
  type UpDirection
} from './interfaces'

class Load3d {
  renderer: THREE.WebGLRenderer
  protected clock: THREE.Clock
  protected animationFrameId: number | null = null
  private loadingPromise: Promise<void> | null = null
  private onContextMenuCallback?: (event: MouseEvent) => void
  private getDimensionsCallback?: () => { width: number; height: number } | null

  eventManager: EventManager
  sceneManager: SceneManager
  cameraManager: CameraManager
  controlsManager: ControlsManager
  lightingManager: LightingManager
  viewHelperManager: ViewHelperManager
  loaderManager: LoaderManager
  modelManager: SceneModelManager
  recordingManager: RecordingManager
  animationManager: AnimationManager

  STATUS_MOUSE_ON_NODE: boolean
  STATUS_MOUSE_ON_SCENE: boolean
  STATUS_MOUSE_ON_VIEWER: boolean
  INITIAL_RENDER_DONE: boolean = false

  targetWidth: number = 0
  targetHeight: number = 0
  targetAspectRatio: number = 1
  isViewerMode: boolean = false

  // Context menu tracking
  private rightMouseDownX: number = 0
  private rightMouseDownY: number = 0
  private rightMouseMoved: boolean = false
  private readonly dragThreshold: number = 5
  private contextMenuAbortController: AbortController | null = null
  private resizeObserver: ResizeObserver | null = null

  constructor(container: Element | HTMLElement, options: Load3DOptions = {}) {
    this.clock = new THREE.Clock()
    this.isViewerMode = options.isViewerMode || false
    this.onContextMenuCallback = options.onContextMenu
    this.getDimensionsCallback = options.getDimensions

    if (options.width && options.height) {
      this.targetWidth = options.width
      this.targetHeight = options.height
      this.targetAspectRatio = options.width / options.height
    }

    this.renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true })
    this.renderer.setSize(300, 300)
    this.renderer.setClearColor(0x282828)
    this.renderer.autoClear = false
    this.renderer.outputColorSpace = THREE.SRGBColorSpace
    this.renderer.domElement.classList.add(
      'absolute',
      'inset-0',
      'h-full',
      'w-full',
      'outline-none'
    )
    container.appendChild(this.renderer.domElement)

    this.eventManager = new EventManager()

    this.sceneManager = new SceneManager(
      this.renderer,
      this.getActiveCamera.bind(this),
      this.getControls.bind(this),
      this.eventManager
    )

    this.cameraManager = new CameraManager(this.renderer, this.eventManager)

    this.controlsManager = new ControlsManager(
      this.renderer,
      this.cameraManager.activeCamera,
      this.eventManager
    )

    this.cameraManager.setControls(this.controlsManager.controls)

    this.lightingManager = new LightingManager(
      this.sceneManager.scene,
      this.eventManager
    )

    this.viewHelperManager = new ViewHelperManager(
      this.renderer,
      this.getActiveCamera.bind(this),
      this.getControls.bind(this),
      this.eventManager
    )

    this.modelManager = new SceneModelManager(
      this.sceneManager.scene,
      this.renderer,
      this.eventManager,
      this.getActiveCamera.bind(this),
      this.setupCamera.bind(this)
    )

    this.loaderManager = new LoaderManager(this.modelManager, this.eventManager)

    this.recordingManager = new RecordingManager(
      this.sceneManager.scene,
      this.renderer,
      this.eventManager
    )

    this.animationManager = new AnimationManager(this.eventManager)
    this.sceneManager.init()
    this.cameraManager.init()
    this.controlsManager.init()
    this.lightingManager.init()
    this.loaderManager.init()
    this.animationManager.init()

    this.viewHelperManager.createViewHelper(container)
    this.viewHelperManager.init()

    this.STATUS_MOUSE_ON_NODE = false
    this.STATUS_MOUSE_ON_SCENE = false
    this.STATUS_MOUSE_ON_VIEWER = false

    this.initContextMenu()
    this.initResizeObserver(container)

    this.handleResize()
    this.startAnimation()

    setTimeout(() => {
      this.forceRender()
    }, 100)
  }

  private initResizeObserver(container: Element | HTMLElement): void {
    if (typeof ResizeObserver === 'undefined') return

    this.resizeObserver?.disconnect()
    this.resizeObserver = new ResizeObserver(() => {
      this.handleResize()
    })
    this.resizeObserver.observe(container)
  }

  /**
   * Initialize context menu on the Three.js canvas
   * Detects right-click vs right-drag to show menu only on click
   */
  private initContextMenu(): void {
    const canvas = this.renderer.domElement

    this.contextMenuAbortController = new AbortController()
    const { signal } = this.contextMenuAbortController

    const mousedownHandler = (e: MouseEvent) => {
      if (e.button === 2) {
        this.rightMouseDownX = e.clientX
        this.rightMouseDownY = e.clientY
        this.rightMouseMoved = false
      }
    }

    const mousemoveHandler = (e: MouseEvent) => {
      if (e.buttons === 2) {
        const dx = Math.abs(e.clientX - this.rightMouseDownX)
        const dy = Math.abs(e.clientY - this.rightMouseDownY)

        if (dx > this.dragThreshold || dy > this.dragThreshold) {
          this.rightMouseMoved = true
        }
      }
    }

    const contextmenuHandler = (e: MouseEvent) => {
      if (this.isViewerMode) return

      const dx = Math.abs(e.clientX - this.rightMouseDownX)
      const dy = Math.abs(e.clientY - this.rightMouseDownY)
      const wasDragging =
        this.rightMouseMoved ||
        dx > this.dragThreshold ||
        dy > this.dragThreshold

      this.rightMouseMoved = false

      if (wasDragging) {
        return
      }

      e.preventDefault()
      e.stopPropagation()

      this.showNodeContextMenu(e)
    }

    canvas.addEventListener('mousedown', mousedownHandler, { signal })
    canvas.addEventListener('mousemove', mousemoveHandler, { signal })
    canvas.addEventListener('contextmenu', contextmenuHandler, { signal })
  }

  private showNodeContextMenu(event: MouseEvent): void {
    if (this.onContextMenuCallback) {
      this.onContextMenuCallback(event)
    }
  }

  getEventManager(): EventManager {
    return this.eventManager
  }

  getSceneManager(): SceneManager {
    return this.sceneManager
  }
  getCameraManager(): CameraManager {
    return this.cameraManager
  }
  getControlsManager(): ControlsManager {
    return this.controlsManager
  }
  getLightingManager(): LightingManager {
    return this.lightingManager
  }
  getViewHelperManager(): ViewHelperManager {
    return this.viewHelperManager
  }
  getLoaderManager(): LoaderManager {
    return this.loaderManager
  }
  getModelManager(): SceneModelManager {
    return this.modelManager
  }
  getRecordingManager(): RecordingManager {
    return this.recordingManager
  }

  getTargetSize(): { width: number; height: number } {
    return {
      width: this.targetWidth,
      height: this.targetHeight
    }
  }

  private shouldMaintainAspectRatio(): boolean {
    return this.isViewerMode || (this.targetWidth > 0 && this.targetHeight > 0)
  }

  forceRender(): void {
    const delta = this.clock.getDelta()
    this.animationManager.update(delta)
    this.viewHelperManager.update(delta)
    this.controlsManager.update()

    this.renderMainScene()

    this.resetViewport()

    if (this.viewHelperManager.viewHelper.render) {
      this.viewHelperManager.viewHelper.render(this.renderer)
    }

    this.INITIAL_RENDER_DONE = true
  }

  renderMainScene(): void {
    const containerWidth = this.renderer.domElement.clientWidth
    const containerHeight = this.renderer.domElement.clientHeight

    if (this.getDimensionsCallback) {
      const dims = this.getDimensionsCallback()
      if (dims) {
        this.targetWidth = dims.width
        this.targetHeight = dims.height
        this.targetAspectRatio = dims.width / dims.height
      }
    }

    if (this.shouldMaintainAspectRatio()) {
      const containerAspectRatio = containerWidth / containerHeight

      let renderWidth: number
      let renderHeight: number
      let offsetX: number = 0
      let offsetY: number = 0

      if (containerAspectRatio > this.targetAspectRatio) {
        renderHeight = containerHeight
        renderWidth = renderHeight * this.targetAspectRatio
        offsetX = (containerWidth - renderWidth) / 2
      } else {
        renderWidth = containerWidth
        renderHeight = renderWidth / this.targetAspectRatio
        offsetY = (containerHeight - renderHeight) / 2
      }

      this.renderer.setViewport(0, 0, containerWidth, containerHeight)
      this.renderer.setScissor(0, 0, containerWidth, containerHeight)
      this.renderer.setScissorTest(true)
      this.renderer.setClearColor(0x0a0a0a)
      this.renderer.clear()

      this.renderer.setViewport(offsetX, offsetY, renderWidth, renderHeight)
      this.renderer.setScissor(offsetX, offsetY, renderWidth, renderHeight)

      const renderAspectRatio = renderWidth / renderHeight
      this.cameraManager.updateAspectRatio(renderAspectRatio)
    } else {
      // No aspect ratio constraint: fill the entire container
      this.renderer.setViewport(0, 0, containerWidth, containerHeight)
      this.renderer.setScissor(0, 0, containerWidth, containerHeight)
      this.renderer.setScissorTest(true)
    }

    this.sceneManager.renderBackground()
    this.renderer.render(
      this.sceneManager.scene,
      this.cameraManager.activeCamera
    )
  }

  resetViewport(): void {
    const width = this.renderer.domElement.clientWidth
    const height = this.renderer.domElement.clientHeight

    this.renderer.setViewport(0, 0, width, height)
    this.renderer.setScissor(0, 0, width, height)
    this.renderer.setScissorTest(false)
  }

  private getActiveCamera(): THREE.Camera {
    return this.cameraManager.activeCamera
  }

  private getControls() {
    return this.controlsManager.controls
  }

  private setupCamera(size: THREE.Vector3): void {
    this.cameraManager.setupForModel(size)
  }

  private startAnimation(): void {
    const animate = () => {
      this.animationFrameId = requestAnimationFrame(animate)

      if (!this.isActive()) {
        return
      }

      const delta = this.clock.getDelta()
      this.animationManager.update(delta)
      this.viewHelperManager.update(delta)
      this.controlsManager.update()

      this.renderMainScene()

      this.resetViewport()

      if (this.viewHelperManager.viewHelper.render) {
        this.viewHelperManager.viewHelper.render(this.renderer)
      }
    }

    animate()
  }

  updateStatusMouseOnNode(onNode: boolean): void {
    this.STATUS_MOUSE_ON_NODE = onNode
  }

  updateStatusMouseOnScene(onScene: boolean): void {
    this.STATUS_MOUSE_ON_SCENE = onScene
  }

  updateStatusMouseOnViewer(onViewer: boolean): void {
    this.STATUS_MOUSE_ON_VIEWER = onViewer
  }

  isActive(): boolean {
    return (
      this.STATUS_MOUSE_ON_NODE ||
      this.STATUS_MOUSE_ON_SCENE ||
      this.STATUS_MOUSE_ON_VIEWER ||
      this.isRecording() ||
      !this.INITIAL_RENDER_DONE ||
      this.animationManager.isAnimationPlaying
    )
  }

  async exportModel(format: string): Promise<void> {
    if (!this.modelManager.currentModel) {
      throw new Error('No model to export')
    }

    const exportMessage = `Exporting as ${format.toUpperCase()}...`
    this.eventManager.emitEvent('exportLoadingStart', exportMessage)

    try {
      const model = this.modelManager.currentModel.clone()

      const originalFileName = this.modelManager.originalFileName || 'model'
      const filename = `${originalFileName}.${format}`

      const originalURL = this.modelManager.originalURL

      await new Promise((resolve) => setTimeout(resolve, 10))

      switch (format) {
        case 'glb':
          await ModelExporter.exportGLB(model, filename, originalURL)
          break
        case 'obj':
          await ModelExporter.exportOBJ(model, filename, originalURL)
          break
        case 'stl':
          ;(await ModelExporter.exportSTL(model, filename), originalURL)
          break
        default:
          throw new Error(`Unsupported export format: ${format}`)
      }

      await new Promise((resolve) => setTimeout(resolve, 10))
    } catch (error) {
      console.error(`Error exporting model as ${format}:`, error)
      throw error
    } finally {
      this.eventManager.emitEvent('exportLoadingEnd', null)
    }
  }

  setBackgroundColor(color: string): void {
    this.sceneManager.setBackgroundColor(color)

    this.forceRender()
  }

  async setBackgroundImage(uploadPath: string): Promise<void> {
    await this.sceneManager.setBackgroundImage(uploadPath)

    if (
      this.sceneManager.backgroundTexture &&
      this.sceneManager.backgroundMesh
    ) {
      const containerWidth = this.renderer.domElement.clientWidth
      const containerHeight = this.renderer.domElement.clientHeight

      if (this.shouldMaintainAspectRatio()) {
        const containerAspectRatio = containerWidth / containerHeight

        let renderWidth: number
        let renderHeight: number

        if (containerAspectRatio > this.targetAspectRatio) {
          renderHeight = containerHeight
          renderWidth = renderHeight * this.targetAspectRatio
        } else {
          renderWidth = containerWidth
          renderHeight = renderWidth / this.targetAspectRatio
        }

        this.sceneManager.updateBackgroundSize(
          this.sceneManager.backgroundTexture,
          this.sceneManager.backgroundMesh,
          renderWidth,
          renderHeight
        )
      } else {
        // No aspect ratio constraints: fill container
        this.sceneManager.updateBackgroundSize(
          this.sceneManager.backgroundTexture,
          this.sceneManager.backgroundMesh,
          containerWidth,
          containerHeight
        )
      }
    }

    this.forceRender()
  }

  removeBackgroundImage(): void {
    this.sceneManager.removeBackgroundImage()

    this.forceRender()
  }

  toggleGrid(showGrid: boolean): void {
    this.sceneManager.toggleGrid(showGrid)
    this.forceRender()
  }

  setBackgroundRenderMode(mode: 'tiled' | 'panorama'): void {
    this.sceneManager.setBackgroundRenderMode(mode)
    this.forceRender()
  }

  toggleCamera(cameraType?: 'perspective' | 'orthographic'): void {
    this.cameraManager.toggleCamera(cameraType)

    this.controlsManager.updateCamera(this.cameraManager.activeCamera)
    this.viewHelperManager.recreateViewHelper()

    this.handleResize()
  }

  getCurrentCameraType(): 'perspective' | 'orthographic' {
    return this.cameraManager.getCurrentCameraType()
  }

  getCurrentModel(): THREE.Object3D | null {
    return this.modelManager.currentModel
  }

  setCameraState(state: CameraState): void {
    this.cameraManager.setCameraState(state)

    this.forceRender()
  }

  getCameraState(): CameraState {
    return this.cameraManager.getCameraState()
  }

  setFOV(fov: number): void {
    this.cameraManager.setFOV(fov)
    this.forceRender()
  }

  setMaterialMode(mode: MaterialMode): void {
    this.modelManager.setMaterialMode(mode)
    this.forceRender()
  }

  async loadModel(url: string, originalFileName?: string): Promise<void> {
    if (this.loadingPromise) {
      try {
        await this.loadingPromise
      } catch (e) {}
    }

    this.loadingPromise = this._loadModelInternal(url, originalFileName)
    return this.loadingPromise
  }

  private async _loadModelInternal(
    url: string,
    originalFileName?: string
  ): Promise<void> {
    this.cameraManager.reset()
    this.controlsManager.reset()
    this.modelManager.clearModel()
    this.animationManager.dispose()

    await this.loaderManager.loadModel(url, originalFileName)

    // Auto-detect and setup animations if present
    if (this.modelManager.currentModel) {
      this.animationManager.setupModelAnimations(
        this.modelManager.currentModel,
        this.modelManager.originalModel
      )
    }

    this.handleResize()

    this.loadingPromise = null
  }

  isSplatModel(): boolean {
    return this.modelManager.containsSplatMesh()
  }

  isPlyModel(): boolean {
    return this.modelManager.originalModel instanceof THREE.BufferGeometry
  }

  clearModel(): void {
    this.animationManager.dispose()
    this.modelManager.clearModel()
    this.forceRender()
  }

  setUpDirection(direction: UpDirection): void {
    this.modelManager.setUpDirection(direction)
    this.forceRender()
  }

  setLightIntensity(intensity: number): void {
    this.lightingManager.setLightIntensity(intensity)
    this.forceRender()
  }

  setTargetSize(width: number, height: number): void {
    this.targetWidth = width
    this.targetHeight = height
    this.targetAspectRatio = width / height
    this.handleResize()
  }

  addEventListener<T>(event: string, callback: EventCallback<T>): void {
    this.eventManager.addEventListener(event, callback)
  }

  removeEventListener<T>(event: string, callback: EventCallback<T>): void {
    this.eventManager.removeEventListener(event, callback)
  }

  refreshViewport(): void {
    this.handleResize()
  }

  handleResize(): void {
    const parentElement = this.renderer?.domElement?.parentElement

    if (!parentElement) {
      console.warn('Parent element not found')
      return
    }

    const containerWidth = parentElement.clientWidth
    const containerHeight = parentElement.clientHeight

    if (this.getDimensionsCallback) {
      const dims = this.getDimensionsCallback()
      if (dims) {
        this.targetWidth = dims.width
        this.targetHeight = dims.height
        this.targetAspectRatio = dims.width / dims.height
      }
    }

    if (this.shouldMaintainAspectRatio()) {
      const containerAspectRatio = containerWidth / containerHeight
      let renderWidth: number
      let renderHeight: number

      if (containerAspectRatio > this.targetAspectRatio) {
        renderHeight = containerHeight
        renderWidth = renderHeight * this.targetAspectRatio
      } else {
        renderWidth = containerWidth
        renderHeight = renderWidth / this.targetAspectRatio
      }

      this.renderer.setSize(containerWidth, containerHeight)
      this.cameraManager.handleResize(renderWidth, renderHeight)
      this.sceneManager.handleResize(renderWidth, renderHeight)
    } else {
      // No aspect ratio constraint: use container dimensions directly
      this.renderer.setSize(containerWidth, containerHeight)
      this.cameraManager.handleResize(containerWidth, containerHeight)
      this.sceneManager.handleResize(containerWidth, containerHeight)
    }

    this.forceRender()
  }

  captureScene(width: number, height: number): Promise<CaptureResult> {
    return this.sceneManager.captureScene(width, height)
  }

  public async startRecording(): Promise<void> {
    this.viewHelperManager.visibleViewHelper(false)

    return this.recordingManager.startRecording(
      this.targetWidth,
      this.targetHeight
    )
  }

  public stopRecording(): void {
    this.viewHelperManager.visibleViewHelper(true)

    this.recordingManager.stopRecording()

    this.eventManager.emitEvent('recordingStatusChange', false)
  }

  public isRecording(): boolean {
    return this.recordingManager.getIsRecording()
  }

  public getRecordingDuration(): number {
    return this.recordingManager.getRecordingDuration()
  }

  public getRecordingData(): string | null {
    return this.recordingManager.getRecordingData()
  }

  public exportRecording(filename?: string): void {
    this.recordingManager.exportRecording(filename)
  }

  public clearRecording(): void {
    this.recordingManager.clearRecording()
  }

  // Animation methods
  public setAnimationSpeed(speed: number): void {
    this.animationManager.setAnimationSpeed(speed)
  }

  public updateSelectedAnimation(index: number): void {
    this.animationManager.updateSelectedAnimation(index)
  }

  public toggleAnimation(play?: boolean): void {
    this.animationManager.toggleAnimation(play)
  }

  public hasAnimations(): boolean {
    return this.animationManager.animationClips.length > 0
  }

  public hasSkeleton(): boolean {
    return this.modelManager.hasSkeleton()
  }

  public setShowSkeleton(show: boolean): void {
    this.modelManager.setShowSkeleton(show)
    this.forceRender()
  }

  public getShowSkeleton(): boolean {
    return this.modelManager.showSkeleton
  }

  public getAnimationTime(): number {
    return this.animationManager.getAnimationTime()
  }

  public getAnimationDuration(): number {
    return this.animationManager.getAnimationDuration()
  }

  public setAnimationTime(time: number): void {
    this.animationManager.setAnimationTime(time)
    this.forceRender()
  }

  public async captureThumbnail(
    width: number = 256,
    height: number = 256
  ): Promise<string> {
    if (!this.modelManager.currentModel) {
      throw new Error('No model loaded for thumbnail capture')
    }

    const savedState = this.cameraManager.getCameraState()
    const savedCameraType = this.cameraManager.getCurrentCameraType()
    const savedGridVisible = this.sceneManager.gridHelper.visible

    try {
      this.sceneManager.gridHelper.visible = false

      if (savedCameraType !== 'perspective') {
        this.cameraManager.toggleCamera('perspective')
      }

      const box = new THREE.Box3().setFromObject(this.modelManager.currentModel)
      const size = box.getSize(new THREE.Vector3())
      const center = box.getCenter(new THREE.Vector3())

      const maxDim = Math.max(size.x, size.y, size.z)
      const distance = maxDim * 1.5

      const cameraPosition = new THREE.Vector3(
        center.x - distance * 0.8,
        center.y + distance * 0.4,
        center.z + distance * 0.3
      )

      this.cameraManager.perspectiveCamera.position.copy(cameraPosition)
      this.cameraManager.perspectiveCamera.lookAt(center)
      this.cameraManager.perspectiveCamera.updateProjectionMatrix()

      if (this.controlsManager.controls) {
        this.controlsManager.controls.target.copy(center)
        this.controlsManager.controls.update()
      }

      const result = await this.sceneManager.captureScene(width, height)
      return result.scene
    } finally {
      this.sceneManager.gridHelper.visible = savedGridVisible

      if (savedCameraType !== 'perspective') {
        this.cameraManager.toggleCamera(savedCameraType)
      }
      this.cameraManager.setCameraState(savedState)
      this.controlsManager.controls?.update()
    }
  }

  public remove(): void {
    if (this.resizeObserver) {
      this.resizeObserver.disconnect()
      this.resizeObserver = null
    }

    if (this.contextMenuAbortController) {
      this.contextMenuAbortController.abort()
      this.contextMenuAbortController = null
    }

    this.renderer.forceContextLoss()
    const canvas = this.renderer.domElement
    const event = new Event('webglcontextlost', {
      bubbles: true,
      cancelable: true
    })
    canvas.dispatchEvent(event)

    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId)
    }

    this.sceneManager.dispose()
    this.cameraManager.dispose()
    this.controlsManager.dispose()
    this.lightingManager.dispose()
    this.viewHelperManager.dispose()
    this.loaderManager.dispose()
    this.modelManager.dispose()
    this.recordingManager.dispose()
    this.animationManager.dispose()

    this.renderer.dispose()
    this.renderer.domElement.remove()
  }
}

export default Load3d
