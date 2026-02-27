// Use type-only imports to avoid pulling THREE.js into the main bundle
// These imports are erased at compile time and don't create runtime dependencies
import type * as THREE from 'three'
import type { OrbitControls } from 'three/examples/jsm/controls/OrbitControls'
import type { ViewHelper } from 'three/examples/jsm/helpers/ViewHelper'
import type { FBXLoader } from 'three/examples/jsm/loaders/FBXLoader'
import type { GLTF, GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader'
import type { MTLLoader } from 'three/examples/jsm/loaders/MTLLoader'
import type { STLLoader } from 'three/examples/jsm/loaders/STLLoader'
import type { OBJLoader2Parallel } from 'wwobjloader2'

export type MaterialMode =
  | 'original'
  | 'pointCloud'
  | 'normal'
  | 'wireframe'
  | 'depth'
export type UpDirection = 'original' | '-x' | '+x' | '-y' | '+y' | '-z' | '+z'
export type CameraType = 'perspective' | 'orthographic'
export type BackgroundRenderModeType = 'tiled' | 'panorama'

export interface CameraState {
  position: THREE.Vector3
  target: THREE.Vector3
  zoom: number
  cameraType: CameraType
}

export interface SceneConfig {
  showGrid: boolean
  backgroundColor: string
  backgroundImage?: string
  backgroundRenderMode?: BackgroundRenderModeType
}

export interface ModelConfig {
  upDirection: UpDirection
  materialMode: MaterialMode
  showSkeleton: boolean
}

export interface CameraConfig {
  cameraType: CameraType
  fov: number
  state?: CameraState
}

export interface LightConfig {
  intensity: number
}

export interface EventCallback<T = unknown> {
  (data: T): void
}

export interface Load3DOptions {
  // Optional target dimensions for aspect ratio control
  width?: number
  height?: number

  // Dynamic dimension provider (called on every render)
  // Use this for reactive dimensions that change over time
  getDimensions?: () => { width: number; height: number } | null

  // Viewer mode flag (affects aspect ratio behavior)
  isViewerMode?: boolean

  // Optional context menu callback
  onContextMenu?: (event: MouseEvent) => void
}

export interface CaptureResult {
  scene: string
  mask: string
  normal: string
}

interface BaseManager {
  init(): void
  dispose(): void
  reset(): void
}

export interface AnimationItem {
  name: string
  index: number
}

export interface SceneManagerInterface extends BaseManager {
  scene: THREE.Scene
  gridHelper: THREE.GridHelper
  toggleGrid(showGrid: boolean): void
  setBackgroundColor(color: string): void
  setBackgroundImage(uploadPath: string): Promise<void>
  removeBackgroundImage(): void
  setBackgroundRenderMode(mode: BackgroundRenderModeType): void
  handleResize(width: number, height: number): void
  captureScene(width: number, height: number): Promise<CaptureResult>
}

export interface CameraManagerInterface extends BaseManager {
  activeCamera: THREE.Camera
  perspectiveCamera: THREE.PerspectiveCamera
  orthographicCamera: THREE.OrthographicCamera
  getCurrentCameraType(): CameraType
  toggleCamera(cameraType?: CameraType): void
  setFOV(fov: number): void
  setCameraState(state: CameraState): void
  getCameraState(): CameraState
  handleResize(width: number, height: number): void
  setControls(controls: OrbitControls): void
}

export interface ControlsManagerInterface extends BaseManager {
  controls: OrbitControls
  handleResize(): void
}

export interface LightingManagerInterface extends BaseManager {
  lights: THREE.Light[]
  setLightIntensity(intensity: number): void
}

export interface ViewHelperManagerInterface extends BaseManager {
  viewHelper: ViewHelper
  viewHelperContainer: HTMLDivElement
  createViewHelper(container: Element | HTMLElement): void
  update(delta: number): void
  handleResize(): void
}

export interface EventManagerInterface {
  addEventListener<T>(event: string, callback: EventCallback<T>): void
  removeEventListener<T>(event: string, callback: EventCallback<T>): void
  emitEvent<T>(event: string, data: T): void
}

export interface AnimationManagerInterface extends BaseManager {
  currentAnimation: THREE.AnimationMixer | null
  animationActions: THREE.AnimationAction[]
  animationClips: THREE.AnimationClip[]
  selectedAnimationIndex: number
  isAnimationPlaying: boolean
  animationSpeed: number

  setupModelAnimations(
    model: THREE.Object3D,
    originalModel: THREE.Object3D | THREE.BufferGeometry | GLTF | null
  ): void
  updateAnimationList(): void
  setAnimationSpeed(speed: number): void
  updateSelectedAnimation(index: number): void
  toggleAnimation(play?: boolean): void
  update(delta: number): void
  getAnimationTime(): number
  getAnimationDuration(): number
  setAnimationTime(time: number): void
}

export interface ModelManagerInterface {
  originalFileName: string | null
  originalURL: string | null
  currentModel: THREE.Object3D | null
  originalModel: THREE.Object3D | THREE.BufferGeometry | GLTF | null
  originalRotation: THREE.Euler | null
  currentUpDirection: UpDirection

  init(): void
  dispose(): void
  clearModel(): void
  reset(): void
  setupModel(model: THREE.Object3D): Promise<void>
  addModelToScene(model: THREE.Object3D): void
  setOriginalModel(model: THREE.Object3D | THREE.BufferGeometry | GLTF): void
  setUpDirection(direction: UpDirection): void
  materialMode: MaterialMode
  originalMaterials: WeakMap<THREE.Mesh, THREE.Material | THREE.Material[]>
  normalMaterial: THREE.MeshNormalMaterial
  standardMaterial: THREE.MeshStandardMaterial
  wireframeMaterial: THREE.MeshBasicMaterial
  depthMaterial: THREE.MeshDepthMaterial
  setMaterialMode(mode: MaterialMode): void
  setupModelMaterials(model: THREE.Object3D): void
}

export interface LoaderManagerInterface {
  gltfLoader: GLTFLoader
  objLoader: OBJLoader2Parallel
  mtlLoader: MTLLoader
  fbxLoader: FBXLoader
  stlLoader: STLLoader

  init(): void
  dispose(): void
  loadModel(url: string, originalFileName?: string): Promise<void>
}
