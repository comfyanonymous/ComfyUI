import { SplatMesh } from '@sparkjsdev/spark'
import * as THREE from 'three'
import { type GLTF } from 'three/examples/jsm/loaders/GLTFLoader'

import {
  type EventManagerInterface,
  type MaterialMode,
  type ModelManagerInterface,
  type UpDirection
} from './interfaces'

export class SceneModelManager implements ModelManagerInterface {
  currentModel: THREE.Object3D | null = null
  originalModel:
    | THREE.Object3D
    | THREE.Group
    | THREE.BufferGeometry
    | GLTF
    | null = null
  originalRotation: THREE.Euler | null = null
  currentUpDirection: UpDirection = 'original'
  materialMode: MaterialMode = 'original'
  originalMaterials: WeakMap<THREE.Mesh, THREE.Material | THREE.Material[]> =
    new WeakMap()
  normalMaterial: THREE.MeshNormalMaterial
  standardMaterial: THREE.MeshStandardMaterial
  wireframeMaterial: THREE.MeshBasicMaterial
  depthMaterial: THREE.MeshDepthMaterial
  originalFileName: string | null = null
  originalURL: string | null = null
  appliedTexture: THREE.Texture | null = null
  textureLoader: THREE.TextureLoader
  skeletonHelper: THREE.SkeletonHelper | null = null
  showSkeleton: boolean = false

  private scene: THREE.Scene
  private renderer: THREE.WebGLRenderer
  private eventManager: EventManagerInterface
  private activeCamera: THREE.Camera
  private setupCamera: (size: THREE.Vector3) => void

  constructor(
    scene: THREE.Scene,
    renderer: THREE.WebGLRenderer,
    eventManager: EventManagerInterface,
    getActiveCamera: () => THREE.Camera,
    setupCamera: (size: THREE.Vector3) => void
  ) {
    this.scene = scene
    this.renderer = renderer
    this.eventManager = eventManager
    this.activeCamera = getActiveCamera()
    this.setupCamera = setupCamera
    this.textureLoader = new THREE.TextureLoader()

    this.normalMaterial = new THREE.MeshNormalMaterial({
      flatShading: false,
      side: THREE.DoubleSide,
      normalScale: new THREE.Vector2(1, 1),
      transparent: false,
      opacity: 1.0
    })

    this.wireframeMaterial = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      wireframe: true,
      transparent: false,
      opacity: 1.0
    })

    this.depthMaterial = new THREE.MeshDepthMaterial({
      depthPacking: THREE.BasicDepthPacking,
      side: THREE.DoubleSide
    })

    this.standardMaterial = this.createSTLMaterial()
  }

  init(): void {}

  dispose(): void {
    this.clearModel()
    this.normalMaterial.dispose()
    this.standardMaterial.dispose()
    this.wireframeMaterial.dispose()
    this.depthMaterial.dispose()

    if (this.appliedTexture) {
      this.appliedTexture.dispose()
      this.appliedTexture = null
    }
  }

  createSTLMaterial(): THREE.MeshStandardMaterial {
    return new THREE.MeshStandardMaterial({
      color: 0x808080,
      metalness: 0.1,
      roughness: 0.8,
      flatShading: false,
      side: THREE.DoubleSide
    })
  }

  private handlePLYModeSwitch(mode: MaterialMode): void {
    if (!(this.originalModel instanceof THREE.BufferGeometry)) {
      return
    }

    const plyGeometry = this.originalModel.clone()
    const hasVertexColors = plyGeometry.attributes.color !== undefined

    // Find and remove ALL MainModel instances by name to ensure deletion
    const oldMainModels: THREE.Object3D[] = []
    this.scene.traverse((obj) => {
      if (obj.name === 'MainModel') {
        oldMainModels.push(obj)
      }
    })

    // Remove and dispose all found MainModels
    oldMainModels.forEach((oldModel) => {
      oldModel.traverse((child) => {
        if (child instanceof THREE.Mesh || child instanceof THREE.Points) {
          child.geometry?.dispose()
          if (Array.isArray(child.material)) {
            child.material.forEach((m) => m.dispose())
          } else {
            child.material?.dispose()
          }
        }
      })
      this.scene.remove(oldModel)
    })

    this.currentModel = null

    let newModel: THREE.Object3D

    if (mode === 'pointCloud') {
      // Use Points rendering for point cloud mode
      plyGeometry.computeBoundingSphere()
      if (plyGeometry.boundingSphere) {
        const center = plyGeometry.boundingSphere.center
        const radius = plyGeometry.boundingSphere.radius

        plyGeometry.translate(-center.x, -center.y, -center.z)

        if (radius > 0) {
          const scale = 1.0 / radius
          plyGeometry.scale(scale, scale, scale)
        }
      }

      const pointMaterial = hasVertexColors
        ? new THREE.PointsMaterial({
            size: 0.005,
            vertexColors: true,
            sizeAttenuation: true
          })
        : new THREE.PointsMaterial({
            size: 0.005,
            color: 0xcccccc,
            sizeAttenuation: true
          })

      const points = new THREE.Points(plyGeometry, pointMaterial)
      newModel = new THREE.Group()
      newModel.add(points)
    } else {
      // Use Mesh rendering for other modes
      let meshMaterial: THREE.Material = hasVertexColors
        ? new THREE.MeshStandardMaterial({
            vertexColors: true,
            metalness: 0.0,
            roughness: 0.5,
            side: THREE.DoubleSide
          })
        : this.standardMaterial.clone()

      if (
        !hasVertexColors &&
        meshMaterial instanceof THREE.MeshStandardMaterial
      ) {
        meshMaterial.side = THREE.DoubleSide
      }

      const mesh = new THREE.Mesh(plyGeometry, meshMaterial)
      this.originalMaterials.set(mesh, meshMaterial)

      newModel = new THREE.Group()
      newModel.add(mesh)

      // Apply the requested material mode
      if (mode === 'normal') {
        mesh.material = new THREE.MeshNormalMaterial({
          flatShading: false,
          side: THREE.DoubleSide
        })
      } else if (mode === 'wireframe') {
        mesh.material = new THREE.MeshBasicMaterial({
          color: 0xffffff,
          wireframe: true
        })
      }
    }

    // Double check: remove any remaining MainModel before adding new one
    const remainingMainModels: THREE.Object3D[] = []
    this.scene.traverse((obj) => {
      if (obj.name === 'MainModel') {
        remainingMainModels.push(obj)
      }
    })
    remainingMainModels.forEach((obj) => this.scene.remove(obj))

    this.currentModel = newModel
    newModel.name = 'MainModel'

    // Setup the new model
    if (mode === 'pointCloud') {
      this.scene.add(newModel)
    } else {
      const box = new THREE.Box3().setFromObject(newModel)
      const size = box.getSize(new THREE.Vector3())
      const center = box.getCenter(new THREE.Vector3())

      const maxDim = Math.max(size.x, size.y, size.z)
      const targetSize = 5
      const scale = targetSize / maxDim
      newModel.scale.multiplyScalar(scale)

      box.setFromObject(newModel)
      box.getCenter(center)
      box.getSize(size)

      newModel.position.set(-center.x, -box.min.y, -center.z)
      this.scene.add(newModel)
    }

    this.eventManager.emitEvent('materialModeChange', mode)
  }

  setMaterialMode(mode: MaterialMode): void {
    if (!this.currentModel || mode === this.materialMode) {
      return
    }

    this.materialMode = mode

    // Handle PLY files specially - they need to be recreated for mode switch
    if (this.originalModel instanceof THREE.BufferGeometry) {
      this.handlePLYModeSwitch(mode)
      return
    }

    if (mode === 'depth') {
      this.renderer.outputColorSpace = THREE.LinearSRGBColorSpace
    } else {
      this.renderer.outputColorSpace = THREE.SRGBColorSpace
    }

    if (this.currentModel) {
      this.currentModel.visible = true
    }

    this.currentModel.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        switch (mode) {
          case 'depth':
            if (!this.originalMaterials.has(child)) {
              this.originalMaterials.set(child, child.material)
            }
            const depthMat = new THREE.MeshDepthMaterial({
              depthPacking: THREE.BasicDepthPacking,
              side: THREE.DoubleSide
            })

            depthMat.onBeforeCompile = (shader) => {
              shader.uniforms.cameraType = {
                value:
                  this.activeCamera instanceof THREE.OrthographicCamera
                    ? 1.0
                    : 0.0
              }

              shader.fragmentShader = `
                uniform float cameraType;
                ${shader.fragmentShader}
              `

              shader.fragmentShader = shader.fragmentShader.replace(
                /gl_FragColor\s*=\s*vec4\(\s*vec3\(\s*1.0\s*-\s*fragCoordZ\s*\)\s*,\s*opacity\s*\)\s*;/,
                `
                  float depth = 1.0 - fragCoordZ;
                  if (cameraType > 0.5) {
                    depth = pow(depth, 400.0);
                  } else {
                    depth = pow(depth, 0.6);
                  }
                  gl_FragColor = vec4(vec3(depth), opacity);
                `
              )
            }

            depthMat.customProgramCacheKey = () => {
              return this.activeCamera instanceof THREE.OrthographicCamera
                ? 'ortho'
                : 'persp'
            }

            child.material = depthMat
            break
          case 'normal':
            if (!this.originalMaterials.has(child)) {
              this.originalMaterials.set(child, child.material)
            }
            child.material = new THREE.MeshNormalMaterial({
              flatShading: false,
              side: THREE.DoubleSide,
              normalScale: new THREE.Vector2(1, 1),
              transparent: false,
              opacity: 1.0
            })
            break
          case 'wireframe':
            if (!this.originalMaterials.has(child)) {
              this.originalMaterials.set(child, child.material)
            }
            child.material = new THREE.MeshBasicMaterial({
              color: 0xffffff,
              wireframe: true,
              transparent: false,
              opacity: 1.0
            })
            break
          case 'original':
          case 'pointCloud':
            const originalMaterial = this.originalMaterials.get(child)
            if (originalMaterial) {
              child.material = originalMaterial
            } else {
              if (this.appliedTexture) {
                child.material = new THREE.MeshStandardMaterial({
                  map: this.appliedTexture,
                  metalness: 0.1,
                  roughness: 0.8,
                  side: THREE.DoubleSide
                })
              } else {
                child.material = this.standardMaterial
              }
            }
            break
        }
      }
    })

    this.eventManager.emitEvent('materialModeChange', mode)
  }

  setupModelMaterials(model: THREE.Object3D): void {
    model.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        this.originalMaterials.set(child, child.material)
      }
    })

    this.setMaterialMode('original')
  }

  clearModel(): void {
    const objectsToRemove: THREE.Object3D[] = []

    this.scene.traverse((object) => {
      const isEnvironmentObject =
        object instanceof THREE.GridHelper ||
        object instanceof THREE.Light ||
        object instanceof THREE.Camera

      if (!isEnvironmentObject) {
        objectsToRemove.push(object)
      }
    })

    objectsToRemove.forEach((obj) => {
      if (obj.parent && obj.parent !== this.scene) {
        obj.parent.remove(obj)
      } else {
        this.scene.remove(obj)
      }

      if (obj instanceof THREE.Mesh) {
        obj.geometry?.dispose()
        if (Array.isArray(obj.material)) {
          obj.material.forEach((material) => material.dispose())
        } else {
          obj.material?.dispose()
        }
      }
    })

    this.reset()
  }

  reset(): void {
    this.currentModel = null
    this.originalModel = null
    this.originalRotation = null
    this.currentUpDirection = 'original'
    this.setMaterialMode('original')
    this.originalFileName = null
    this.originalURL = null

    if (this.appliedTexture) {
      this.appliedTexture.dispose()
      this.appliedTexture = null
    }

    if (this.skeletonHelper) {
      this.scene.remove(this.skeletonHelper)
      this.skeletonHelper.dispose()
      this.skeletonHelper = null
    }
    this.showSkeleton = false

    this.originalMaterials = new WeakMap()
  }

  hasSkeleton(): boolean {
    if (!this.currentModel) return false
    let found = false
    this.currentModel.traverse((child) => {
      if (child instanceof THREE.SkinnedMesh && child.skeleton) {
        found = true
      }
    })
    return found
  }

  setShowSkeleton(show: boolean): void {
    this.showSkeleton = show

    if (show) {
      if (!this.skeletonHelper && this.currentModel) {
        let rootBone: THREE.Bone | null = null
        this.currentModel.traverse((child) => {
          if (child instanceof THREE.Bone && !rootBone) {
            if (!(child.parent instanceof THREE.Bone)) {
              rootBone = child
            }
          }
        })

        if (rootBone) {
          this.skeletonHelper = new THREE.SkeletonHelper(rootBone)
          this.scene.add(this.skeletonHelper)
        } else {
          let skinnedMesh: THREE.SkinnedMesh | null = null
          this.currentModel.traverse((child) => {
            if (child instanceof THREE.SkinnedMesh && !skinnedMesh) {
              skinnedMesh = child
            }
          })

          if (skinnedMesh) {
            this.skeletonHelper = new THREE.SkeletonHelper(skinnedMesh)
            this.scene.add(this.skeletonHelper)
          }
        }
      } else if (this.skeletonHelper) {
        this.skeletonHelper.visible = true
      }
    } else {
      if (this.skeletonHelper) {
        this.skeletonHelper.visible = false
      }
    }

    this.eventManager.emitEvent('skeletonVisibilityChange', show)
  }

  addModelToScene(model: THREE.Object3D): void {
    this.currentModel = model
    model.name = 'MainModel'

    this.scene.add(this.currentModel)
  }

  async setupModel(model: THREE.Object3D): Promise<void> {
    this.currentModel = model
    model.name = 'MainModel'

    // Check if model is or contains a SplatMesh (3D Gaussian Splatting)
    const isSplatModel = this.containsSplatMesh(model)

    if (isSplatModel) {
      // SplatMesh handles its own rendering, just add to scene
      this.scene.add(model)
      // Set a default camera distance for splat models
      this.setupCamera(new THREE.Vector3(5, 5, 5))
      return
    }

    const box = new THREE.Box3().setFromObject(model)
    const size = box.getSize(new THREE.Vector3())
    const center = box.getCenter(new THREE.Vector3())

    const maxDim = Math.max(size.x, size.y, size.z)
    const targetSize = 5
    const scale = targetSize / maxDim
    model.scale.multiplyScalar(scale)

    box.setFromObject(model)
    box.getCenter(center)
    box.getSize(size)

    model.position.set(-center.x, -box.min.y, -center.z)

    this.scene.add(model)

    if (this.materialMode !== 'original') {
      this.setMaterialMode(this.materialMode)
    }

    if (this.currentUpDirection !== 'original') {
      this.setUpDirection(this.currentUpDirection)
    }
    this.setupModelMaterials(model)

    this.setupCamera(size)
  }

  containsSplatMesh(model?: THREE.Object3D | null): boolean {
    const target = model ?? this.currentModel
    if (!target) return false
    if (target instanceof SplatMesh) return true
    let found = false
    target.traverse((child) => {
      if (child instanceof SplatMesh) found = true
    })
    return found
  }

  setOriginalModel(model: THREE.Object3D | THREE.BufferGeometry | GLTF): void {
    this.originalModel = model
  }

  setUpDirection(direction: UpDirection): void {
    if (!this.currentModel) return

    if (!this.originalRotation && this.currentModel.rotation) {
      this.originalRotation = this.currentModel.rotation.clone()
    }

    this.currentUpDirection = direction

    if (this.originalRotation) {
      this.currentModel.rotation.copy(this.originalRotation)
    }

    switch (direction) {
      case 'original':
        break
      case '-x':
        this.currentModel.rotation.z = Math.PI / 2
        break
      case '+x':
        this.currentModel.rotation.z = -Math.PI / 2
        break
      case '-y':
        this.currentModel.rotation.x = Math.PI
        break
      case '+y':
        break
      case '-z':
        this.currentModel.rotation.x = Math.PI / 2
        break
      case '+z':
        this.currentModel.rotation.x = -Math.PI / 2
        break
    }

    this.eventManager.emitEvent('upDirectionChange', direction)
  }
}
