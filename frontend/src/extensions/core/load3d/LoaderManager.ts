import { SplatMesh } from '@sparkjsdev/spark'
import * as THREE from 'three'
import { FBXLoader } from 'three/examples/jsm/loaders/FBXLoader'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader'
import { MTLLoader } from 'three/examples/jsm/loaders/MTLLoader'
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader'
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader'
import { MtlObjBridge, OBJLoader2Parallel } from 'wwobjloader2'
// Use pre-bundled worker module (has all dependencies included)
// The unbundled 'wwobjloader2/worker' has ES imports that fail in production builds
import OBJLoader2WorkerUrl from 'wwobjloader2/bundle/worker/module?url'

import { t } from '@/i18n'
import { useSettingStore } from '@/platform/settings/settingStore'
import { useToastStore } from '@/platform/updates/common/toastStore'
import { api } from '@/scripts/api'
import { isPLYAsciiFormat } from '@/scripts/metadata/ply'

import {
  type EventManagerInterface,
  type LoaderManagerInterface,
  type ModelManagerInterface
} from './interfaces'
import { FastPLYLoader } from './loader/FastPLYLoader'

export class LoaderManager implements LoaderManagerInterface {
  gltfLoader: GLTFLoader
  objLoader: OBJLoader2Parallel
  mtlLoader: MTLLoader
  fbxLoader: FBXLoader
  stlLoader: STLLoader
  plyLoader: PLYLoader
  fastPlyLoader: FastPLYLoader

  private modelManager: ModelManagerInterface
  private eventManager: EventManagerInterface
  private currentLoadId: number = 0

  constructor(
    modelManager: ModelManagerInterface,
    eventManager: EventManagerInterface
  ) {
    this.modelManager = modelManager
    this.eventManager = eventManager

    this.gltfLoader = new GLTFLoader()
    this.objLoader = new OBJLoader2Parallel()
    // Set worker URL for Vite compatibility
    this.objLoader.setWorkerUrl(
      true,
      new URL(OBJLoader2WorkerUrl, import.meta.url)
    )
    this.mtlLoader = new MTLLoader()
    this.fbxLoader = new FBXLoader()
    this.stlLoader = new STLLoader()
    this.plyLoader = new PLYLoader()
    this.fastPlyLoader = new FastPLYLoader()
  }

  init(): void {}

  dispose(): void {}

  async loadModel(url: string, originalFileName?: string): Promise<void> {
    const loadId = ++this.currentLoadId

    try {
      this.eventManager.emitEvent('modelLoadingStart', null)

      this.modelManager.clearModel()

      this.modelManager.originalURL = url

      let fileExtension: string | undefined
      if (originalFileName) {
        fileExtension = originalFileName.split('.').pop()?.toLowerCase()

        this.modelManager.originalFileName =
          originalFileName.split('/').pop()?.split('.')[0] || 'model'
      } else {
        const filename = new URLSearchParams(url.split('?')[1]).get('filename')
        fileExtension = filename?.split('.').pop()?.toLowerCase()

        if (filename) {
          this.modelManager.originalFileName = filename.split('.')[0] || 'model'
        } else {
          this.modelManager.originalFileName = 'model'
        }
      }

      if (!fileExtension) {
        useToastStore().addAlert(t('toastMessages.couldNotDetermineFileType'))
        return
      }

      const model = await this.loadModelInternal(url, fileExtension)

      if (loadId !== this.currentLoadId) {
        return
      }

      if (model) {
        await this.modelManager.setupModel(model)
      }

      this.eventManager.emitEvent('modelLoadingEnd', null)
    } catch (error) {
      if (loadId === this.currentLoadId) {
        this.eventManager.emitEvent('modelLoadingEnd', null)
        console.error('Error loading model:', error)
        useToastStore().addAlert(t('toastMessages.errorLoadingModel'))
      }
    }
  }

  private async loadModelInternal(
    url: string,
    fileExtension: string
  ): Promise<THREE.Object3D | null> {
    let model: THREE.Object3D | null = null

    const params = new URLSearchParams(url.split('?')[1])

    const filename = params.get('filename')

    if (!filename) {
      console.error('Missing filename in URL:', url)

      return null
    }

    const loadRootFolder = params.get('type') === 'output' ? 'output' : 'input'

    const subfolder = params.get('subfolder') ?? ''

    const path =
      'api/view?type=' +
      loadRootFolder +
      '&subfolder=' +
      encodeURIComponent(subfolder) +
      '&filename='

    switch (fileExtension) {
      case 'stl':
        this.stlLoader.setPath(path)
        const geometry = await this.stlLoader.loadAsync(filename)
        this.modelManager.setOriginalModel(geometry)
        geometry.computeVertexNormals()

        const mesh = new THREE.Mesh(
          geometry,
          this.modelManager.standardMaterial
        )

        const group = new THREE.Group()
        group.add(mesh)
        model = group
        break

      case 'fbx':
        this.fbxLoader.setPath(path)

        const fbxModel = await this.fbxLoader.loadAsync(filename)

        this.modelManager.setOriginalModel(fbxModel)
        model = fbxModel

        fbxModel.traverse((child) => {
          if (child instanceof THREE.Mesh) {
            this.modelManager.originalMaterials.set(child, child.material)

            if (child instanceof THREE.SkinnedMesh) {
              child.frustumCulled = false
            }
          }
        })
        break

      case 'obj':
        if (this.modelManager.materialMode === 'original') {
          try {
            this.mtlLoader.setPath(path)

            const mtlFileName = filename.replace(/\.obj$/, '.mtl')

            const materials = await this.mtlLoader.loadAsync(mtlFileName)
            materials.preload()
            const materialsFromMtl =
              MtlObjBridge.addMaterialsFromMtlLoader(materials)
            this.objLoader.setMaterials(materialsFromMtl)
          } catch (e) {
            console.log(
              'No MTL file found or error loading it, continuing without materials'
            )
          }
        }

        // OBJLoader2Parallel uses Web Worker for parsing (non-blocking)
        const objUrl = path + encodeURIComponent(filename)
        model = await this.objLoader.loadAsync(objUrl)

        model.traverse((child) => {
          if (child instanceof THREE.Mesh) {
            this.modelManager.originalMaterials.set(child, child.material)
          }
        })
        break

      case 'gltf':
      case 'glb':
        this.gltfLoader.setPath(path)
        const gltf = await this.gltfLoader.loadAsync(filename)

        this.modelManager.setOriginalModel(gltf)
        model = gltf.scene

        gltf.scene.traverse((child) => {
          if (child instanceof THREE.Mesh) {
            child.geometry.computeVertexNormals()
            this.modelManager.originalMaterials.set(child, child.material)

            if (child instanceof THREE.SkinnedMesh) {
              child.frustumCulled = false
            }
          }
        })
        break

      case 'ply':
        model = await this.loadPLY(path, filename)
        break

      case 'spz':
      case 'splat':
      case 'ksplat':
        model = await this.loadSplat(path, filename)
        break
    }

    return model
  }

  private async fetchModelData(path: string, filename: string) {
    const route =
      '/' + path.replace(/^api\//, '') + encodeURIComponent(filename)
    const response = await api.fetchApi(route)
    if (!response.ok) {
      throw new Error(`Failed to fetch model: ${response.status}`)
    }
    return response.arrayBuffer()
  }

  private async loadSplat(
    path: string,
    filename: string
  ): Promise<THREE.Object3D> {
    const arrayBuffer = await this.fetchModelData(path, filename)

    const splatMesh = new SplatMesh({ fileBytes: arrayBuffer })
    this.modelManager.setOriginalModel(splatMesh)
    const splatGroup = new THREE.Group()
    splatGroup.add(splatMesh)
    return splatGroup
  }

  private async loadPLY(
    path: string,
    filename: string
  ): Promise<THREE.Object3D | null> {
    const plyEngine = useSettingStore().get('Comfy.Load3D.PLYEngine') as string

    if (plyEngine === 'sparkjs') {
      return this.loadSplat(path, filename)
    }

    // Use Three.js PLYLoader or FastPLYLoader for point cloud PLY files
    const arrayBuffer = await this.fetchModelData(path, filename)

    const isASCII = isPLYAsciiFormat(arrayBuffer)

    let plyGeometry: THREE.BufferGeometry

    if (isASCII && plyEngine === 'fastply') {
      plyGeometry = this.fastPlyLoader.parse(arrayBuffer)
    } else {
      this.plyLoader.setPath(path)
      plyGeometry = this.plyLoader.parse(arrayBuffer)
    }

    this.modelManager.setOriginalModel(plyGeometry)
    plyGeometry.computeVertexNormals()

    const hasVertexColors = plyGeometry.attributes.color !== undefined
    const materialMode = this.modelManager.materialMode

    // Use Points rendering for pointCloud mode (better for point clouds)
    if (materialMode === 'pointCloud') {
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

      const plyPoints = new THREE.Points(plyGeometry, pointMaterial)
      this.modelManager.originalMaterials.set(
        plyPoints as unknown as THREE.Mesh,
        pointMaterial
      )

      const plyGroup = new THREE.Group()
      plyGroup.add(plyPoints)
      return plyGroup
    }

    // Use Mesh rendering for other modes
    let plyMaterial: THREE.Material

    if (hasVertexColors) {
      plyMaterial = new THREE.MeshStandardMaterial({
        vertexColors: true,
        metalness: 0.0,
        roughness: 0.5,
        side: THREE.DoubleSide
      })
    } else {
      plyMaterial = this.modelManager.standardMaterial.clone()
      plyMaterial.side = THREE.DoubleSide
    }

    const plyMesh = new THREE.Mesh(plyGeometry, plyMaterial)
    this.modelManager.originalMaterials.set(plyMesh, plyMaterial)

    const plyGroup = new THREE.Group()
    plyGroup.add(plyMesh)
    return plyGroup
  }
}
