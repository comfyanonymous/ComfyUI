import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls'

import {
  type CameraManagerInterface,
  type CameraState,
  type CameraType,
  type EventManagerInterface
} from './interfaces'

export class CameraManager implements CameraManagerInterface {
  perspectiveCamera: THREE.PerspectiveCamera
  orthographicCamera: THREE.OrthographicCamera
  activeCamera: THREE.Camera

  private eventManager: EventManagerInterface

  private controls: OrbitControls | null = null

  DEFAULT_DISTANCE = 10
  DEFAULT_LOOK_AT = 0

  DEFAULT_CAMERA = {
    near: 0.01,
    far: 10000
  }

  DEFAULT_PERSPECTIVE_CAMERA = {
    fov: 35,
    aspect: 1
  }

  DEFAULT_FRUSTUM_SIZE = 10

  DEFAULT_ORTHOGRAPHIC_CAMERA = {
    left: -this.DEFAULT_FRUSTUM_SIZE / 2,
    right: this.DEFAULT_FRUSTUM_SIZE / 2,
    top: this.DEFAULT_FRUSTUM_SIZE / 2,
    bottom: -this.DEFAULT_FRUSTUM_SIZE / 2
  }

  constructor(
    _renderer: THREE.WebGLRenderer,
    eventManager: EventManagerInterface
  ) {
    this.eventManager = eventManager

    this.perspectiveCamera = new THREE.PerspectiveCamera(
      this.DEFAULT_PERSPECTIVE_CAMERA.fov,
      this.DEFAULT_PERSPECTIVE_CAMERA.aspect,
      this.DEFAULT_CAMERA.near,
      this.DEFAULT_CAMERA.far
    )

    this.orthographicCamera = new THREE.OrthographicCamera(
      this.DEFAULT_ORTHOGRAPHIC_CAMERA.left,
      this.DEFAULT_ORTHOGRAPHIC_CAMERA.right,
      this.DEFAULT_ORTHOGRAPHIC_CAMERA.top,
      this.DEFAULT_ORTHOGRAPHIC_CAMERA.bottom,
      this.DEFAULT_CAMERA.near,
      this.DEFAULT_CAMERA.far
    )

    this.reset()

    this.activeCamera = this.perspectiveCamera
  }

  init(): void {}

  dispose(): void {}

  setControls(controls: OrbitControls): void {
    this.controls = controls

    if (this.controls) {
      this.controls.addEventListener('end', () => {
        this.eventManager.emitEvent('cameraChanged', this.getCameraState())
      })
    }
  }

  getCurrentCameraType(): CameraType {
    return this.activeCamera === this.perspectiveCamera
      ? 'perspective'
      : 'orthographic'
  }

  toggleCamera(cameraType?: CameraType): void {
    const oldCamera = this.activeCamera

    const position = oldCamera.position.clone()
    const rotation = oldCamera.rotation.clone()
    const target = this.controls?.target.clone() || new THREE.Vector3()

    const oldZoom =
      oldCamera instanceof THREE.OrthographicCamera
        ? oldCamera.zoom
        : (oldCamera as THREE.PerspectiveCamera).zoom

    if (!cameraType) {
      this.activeCamera =
        oldCamera === this.perspectiveCamera
          ? this.orthographicCamera
          : this.perspectiveCamera
    } else {
      this.activeCamera =
        cameraType === 'perspective'
          ? this.perspectiveCamera
          : this.orthographicCamera

      if (oldCamera === this.activeCamera) {
        return
      }
    }

    this.activeCamera.position.copy(position)
    this.activeCamera.rotation.copy(rotation)

    if (this.activeCamera instanceof THREE.OrthographicCamera) {
      this.activeCamera.zoom = oldZoom
      this.activeCamera.updateProjectionMatrix()
    } else if (this.activeCamera instanceof THREE.PerspectiveCamera) {
      this.activeCamera.zoom = oldZoom
      this.activeCamera.updateProjectionMatrix()
    }

    if (this.controls) {
      this.controls.object = this.activeCamera
      this.controls.target.copy(target)
      this.controls.update()
    }

    this.eventManager.emitEvent('cameraTypeChange', cameraType)
  }

  setFOV(fov: number): void {
    if (this.activeCamera === this.perspectiveCamera) {
      this.perspectiveCamera.fov = fov
      this.perspectiveCamera.updateProjectionMatrix()
    }

    this.eventManager.emitEvent('fovChange', fov)
  }

  getCameraState(): CameraState {
    return {
      position: this.activeCamera.position.clone(),
      target: this.controls?.target.clone() || new THREE.Vector3(),
      zoom:
        this.activeCamera instanceof THREE.OrthographicCamera
          ? this.activeCamera.zoom
          : (this.activeCamera as THREE.PerspectiveCamera).zoom,
      cameraType: this.getCurrentCameraType()
    }
  }

  setCameraState(state: CameraState): void {
    this.activeCamera.position.copy(state.position)

    this.controls?.target.copy(state.target)

    if (this.activeCamera instanceof THREE.OrthographicCamera) {
      this.activeCamera.zoom = state.zoom
      this.activeCamera.updateProjectionMatrix()
    } else if (this.activeCamera instanceof THREE.PerspectiveCamera) {
      this.activeCamera.zoom = state.zoom
      this.activeCamera.updateProjectionMatrix()
    }

    this.controls?.update()
  }

  handleResize(width: number, height: number): void {
    const aspect = width / height
    this.updateAspectRatio(aspect)
  }

  updateAspectRatio(aspect: number): void {
    if (this.activeCamera === this.perspectiveCamera) {
      this.perspectiveCamera.aspect = aspect
      this.perspectiveCamera.updateProjectionMatrix()
    } else {
      const frustumSize = 10
      this.orthographicCamera.left = (-frustumSize * aspect) / 2
      this.orthographicCamera.right = (frustumSize * aspect) / 2
      this.orthographicCamera.top = frustumSize / 2
      this.orthographicCamera.bottom = -frustumSize / 2
      this.orthographicCamera.updateProjectionMatrix()
    }
  }

  setupForModel(size: THREE.Vector3): void {
    const distance = Math.max(size.x, size.z) * 2
    const height = size.y * 2

    this.perspectiveCamera.position.set(distance, height, distance)
    this.orthographicCamera.position.set(distance, height, distance)

    if (this.activeCamera === this.perspectiveCamera) {
      this.perspectiveCamera.lookAt(0, size.y / 2, 0)
      this.perspectiveCamera.updateProjectionMatrix()
    } else {
      const frustumSize = Math.max(size.x, size.y, size.z) * 2
      const aspect = this.perspectiveCamera.aspect
      this.orthographicCamera.left = (-frustumSize * aspect) / 2
      this.orthographicCamera.right = (frustumSize * aspect) / 2
      this.orthographicCamera.top = frustumSize / 2
      this.orthographicCamera.bottom = -frustumSize / 2
      this.orthographicCamera.lookAt(0, size.y / 2, 0)
      this.orthographicCamera.updateProjectionMatrix()
    }

    this.controls?.target.set(0, size.y / 2, 0)
    this.controls?.update()
  }

  reset(): void {
    this.perspectiveCamera.position.set(
      this.DEFAULT_DISTANCE,
      this.DEFAULT_DISTANCE,
      this.DEFAULT_DISTANCE
    )

    this.orthographicCamera.position.set(
      this.DEFAULT_DISTANCE,
      this.DEFAULT_DISTANCE,
      this.DEFAULT_DISTANCE
    )

    this.perspectiveCamera.lookAt(
      this.DEFAULT_LOOK_AT,
      this.DEFAULT_LOOK_AT,
      this.DEFAULT_LOOK_AT
    )
    this.orthographicCamera.lookAt(
      this.DEFAULT_LOOK_AT,
      this.DEFAULT_LOOK_AT,
      this.DEFAULT_LOOK_AT
    )

    this.perspectiveCamera.updateProjectionMatrix()
    this.orthographicCamera.updateProjectionMatrix()
  }
}
