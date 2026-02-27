import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls'

import {
  type ControlsManagerInterface,
  type EventManagerInterface
} from './interfaces'

export class ControlsManager implements ControlsManagerInterface {
  controls: OrbitControls
  private eventManager: EventManagerInterface
  private camera: THREE.Camera

  constructor(
    renderer: THREE.WebGLRenderer,
    camera: THREE.Camera,
    eventManager: EventManagerInterface
  ) {
    this.eventManager = eventManager
    this.camera = camera

    const container = renderer.domElement.parentElement || renderer.domElement
    this.controls = new OrbitControls(camera, container)
    this.controls.enableDamping = true
  }

  init(): void {
    this.controls.addEventListener('end', () => {
      const cameraState = {
        position: this.camera.position.clone(),
        target: this.controls.target.clone(),
        zoom:
          this.camera instanceof THREE.OrthographicCamera
            ? (this.camera as THREE.OrthographicCamera).zoom
            : (this.camera as THREE.PerspectiveCamera).zoom,
        cameraType:
          this.camera instanceof THREE.PerspectiveCamera
            ? 'perspective'
            : 'orthographic'
      }

      this.eventManager.emitEvent('cameraChanged', cameraState)
    })
  }

  dispose(): void {
    this.controls.dispose()
  }

  handleResize(): void {}

  update(): void {
    this.controls.update()
  }

  updateCamera(camera: THREE.Camera): void {
    const position = this.controls.object.position.clone()
    const target = this.controls.target.clone()

    this.camera = camera
    this.controls.object = camera
    this.controls.target = target
    camera.position.copy(position)
    this.controls.update()
  }
  reset(): void {
    this.controls.target.set(0, 0, 0)
    this.controls.update()
  }
}
