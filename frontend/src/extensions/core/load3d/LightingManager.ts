import * as THREE from 'three'

import {
  type EventManagerInterface,
  type LightingManagerInterface
} from './interfaces'

export class LightingManager implements LightingManagerInterface {
  lights: THREE.Light[] = []
  currentIntensity: number = 3
  private scene: THREE.Scene
  private eventManager: EventManagerInterface

  constructor(scene: THREE.Scene, eventManager: EventManagerInterface) {
    this.scene = scene
    this.eventManager = eventManager
  }

  init(): void {
    this.setupLights()
  }

  dispose(): void {
    this.lights.forEach((light) => {
      this.scene.remove(light)
    })
    this.lights = []
  }

  setupLights(): void {
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5)
    this.scene.add(ambientLight)
    this.lights.push(ambientLight)

    const mainLight = new THREE.DirectionalLight(0xffffff, 0.8)
    mainLight.position.set(0, 10, 10)
    this.scene.add(mainLight)
    this.lights.push(mainLight)

    const backLight = new THREE.DirectionalLight(0xffffff, 0.5)
    backLight.position.set(0, 10, -10)
    this.scene.add(backLight)
    this.lights.push(backLight)

    const leftFillLight = new THREE.DirectionalLight(0xffffff, 0.3)
    leftFillLight.position.set(-10, 0, 0)
    this.scene.add(leftFillLight)
    this.lights.push(leftFillLight)

    const rightFillLight = new THREE.DirectionalLight(0xffffff, 0.3)
    rightFillLight.position.set(10, 0, 0)
    this.scene.add(rightFillLight)
    this.lights.push(rightFillLight)

    const bottomLight = new THREE.DirectionalLight(0xffffff, 0.2)
    bottomLight.position.set(0, -10, 0)
    this.scene.add(bottomLight)
    this.lights.push(bottomLight)
  }

  setLightIntensity(intensity: number): void {
    this.currentIntensity = intensity
    this.lights.forEach((light) => {
      if (light instanceof THREE.DirectionalLight) {
        if (light === this.lights[1]) {
          light.intensity = intensity * 0.8
        } else if (light === this.lights[2]) {
          light.intensity = intensity * 0.5
        } else if (light === this.lights[5]) {
          light.intensity = intensity * 0.2
        } else {
          light.intensity = intensity * 0.3
        }
      } else if (light instanceof THREE.AmbientLight) {
        light.intensity = intensity * 0.5
      }
    })

    this.eventManager.emitEvent('lightIntensityChange', intensity)
  }

  reset(): void {}
}
