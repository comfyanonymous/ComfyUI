import * as THREE from 'three'
import type { GLTF } from 'three/examples/jsm/loaders/GLTFLoader'

import type {
  AnimationItem,
  AnimationManagerInterface,
  EventManagerInterface
} from '@/extensions/core/load3d/interfaces'

export class AnimationManager implements AnimationManagerInterface {
  currentAnimation: THREE.AnimationMixer | null = null
  animationActions: THREE.AnimationAction[] = []
  animationClips: THREE.AnimationClip[] = []
  selectedAnimationIndex: number = 0
  isAnimationPlaying: boolean = false
  animationSpeed: number = 1.0

  private eventManager: EventManagerInterface

  constructor(eventManager: EventManagerInterface) {
    this.eventManager = eventManager
  }

  init(): void {}

  dispose(): void {
    if (this.currentAnimation) {
      this.animationActions.forEach((action) => {
        action.stop()
      })
      this.currentAnimation = null
    }
    this.animationActions = []
    this.animationClips = []
    this.selectedAnimationIndex = 0
    this.isAnimationPlaying = false
    this.animationSpeed = 1.0

    this.eventManager.emitEvent('animationListChange', [])
  }

  setupModelAnimations(
    model: THREE.Object3D,
    originalModel: THREE.Object3D | THREE.BufferGeometry | GLTF | null
  ): void {
    if (this.currentAnimation) {
      this.currentAnimation.stopAllAction()
      this.animationActions = []
    }

    let animations: THREE.AnimationClip[] = []
    if (model.animations?.length > 0) {
      animations = model.animations
    } else if (
      originalModel &&
      'animations' in originalModel &&
      Array.isArray(originalModel.animations)
    ) {
      animations = originalModel.animations
    }

    if (animations.length > 0) {
      this.animationClips = animations

      this.currentAnimation = new THREE.AnimationMixer(model)

      if (this.animationClips.length > 0) {
        this.updateSelectedAnimation(0)
      }
    } else {
      this.animationClips = []
    }

    this.updateAnimationList()
  }

  updateAnimationList(): void {
    let updatedAnimationList: AnimationItem[] = []

    if (this.animationClips.length > 0) {
      updatedAnimationList = this.animationClips.map((clip, index) => ({
        name: clip.name || `Animation ${index + 1}`,
        index
      }))
    }

    this.eventManager.emitEvent('animationListChange', updatedAnimationList)
  }

  setAnimationSpeed(speed: number): void {
    this.animationSpeed = speed
    this.animationActions.forEach((action) => {
      action.setEffectiveTimeScale(speed)
    })
  }

  updateSelectedAnimation(index: number): void {
    if (
      !this.currentAnimation ||
      !this.animationClips ||
      index >= this.animationClips.length
    ) {
      console.warn('Invalid animation update request')
      return
    }

    this.animationActions.forEach((action) => {
      action.stop()
    })
    this.currentAnimation.stopAllAction()
    this.animationActions = []

    this.selectedAnimationIndex = index
    const clip = this.animationClips[index]

    const action = this.currentAnimation.clipAction(clip)

    action.setEffectiveTimeScale(this.animationSpeed)

    action.reset()
    action.clampWhenFinished = false
    action.loop = THREE.LoopRepeat

    if (this.isAnimationPlaying) {
      action.play()
    } else {
      action.play()
      action.paused = true
    }

    this.animationActions = [action]

    // Emit initial progress to set duration
    this.eventManager.emitEvent('animationProgressChange', {
      progress: 0,
      currentTime: 0,
      duration: clip.duration
    })
  }

  toggleAnimation(play?: boolean): void {
    if (!this.currentAnimation || this.animationActions.length === 0) {
      console.warn('No animation to toggle')
      return
    }

    this.isAnimationPlaying = play ?? !this.isAnimationPlaying

    this.animationActions.forEach((action) => {
      if (this.isAnimationPlaying) {
        action.paused = false
        if (action.time === 0 || action.time === action.getClip().duration) {
          action.reset()
        }
      } else {
        action.paused = true
      }
    })
  }

  update(delta: number): void {
    if (this.currentAnimation && this.isAnimationPlaying) {
      this.currentAnimation.update(delta)

      if (this.animationActions.length > 0) {
        const action = this.animationActions[0]
        const clip = action.getClip()
        const progress = (action.time / clip.duration) * 100
        this.eventManager.emitEvent('animationProgressChange', {
          progress,
          currentTime: action.time,
          duration: clip.duration
        })
      }
    }
  }

  getAnimationTime(): number {
    if (this.animationActions.length === 0) return 0
    return this.animationActions[0].time
  }

  getAnimationDuration(): number {
    if (this.animationActions.length === 0) return 0
    return this.animationActions[0].getClip().duration
  }

  setAnimationTime(time: number): void {
    if (this.animationActions.length === 0) return
    const duration = this.getAnimationDuration()
    const clampedTime = Math.max(0, Math.min(time, duration))

    // Temporarily unpause to allow time update, then restore
    const wasPaused = this.animationActions.map((action) => action.paused)
    this.animationActions.forEach((action) => {
      action.paused = false
      action.time = clampedTime
    })

    if (this.currentAnimation) {
      this.currentAnimation.setTime(clampedTime)
      this.currentAnimation.update(0)
    }

    // Restore paused state
    this.animationActions.forEach((action, i) => {
      action.paused = wasPaused[i]
    })

    this.eventManager.emitEvent('animationProgressChange', {
      progress: (clampedTime / duration) * 100,
      currentTime: clampedTime,
      duration
    })
  }

  reset(): void {}
}
