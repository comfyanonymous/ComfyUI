import type { ElectronAPI } from '@comfyorg/comfyui-electron-types'

export function isElectron() {
  return 'electronAPI' in window && window.electronAPI !== undefined
}

export function electronAPI() {
  return (window as any).electronAPI as ElectronAPI
}

export function isNativeWindow() {
  return isElectron() && !!window.navigator.windowControlsOverlay?.visible
}
