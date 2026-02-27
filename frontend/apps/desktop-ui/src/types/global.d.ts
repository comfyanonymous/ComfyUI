declare global {
  interface Navigator {
    /**
     * Desktop app uses windowControlsOverlay to decide if it is in a custom window.
     */
    windowControlsOverlay?: {
      visible: boolean
    }
  }
}

export {}
