import { isValidUrl } from '@comfyorg/shared-frontend-utils/formatUtil'

import { electronAPI } from './envUtil'

/**
 * Check if a mirror is reachable from the electron App.
 * @param mirror - The mirror to check.
 * @returns True if the mirror is reachable, false otherwise.
 */
export const checkMirrorReachable = async (mirror: string) => {
  return (
    isValidUrl(mirror) && (await electronAPI().NetWork.canAccessUrl(mirror))
  )
}
