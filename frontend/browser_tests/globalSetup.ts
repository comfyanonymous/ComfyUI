import type { FullConfig } from '@playwright/test'
import dotenv from 'dotenv'

import { backupPath } from './utils/backupUtils'

dotenv.config()

export default function globalSetup(_config: FullConfig) {
  if (!process.env.CI) {
    if (process.env.TEST_COMFYUI_DIR) {
      backupPath([process.env.TEST_COMFYUI_DIR, 'user'])
      backupPath([process.env.TEST_COMFYUI_DIR, 'models'], {
        renameAndReplaceWithScaffolding: true
      })
    } else {
      console.warn(
        'Set TEST_COMFYUI_DIR in .env to prevent user data (settings, workflows, etc.) from being overwritten'
      )
    }
  }
}
