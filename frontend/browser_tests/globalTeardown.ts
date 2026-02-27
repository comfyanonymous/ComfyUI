import type { FullConfig } from '@playwright/test'
import dotenv from 'dotenv'

import { writePerfReport } from './helpers/perfReporter'
import { restorePath } from './utils/backupUtils'

dotenv.config()

export default function globalTeardown(_config: FullConfig) {
  writePerfReport()

  if (!process.env.CI && process.env.TEST_COMFYUI_DIR) {
    restorePath([process.env.TEST_COMFYUI_DIR, 'user'])
    restorePath([process.env.TEST_COMFYUI_DIR, 'models'])
  }
}
