import { vi } from 'vitest'
import 'vue'

// Mock @sparkjsdev/spark which uses WASM that doesn't work in Node.js
vi.mock('@sparkjsdev/spark', () => ({
  SplatMesh: class SplatMesh {
    constructor() {}
  }
}))

// Augment Window interface for tests
declare global {
  interface Window {
    __CONFIG__: {
      mixpanel_token?: string
      require_whitelist?: boolean
      subscription_required?: boolean
      max_upload_size?: number
      comfy_api_base_url?: string
      comfy_platform_base_url?: string
      firebase_config?: {
        apiKey: string
        authDomain: string
        databaseURL?: string
        projectId: string
        storageBucket: string
        messagingSenderId: string
        appId: string
        measurementId?: string
      }
      server_health_alert?: {
        message: string
        tooltip?: string
        severity?: 'info' | 'warning' | 'error'
        badge?: string
      }
    }
  }
}

// Define global variables for tests
globalThis.__COMFYUI_FRONTEND_VERSION__ = '1.24.0'
globalThis.__SENTRY_ENABLED__ = false
globalThis.__SENTRY_DSN__ = ''
globalThis.__ALGOLIA_APP_ID__ = ''
globalThis.__ALGOLIA_API_KEY__ = ''
globalThis.__USE_PROD_CONFIG__ = false
globalThis.__DISTRIBUTION__ = 'localhost'
globalThis.__IS_NIGHTLY__ = false

// Define runtime config for tests
window.__CONFIG__ = {
  subscription_required: true,
  mixpanel_token: 'test-token',
  comfy_api_base_url: 'https://stagingapi.comfy.org',
  comfy_platform_base_url: 'https://stagingplatform.comfy.org',
  firebase_config: {
    apiKey: 'test',
    authDomain: 'test.firebaseapp.com',
    projectId: 'test',
    storageBucket: 'test.appspot.com',
    messagingSenderId: '123',
    appId: '123'
  }
}

// Mock Worker for extendable-media-recorder
globalThis.Worker = vi.fn().mockImplementation(() => ({
  postMessage: vi.fn(),
  terminate: vi.fn(),
  addEventListener: vi.fn(),
  removeEventListener: vi.fn(),
  dispatchEvent: vi.fn()
}))
