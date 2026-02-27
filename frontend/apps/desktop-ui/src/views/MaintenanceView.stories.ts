// eslint-disable-next-line storybook/no-renderer-packages
import type { Meta, StoryObj } from '@storybook/vue3'
import { defineAsyncComponent } from 'vue'

type UnsafeReason = 'appInstallDir' | 'updaterCache' | 'oneDrive' | null
type ValidationIssueState = 'OK' | 'warning' | 'error' | 'skipped'

type ValidationState = {
  inProgress: boolean
  installState: string
  basePath?: ValidationIssueState
  unsafeBasePath: boolean
  unsafeBasePathReason: UnsafeReason
  venvDirectory?: ValidationIssueState
  pythonInterpreter?: ValidationIssueState
  pythonPackages?: ValidationIssueState
  uv?: ValidationIssueState
  git?: ValidationIssueState
  vcRedist?: ValidationIssueState
  upgradePackages?: ValidationIssueState
}

const validationState: ValidationState = {
  inProgress: false,
  installState: 'installed',
  basePath: 'OK',
  unsafeBasePath: false,
  unsafeBasePathReason: null,
  venvDirectory: 'OK',
  pythonInterpreter: 'OK',
  pythonPackages: 'OK',
  uv: 'OK',
  git: 'OK',
  vcRedist: 'OK',
  upgradePackages: 'OK'
}

const createMockElectronAPI = () => {
  const logListeners: Array<(message: string) => void> = []

  const getValidationUpdate = () => ({
    ...validationState
  })

  return {
    getPlatform: () => 'darwin',
    changeTheme: (_theme: unknown) => {},
    onLogMessage: (listener: (message: string) => void) => {
      logListeners.push(listener)
    },
    showContextMenu: (_options: unknown) => {},
    Events: {
      trackEvent: (_eventName: string, _data?: unknown) => {}
    },
    Validation: {
      onUpdate: (_callback: (update: unknown) => void) => {},
      async getStatus() {
        return getValidationUpdate()
      },
      async validateInstallation(callback: (update: unknown) => void) {
        callback(getValidationUpdate())
      },
      async complete() {
        // Only allow completion when the base path is safe
        return !validationState.unsafeBasePath
      },
      dispose: () => {}
    },
    setBasePath: () => Promise.resolve(true),
    reinstall: () => Promise.resolve(),
    uv: {
      installRequirements: () => Promise.resolve(),
      clearCache: () => Promise.resolve(),
      resetVenv: () => Promise.resolve()
    }
  }
}

const ensureElectronAPI = () => {
  const globalWindow = window as { electronAPI?: unknown }
  if (!globalWindow.electronAPI) {
    globalWindow.electronAPI = createMockElectronAPI()
  }

  return globalWindow.electronAPI
}

const MaintenanceView = defineAsyncComponent(async () => {
  ensureElectronAPI()
  const module = await import('./MaintenanceView.vue')
  return module.default
})

const meta: Meta<typeof MaintenanceView> = {
  title: 'Desktop/Views/MaintenanceView',
  component: MaintenanceView,
  parameters: {
    layout: 'fullscreen',
    backgrounds: {
      default: 'dark',
      values: [
        { name: 'dark', value: '#0a0a0a' },
        { name: 'neutral-900', value: '#171717' },
        { name: 'neutral-950', value: '#0a0a0a' }
      ]
    }
  }
}

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
  name: 'All tasks OK',
  render: () => ({
    components: { MaintenanceView },
    setup() {
      validationState.inProgress = false
      validationState.installState = 'installed'
      validationState.basePath = 'OK'
      validationState.unsafeBasePath = false
      validationState.unsafeBasePathReason = null
      validationState.venvDirectory = 'OK'
      validationState.pythonInterpreter = 'OK'
      validationState.pythonPackages = 'OK'
      validationState.uv = 'OK'
      validationState.git = 'OK'
      validationState.vcRedist = 'OK'
      validationState.upgradePackages = 'OK'
      ensureElectronAPI()
      return {}
    },
    template: '<MaintenanceView />'
  })
}

export const UnsafeBasePathOneDrive: Story = {
  name: 'Unsafe base path (OneDrive)',
  render: () => ({
    components: { MaintenanceView },
    setup() {
      validationState.inProgress = false
      validationState.installState = 'installed'
      validationState.basePath = 'error'
      validationState.unsafeBasePath = true
      validationState.unsafeBasePathReason = 'oneDrive'
      validationState.venvDirectory = 'OK'
      validationState.pythonInterpreter = 'OK'
      validationState.pythonPackages = 'OK'
      validationState.uv = 'OK'
      validationState.git = 'OK'
      validationState.vcRedist = 'OK'
      validationState.upgradePackages = 'OK'
      ensureElectronAPI()
      return {}
    },
    template: '<MaintenanceView />'
  })
}
