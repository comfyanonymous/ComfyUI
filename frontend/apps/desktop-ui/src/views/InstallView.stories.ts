// eslint-disable-next-line storybook/no-renderer-packages
import type { Meta, StoryObj } from '@storybook/vue3'
import type { ElectronAPI } from '@comfyorg/comfyui-electron-types'
import { nextTick, provide } from 'vue'
import type { ElectronWindow } from '@/utils/envUtil'
import { createMemoryHistory, createRouter } from 'vue-router'

import InstallView from './InstallView.vue'

// Create a mock router for stories
const createMockRouter = () =>
  createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: '/', component: { template: '<div>Home</div>' } },
      {
        path: '/server-start',
        component: { template: '<div>Server Start</div>' }
      },
      {
        path: '/manual-configuration',
        component: { template: '<div>Manual Configuration</div>' }
      }
    ]
  })

const meta: Meta<typeof InstallView> = {
  title: 'Desktop/Views/InstallView',
  component: InstallView,
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
  },
  decorators: [
    (story) => {
      // Create router for this story
      const router = createMockRouter()

      // Mock electron API
      ;(window as ElectronWindow).electronAPI = {
        getPlatform: () => 'darwin',
        Config: {
          getDetectedGpu: () => Promise.resolve('mps')
        },
        Events: {
          trackEvent: (
            _eventName: string,
            _data?: Record<string, unknown>
          ) => {}
        },
        installComfyUI: (
          _options: Parameters<ElectronAPI['installComfyUI']>[0]
        ) => {},
        changeTheme: (_theme: Parameters<ElectronAPI['changeTheme']>[0]) => {},
        getSystemPaths: () =>
          Promise.resolve({
            defaultInstallPath: '/Users/username/ComfyUI'
          }),
        validateInstallPath: () =>
          Promise.resolve({
            isValid: true,
            exists: false,
            canWrite: true,
            freeSpace: 100000000000,
            requiredSpace: 10000000000,
            isNonDefaultDrive: false
          }),
        validateComfyUISource: () =>
          Promise.resolve({
            isValid: true
          }),
        showDirectoryPicker: () => Promise.resolve('/Users/username/ComfyUI')
      }

      return {
        setup() {
          // Provide router for all child components
          provide('router', router)
          return {
            story
          }
        },
        template: '<div style="width: 100vw; height: 100vh;"><story /></div>'
      }
    }
  ]
}

export default meta
type Story = StoryObj<typeof meta>

// Default story - start at GPU selection
export const GpuSelection: Story = {
  render: () => ({
    components: { InstallView },
    setup() {
      // The component will automatically start at step 1
      return {}
    },
    template: '<InstallView />'
  })
}

// Story showing the install location step
export const InstallLocation: Story = {
  render: () => ({
    components: { InstallView },
    setup() {
      return {}
    },
    async mounted() {
      // Wait for component to be fully mounted
      await nextTick()

      // Select Apple Metal option to enable navigation
      const hardwareOptions = this.$el.querySelectorAll(
        '.p-selectbutton-option'
      )
      if (hardwareOptions.length > 0) {
        hardwareOptions[0].click() // Click Apple Metal (first option)
      }

      await nextTick()

      // Click Next to go to step 2
      const buttons = Array.from(
        this.$el.querySelectorAll('button')
      ) as HTMLButtonElement[]
      const nextBtn = buttons.find((btn) => btn.textContent?.includes('Next'))
      if (nextBtn) {
        nextBtn.click()
      }
    },
    template: '<InstallView />'
  })
}

// Story showing the migration step (currently empty)
export const MigrationStep: Story = {
  render: () => ({
    components: { InstallView },
    setup() {
      return {}
    },
    async mounted() {
      // Wait for component to be fully mounted
      await nextTick()

      // Select Apple Metal option to enable navigation
      const hardwareOptions = this.$el.querySelectorAll(
        '.p-selectbutton-option'
      )
      if (hardwareOptions.length > 0) {
        hardwareOptions[0].click() // Click Apple Metal (first option)
      }

      await nextTick()

      // Click Next to go to step 2
      const buttons1 = Array.from(
        this.$el.querySelectorAll('button')
      ) as HTMLButtonElement[]
      const nextBtn1 = buttons1.find((btn) => btn.textContent?.includes('Next'))
      if (nextBtn1) {
        nextBtn1.click()
      }

      await nextTick()

      // Click Next again to go to step 3
      const buttons2 = Array.from(
        this.$el.querySelectorAll('button')
      ) as HTMLButtonElement[]
      const nextBtn2 = buttons2.find((btn) => btn.textContent?.includes('Next'))
      if (nextBtn2) {
        nextBtn2.click()
      }
    },
    template: '<InstallView />'
  })
}

// Story showing the desktop settings configuration
export const DesktopSettings: Story = {
  render: () => ({
    components: { InstallView },
    setup() {
      return {}
    },
    async mounted() {
      // Wait for component to be fully mounted
      await nextTick()

      // Select Apple Metal option to enable navigation
      const hardwareOptions = this.$el.querySelectorAll(
        '.p-selectbutton-option'
      )
      if (hardwareOptions.length > 0) {
        hardwareOptions[0].click() // Click Apple Metal (first option)
      }

      await nextTick()

      // Click Next to go to step 2
      const buttons1 = Array.from(
        this.$el.querySelectorAll('button')
      ) as HTMLButtonElement[]
      const nextBtn1 = buttons1.find((btn) => btn.textContent?.includes('Next'))
      if (nextBtn1) {
        nextBtn1.click()
      }

      await nextTick()

      // Click Next again to go to step 3
      const buttons2 = Array.from(
        this.$el.querySelectorAll('button')
      ) as HTMLButtonElement[]
      const nextBtn2 = buttons2.find((btn) => btn.textContent?.includes('Next'))
      if (nextBtn2) {
        nextBtn2.click()
      }

      await nextTick()

      // Click Next again to go to step 4
      const buttons3 = Array.from(
        this.$el.querySelectorAll('button')
      ) as HTMLButtonElement[]
      const nextBtn3 = buttons3.find((btn) => btn.textContent?.includes('Next'))
      if (nextBtn3) {
        nextBtn3.click()
      }
    },
    template: '<InstallView />'
  })
}

// Story with Windows platform (no Apple Metal option)
export const WindowsPlatform: Story = {
  render: () => {
    // Override the platform to Windows
    ;(window as ElectronWindow).electronAPI.getPlatform = () => 'win32'
    ;(window as ElectronWindow).electronAPI.Config.getDetectedGpu = () =>
      Promise.resolve('nvidia')

    return {
      components: { InstallView },
      setup() {
        return {}
      },
      template: '<InstallView />'
    }
  }
}

// Story with macOS platform (Apple Metal option)
export const MacOSPlatform: Story = {
  name: 'macOS Platform',
  render: () => {
    // Override the platform to macOS
    ;(window as ElectronWindow).electronAPI.getPlatform = () => 'darwin'
    ;(window as ElectronWindow).electronAPI.Config.getDetectedGpu = () =>
      Promise.resolve('mps')

    return {
      components: { InstallView },
      setup() {
        return {}
      },
      template: '<InstallView />'
    }
  }
}

// Story with CPU selected
export const CpuSelected: Story = {
  render: () => ({
    components: { InstallView },
    setup() {
      return {}
    },
    async mounted() {
      // Wait for component to be fully mounted
      await nextTick()

      // Find and click the CPU hardware option
      const hardwareButtons = this.$el.querySelectorAll('.hardware-option')
      // CPU is the button with "CPU" text
      for (const button of hardwareButtons) {
        if (button.textContent?.includes('CPU')) {
          button.click()
          break
        }
      }
    },
    template: '<InstallView />'
  })
}

// Story with manual install selected
export const ManualInstall: Story = {
  render: () => ({
    components: { InstallView },
    setup() {
      return {}
    },
    async mounted() {
      // Wait for component to be fully mounted
      await nextTick()

      // Find and click the Manual Install hardware option
      const hardwareButtons = this.$el.querySelectorAll('.hardware-option')
      // Manual Install is the button with "Manual Install" text
      for (const button of hardwareButtons) {
        if (button.textContent?.includes('Manual Install')) {
          button.click()
          break
        }
      }
    },
    template: '<InstallView />'
  })
}

// Story with error state (invalid install path)
export const ErrorState: Story = {
  render: () => {
    // Override validation to return an error
    ;(window as ElectronWindow).electronAPI.validateInstallPath = () =>
      Promise.resolve({
        isValid: false,
        exists: false,
        canWrite: false,
        freeSpace: 100000000000,
        requiredSpace: 10000000000,
        isNonDefaultDrive: false,
        error: 'Story mock: Example error state'
      })

    return {
      components: { InstallView },
      setup() {
        return {}
      },
      async mounted() {
        // Wait for component to be fully mounted
        await nextTick()

        // Select Apple Metal option to enable navigation
        const hardwareOptions = this.$el.querySelectorAll(
          '.p-selectbutton-option'
        )
        if (hardwareOptions.length > 0) {
          hardwareOptions[0].click() // Click Apple Metal (first option)
        }

        await nextTick()

        // Click Next to go to step 2 where error will be shown
        const buttons = Array.from(
          this.$el.querySelectorAll('button')
        ) as HTMLButtonElement[]
        const nextBtn = buttons.find((btn) => btn.textContent?.includes('Next'))
        if (nextBtn) {
          nextBtn.click()
        }
      },
      template: '<InstallView />'
    }
  }
}

// Story with warning state (non-default drive)
export const WarningState: Story = {
  render: () => {
    // Override validation to return a warning about non-default drive
    ;(window as ElectronWindow).electronAPI.validateInstallPath = () =>
      Promise.resolve({
        isValid: true,
        exists: false,
        canWrite: true,
        freeSpace: 500_000_000_000,
        requiredSpace: 10_000_000_000,
        isNonDefaultDrive: true
      })

    return {
      components: { InstallView },
      setup() {
        return {}
      },
      async mounted() {
        // Wait for component to be fully mounted
        await nextTick()

        // Select Apple Metal option to enable navigation
        const hardwareOptions = this.$el.querySelectorAll('.hardware-option')
        if (hardwareOptions.length > 0) {
          hardwareOptions[0].click() // Click Apple Metal (first option)
        }

        await nextTick()

        // Click Next to go to step 2 where warning will be shown
        const buttons = Array.from(
          this.$el.querySelectorAll('button')
        ) as HTMLButtonElement[]
        const nextBtn = buttons.find((btn) => btn.textContent?.includes('Next'))
        if (nextBtn) {
          nextBtn.click()
        }
      },
      template: '<InstallView />'
    }
  }
}
