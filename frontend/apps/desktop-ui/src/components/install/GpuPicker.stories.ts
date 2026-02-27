// eslint-disable-next-line storybook/no-renderer-packages
import type { Meta, StoryObj } from '@storybook/vue3'
import type {
  ElectronAPI,
  TorchDeviceType
} from '@comfyorg/comfyui-electron-types'
import { ref } from 'vue'

import GpuPicker from './GpuPicker.vue'

type Platform = ReturnType<ElectronAPI['getPlatform']>
type ElectronAPIStub = Pick<ElectronAPI, 'getPlatform'>
type WindowWithElectron = Window & { electronAPI?: ElectronAPIStub }

const meta: Meta<typeof GpuPicker> = {
  title: 'Desktop/Components/GpuPicker',
  component: GpuPicker,
  parameters: {
    layout: 'padded',
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

function createElectronDecorator(platform: Platform) {
  function getPlatform() {
    return platform
  }

  return function ElectronDecorator() {
    const windowWithElectron = window as WindowWithElectron
    windowWithElectron.electronAPI = { getPlatform }
    return { template: '<story />' }
  }
}

function renderWithDevice(device: TorchDeviceType | null) {
  return function Render() {
    return {
      components: { GpuPicker },
      setup() {
        const selected = ref<TorchDeviceType | null>(device)
        return { selected }
      },
      template: `
        <div class="min-h-screen bg-neutral-950 p-8">
          <GpuPicker v-model:device="selected" />
        </div>
      `
    }
  }
}

const windowsDecorator = createElectronDecorator('win32')
const macDecorator = createElectronDecorator('darwin')

export const WindowsNvidiaSelected: Story = {
  decorators: [windowsDecorator],
  render: renderWithDevice('nvidia')
}

export const WindowsAmdSelected: Story = {
  decorators: [windowsDecorator],
  render: renderWithDevice('amd')
}

export const WindowsCpuSelected: Story = {
  decorators: [windowsDecorator],
  render: renderWithDevice('cpu')
}

export const MacMpsSelected: Story = {
  decorators: [macDecorator],
  render: renderWithDevice('mps')
}
