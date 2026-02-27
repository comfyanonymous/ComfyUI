import type { VueWrapper } from '@vue/test-utils'
import { mount } from '@vue/test-utils'
import { createTestingPinia } from '@pinia/testing'
import Button from '@/components/ui/button/Button.vue'
import PrimeVue from 'primevue/config'
import Listbox from 'primevue/listbox'
import Select from 'primevue/select'
import Tooltip from 'primevue/tooltip'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick } from 'vue'
import { createI18n } from 'vue-i18n'

import VerifiedIcon from '@/components/icons/VerifiedIcon.vue'
import enMessages from '@/locales/en/main.json' with { type: 'json' }

// SelectedVersion is now using direct strings instead of enum

import PackVersionSelectorPopover from './PackVersionSelectorPopover.vue'

interface PackVersionSelectorVM {
  getVersionCompatibility: (version: string) => unknown
}

function getVM(wrapper: VueWrapper): PackVersionSelectorVM {
  return wrapper.vm as Partial<PackVersionSelectorVM> as PackVersionSelectorVM
}

// Default mock versions for reference
const defaultMockVersions = [
  {
    version: '1.0.0',
    createdAt: '2023-01-01',
    supported_os: ['windows', 'linux'],
    supported_accelerators: ['CPU'],
    supported_comfyui_version: '>=0.1.0',
    supported_comfyui_frontend_version: '>=1.0.0',
    supported_python_version: '>=3.8',
    is_banned: false,
    has_registry_data: true
  },
  { version: '0.9.0', createdAt: '2022-12-01' },
  { version: '0.8.0', createdAt: '2022-11-01' }
]

const mockNodePack = {
  id: 'test-pack',
  name: 'Test Pack',
  latest_version: {
    version: '1.0.0',
    supported_os: ['windows', 'linux'],
    supported_accelerators: ['CPU'],
    supported_comfyui_version: '>=0.1.0',
    supported_comfyui_frontend_version: '>=1.0.0',
    supported_python_version: '>=3.8',
    is_banned: false,
    has_registry_data: true
  },
  repository: 'https://github.com/user/repo',
  has_registry_data: true
}

// Create mock functions
const mockGetPackVersions = vi.fn()
const mockInstallPack = vi.fn().mockResolvedValue(undefined)
const mockCheckNodeCompatibility = vi.fn()
const mockIsPackInstalled = vi.fn(() => false)
const mockGetInstalledPackVersion = vi.fn(() => undefined)

// Mock the registry service
vi.mock('@/services/comfyRegistryService', () => ({
  useComfyRegistryService: vi.fn(() => ({
    getPackVersions: mockGetPackVersions
  }))
}))

// Mock the manager store
vi.mock('@/workbench/extensions/manager/stores/comfyManagerStore', () => ({
  useComfyManagerStore: vi.fn(() => ({
    installPack: {
      call: mockInstallPack,
      clear: vi.fn()
    },
    isPackInstalled: mockIsPackInstalled,
    getInstalledPackVersion: mockGetInstalledPackVersion
  }))
}))

// Mock the conflict detection composable
vi.mock(
  '@/workbench/extensions/manager/composables/useConflictDetection',
  () => ({
    useConflictDetection: vi.fn(() => ({
      checkNodeCompatibility: mockCheckNodeCompatibility
    }))
  })
)

const waitForPromises = async () => {
  await new Promise((resolve) => setTimeout(resolve, 16))
  await nextTick()
}

describe('PackVersionSelectorPopover', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockGetPackVersions.mockReset()
    mockInstallPack.mockReset().mockResolvedValue(undefined)
    mockCheckNodeCompatibility
      .mockReset()
      .mockReturnValue({ hasConflict: false, conflicts: [] })
    mockIsPackInstalled.mockReset().mockReturnValue(false)
    mockGetInstalledPackVersion.mockReset().mockReturnValue(undefined)
  })

  const mountComponent = ({
    props = {}
  }: { props?: Record<string, unknown> } = {}): VueWrapper => {
    const i18n = createI18n({
      legacy: false,
      locale: 'en',
      messages: { en: enMessages }
    })

    return mount(PackVersionSelectorPopover, {
      props: {
        nodePack: mockNodePack,
        ...props
      },
      global: {
        plugins: [PrimeVue, createTestingPinia({ stubActions: false }), i18n],
        components: {
          Listbox,
          VerifiedIcon,
          Select
        },
        directives: {
          tooltip: Tooltip
        }
      }
    })
  }

  it('fetches versions on mount', async () => {
    // Set up the mock for this specific test
    mockGetPackVersions.mockResolvedValueOnce(defaultMockVersions)

    mountComponent()
    await waitForPromises()

    expect(mockGetPackVersions).toHaveBeenCalledWith(mockNodePack.id)
  })

  it('shows loading state while fetching versions', async () => {
    // Delay the promise resolution
    mockGetPackVersions.mockImplementationOnce(
      () =>
        new Promise((resolve) =>
          setTimeout(() => resolve(defaultMockVersions), 1000)
        )
    )

    const wrapper = mountComponent()

    expect(wrapper.text()).toContain('Loading versions...')
  })

  it('displays special options and version options in the listbox', async () => {
    // Set up the mock for this specific test
    mockGetPackVersions.mockResolvedValueOnce(defaultMockVersions)

    const wrapper = mountComponent()
    await waitForPromises()

    const listbox = wrapper.findComponent(Listbox)
    expect(listbox.exists()).toBe(true)

    const options = listbox.props('options')!
    // Check that we have both special options and version options
    // Latest version (1.0.0) should be excluded from the version list to avoid duplication
    expect(options.length).toBe(defaultMockVersions.length + 1) // 2 special options + version options minus 1 duplicate

    // Check that special options exist
    expect(options.some((o) => o.value === 'nightly')).toBe(true)
    expect(options.some((o) => o.value === 'latest')).toBe(true)

    // Check that version options exist (excluding latest version 1.0.0)
    expect(options.some((o) => o.value === '1.0.0')).toBe(false) // Should be excluded as it's the latest
    expect(options.some((o) => o.value === '0.9.0')).toBe(true)
    expect(options.some((o) => o.value === '0.8.0')).toBe(true)
  })

  it('emits cancel event when cancel button is clicked', async () => {
    // Set up the mock for this specific test
    mockGetPackVersions.mockResolvedValueOnce(defaultMockVersions)

    const wrapper = mountComponent()
    await waitForPromises()

    const cancelButton = wrapper.findAllComponents(Button)[0]
    await cancelButton.trigger('click')

    expect(wrapper.emitted('cancel')).toBeTruthy()
  })

  it('calls installPack and emits submit when install button is clicked', async () => {
    // Set up the mock for this specific test
    mockGetPackVersions.mockResolvedValueOnce(defaultMockVersions)

    const wrapper = mountComponent()
    await waitForPromises()

    // Set the selected version
    await wrapper.findComponent(Listbox).setValue('0.9.0')

    const installButton = wrapper.findAllComponents(Button)[1]
    await installButton.trigger('click')

    // Check that installPack was called with the correct parameters
    expect(mockInstallPack).toHaveBeenCalledWith(
      expect.objectContaining({
        id: mockNodePack.id,
        repository: mockNodePack.repository,
        version: '0.9.0',
        selected_version: '0.9.0'
      })
    )

    // Check that submit was emitted
    expect(wrapper.emitted('submit')).toBeTruthy()
  })

  it('is reactive to nodePack prop changes', async () => {
    // Set up the mock for the initial fetch
    mockGetPackVersions.mockResolvedValueOnce(defaultMockVersions)

    const wrapper = mountComponent()
    await waitForPromises()

    // Set up the mock for the second fetch after prop change
    mockGetPackVersions.mockResolvedValueOnce(defaultMockVersions)

    // Update the nodePack prop
    const newNodePack = { ...mockNodePack, id: 'new-test-pack' }
    await wrapper.setProps({ nodePack: newNodePack })
    await waitForPromises()

    // Should fetch versions for the new nodePack
    expect(mockGetPackVersions).toHaveBeenCalledWith(newNodePack.id)
  })

  describe('nodePack.id changes', () => {
    it('re-fetches versions when nodePack.id changes', async () => {
      // Set up the mock for the initial fetch
      mockGetPackVersions.mockResolvedValueOnce(defaultMockVersions)

      const wrapper = mountComponent()
      await waitForPromises()

      // Verify initial fetch
      expect(mockGetPackVersions).toHaveBeenCalledTimes(1)
      expect(mockGetPackVersions).toHaveBeenCalledWith(mockNodePack.id)

      // Set up the mock for the second fetch
      const newVersions = [
        { version: '2.0.0', createdAt: '2023-06-01' },
        { version: '1.9.0', createdAt: '2023-05-01' }
      ]
      mockGetPackVersions.mockResolvedValueOnce(newVersions)

      // Update the nodePack with a new ID
      const newNodePack = {
        ...mockNodePack,
        id: 'different-pack',
        name: 'Different Pack'
      }
      await wrapper.setProps({ nodePack: newNodePack })
      await waitForPromises()

      // Should fetch versions for the new nodePack
      expect(mockGetPackVersions).toHaveBeenCalledTimes(2)
      expect(mockGetPackVersions).toHaveBeenLastCalledWith(newNodePack.id)

      // Check that new versions are displayed
      const listbox = wrapper.findComponent(Listbox)
      const options = listbox.props('options')!
      expect(options.some((o) => o.value === '2.0.0')).toBe(true)
      expect(options.some((o) => o.value === '1.9.0')).toBe(true)
    })

    it('does not re-fetch when nodePack changes but id remains the same', async () => {
      // Set up the mock for the initial fetch
      mockGetPackVersions.mockResolvedValueOnce(defaultMockVersions)

      const wrapper = mountComponent()
      await waitForPromises()

      // Verify initial fetch
      expect(mockGetPackVersions).toHaveBeenCalledTimes(1)

      // Update the nodePack with same ID but different properties
      const updatedNodePack = {
        ...mockNodePack,
        name: 'Updated Test Pack',
        description: 'New description'
      }
      await wrapper.setProps({ nodePack: updatedNodePack })
      await waitForPromises()

      // Should NOT fetch versions again
      expect(mockGetPackVersions).toHaveBeenCalledTimes(1)
    })

    it('maintains selected version when switching to a new pack', async () => {
      // Set up the mock for the initial fetch
      mockGetPackVersions.mockResolvedValueOnce(defaultMockVersions)

      const wrapper = mountComponent()
      await waitForPromises()

      // Select a specific version
      const listbox = wrapper.findComponent(Listbox)
      await listbox.setValue('0.9.0')
      expect(listbox.props('modelValue')).toBe('0.9.0')

      // Set up the mock for the second fetch
      mockGetPackVersions.mockResolvedValueOnce([
        { version: '3.0.0', createdAt: '2023-07-01' },
        { version: '0.9.0', createdAt: '2023-04-01' }
      ])

      // Update to a new pack that also has version 0.9.0
      const newNodePack = {
        id: 'another-pack',
        name: 'Another Pack',
        latest_version: { version: '3.0.0' }
      }
      await wrapper.setProps({ nodePack: newNodePack })
      await waitForPromises()

      // Selected version should remain the same if available
      expect(listbox.props('modelValue')).toBe('0.9.0')
    })
  })

  describe('Unclaimed GitHub packs handling', () => {
    it('falls back to nightly when no versions exist', async () => {
      // Set up the mock to return versions
      mockGetPackVersions.mockResolvedValueOnce(defaultMockVersions)

      const packWithRepo = {
        ...mockNodePack,
        latest_version: undefined
      }

      const wrapper = mountComponent({
        props: {
          nodePack: packWithRepo
        }
      })

      await waitForPromises()
      const listbox = wrapper.findComponent(Listbox)
      expect(listbox.exists()).toBe(true)
      expect(listbox.props('modelValue')).toBe('nightly')
    })

    it('defaults to nightly when publisher name is "Unclaimed"', async () => {
      // Set up the mock to return versions
      mockGetPackVersions.mockResolvedValueOnce(defaultMockVersions)

      const unclaimedNodePack = {
        ...mockNodePack,
        publisher: { name: 'Unclaimed' }
      }

      const wrapper = mountComponent({
        props: {
          nodePack: unclaimedNodePack
        }
      })

      await waitForPromises()
      const listbox = wrapper.findComponent(Listbox)
      expect(listbox.exists()).toBe(true)
      expect(listbox.props('modelValue')).toBe('nightly')
    })
  })

  describe('version compatibility checking', () => {
    it('shows warning icon for incompatible versions', async () => {
      // Set up the mock for versions
      mockGetPackVersions.mockResolvedValueOnce(defaultMockVersions)

      // Mock compatibility check to return conflict for specific version
      mockCheckNodeCompatibility.mockImplementation((versionData) => {
        if (versionData.supported_os?.includes('linux')) {
          return {
            hasConflict: true,
            conflicts: [
              {
                type: 'os',
                current_value: 'windows',
                required_value: 'linux'
              }
            ]
          }
        }
        return { hasConflict: false, conflicts: [] }
      })

      const nodePackWithCompatibility = {
        ...mockNodePack,
        supported_os: ['linux'],
        supported_accelerators: ['CUDA']
      }

      const wrapper = mountComponent({
        props: { nodePack: nodePackWithCompatibility }
      })
      await waitForPromises()

      // Check that compatibility checking function was called
      expect(mockCheckNodeCompatibility).toHaveBeenCalled()

      // The warning icon should be shown for incompatible versions
      const warningIcons = wrapper.findAll('.icon-\\[lucide--triangle-alert\\]')
      expect(warningIcons.length).toBeGreaterThan(0)
    })

    it('shows verified icon for compatible versions', async () => {
      // Set up the mock for versions
      mockGetPackVersions.mockResolvedValueOnce(defaultMockVersions)

      // Mock compatibility check to return no conflicts
      mockCheckNodeCompatibility.mockReturnValue({
        hasConflict: false,
        conflicts: []
      })

      const wrapper = mountComponent()
      await waitForPromises()

      // Check that compatibility checking function was called
      expect(mockCheckNodeCompatibility).toHaveBeenCalled()

      // The verified icon should be shown for compatible versions
      // Look for the VerifiedIcon component or SVG elements
      const verifiedIcons = wrapper.findAll('svg')
      expect(verifiedIcons.length).toBeGreaterThan(0)
    })

    it('calls checkVersionCompatibility with correct version data', async () => {
      // Set up the mock for versions with specific supported data
      const versionsWithCompatibility = [
        {
          version: '1.0.0',
          supported_os: ['windows', 'linux'],
          supported_accelerators: ['CUDA', 'CPU'],
          supported_comfyui_version: '>=0.1.0',
          supported_comfyui_frontend_version: '>=1.0.0'
        }
      ]
      mockGetPackVersions.mockResolvedValueOnce(versionsWithCompatibility)

      const nodePackWithCompatibility = {
        ...mockNodePack,
        supported_os: ['windows'],
        supported_accelerators: ['CPU'],
        supported_comfyui_version: '>=0.1.0',
        supported_comfyui_frontend_version: '>=1.0.0',
        latest_version: {
          version: '1.0.0',
          supported_os: ['windows', 'linux'],
          supported_accelerators: ['CPU'], // latest_version data takes precedence
          supported_comfyui_version: '>=0.1.0',
          supported_comfyui_frontend_version: '>=1.0.0',
          supported_python_version: '>=3.8',
          is_banned: false,
          has_registry_data: true
        }
      }

      const wrapper = mountComponent({
        props: { nodePack: nodePackWithCompatibility }
      })
      await waitForPromises()

      // Clear previous calls from component mounting/rendering
      mockCheckNodeCompatibility.mockClear()

      // Trigger compatibility check by accessing getVersionCompatibility
      const vm = getVM(wrapper)
      vm.getVersionCompatibility('1.0.0')

      // Verify that checkNodeCompatibility was called with correct data
      // Since 1.0.0 is the latest version, it should use latest_version data
      expect(mockCheckNodeCompatibility).toHaveBeenCalledWith({
        supported_os: ['windows', 'linux'],
        supported_accelerators: ['CPU'], // latest_version data takes precedence
        supported_comfyui_version: '>=0.1.0',
        supported_comfyui_frontend_version: '>=1.0.0',
        supported_python_version: '>=3.8',
        is_banned: false,
        has_registry_data: true,
        version: '1.0.0'
      })
    })

    it('shows version conflict warnings for ComfyUI and frontend versions', async () => {
      // Set up the mock for versions
      mockGetPackVersions.mockResolvedValueOnce(defaultMockVersions)

      // Mock compatibility check to return version conflicts
      mockCheckNodeCompatibility.mockImplementation((versionData) => {
        const conflicts = []
        if (versionData.supported_comfyui_version) {
          conflicts.push({
            type: 'comfyui_version',
            current_value: '0.5.0',
            required_value: versionData.supported_comfyui_version
          })
        }
        if (versionData.supported_comfyui_frontend_version) {
          conflicts.push({
            type: 'frontend_version',
            current_value: '1.0.0',
            required_value: versionData.supported_comfyui_frontend_version
          })
        }
        return {
          hasConflict: conflicts.length > 0,
          conflicts
        }
      })

      const nodePackWithVersionRequirements = {
        ...mockNodePack,
        supported_comfyui_version: '>=1.0.0',
        supported_comfyui_frontend_version: '>=2.0.0'
      }

      const wrapper = mountComponent({
        props: { nodePack: nodePackWithVersionRequirements }
      })
      await waitForPromises()

      // Check that compatibility checking function was called
      expect(mockCheckNodeCompatibility).toHaveBeenCalled()

      // The warning icon should be shown for version incompatible packages
      const warningIcons = wrapper.findAll('.icon-\\[lucide--triangle-alert\\]')
      expect(warningIcons.length).toBeGreaterThan(0)
    })

    it('handles latest and nightly versions using nodePack data', async () => {
      // Set up the mock for versions
      mockGetPackVersions.mockResolvedValueOnce(defaultMockVersions)

      const nodePackWithCompatibility = {
        ...mockNodePack,
        supported_os: ['windows'],
        supported_accelerators: ['CPU'],
        supported_comfyui_version: '>=0.1.0',
        supported_comfyui_frontend_version: '>=1.0.0',
        latest_version: {
          ...mockNodePack.latest_version,
          supported_os: ['windows'], // Match nodePack data for test consistency
          supported_accelerators: ['CPU'], // Match nodePack data for test consistency
          supported_python_version: '>=3.8',
          is_banned: false,
          has_registry_data: true
        }
      }

      const wrapper = mountComponent({
        props: { nodePack: nodePackWithCompatibility }
      })
      await waitForPromises()

      const vm = getVM(wrapper)

      // Clear previous calls from component mounting/rendering
      mockCheckNodeCompatibility.mockClear()

      // Test latest version
      vm.getVersionCompatibility('latest')
      expect(mockCheckNodeCompatibility).toHaveBeenCalledWith({
        supported_os: ['windows'],
        supported_accelerators: ['CPU'],
        supported_comfyui_version: '>=0.1.0',
        supported_comfyui_frontend_version: '>=1.0.0',
        supported_python_version: '>=3.8',
        is_banned: false,
        has_registry_data: true,
        version: '1.0.0'
      })

      // Clear for next test call
      mockCheckNodeCompatibility.mockClear()

      // Test nightly version
      vm.getVersionCompatibility('nightly')
      expect(mockCheckNodeCompatibility).toHaveBeenCalledWith({
        id: 'test-pack',
        name: 'Test Pack',
        supported_os: ['windows'],
        supported_accelerators: ['CPU'],
        supported_comfyui_version: '>=0.1.0',
        supported_comfyui_frontend_version: '>=1.0.0',
        repository: 'https://github.com/user/repo',
        has_registry_data: true,
        latest_version: {
          supported_os: ['windows'],
          supported_accelerators: ['CPU'],
          supported_python_version: '>=3.8',
          is_banned: false,
          has_registry_data: true,
          version: '1.0.0',
          supported_comfyui_version: '>=0.1.0',
          supported_comfyui_frontend_version: '>=1.0.0'
        }
      })
    })

    it('shows banned package warnings', async () => {
      // Set up the mock for versions
      mockGetPackVersions.mockResolvedValueOnce(defaultMockVersions)

      // Mock compatibility check to return banned conflicts
      mockCheckNodeCompatibility.mockImplementation((versionData) => {
        if (versionData.is_banned === true) {
          return {
            hasConflict: true,
            conflicts: [
              {
                type: 'banned',
                current_value: 'installed',
                required_value: 'not_banned'
              }
            ]
          }
        }
        return { hasConflict: false, conflicts: [] }
      })

      const bannedNodePack = {
        ...mockNodePack,
        is_banned: true,
        latest_version: {
          ...mockNodePack.latest_version,
          is_banned: true
        }
      }

      const wrapper = mountComponent({
        props: { nodePack: bannedNodePack }
      })
      await waitForPromises()

      // Check that compatibility checking function was called
      expect(mockCheckNodeCompatibility).toHaveBeenCalled()

      // Open the dropdown to see the options
      const select = wrapper.find('.p-select')
      if (!select.exists()) {
        // Try alternative selector
        const selectButton = wrapper.find('[aria-haspopup="listbox"]')
        if (selectButton.exists()) {
          await selectButton.trigger('click')
        }
      } else {
        await select.trigger('click')
      }
      await wrapper.vm.$nextTick()

      // The warning icon should be shown for banned packages in the dropdown options
      const warningIcons = wrapper.findAll('.icon-\\[lucide--triangle-alert\\]')
      expect(warningIcons.length).toBeGreaterThan(0)
    })

    it('shows security pending warnings', async () => {
      // Set up the mock for versions
      mockGetPackVersions.mockResolvedValueOnce(defaultMockVersions)

      // Mock compatibility check to return security pending conflicts
      mockCheckNodeCompatibility.mockImplementation((versionData) => {
        if (versionData.has_registry_data === false) {
          return {
            hasConflict: true,
            conflicts: [
              {
                type: 'pending',
                current_value: 'no_registry_data',
                required_value: 'registry_data_available'
              }
            ]
          }
        }
        return { hasConflict: false, conflicts: [] }
      })

      const securityPendingNodePack = {
        ...mockNodePack,
        has_registry_data: false,
        latest_version: {
          ...mockNodePack.latest_version,
          has_registry_data: false
        }
      }

      const wrapper = mountComponent({
        props: { nodePack: securityPendingNodePack }
      })
      await waitForPromises()

      // Check that compatibility checking function was called
      expect(mockCheckNodeCompatibility).toHaveBeenCalled()

      // The warning icon should be shown for security pending packages
      const warningIcons = wrapper.findAll('.icon-\\[lucide--triangle-alert\\]')
      expect(warningIcons.length).toBeGreaterThan(0)
    })
  })
})
