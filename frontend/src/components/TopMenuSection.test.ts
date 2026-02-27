import { createTestingPinia } from '@pinia/testing'
import { mount } from '@vue/test-utils'
import type { MenuItem } from 'primevue/menuitem'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { computed, defineComponent, h, nextTick, onMounted, ref } from 'vue'
import type { Component } from 'vue'
import { createI18n } from 'vue-i18n'

import TopMenuSection from '@/components/TopMenuSection.vue'
import QueueNotificationBannerHost from '@/components/queue/QueueNotificationBannerHost.vue'
import CurrentUserButton from '@/components/topbar/CurrentUserButton.vue'
import LoginButton from '@/components/topbar/LoginButton.vue'
import type {
  JobListItem,
  JobStatus
} from '@/platform/remote/comfyui/jobs/jobTypes'
import { useSettingStore } from '@/platform/settings/settingStore'
import { useCommandStore } from '@/stores/commandStore'
import { useExecutionStore } from '@/stores/executionStore'
import { TaskItemImpl, useQueueStore } from '@/stores/queueStore'
import { useSidebarTabStore } from '@/stores/workspace/sidebarTabStore'

const mockData = vi.hoisted(() => ({
  isLoggedIn: false,
  isDesktop: false,
  setShowConflictRedDot: (_value: boolean) => {}
}))

vi.mock('@/composables/auth/useCurrentUser', () => ({
  useCurrentUser: () => {
    return {
      isLoggedIn: computed(() => mockData.isLoggedIn)
    }
  }
}))

vi.mock('@/platform/distribution/types', () => ({
  isCloud: false,
  isNightly: false,
  get isDesktop() {
    return mockData.isDesktop
  }
}))

vi.mock('@/platform/updates/common/releaseStore', () => ({
  useReleaseStore: () => ({
    shouldShowRedDot: computed(() => true)
  })
}))

vi.mock(
  '@/workbench/extensions/manager/composables/useConflictAcknowledgment',
  () => {
    const shouldShowConflictRedDot = ref(false)
    mockData.setShowConflictRedDot = (value: boolean) => {
      shouldShowConflictRedDot.value = value
    }

    return {
      useConflictAcknowledgment: () => ({
        shouldShowRedDot: shouldShowConflictRedDot
      })
    }
  }
)

vi.mock('@/workbench/extensions/manager/composables/useManagerState', () => ({
  useManagerState: () => ({
    shouldShowManagerButtons: computed(() => true),
    openManager: vi.fn()
  })
}))

vi.mock('@/stores/firebaseAuthStore', () => ({
  useFirebaseAuthStore: vi.fn(() => ({
    currentUser: null,
    loading: false
  }))
}))

type WrapperOptions = {
  pinia?: ReturnType<typeof createTestingPinia>
  stubs?: Record<string, boolean | Component>
  attachTo?: HTMLElement
}

function createWrapper({
  pinia = createTestingPinia({ createSpy: vi.fn }),
  stubs = {},
  attachTo
}: WrapperOptions = {}) {
  const i18n = createI18n({
    legacy: false,
    locale: 'en',
    messages: {
      en: {
        sideToolbar: {
          queueProgressOverlay: {
            viewJobHistory: 'View job history',
            expandCollapsedQueue: 'Expand collapsed queue',
            activeJobsShort: '{count} active | {count} active',
            clearQueueTooltip: 'Clear queue'
          }
        }
      }
    }
  })

  return mount(TopMenuSection, {
    attachTo,
    global: {
      plugins: [pinia, i18n],
      stubs: {
        SubgraphBreadcrumb: true,
        QueueProgressOverlay: true,
        QueueInlineProgressSummary: true,
        QueueNotificationBannerHost: true,
        CurrentUserButton: true,
        LoginButton: true,
        ContextMenu: {
          name: 'ContextMenu',
          props: ['model'],
          template: '<div />'
        },
        ...stubs
      },
      directives: {
        tooltip: () => {}
      }
    }
  })
}

function createJob(id: string, status: JobStatus): JobListItem {
  return {
    id,
    status,
    create_time: 0,
    priority: 0
  }
}

function createTask(id: string, status: JobStatus): TaskItemImpl {
  return new TaskItemImpl(createJob(id, status))
}

function createComfyActionbarStub(actionbarTarget: HTMLElement) {
  return defineComponent({
    name: 'ComfyActionbar',
    setup(_, { emit }) {
      onMounted(() => {
        emit('update:progressTarget', actionbarTarget)
      })
      return () => h('div')
    }
  })
}

describe('TopMenuSection', () => {
  beforeEach(() => {
    vi.resetAllMocks()
    localStorage.clear()
    mockData.isDesktop = false
    mockData.isLoggedIn = false
    mockData.setShowConflictRedDot(false)
  })

  describe('authentication state', () => {
    describe('when user is logged in', () => {
      beforeEach(() => {
        mockData.isLoggedIn = true
      })

      it('should display CurrentUserButton and not display LoginButton', () => {
        const wrapper = createWrapper()
        expect(wrapper.findComponent(CurrentUserButton).exists()).toBe(true)
        expect(wrapper.findComponent(LoginButton).exists()).toBe(false)
      })
    })

    describe('when user is not logged in', () => {
      beforeEach(() => {
        mockData.isLoggedIn = false
      })

      describe('on desktop platform', () => {
        it('should display LoginButton and not display CurrentUserButton', () => {
          mockData.isDesktop = true
          const wrapper = createWrapper()
          expect(wrapper.findComponent(LoginButton).exists()).toBe(true)
          expect(wrapper.findComponent(CurrentUserButton).exists()).toBe(false)
        })
      })

      describe('on web platform', () => {
        it('should not display CurrentUserButton and not display LoginButton', () => {
          const wrapper = createWrapper()
          expect(wrapper.findComponent(CurrentUserButton).exists()).toBe(false)
          expect(wrapper.findComponent(LoginButton).exists()).toBe(false)
        })
      })
    })
  })

  it('shows the active jobs label with the current count', async () => {
    const wrapper = createWrapper()
    const queueStore = useQueueStore()
    queueStore.pendingTasks = [createTask('pending-1', 'pending')]
    queueStore.runningTasks = [
      createTask('running-1', 'in_progress'),
      createTask('running-2', 'in_progress')
    ]

    await nextTick()

    const queueButton = wrapper.find('[data-testid="queue-overlay-toggle"]')
    expect(queueButton.text()).toContain('3 active')
    expect(wrapper.find('[data-testid="active-jobs-indicator"]').exists()).toBe(
      true
    )
  })

  it('hides the active jobs indicator when no jobs are active', () => {
    const wrapper = createWrapper()

    expect(wrapper.find('[data-testid="active-jobs-indicator"]').exists()).toBe(
      false
    )
  })

  it('hides queue progress overlay when QPO V2 is enabled', async () => {
    const pinia = createTestingPinia({ createSpy: vi.fn })
    const settingStore = useSettingStore(pinia)
    vi.mocked(settingStore.get).mockImplementation((key) =>
      key === 'Comfy.Queue.QPOV2' ? true : undefined
    )
    const wrapper = createWrapper({ pinia })

    await nextTick()

    expect(wrapper.find('[data-testid="queue-overlay-toggle"]').exists()).toBe(
      true
    )
    expect(
      wrapper.findComponent({ name: 'QueueProgressOverlay' }).exists()
    ).toBe(false)
  })

  it('toggles the queue progress overlay when QPO V2 is disabled', async () => {
    const pinia = createTestingPinia({ createSpy: vi.fn, stubActions: false })
    const settingStore = useSettingStore(pinia)
    vi.mocked(settingStore.get).mockImplementation((key) =>
      key === 'Comfy.Queue.QPOV2' ? false : undefined
    )
    const wrapper = createWrapper({ pinia })
    const commandStore = useCommandStore(pinia)

    await wrapper.find('[data-testid="queue-overlay-toggle"]').trigger('click')

    expect(commandStore.execute).toHaveBeenCalledWith(
      'Comfy.Queue.ToggleOverlay'
    )
  })

  it('opens the job history sidebar tab when QPO V2 is enabled', async () => {
    const pinia = createTestingPinia({ createSpy: vi.fn, stubActions: false })
    const settingStore = useSettingStore(pinia)
    vi.mocked(settingStore.get).mockImplementation((key) =>
      key === 'Comfy.Queue.QPOV2' ? true : undefined
    )
    const wrapper = createWrapper({ pinia })
    const sidebarTabStore = useSidebarTabStore(pinia)

    await wrapper.find('[data-testid="queue-overlay-toggle"]').trigger('click')

    expect(sidebarTabStore.activeSidebarTabId).toBe('job-history')
  })

  it('toggles the job history sidebar tab when QPO V2 is enabled', async () => {
    const pinia = createTestingPinia({ createSpy: vi.fn, stubActions: false })
    const settingStore = useSettingStore(pinia)
    vi.mocked(settingStore.get).mockImplementation((key) =>
      key === 'Comfy.Queue.QPOV2' ? true : undefined
    )
    const wrapper = createWrapper({ pinia })
    const sidebarTabStore = useSidebarTabStore(pinia)
    const toggleButton = wrapper.find('[data-testid="queue-overlay-toggle"]')

    await toggleButton.trigger('click')
    expect(sidebarTabStore.activeSidebarTabId).toBe('job-history')

    await toggleButton.trigger('click')
    expect(sidebarTabStore.activeSidebarTabId).toBe(null)
  })

  describe('inline progress summary', () => {
    const configureSettings = (
      pinia: ReturnType<typeof createTestingPinia>,
      qpoV2Enabled: boolean
    ) => {
      const settingStore = useSettingStore(pinia)
      vi.mocked(settingStore.get).mockImplementation((key) => {
        if (key === 'Comfy.Queue.QPOV2') return qpoV2Enabled
        if (key === 'Comfy.UseNewMenu') return 'Top'
        return undefined
      })
    }

    it('renders inline progress summary when QPO V2 is enabled', async () => {
      const pinia = createTestingPinia({ createSpy: vi.fn })
      configureSettings(pinia, true)

      const wrapper = createWrapper({ pinia })

      await nextTick()

      expect(
        wrapper.findComponent({ name: 'QueueInlineProgressSummary' }).exists()
      ).toBe(true)
    })

    it('does not render inline progress summary when QPO V2 is disabled', async () => {
      const pinia = createTestingPinia({ createSpy: vi.fn })
      configureSettings(pinia, false)

      const wrapper = createWrapper({ pinia })

      await nextTick()

      expect(
        wrapper.findComponent({ name: 'QueueInlineProgressSummary' }).exists()
      ).toBe(false)
    })

    it('teleports inline progress summary when actionbar is floating', async () => {
      localStorage.setItem('Comfy.MenuPosition.Docked', 'false')
      const actionbarTarget = document.createElement('div')
      document.body.appendChild(actionbarTarget)
      const pinia = createTestingPinia({ createSpy: vi.fn })
      configureSettings(pinia, true)
      const executionStore = useExecutionStore(pinia)
      executionStore.activeJobId = 'job-1'

      const ComfyActionbarStub = createComfyActionbarStub(actionbarTarget)

      const wrapper = createWrapper({
        pinia,
        attachTo: document.body,
        stubs: {
          ComfyActionbar: ComfyActionbarStub,
          QueueInlineProgressSummary: false
        }
      })

      try {
        await nextTick()

        expect(actionbarTarget.querySelector('[role="status"]')).not.toBeNull()
      } finally {
        wrapper.unmount()
        actionbarTarget.remove()
      }
    })
  })

  describe(QueueNotificationBannerHost, () => {
    const configureSettings = (
      pinia: ReturnType<typeof createTestingPinia>,
      qpoV2Enabled: boolean
    ) => {
      const settingStore = useSettingStore(pinia)
      vi.mocked(settingStore.get).mockImplementation((key) => {
        if (key === 'Comfy.Queue.QPOV2') return qpoV2Enabled
        if (key === 'Comfy.UseNewMenu') return 'Top'
        return undefined
      })
    }

    it('renders queue notification banners when QPO V2 is enabled', async () => {
      const pinia = createTestingPinia({ createSpy: vi.fn })
      configureSettings(pinia, true)

      const wrapper = createWrapper({ pinia })

      await nextTick()

      expect(
        wrapper.findComponent({ name: 'QueueNotificationBannerHost' }).exists()
      ).toBe(true)
    })

    it('renders queue notification banners when QPO V2 is disabled', async () => {
      const pinia = createTestingPinia({ createSpy: vi.fn })
      configureSettings(pinia, false)

      const wrapper = createWrapper({ pinia })

      await nextTick()

      expect(
        wrapper.findComponent({ name: 'QueueNotificationBannerHost' }).exists()
      ).toBe(true)
    })

    it('renders inline summary above banners when both are visible', async () => {
      const pinia = createTestingPinia({ createSpy: vi.fn })
      configureSettings(pinia, true)
      const wrapper = createWrapper({ pinia })

      await nextTick()

      const html = wrapper.html()
      const inlineSummaryIndex = html.indexOf(
        'queue-inline-progress-summary-stub'
      )
      const queueBannerIndex = html.indexOf(
        'queue-notification-banner-host-stub'
      )

      expect(inlineSummaryIndex).toBeGreaterThan(-1)
      expect(queueBannerIndex).toBeGreaterThan(-1)
      expect(inlineSummaryIndex).toBeLessThan(queueBannerIndex)
    })

    it('does not teleport queue notification banners when actionbar is floating', async () => {
      localStorage.setItem('Comfy.MenuPosition.Docked', 'false')
      const actionbarTarget = document.createElement('div')
      document.body.appendChild(actionbarTarget)
      const pinia = createTestingPinia({ createSpy: vi.fn })
      configureSettings(pinia, true)
      const executionStore = useExecutionStore(pinia)
      executionStore.activeJobId = 'job-1'

      const ComfyActionbarStub = createComfyActionbarStub(actionbarTarget)

      const wrapper = createWrapper({
        pinia,
        attachTo: document.body,
        stubs: {
          ComfyActionbar: ComfyActionbarStub,
          QueueNotificationBannerHost: true
        }
      })

      try {
        await nextTick()

        expect(
          actionbarTarget.querySelector('queue-notification-banner-host-stub')
        ).toBeNull()
        expect(
          wrapper
            .findComponent({ name: 'QueueNotificationBannerHost' })
            .exists()
        ).toBe(true)
      } finally {
        wrapper.unmount()
        actionbarTarget.remove()
      }
    })
  })

  it('disables the clear queue context menu item when no queued jobs exist', () => {
    const wrapper = createWrapper()
    const menu = wrapper.findComponent({ name: 'ContextMenu' })
    const model = menu.props('model') as MenuItem[]
    expect(model[0]?.label).toBe('Clear queue')
    expect(model[0]?.disabled).toBe(true)
  })

  it('enables the clear queue context menu item when queued jobs exist', async () => {
    const wrapper = createWrapper()
    const queueStore = useQueueStore()
    queueStore.pendingTasks = [createTask('pending-1', 'pending')]

    await nextTick()

    const menu = wrapper.findComponent({ name: 'ContextMenu' })
    const model = menu.props('model') as MenuItem[]
    expect(model[0]?.disabled).toBe(false)
  })

  it('shows manager red dot only for manager conflicts', async () => {
    const wrapper = createWrapper()

    // Release red dot is mocked as true globally for this test file.
    expect(wrapper.find('span.bg-red-500').exists()).toBe(false)

    mockData.setShowConflictRedDot(true)
    await nextTick()

    expect(wrapper.find('span.bg-red-500').exists()).toBe(true)
  })
})
