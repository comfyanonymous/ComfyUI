import { createTestingPinia } from '@pinia/testing'
import { mount } from '@vue/test-utils'
import Badge from 'primevue/badge'
import Button from '@/components/ui/button/Button.vue'
import Column from 'primevue/column'
import PrimeVue from 'primevue/config'
import DataTable from 'primevue/datatable'
import Message from 'primevue/message'
import ProgressSpinner from 'primevue/progressspinner'
import Tooltip from 'primevue/tooltip'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick } from 'vue'
import { createI18n } from 'vue-i18n'

import type { AuditLog } from '@/services/customerEventsService'
import { EventType } from '@/services/customerEventsService'

import UsageLogsTable from './UsageLogsTable.vue'

type ComponentInstance = InstanceType<typeof UsageLogsTable> & {
  loading: boolean
  error: string | null
  events: Partial<AuditLog>[]
  pagination: {
    page: number
    limit: number
    total: number
    totalPages: number
  }
  dataTableFirst: number
  tooltipContentMap: Map<string, string>
  loadEvents: () => Promise<void>
  refresh: () => Promise<void>
  onPageChange: (event: { page: number }) => void
}

// Mock the customerEventsService
const mockCustomerEventsService = vi.hoisted(() => ({
  getMyEvents: vi.fn(),
  formatEventType: vi.fn(),
  getEventSeverity: vi.fn(),
  formatAmount: vi.fn(),
  formatDate: vi.fn(),
  hasAdditionalInfo: vi.fn(),
  getTooltipContent: vi.fn(),
  error: { value: null },
  isLoading: { value: false }
}))

vi.mock('@/services/customerEventsService', () => ({
  useCustomerEventsService: () => mockCustomerEventsService,
  EventType: {
    CREDIT_ADDED: 'credit_added',
    ACCOUNT_CREATED: 'account_created',
    API_USAGE_STARTED: 'api_usage_started',
    API_USAGE_COMPLETED: 'api_usage_completed'
  }
}))

// Create i18n instance
const i18n = createI18n({
  legacy: false,
  locale: 'en',
  messages: {
    en: {
      credits: {
        eventType: 'Event Type',
        details: 'Details',
        time: 'Time',
        additionalInfo: 'Additional Info',
        added: 'Added',
        accountInitialized: 'Account initialized',
        model: 'Model'
      }
    }
  }
})

describe('UsageLogsTable', () => {
  const mockEventsResponse = {
    events: [
      {
        event_id: 'event-1',
        event_type: 'credit_added',
        params: {
          amount: 1000,
          transaction_id: 'txn-123'
        },
        createdAt: '2024-01-01T10:00:00Z'
      },
      {
        event_id: 'event-2',
        event_type: 'api_usage_completed',
        params: {
          api_name: 'Image Generation',
          model: 'sdxl-base',
          duration: 5000
        },
        createdAt: '2024-01-02T10:00:00Z'
      }
    ],
    total: 2,
    page: 1,
    limit: 7,
    totalPages: 1
  }

  beforeEach(() => {
    vi.clearAllMocks()

    // Setup default service mock implementations
    mockCustomerEventsService.getMyEvents.mockResolvedValue(mockEventsResponse)
    mockCustomerEventsService.formatEventType.mockImplementation((type) => {
      switch (type) {
        case EventType.CREDIT_ADDED:
          return 'Credits Added'
        case EventType.ACCOUNT_CREATED:
          return 'Account Created'
        case EventType.API_USAGE_COMPLETED:
          return 'API Usage'
        default:
          return type
      }
    })
    mockCustomerEventsService.getEventSeverity.mockImplementation((type) => {
      switch (type) {
        case EventType.CREDIT_ADDED:
          return 'success'
        case EventType.ACCOUNT_CREATED:
          return 'info'
        case EventType.API_USAGE_COMPLETED:
          return 'warning'
        default:
          return 'info'
      }
    })
    mockCustomerEventsService.formatAmount.mockImplementation((amount) => {
      if (!amount) return '0.00'
      return (amount / 100).toFixed(2)
    })
    mockCustomerEventsService.formatDate.mockImplementation((dateString) => {
      return new Date(dateString).toLocaleDateString()
    })
    mockCustomerEventsService.hasAdditionalInfo.mockImplementation((event) => {
      const { amount, api_name, model, ...otherParams } = event.params || {}
      return Object.keys(otherParams).length > 0
    })
    mockCustomerEventsService.getTooltipContent.mockImplementation(() => {
      return '<strong>Transaction Id:</strong> txn-123'
    })
    mockCustomerEventsService.error.value = null
    mockCustomerEventsService.isLoading.value = false
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  const mountComponent = (options = {}) => {
    return mount(UsageLogsTable, {
      global: {
        plugins: [PrimeVue, i18n, createTestingPinia()],
        components: {
          DataTable,
          Column,
          Badge,
          Button,
          Message,
          ProgressSpinner
        },
        directives: {
          tooltip: Tooltip
        }
      },
      ...options
    })
  }

  describe('loading states', () => {
    it('shows loading spinner when loading is true', async () => {
      const wrapper = mountComponent()
      const vm = wrapper.vm as ComponentInstance
      vm.loading = true
      await nextTick()

      expect(wrapper.findComponent(ProgressSpinner).exists()).toBe(true)
      expect(wrapper.findComponent(DataTable).exists()).toBe(false)
    })

    it('shows error message when error exists', async () => {
      const wrapper = mountComponent()
      const vm = wrapper.vm as ComponentInstance
      vm.error = 'Failed to load events'
      vm.loading = false
      await nextTick()

      const messageComponent = wrapper.findComponent(Message)
      expect(messageComponent.exists()).toBe(true)
      expect(messageComponent.props('severity')).toBe('error')
      expect(messageComponent.text()).toContain('Failed to load events')
    })

    it('shows data table when loaded successfully', async () => {
      const wrapper = mountComponent()

      const vm = wrapper.vm as ComponentInstance
      // Wait for component to mount and load data
      await wrapper.vm.$nextTick()
      await new Promise((resolve) => setTimeout(resolve, 0))

      vm.loading = false
      vm.events = mockEventsResponse.events
      await nextTick()

      expect(wrapper.findComponent(DataTable).exists()).toBe(true)
      expect(wrapper.findComponent(ProgressSpinner).exists()).toBe(false)
      expect(wrapper.findComponent(Message).exists()).toBe(false)
    })
  })

  describe('data rendering', () => {
    it('renders events data correctly', async () => {
      const wrapper = mountComponent()
      const vm = wrapper.vm as ComponentInstance
      vm.loading = false
      vm.events = mockEventsResponse.events
      await nextTick()

      const dataTable = wrapper.findComponent(DataTable)
      expect(dataTable.props('value')).toEqual(mockEventsResponse.events)
      expect(dataTable.props('rows')).toBe(7)
      expect(dataTable.props('paginator')).toBe(true)
      expect(dataTable.props('lazy')).toBe(true)
    })

    it('renders badge for event types correctly', async () => {
      const wrapper = mountComponent()
      const vm = wrapper.vm as ComponentInstance
      vm.loading = false
      vm.events = mockEventsResponse.events
      await nextTick()

      const badges = wrapper.findAllComponents(Badge)
      expect(badges.length).toBeGreaterThan(0)

      // Check if formatEventType and getEventSeverity are called
      expect(mockCustomerEventsService.formatEventType).toHaveBeenCalled()
      expect(mockCustomerEventsService.getEventSeverity).toHaveBeenCalled()
    })

    it('renders different event details based on event type', async () => {
      const wrapper = mountComponent()
      const vm = wrapper.vm as ComponentInstance
      vm.loading = false
      vm.events = mockEventsResponse.events
      await nextTick()

      // Check if formatAmount is called for credit_added events
      expect(mockCustomerEventsService.formatAmount).toHaveBeenCalled()
    })

    it('renders tooltip buttons for events with additional info', async () => {
      mockCustomerEventsService.hasAdditionalInfo.mockReturnValue(true)

      const wrapper = mountComponent()
      const vm = wrapper.vm as ComponentInstance
      vm.loading = false
      vm.events = mockEventsResponse.events
      await nextTick()

      expect(mockCustomerEventsService.hasAdditionalInfo).toHaveBeenCalled()
    })
  })

  describe('pagination', () => {
    it('handles page change correctly', async () => {
      const wrapper = mountComponent()
      const vm = wrapper.vm as ComponentInstance
      vm.loading = false
      vm.events = mockEventsResponse.events
      await nextTick()

      // Simulate page change
      const dataTable = wrapper.findComponent(DataTable)
      await dataTable.vm.$emit('page', { page: 1 })

      expect(vm.pagination.page).toBe(1) // page + 1
      expect(mockCustomerEventsService.getMyEvents).toHaveBeenCalledWith({
        page: 2,
        limit: 7
      })
    })

    it('calculates dataTableFirst correctly', async () => {
      const wrapper = mountComponent()
      const vm = wrapper.vm as ComponentInstance
      vm.pagination = { page: 2, limit: 7, total: 20, totalPages: 3 }
      await nextTick()

      expect(vm.dataTableFirst).toBe(7) // (2-1) * 7
    })
  })

  describe('tooltip functionality', () => {
    it('generates tooltip content map correctly', async () => {
      mockCustomerEventsService.hasAdditionalInfo.mockReturnValue(true)
      mockCustomerEventsService.getTooltipContent.mockReturnValue(
        '<strong>Test:</strong> value'
      )

      const wrapper = mountComponent()
      const vm = wrapper.vm as ComponentInstance

      vm.loading = false
      vm.events = mockEventsResponse.events
      await nextTick()

      const tooltipMap = vm.tooltipContentMap
      expect(tooltipMap.get('event-1')).toBe('<strong>Test:</strong> value')
    })

    it('excludes events without additional info from tooltip map', async () => {
      mockCustomerEventsService.hasAdditionalInfo.mockReturnValue(false)

      const wrapper = mountComponent()
      const vm = wrapper.vm as ComponentInstance

      vm.loading = false
      vm.events = mockEventsResponse.events
      await nextTick()

      const tooltipMap = vm.tooltipContentMap
      expect(tooltipMap.size).toBe(0)
    })
  })

  describe('component methods', () => {
    it('exposes refresh method', () => {
      const wrapper = mountComponent()

      expect(typeof wrapper.vm.refresh).toBe('function')
    })

    it('resets to first page on refresh', async () => {
      const wrapper = mountComponent()
      const vm = wrapper.vm as ComponentInstance

      vm.pagination.page = 3

      await vm.refresh()

      expect(vm.pagination.page).toBe(1)
      expect(mockCustomerEventsService.getMyEvents).toHaveBeenCalledWith({
        page: 1,
        limit: 7
      })
    })
  })

  describe('component lifecycle', () => {
    it('initializes with correct default values', () => {
      const wrapper = mountComponent()

      const vm = wrapper.vm as ComponentInstance

      expect(vm.events).toEqual([])
      expect(vm.loading).toBe(true)
      expect(vm.error).toBeNull()
      expect(vm.pagination).toEqual({
        page: 1,
        limit: 7,
        total: 0,
        totalPages: 0
      })
    })
  })

  describe('EventType integration', () => {
    it('uses EventType enum in template conditions', async () => {
      const wrapper = mountComponent()
      const vm = wrapper.vm as ComponentInstance

      vm.loading = false
      vm.events = [
        {
          event_id: 'event-1',
          event_type: EventType.CREDIT_ADDED,
          params: { amount: 1000 },
          createdAt: '2024-01-01T10:00:00Z'
        }
      ]
      await nextTick()

      // Verify that the component can access EventType enum
      expect(EventType.CREDIT_ADDED).toBe('credit_added')
      expect(EventType.ACCOUNT_CREATED).toBe('account_created')
      expect(EventType.API_USAGE_COMPLETED).toBe('api_usage_completed')
    })
  })
})
