<template>
  <div>
    <div v-if="loading" class="flex items-center justify-center p-8">
      <ProgressSpinner />
    </div>
    <div v-else-if="error" class="p-4">
      <Message severity="error" :closable="false">{{ error }}</Message>
    </div>
    <DataTable
      v-else
      :value="events"
      :paginator="true"
      :rows="pagination.limit"
      :total-records="pagination.total"
      :first="dataTableFirst"
      :lazy="true"
      class="p-datatable-sm custom-datatable"
      @page="onPageChange"
    >
      <Column field="event_type" :header="$t('credits.eventType')">
        <template #body="{ data }">
          <Badge
            :value="customerEventService.formatEventType(data.event_type)"
            :severity="customerEventService.getEventSeverity(data.event_type)"
          />
        </template>
      </Column>
      <Column field="details" :header="$t('credits.details')">
        <template #body="{ data }">
          <div class="event-details">
            <!-- Credits Added -->
            <template v-if="data.event_type === EventType.CREDIT_ADDED">
              <div class="font-semibold text-green-500">
                {{ $t('credits.added') }} ${{
                  customerEventService.formatAmount(data.params?.amount)
                }}
              </div>
            </template>

            <!-- Account Created -->
            <template v-else-if="data.event_type === EventType.ACCOUNT_CREATED">
              <div>{{ $t('credits.accountInitialized') }}</div>
            </template>

            <!-- API Usage -->
            <template
              v-else-if="data.event_type === EventType.API_USAGE_COMPLETED"
            >
              <div class="flex flex-col gap-1">
                <div class="font-semibold">
                  {{ data.params?.api_name || 'API' }}
                </div>
                <div class="text-sm text-smoke-400">
                  {{ $t('credits.model') }}: {{ data.params?.model || '-' }}
                </div>
              </div>
            </template>
          </div>
        </template>
      </Column>
      <Column field="createdAt" :header="$t('credits.time')">
        <template #body="{ data }">
          {{ customerEventService.formatDate(data.createdAt) }}
        </template>
      </Column>
      <Column field="params" :header="$t('credits.additionalInfo')">
        <template #body="{ data }">
          <Button
            v-if="customerEventService.hasAdditionalInfo(data)"
            v-tooltip.top="{
              escape: false,
              value: tooltipContentMap.get(data.event_id) || '',
              pt: {
                text: {
                  style: {
                    width: 'max-content !important'
                  }
                }
              }
            }"
            variant="textonly"
            size="icon-sm"
            :aria-label="$t('credits.additionalInfo')"
          >
            <i class="pi pi-info-circle" />
          </Button>
        </template>
      </Column>
    </DataTable>
  </div>
</template>

<script setup lang="ts">
import Badge from 'primevue/badge'
import Column from 'primevue/column'
import DataTable from 'primevue/datatable'
import Message from 'primevue/message'
import ProgressSpinner from 'primevue/progressspinner'
import { computed, ref } from 'vue'

import Button from '@/components/ui/button/Button.vue'
import { useTelemetry } from '@/platform/telemetry'
import type { AuditLog } from '@/services/customerEventsService'
import {
  EventType,
  useCustomerEventsService
} from '@/services/customerEventsService'

const events = ref<AuditLog[]>([])
const loading = ref(true)
const error = ref<string | null>(null)

const customerEventService = useCustomerEventsService()

const pagination = ref({
  page: 1,
  limit: 7,
  total: 0,
  totalPages: 0
})

const dataTableFirst = computed(
  () => (pagination.value.page - 1) * pagination.value.limit
)

const tooltipContentMap = computed(() => {
  const map = new Map<string, string>()
  events.value.forEach((event) => {
    if (customerEventService.hasAdditionalInfo(event) && event.event_id) {
      map.set(event.event_id, customerEventService.getTooltipContent(event))
    }
  })
  return map
})

const loadEvents = async () => {
  loading.value = true
  error.value = null

  try {
    const response = await customerEventService.getMyEvents({
      page: pagination.value.page,
      limit: pagination.value.limit
    })

    if (response) {
      if (response.events) {
        events.value = response.events
      }

      if (response.page) {
        pagination.value.page = response.page
      }

      if (response.limit) {
        pagination.value.limit = response.limit
      }

      if (response.total) {
        pagination.value.total = response.total
      }

      if (response.totalPages) {
        pagination.value.totalPages = response.totalPages
      }

      // Check if a pending top-up has completed
      useTelemetry()?.checkForCompletedTopup(response.events)
    } else {
      error.value = customerEventService.error.value || 'Failed to load events'
    }
  } catch (err) {
    error.value = err instanceof Error ? err.message : 'Unknown error'
    console.error('Error loading events:', err)
  } finally {
    loading.value = false
  }
}

const onPageChange = (event: { page: number }) => {
  pagination.value.page = event.page + 1
  loadEvents().catch((error) => {
    console.error('Error loading events:', error)
  })
}

const refresh = async () => {
  pagination.value.page = 1
  await loadEvents()
}

defineExpose({
  refresh
})
</script>
