<template>
  <div class="credits-container h-full">
    <!-- Legacy Design -->
    <div class="flex h-full flex-col">
      <h2 class="mb-2 text-2xl font-bold">
        {{ $t('credits.credits') }}
      </h2>

      <Divider />

      <div class="flex flex-col gap-2">
        <h3 class="text-sm font-medium text-muted">
          {{ $t('credits.yourCreditBalance') }}
        </h3>
        <div class="flex items-center justify-between">
          <UserCredit text-class="text-3xl font-bold" />
          <Skeleton v-if="loading" width="2rem" height="2rem" />
          <Button
            v-else-if="isActiveSubscription"
            :loading="loading"
            @click="handlePurchaseCreditsClick"
          >
            {{ $t('credits.purchaseCredits') }}
          </Button>
        </div>
        <div class="flex flex-row items-center">
          <Skeleton
            v-if="balanceLoading"
            width="12rem"
            height="1rem"
            class="text-xs"
          />
          <div v-else-if="formattedLastUpdateTime" class="text-xs text-muted">
            {{ $t('credits.lastUpdated') }}: {{ formattedLastUpdateTime }}
          </div>
          <Button
            variant="muted-textonly"
            size="icon-sm"
            :aria-label="$t('g.refresh')"
            @click="() => authActions.fetchBalance()"
          >
            <i class="pi pi-refresh" />
          </Button>
        </div>
      </div>

      <div class="flex items-center justify-between">
        <h3>{{ $t('credits.activity') }}</h3>
        <Button
          variant="muted-textonly"
          :loading="loading"
          @click="handleCreditsHistoryClick"
        >
          <i class="pi pi-arrow-up-right" />
          {{ $t('credits.invoiceHistory') }}
        </Button>
      </div>

      <template v-if="creditHistory.length > 0">
        <div class="grow">
          <DataTable :value="creditHistory" :show-headers="false">
            <Column field="title" :header="$t('g.name')">
              <template #body="{ data }">
                <div class="text-sm font-medium">{{ data.title }}</div>
                <div class="text-xs text-muted">{{ data.timestamp }}</div>
              </template>
            </Column>
            <Column field="amount" :header="$t('g.amount')">
              <template #body="{ data }">
                <div
                  :class="[
                    'text-center text-base font-medium',
                    data.isPositive ? 'text-sky-500' : 'text-red-400'
                  ]"
                >
                  {{ data.isPositive ? '+' : '-' }}${{
                    formatMetronomeCurrency(data.amount, 'usd')
                  }}
                </div>
              </template>
            </Column>
          </DataTable>
        </div>
      </template>

      <Divider />

      <UsageLogsTable ref="usageLogsTableRef" />

      <div class="flex flex-row gap-2">
        <Button variant="muted-textonly" @click="handleFaqClick">
          <i class="pi pi-question-circle" />
          {{ $t('credits.faqs') }}
        </Button>
        <Button variant="muted-textonly" @click="handleOpenPartnerNodesInfo">
          <i class="pi pi-question-circle" />
          {{ $t('subscription.partnerNodesCredits') }}
        </Button>
        <Button variant="muted-textonly" @click="handleMessageSupport">
          <i class="pi pi-comments" />
          {{ $t('credits.messageSupport') }}
        </Button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import Column from 'primevue/column'
import DataTable from 'primevue/datatable'
import Divider from 'primevue/divider'
import Skeleton from 'primevue/skeleton'
import { computed, ref, watch } from 'vue'

import UserCredit from '@/components/common/UserCredit.vue'
import UsageLogsTable from '@/components/dialog/content/setting/UsageLogsTable.vue'
import Button from '@/components/ui/button/Button.vue'
import { useBillingContext } from '@/composables/billing/useBillingContext'
import { useFirebaseAuthActions } from '@/composables/auth/useFirebaseAuthActions'
import { useExternalLink } from '@/composables/useExternalLink'
import { useTelemetry } from '@/platform/telemetry'
import { useDialogService } from '@/services/dialogService'
import { useCommandStore } from '@/stores/commandStore'
import { useFirebaseAuthStore } from '@/stores/firebaseAuthStore'
import { formatMetronomeCurrency } from '@/utils/formatUtil'

interface CreditHistoryItemData {
  title: string
  timestamp: string
  amount: number
  isPositive: boolean
}

const { buildDocsUrl, docsPaths } = useExternalLink()
const dialogService = useDialogService()
const authStore = useFirebaseAuthStore()
const authActions = useFirebaseAuthActions()
const commandStore = useCommandStore()
const telemetry = useTelemetry()
const { isActiveSubscription } = useBillingContext()
const loading = computed(() => authStore.loading)
const balanceLoading = computed(() => authStore.isFetchingBalance)

const usageLogsTableRef = ref<InstanceType<typeof UsageLogsTable> | null>(null)

const formattedLastUpdateTime = computed(() =>
  authStore.lastBalanceUpdateTime
    ? authStore.lastBalanceUpdateTime.toLocaleString()
    : ''
)

watch(
  () => authStore.lastBalanceUpdateTime,
  (newTime, oldTime) => {
    if (newTime && newTime !== oldTime && usageLogsTableRef.value) {
      usageLogsTableRef.value.refresh()
    }
  }
)

const handlePurchaseCreditsClick = () => {
  // Track purchase credits entry from Settings > Credits panel
  useTelemetry()?.trackAddApiCreditButtonClicked()
  dialogService.showTopUpCreditsDialog()
}

const handleCreditsHistoryClick = async () => {
  await authActions.accessBillingPortal()
}

const handleMessageSupport = async () => {
  telemetry?.trackHelpResourceClicked({
    resource_type: 'help_feedback',
    is_external: true,
    source: 'credits_panel'
  })
  await commandStore.execute('Comfy.ContactSupport')
}

const handleFaqClick = () => {
  window.open(
    buildDocsUrl('/tutorials/api-nodes/faq', { includeLocale: true }),
    '_blank'
  )
}

const handleOpenPartnerNodesInfo = () => {
  window.open(
    buildDocsUrl(docsPaths.partnerNodesPricing, { includeLocale: true }),
    '_blank'
  )
}

const creditHistory = ref<CreditHistoryItemData[]>([])
</script>
