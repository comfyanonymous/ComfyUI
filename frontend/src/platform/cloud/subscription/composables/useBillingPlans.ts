import { computed, ref } from 'vue'

import type { Plan } from '@/platform/workspace/api/workspaceApi'
import { workspaceApi } from '@/platform/workspace/api/workspaceApi'

const plans = ref<Plan[]>([])
const currentPlanSlug = ref<string | null>(null)
const isLoading = ref(false)
const error = ref<string | null>(null)

export function useBillingPlans() {
  async function fetchPlans() {
    if (isLoading.value) return

    isLoading.value = true
    error.value = null

    try {
      const response = await workspaceApi.getBillingPlans()
      plans.value = response.plans
      currentPlanSlug.value = response.current_plan_slug ?? null
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Failed to fetch plans'
      console.error('[useBillingPlans] Failed to fetch plans:', err)
    } finally {
      isLoading.value = false
    }
  }

  const monthlyPlans = computed(() =>
    plans.value.filter((p) => p.duration === 'MONTHLY')
  )

  const annualPlans = computed(() =>
    plans.value.filter((p) => p.duration === 'ANNUAL')
  )

  function getPlanBySlug(slug: string) {
    return plans.value.find((p) => p.slug === slug)
  }

  function getPlansForTier(tier: Plan['tier']) {
    return plans.value.filter((p) => p.tier === tier)
  }

  const isCurrentPlan = (slug: string) => currentPlanSlug.value === slug

  return {
    plans,
    currentPlanSlug,
    isLoading,
    error,
    monthlyPlans,
    annualPlans,
    fetchPlans,
    getPlanBySlug,
    getPlansForTier,
    isCurrentPlan
  }
}
