import type { TierKey } from '@/platform/cloud/subscription/constants/tierPricing'

export type BillingCycle = 'monthly' | 'yearly'

type RankedTierKey = Exclude<TierKey, 'founder' | 'free'>
type RankedPlanKey = `${BillingCycle}-${RankedTierKey}`

interface PlanDescriptor {
  tierKey: TierKey
  billingCycle: BillingCycle
}

const PLAN_ORDER: RankedPlanKey[] = [
  'yearly-pro',
  'yearly-creator',
  'yearly-standard',
  'monthly-pro',
  'monthly-creator',
  'monthly-standard'
]

const PLAN_RANK = PLAN_ORDER.reduce<Map<RankedPlanKey, number>>(
  (acc, plan, index) => acc.set(plan, index),
  new Map()
)

const toRankedPlanKey = (
  tierKey: TierKey,
  billingCycle: BillingCycle
): RankedPlanKey | null => {
  if (tierKey === 'founder' || tierKey === 'free') return null
  return `${billingCycle}-${tierKey}` as RankedPlanKey
}

export const getPlanRank = ({
  tierKey,
  billingCycle
}: PlanDescriptor): number => {
  const planKey = toRankedPlanKey(tierKey, billingCycle)
  if (!planKey) return Number.POSITIVE_INFINITY

  return PLAN_RANK.get(planKey) ?? Number.POSITIVE_INFINITY
}

interface DowngradeCheckParams {
  current: PlanDescriptor
  target: PlanDescriptor
}

export const isPlanDowngrade = ({
  current,
  target
}: DowngradeCheckParams): boolean => {
  const currentRank = getPlanRank(current)
  const targetRank = getPlanRank(target)

  return targetRank > currentRank
}
