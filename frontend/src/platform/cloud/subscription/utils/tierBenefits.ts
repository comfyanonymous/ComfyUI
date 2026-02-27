import {
  getTierCredits,
  getTierFeatures
} from '@/platform/cloud/subscription/constants/tierPricing'
import type { TierKey } from '@/platform/cloud/subscription/constants/tierPricing'

type BenefitType = 'metric' | 'feature' | 'icon'

export interface TierBenefit {
  key: string
  type: BenefitType
  label: string
  value?: string
  icon?: string
}

export function getCommonTierBenefits(
  key: TierKey,
  t: (key: string, params?: Record<string, unknown>) => string,
  n: (value: number) => string
): TierBenefit[] {
  const benefits: TierBenefit[] = []
  const isFree = key === 'free'

  if (isFree) {
    const credits = getTierCredits(key)
    if (credits !== null) {
      benefits.push({
        key: 'monthlyCredits',
        type: 'metric',
        value: n(credits),
        label: t('subscription.monthlyCreditsLabel')
      })
    }
  }

  benefits.push({
    key: 'maxDuration',
    type: 'metric',
    value: t(`subscription.maxDuration.${key}`),
    label: t('subscription.maxDurationLabel')
  })

  benefits.push({
    key: 'gpu',
    type: 'feature',
    label: t('subscription.gpuLabel')
  })

  if (!isFree) {
    benefits.push({
      key: 'addCredits',
      type: 'feature',
      label: t('subscription.addCreditsLabel')
    })
  }

  if (getTierFeatures(key).customLoRAs) {
    benefits.push({
      key: 'customLoRAs',
      type: 'feature',
      label: t('subscription.customLoRAsLabel')
    })
  }

  return benefits
}
