import { describe, expect, it } from 'vitest'

import { getPlanRank, isPlanDowngrade } from './subscriptionTierRank'

describe('subscriptionTierRank', () => {
  it('returns consistent order for ranked plans', () => {
    const yearlyPro = getPlanRank({ tierKey: 'pro', billingCycle: 'yearly' })
    const monthlyStandard = getPlanRank({
      tierKey: 'standard',
      billingCycle: 'monthly'
    })

    expect(yearlyPro).toBeLessThan(monthlyStandard)
  })

  it('identifies downgrades correctly', () => {
    const result = isPlanDowngrade({
      current: { tierKey: 'pro', billingCycle: 'yearly' },
      target: { tierKey: 'creator', billingCycle: 'monthly' }
    })

    expect(result).toBe(true)
  })

  it('treats lateral or upgrade moves as non-downgrades', () => {
    expect(
      isPlanDowngrade({
        current: { tierKey: 'standard', billingCycle: 'monthly' },
        target: { tierKey: 'creator', billingCycle: 'monthly' }
      })
    ).toBe(false)

    expect(
      isPlanDowngrade({
        current: { tierKey: 'creator', billingCycle: 'monthly' },
        target: { tierKey: 'creator', billingCycle: 'monthly' }
      })
    ).toBe(false)
  })

  it('treats unknown plans (e.g., founder) as non-downgrade cases', () => {
    const result = isPlanDowngrade({
      current: { tierKey: 'founder', billingCycle: 'monthly' },
      target: { tierKey: 'standard', billingCycle: 'monthly' }
    })

    expect(result).toBe(false)
  })
})
