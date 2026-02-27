import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import type { FeatureSurveyConfig } from './useSurveyEligibility'

import {
  FEATURE_SURVEYS,
  getEnabledSurveys,
  getSurveyConfig
} from './surveyRegistry'

const TEST_FEATURE_ID = '__test-feature__'
const TEST_CONFIG: FeatureSurveyConfig = {
  featureId: TEST_FEATURE_ID,
  typeformId: 'test-form-123',
  triggerThreshold: 5,
  delayMs: 3000,
  enabled: true
}

describe('surveyRegistry', () => {
  let originalEntries: Record<string, FeatureSurveyConfig>

  beforeEach(() => {
    originalEntries = { ...FEATURE_SURVEYS }
    FEATURE_SURVEYS[TEST_FEATURE_ID] = TEST_CONFIG
  })

  afterEach(() => {
    for (const key of Object.keys(FEATURE_SURVEYS)) {
      delete FEATURE_SURVEYS[key]
    }
    Object.assign(FEATURE_SURVEYS, originalEntries)
  })

  describe('getSurveyConfig', () => {
    it('returns undefined for unknown feature', () => {
      expect(getSurveyConfig('nonexistent-feature')).toBeUndefined()
    })

    it('returns config for registered feature', () => {
      const config = getSurveyConfig(TEST_FEATURE_ID)
      expect(config).toEqual(TEST_CONFIG)
    })
  })

  describe('getEnabledSurveys', () => {
    it('includes surveys with enabled: true', () => {
      const enabled = getEnabledSurveys()
      expect(enabled).toContainEqual(TEST_CONFIG)
    })

    it('includes surveys where enabled is undefined', () => {
      const implicitlyEnabled: FeatureSurveyConfig = {
        featureId: '__implicit__',
        typeformId: 'form-456'
      }
      FEATURE_SURVEYS['__implicit__'] = implicitlyEnabled

      const enabled = getEnabledSurveys()
      expect(enabled).toContainEqual(implicitlyEnabled)
    })

    it('excludes surveys with enabled: false', () => {
      const disabledConfig: FeatureSurveyConfig = {
        featureId: '__disabled__',
        typeformId: 'form-789',
        enabled: false
      }
      FEATURE_SURVEYS['__disabled__'] = disabledConfig

      const enabled = getEnabledSurveys()
      expect(enabled).not.toContainEqual(disabledConfig)
    })
  })
})
