import type { FeatureSurveyConfig } from './useSurveyEligibility'

/**
 * Registry of all feature surveys.
 * Add new surveys here when targeting specific features for feedback.
 */
export const FEATURE_SURVEYS: Record<string, FeatureSurveyConfig> = {}

export function getSurveyConfig(
  featureId: string
): FeatureSurveyConfig | undefined {
  return FEATURE_SURVEYS[featureId]
}

export function getEnabledSurveys(): FeatureSurveyConfig[] {
  return Object.values(FEATURE_SURVEYS).filter(
    (config) => config.enabled !== false
  )
}
