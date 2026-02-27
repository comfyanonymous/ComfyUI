import type { TelemetryEventName } from '@/platform/telemetry/types'

/**
 * Server health alert configuration from the backend
 */
type ServerHealthAlert = {
  message: string
  tooltip?: string
  severity?: 'info' | 'warning' | 'error'
  badge?: string
}

type FirebaseRuntimeConfig = {
  apiKey: string
  authDomain: string
  databaseURL?: string
  projectId: string
  storageBucket: string
  messagingSenderId: string
  appId: string
  measurementId?: string
}

/**
 * Remote configuration type
 * Configuration fetched from the server at runtime
 */
export type RemoteConfig = {
  gtm_container_id?: string
  ga_measurement_id?: string
  mixpanel_token?: string
  subscription_required?: boolean
  server_health_alert?: ServerHealthAlert
  max_upload_size?: number
  comfy_api_base_url?: string
  comfy_platform_base_url?: string
  firebase_config?: FirebaseRuntimeConfig
  telemetry_disabled_events?: TelemetryEventName[]
  model_upload_button_enabled?: boolean
  asset_rename_enabled?: boolean
  private_models_enabled?: boolean
  onboarding_survey_enabled?: boolean
  linear_toggle_enabled?: boolean
  team_workspaces_enabled?: boolean
  user_secrets_enabled?: boolean
  node_library_essentials_enabled?: boolean
  free_tier_credits?: number
  new_free_tier_subscriptions?: boolean
}
