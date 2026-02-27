import type { SecretProvider } from './types'

interface ProviderConfig {
  value: SecretProvider
  label: string
  logo: string
}

export const SECRET_PROVIDERS: ProviderConfig[] = [
  {
    value: 'huggingface',
    label: 'HuggingFace',
    logo: '/assets/images/hf-logo.svg'
  },
  { value: 'civitai', label: 'Civitai', logo: '/assets/images/civitai.svg' }
] as const

export function getProviderLabel(provider: SecretProvider | undefined): string {
  if (!provider) return ''
  const config = SECRET_PROVIDERS.find((p) => p.value === provider)
  return config?.label ?? provider
}

export function getProviderLogo(
  provider: SecretProvider | undefined
): string | undefined {
  if (!provider) return undefined
  const config = SECRET_PROVIDERS.find((p) => p.value === provider)
  return config?.logo
}
