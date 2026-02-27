const ACRONYM_TAGS = new Set(['VAE', 'CLIP', 'GLIGEN'])

export function formatCategoryLabel(raw?: string): string {
  if (!raw) return 'Models'

  // Special display name mappings
  if (raw === 'diffusion_models') return 'Diffusion'

  return raw
    .split('_')
    .map((segment) => {
      const upper = segment.toUpperCase()
      if (ACRONYM_TAGS.has(upper)) return upper

      const lower = segment.toLowerCase()
      return lower.charAt(0).toUpperCase() + lower.slice(1)
    })
    .join(' ')
}
