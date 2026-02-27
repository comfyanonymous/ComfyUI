import { t } from '@/i18n'
import { useExtensionService } from '@/services/extensionService'
import type { TopbarBadge } from '@/types/comfy'

const badges: TopbarBadge[] = [
  {
    text: t('nightly.badge.label'),
    label: t('g.nightly'),
    variant: 'warning',
    tooltip: t('nightly.badge.tooltip')
  }
]

useExtensionService().registerExtension({
  name: 'Comfy.Nightly.Badges',
  topbarBadges: badges
})
