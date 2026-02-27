import { t } from '@/i18n'
import { getDistribution, ZENDESK_FIELDS } from '@/platform/support/config'
import { useExtensionService } from '@/services/extensionService'
import type { ActionBarButton } from '@/types/comfy'

const ZENDESK_BASE_URL = 'https://support.comfy.org/hc/en-us/requests/new'
const ZENDESK_FEEDBACK_FORM_ID = '43066738713236'

const distribution = getDistribution()
const params = new URLSearchParams({
  ticket_form_id: ZENDESK_FEEDBACK_FORM_ID,
  [ZENDESK_FIELDS.DISTRIBUTION]: distribution
})
const feedbackUrl = `${ZENDESK_BASE_URL}?${params.toString()}`

const buttons: ActionBarButton[] = [
  {
    icon: 'icon-[lucide--message-circle-question-mark]',
    label: t('actionbar.feedback'),
    tooltip: t('actionbar.feedbackTooltip'),
    onClick: () => {
      window.open(feedbackUrl, '_blank', 'noopener,noreferrer')
    }
  }
]

useExtensionService().registerExtension({
  name: 'Comfy.Cloud.FeedbackButton',
  actionBarButtons: buttons
})
