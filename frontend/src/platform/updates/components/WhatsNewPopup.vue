<template>
  <div v-if="shouldShow" class="whats-new-popup-container left-4">
    <div class="whats-new-popup" @click.stop>
      <!-- Close Button -->
      <Button
        class="close-button absolute top-2 right-2 z-10 w-8 h-8 p-2 rounded-lg opacity-50"
        :aria-label="$t('g.close')"
        size="icon-sm"
        variant="muted-textonly"
        @click="closePopup"
      >
        <i class="icon-[lucide--x]" />
      </Button>

      <!-- Modal Body -->
      <div class="modal-body flex flex-col gap-4 px-0 pt-0 pb-2 flex-1">
        <!-- Release Content -->
        <div
          class="content-text max-h-96 overflow-y-auto"
          v-html="formattedContent"
        ></div>
      </div>

      <!-- Modal Footer -->
      <div
        class="modal-footer flex justify-between items-center gap-4 px-4 pb-4"
      >
        <a
          class="learn-more-link flex items-center gap-2 text-sm font-normal py-1"
          :href="changelogUrl"
          target="_blank"
          rel="noopener noreferrer"
          @click="closePopup"
        >
          <i class="icon-[lucide--external-link]"></i>
          {{ $t('whatsNewPopup.learnMore') }}
        </a>
        <div class="footer-actions flex items-center gap-4">
          <Button
            class="h-8"
            size="sm"
            variant="muted-textonly"
            @click="closePopup"
          >
            {{ $t('whatsNewPopup.later') }}
          </Button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { default as DOMPurify } from 'dompurify'
import { computed, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'

import Button from '@/components/ui/button/Button.vue'
import { useExternalLink } from '@/composables/useExternalLink'
import { formatVersionAnchor } from '@/utils/formatUtil'
import { renderMarkdownToHtml } from '@/utils/markdownRendererUtil'

import type { ReleaseNote } from '../common/releaseService'
import { useReleaseStore } from '../common/releaseStore'

const { buildDocsUrl } = useExternalLink()
const releaseStore = useReleaseStore()
const { t } = useI18n()

// Define emits
const emit = defineEmits<{
  'whats-new-dismissed': []
}>()

// Local state for dismissed status
const isDismissed = ref(false)

// Get latest release from store
const latestRelease = computed<ReleaseNote | null>(() => {
  return releaseStore.recentRelease
})

// Show popup when on latest version and not dismissed
const shouldShow = computed(
  () => releaseStore.shouldShowPopup && !isDismissed.value
)

// Generate changelog URL with version anchor (language-aware)
const changelogUrl = computed(() => {
  const changelogBaseUrl = buildDocsUrl('/changelog', { includeLocale: true })
  if (latestRelease.value?.version) {
    const versionAnchor = formatVersionAnchor(latestRelease.value.version)
    return `${changelogBaseUrl}#${versionAnchor}`
  }
  return changelogBaseUrl
})

const formattedContent = computed(() => {
  if (!latestRelease.value?.content) {
    return DOMPurify.sanitize(`<p>${t('whatsNewPopup.noReleaseNotes')}</p>`)
  }

  try {
    const markdown = latestRelease.value.content

    // Check if content is meaningful (not just whitespace)
    const trimmedContent = markdown.trim()
    if (!trimmedContent || trimmedContent.replace(/\s+/g, '') === '') {
      return DOMPurify.sanitize(`<p>${t('whatsNewPopup.noReleaseNotes')}</p>`)
    }

    // Extract image and remaining content separately
    const imageMatch = markdown.match(/!\[.*?\]\(.*?\)/)
    const image = imageMatch ? imageMatch[0] : ''

    // Remove image from content but keep original title
    const contentWithoutImage = markdown.replace(/!\[.*?\]\(.*?\)/, '').trim()

    // Reorder: image first, then original content
    const reorderedContent = [image, contentWithoutImage]
      .filter(Boolean)
      .join('\n\n')

    // renderMarkdownToHtml already sanitizes with DOMPurify, so this is safe
    return renderMarkdownToHtml(reorderedContent)
  } catch (error) {
    console.error('Error parsing markdown:', error)
    // Fallback to plain text with line breaks - sanitize the HTML we create
    const fallbackContent = latestRelease.value.content.replace(/\n/g, '<br>')
    return fallbackContent.trim()
      ? DOMPurify.sanitize(fallbackContent)
      : DOMPurify.sanitize(`<p>${t('whatsNewPopup.noReleaseNotes')}</p>`)
  }
})

const show = () => {
  isDismissed.value = false
}

const hide = () => {
  isDismissed.value = true
  emit('whats-new-dismissed')
}

const closePopup = async () => {
  // Mark "what's new" seen when popup is closed
  if (latestRelease.value) {
    await releaseStore.handleWhatsNewSeen(latestRelease.value.version)
  }
  hide()
}

// Initialize on mount
onMounted(async () => {
  // Fetch releases if not already loaded
  if (!releaseStore.releases.length) {
    await releaseStore.fetchReleases()
  }
})

// Expose methods for parent component and tests
defineExpose({
  show,
  hide,
  closePopup
})
</script>

<style scoped>
/* Popup container - positioning handled by parent */
.whats-new-popup-container {
  --whats-new-popup-bottom: 1rem;

  position: absolute;
  bottom: var(--whats-new-popup-bottom);
  z-index: 1000;
  pointer-events: auto;
}

.whats-new-popup {
  background: var(--interface-menu-surface);
  border-radius: 8px;
  max-width: 400px;
  width: 400px;
  border: 1px solid var(--interface-menu-stroke);
  box-shadow: 1px 1px 8px 0 rgb(0 0 0 / 0.2);
  position: relative;
  display: flex;
  flex-direction: column;
}

/* Modal Body */
.modal-body {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  padding: 0;
  flex: 1;
}

.modal-header {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.content-text {
  color: var(--text-primary);
  font-size: 14px;
  line-height: 1.5;
  word-wrap: break-word;
  padding: 0 1rem;
}

/* Style the markdown content */
/* Title */
.content-text :deep(*) {
  box-sizing: border-box;
}

.content-text :deep(h1) {
  color: var(--text-secondary);
  font-family: Inter, sans-serif;
  font-size: 14px;
  font-weight: 400;
  margin-top: 1rem;
  margin-bottom: 8px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

/* What's new title - targets h2 or strong text after h1 */
.content-text :deep(h2),
.content-text :deep(h1 + p strong) {
  color: var(--text-primary);
  font-family: Inter, sans-serif;
  font-size: 14px;
  font-weight: 600;
  margin: 0 0 8px;
  line-height: 1.429;
}

/* Regular paragraphs - short description */
.content-text :deep(p) {
  color: var(--text-secondary);
  font-family: Inter, sans-serif;
  margin: 1rem 0;
}

/* List */
.content-text :deep(ul),
.content-text :deep(ol) {
  margin-bottom: 0;
  padding-left: 0;
  list-style: none;
}

.content-text :deep(ul:first-child),
.content-text :deep(ol:first-child) {
  margin-top: 0;
}

.content-text :deep(ul:last-child),
.content-text :deep(ol:last-child) {
  margin-bottom: 0;
}

.content-text :deep(li) {
  margin-bottom: 6px;
  position: relative;
  padding-left: 18px;
  color: var(--text-secondary);
  font-family: Inter, sans-serif;
  font-size: 14px;
  font-weight: 400;
  line-height: 1.2102;
}

.content-text :deep(li:last-child) {
  margin-bottom: 0;
}

.content-text :deep(li::before) {
  content: '';
  position: absolute;
  left: 4px;
  top: 7px;
  width: 6px;
  height: 6px;
  border: 2px solid var(--text-secondary);
  border-radius: 50%;
  background: transparent;
}

.content-text :deep(li strong) {
  color: var(--text-secondary);
  font-family: Inter, sans-serif;
  font-size: 14px;
  font-weight: 400;
  line-height: 1.2102;
  margin-right: 4px;
}

.content-text :deep(li p) {
  margin: 2px 0 0;
  display: inline;
}

/* Code styling */
.content-text :deep(code) {
  background-color: var(--input-surface);
  border: 1px solid var(--interface-menu-stroke);
  border-radius: 4px;
  padding: 2px 6px;
  color: var(--text-primary);
  white-space: nowrap;
}

.content-text :deep(img) {
  width: 100%;
  height: 200px;
  margin: 0 0 16px;
  object-fit: cover;
  display: block;
  border-radius: 8px;
}

.content-text :deep(img:first-child) {
  margin: -1rem -1rem 16px;
  width: calc(100% + 2rem);
  border-top-left-radius: 8px;
  border-top-right-radius: 8px;
  border-bottom-left-radius: 0;
  border-bottom-right-radius: 0;
}

/* Add border to content when image is present */
.content-text:has(img:first-child) {
  border-left: 1px solid var(--interface-menu-stroke);
  border-right: 1px solid var(--interface-menu-stroke);
  border-top: 1px solid var(--interface-menu-stroke);
  border-bottom-left-radius: 8px;
  border-bottom-right-radius: 8px;
  margin: -1px;
  margin-bottom: 0;
}

.content-text :deep(img + h1) {
  margin-top: 0;
}

/* Secondary headings */
.content-text :deep(h3) {
  color: var(--text-primary);
  font-family: Inter, sans-serif;
  font-size: 16px;
  font-weight: 600;
  margin: 16px 0 8px;
  line-height: 1.4;
}

/* Modal Footer */
.modal-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
  padding: 16px;
  border-top: none;
}

.footer-actions {
  display: flex;
  align-items: center;
  gap: 16px;
}

.learn-more-link {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--text-secondary);
  font-size: 14px;
  font-weight: 400;
  line-height: 1.2102;
  text-decoration: none;
  padding: 4px 0;
}

.learn-more-link:hover {
  color: var(--text-primary);
}

.learn-more-link i {
  width: 16px;
  height: 16px;
}

.action-secondary {
  height: 32px;
  padding: 4px 0;
  background: transparent;
  border: none;
  color: var(--text-secondary);
  font-size: 14px;
  font-weight: 400;
  line-height: 1.2102;
  cursor: pointer;
  border-radius: 4px;
}

.action-secondary:hover {
  color: var(--text-primary);
}

.action-primary {
  height: 40px;
  padding: 8px 16px;
  background: var(--interface-menu-component-surface-hovered);
  border-radius: 8px;
  border: none;
  color: var(--text-primary);
  font-size: 14px;
  font-weight: 400;
  line-height: 1.2102;
  cursor: pointer;
}

.action-primary:hover {
  background: var(--button-hover-surface);
}
</style>
