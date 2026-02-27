<template>
  <div
    class="help-center-menu flex flex-col items-start gap-1"
    role="menu"
    :aria-label="$t('help.helpCenterMenu')"
  >
    <!-- Main Menu Items -->
    <div class="w-full">
      <nav class="flex w-full flex-col gap-2" role="menubar">
        <button
          v-for="menuItem in menuItems"
          v-show="menuItem.visible !== false"
          :key="menuItem.key"
          type="button"
          class="help-menu-item"
          :class="{ 'more-item': menuItem.key === 'more' }"
          role="menuitem"
          @click="menuItem.action"
          @mouseenter="onMenuItemHover(menuItem.key, $event)"
          @mouseleave="onMenuItemLeave(menuItem.key)"
        >
          <div class="help-menu-icon-container">
            <div class="help-menu-icon">
              <component
                :is="menuItem.icon"
                v-if="typeof menuItem.icon === 'object'"
                :size="16"
              />
              <i v-else :class="menuItem.icon" />
            </div>
            <div v-if="menuItem.showRedDot" class="menu-red-dot" />
          </div>
          <span class="menu-label">{{ menuItem.label }}</span>
          <i
            v-if="menuItem.showExternalIcon"
            class="icon-[lucide--external-link] text-primary w-4 h-4 ml-auto"
          />
          <i
            v-if="menuItem.key === 'more'"
            class="pi pi-chevron-right ml-auto"
          />
        </button>
      </nav>
      <div
        class="flex h-4 flex-col items-center justify-between self-stretch p-2"
      >
        <div class="w-full border-b border-interface-menu-stroke" />
      </div>
    </div>

    <!-- More Submenu -->
    <Teleport to="body">
      <div
        v-if="isSubmenuVisible"
        ref="submenuRef"
        class="more-submenu"
        :style="submenuStyle"
        @mouseenter="onSubmenuHover"
        @mouseleave="onSubmenuLeave"
      >
        <template
          v-for="submenuItem in moreMenuItem?.items"
          :key="submenuItem.key"
        >
          <div
            v-if="submenuItem.type === 'divider'"
            v-show="submenuItem.visible !== false"
            class="submenu-divider"
          />
          <button
            v-else
            v-show="submenuItem.visible !== false"
            type="button"
            class="help-menu-item submenu-item"
            role="menuitem"
            @click="submenuItem.action"
          >
            <span class="menu-label">{{ submenuItem.label }}</span>
          </button>
        </template>
      </div>
    </Teleport>

    <!-- What's New Section -->
    <section
      v-if="showVersionUpdates"
      class="w-full"
      data-testid="whats-new-section"
    >
      <h3
        class="section-description flex items-center gap-2.5 self-stretch px-8 pt-2 pb-2"
      >
        {{ $t('helpCenter.whatsNew') }}
      </h3>

      <!-- Release Items -->
      <div
        v-if="hasReleases"
        role="group"
        :aria-label="$t('help.recentReleases')"
      >
        <article
          v-for="release in releaseStore.recentReleases"
          :key="release.id || release.version"
          class="release-menu-item flex h-12 min-h-6 cursor-pointer items-center gap-2 self-stretch rounded p-2 transition-colors hover:bg-interface-menu-component-surface-hovered"
          role="button"
          tabindex="0"
          @click="onReleaseClick(release)"
          @keydown.enter="onReleaseClick(release)"
          @keydown.space.prevent="onReleaseClick(release)"
        >
          <i class="help-menu-icon icon-[lucide--package]" aria-hidden="true" />
          <div class="release-content">
            <span class="release-title">
              {{
                $t('g.releaseTitle', {
                  package: 'Comfy',
                  version: release.version
                })
              }}
            </span>
            <time class="release-date" :datetime="release.published_at">
              <span class="normal-state">
                {{ formatReleaseDate(release.published_at) }}
              </span>
              <span class="hover-state">
                {{ $t('helpCenter.clickToLearnMore') }}
              </span>
            </time>
          </div>
        </article>
      </div>

      <!-- Loading State -->
      <div
        v-else-if="releaseStore.isLoading"
        class="help-menu-item"
        role="status"
        aria-live="polite"
      >
        <i class="pi pi-spin pi-spinner help-menu-icon" aria-hidden="true" />
        <span>{{ $t('helpCenter.loadingReleases') }}</span>
      </div>

      <!-- No Releases State -->
      <div v-else class="help-menu-item" role="status">
        <i class="pi pi-info-circle help-menu-icon" aria-hidden="true" />
        <span>{{ $t('helpCenter.noRecentReleases') }}</span>
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { useToast } from 'primevue/usetoast'
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from 'vue'
import type { CSSProperties, Component } from 'vue'
import { useI18n } from 'vue-i18n'

import PuzzleIcon from '@/components/icons/PuzzleIcon.vue'
import { useExternalLink } from '@/composables/useExternalLink'
import { isCloud, isDesktop } from '@/platform/distribution/types'
import { useSettingStore } from '@/platform/settings/settingStore'
import { useTelemetry } from '@/platform/telemetry'
import type { ReleaseNote } from '@/platform/updates/common/releaseService'
import { useReleaseStore } from '@/platform/updates/common/releaseStore'
import { useCommandStore } from '@/stores/commandStore'
import { electronAPI } from '@/utils/envUtil'
import { formatVersionAnchor } from '@/utils/formatUtil'
import { useConflictAcknowledgment } from '@/workbench/extensions/manager/composables/useConflictAcknowledgment'
import { useManagerState } from '@/workbench/extensions/manager/composables/useManagerState'
import { useComfyManagerService } from '@/workbench/extensions/manager/services/comfyManagerService'
import { ManagerTab } from '@/workbench/extensions/manager/types/comfyManagerTypes'

// Types
interface MenuItem {
  key: string
  icon?: string | Component
  label?: string
  action?: () => void
  visible?: boolean
  type?: 'item' | 'divider'
  items?: MenuItem[]
  showRedDot?: boolean
  showExternalIcon?: boolean
}

// Constants
const TIME_UNITS = {
  MINUTE: 60 * 1000,
  HOUR: 60 * 60 * 1000,
  DAY: 24 * 60 * 60 * 1000,
  WEEK: 7 * 24 * 60 * 60 * 1000,
  MONTH: 30 * 24 * 60 * 60 * 1000,
  YEAR: 365 * 24 * 60 * 60 * 1000
} as const

const SUBMENU_CONFIG = {
  DELAY_MS: 100,
  OFFSET_PX: 8,
  Z_INDEX: 10001
} as const

// Composables
const { t } = useI18n()
const toast = useToast()
const { staticUrls, buildDocsUrl } = useExternalLink()
const releaseStore = useReleaseStore()
const commandStore = useCommandStore()
const settingStore = useSettingStore()
const telemetry = useTelemetry()

// Track when help center was opened
const openedAt = ref(Date.now())

// Emits
const emit = defineEmits<{
  close: []
}>()

// State
const isSubmenuVisible = ref(false)
const submenuRef = ref<HTMLElement | null>(null)
const submenuStyle = ref<CSSProperties>({})
let hoverTimeout: number | null = null

// Computed
const hasReleases = computed(() => releaseStore.releases.length > 0)
const showVersionUpdates = computed(() =>
  settingStore.get('Comfy.Notification.ShowVersionUpdates')
)

// Use conflict acknowledgment state from composable
const { shouldShowRedDot: shouldShowManagerRedDot } =
  useConflictAcknowledgment()
const { isNewManagerUI } = useManagerState()

const moreItems = computed<MenuItem[]>(() => {
  const allMoreItems: MenuItem[] = [
    {
      key: 'desktop-guide',
      type: 'item',
      label: t('helpCenter.desktopUserGuide'),
      visible: isDesktop,
      action: () => {
        trackResourceClick('docs', true)
        openExternalLink(
          buildDocsUrl('/installation/desktop', {
            includeLocale: true,
            platform: true
          })
        )
        emit('close')
      }
    },
    {
      key: 'dev-tools',
      type: 'item',
      label: t('helpCenter.openDevTools'),
      visible: isDesktop,
      action: () => {
        openDevTools()
        emit('close')
      }
    },
    {
      key: 'divider-1',
      type: 'divider',
      visible: isDesktop
    },
    {
      key: 'reinstall',
      type: 'item',
      label: t('helpCenter.reinstall'),
      visible: isDesktop,
      action: () => {
        onReinstall()
        emit('close')
      }
    }
  ]

  // Filter for visible items only
  return allMoreItems.filter((item) => item.visible !== false)
})

const hasVisibleMoreItems = computed(() => {
  return !!moreItems.value.length
})

const moreMenuItem = computed(() =>
  menuItems.value.find((item) => item.key === 'more')
)

const menuItems = computed<MenuItem[]>(() => {
  const items: MenuItem[] = [
    {
      key: 'feedback',
      type: 'item',
      icon: 'icon-[lucide--clipboard-pen]',
      label: t('helpCenter.feedback'),
      action: () => {
        trackResourceClick('help_feedback', false)
        void commandStore.execute('Comfy.ContactSupport')
        emit('close')
      }
    },
    {
      key: 'help',
      type: 'item',
      icon: 'icon-[lucide--message-circle-question]',
      label: t('helpCenter.help'),
      action: () => {
        trackResourceClick('help_feedback', false)
        void commandStore.execute('Comfy.ContactSupport')
        emit('close')
      }
    },
    {
      key: 'docs',
      type: 'item',
      icon: 'icon-[lucide--book-open]',
      label: t('helpCenter.docs'),
      showExternalIcon: true,
      action: () => {
        trackResourceClick('docs', true)
        const path = isCloud ? '/get_started/cloud' : '/'
        openExternalLink(buildDocsUrl(path, { includeLocale: true }))
        emit('close')
      }
    },
    {
      key: 'discord',
      type: 'item',
      icon: 'pi pi-discord',
      label: 'Discord',
      showExternalIcon: true,
      action: () => {
        trackResourceClick('discord', true)
        openExternalLink(staticUrls.discord)
        emit('close')
      }
    },
    {
      key: 'github',
      type: 'item',
      icon: 'icon-[lucide--github]',
      label: t('helpCenter.github'),
      showExternalIcon: true,
      action: () => {
        trackResourceClick('github', true)
        openExternalLink(staticUrls.github)
        emit('close')
      }
    }
  ]

  // Extension manager - only in non-cloud distributions
  if (!isCloud) {
    items.push({
      key: 'manager',
      type: 'item',
      icon: PuzzleIcon,
      label: t('helpCenter.managerExtension'),
      showRedDot: shouldShowManagerRedDot.value,
      action: async () => {
        trackResourceClick('manager', false)
        await useManagerState().openManager({
          initialTab: ManagerTab.All,
          showToastOnLegacyError: false
        })
        emit('close')
      }
    })
  }
  // Update ComfyUI - only for non-desktop, non-cloud with new manager UI
  if (!isDesktop && !isCloud && isNewManagerUI.value) {
    items.push({
      key: 'update-comfyui',
      type: 'item',
      icon: 'icon-[lucide--download]',
      label: t('helpCenter.updateComfyUI'),
      action: () => {
        onUpdateComfyUI()
        emit('close')
      }
    })
  }

  items.push({
    key: 'more',
    type: 'item',
    icon: '',
    label: t('helpCenter.more'),
    visible: hasVisibleMoreItems.value,
    action: () => {}, // No action for more item
    items: moreItems.value
  })

  return items
})

// Utility Functions
const trackResourceClick = (
  resourceType:
    | 'docs'
    | 'discord'
    | 'github'
    | 'help_feedback'
    | 'manager'
    | 'release_notes',
  isExternal: boolean
): void => {
  telemetry?.trackHelpResourceClicked({
    resource_type: resourceType,
    is_external: isExternal,
    source: 'help_center'
  })
}

const openExternalLink = (url: string): void => {
  window.open(url, '_blank', 'noopener,noreferrer')
}

const clearHoverTimeout = (): void => {
  if (hoverTimeout) {
    clearTimeout(hoverTimeout)
    hoverTimeout = null
  }
}

const calculateSubmenuPosition = (button: HTMLElement): CSSProperties => {
  const rect = button.getBoundingClientRect()
  const submenuWidth = 210 // Width defined in CSS

  // Get actual submenu height if available, otherwise estimate based on visible item count
  const visibleItemCount =
    moreMenuItem.value?.items?.filter((item) => item.visible !== false)
      .length || 0
  const estimatedHeight = visibleItemCount * 48 + 16 // ~48px per item + padding
  const submenuHeight = submenuRef.value?.offsetHeight || estimatedHeight

  // Get viewport dimensions
  const viewportWidth = window.innerWidth
  const viewportHeight = window.innerHeight

  // Calculate basic position (aligned with button)
  let top = rect.top
  let left = rect.right + SUBMENU_CONFIG.OFFSET_PX

  // Check if submenu would overflow viewport on the right
  if (left + submenuWidth > viewportWidth) {
    // Position submenu to the left of the button instead
    left = rect.left - submenuWidth - SUBMENU_CONFIG.OFFSET_PX
  }

  // Check if submenu would overflow viewport at the bottom
  if (top + submenuHeight > viewportHeight) {
    // Position submenu above the button, aligned to bottom
    top = Math.max(
      SUBMENU_CONFIG.OFFSET_PX, // Minimum distance from top of viewport
      rect.bottom - submenuHeight
    )
  }

  // Ensure submenu doesn't go above viewport
  if (top < SUBMENU_CONFIG.OFFSET_PX) {
    top = SUBMENU_CONFIG.OFFSET_PX
  }

  top -= 8

  return {
    position: 'fixed',
    top: `${top}px`,
    left: `${left}px`,
    zIndex: SUBMENU_CONFIG.Z_INDEX
  }
}

const formatReleaseDate = (dateString?: string): string => {
  if (!dateString) return 'date'

  const date = new Date(dateString)
  const now = new Date()
  const diffTime = Math.abs(now.getTime() - date.getTime())

  const timeUnits = [
    { unit: TIME_UNITS.YEAR, key: 'yearsAgo' },
    { unit: TIME_UNITS.MONTH, key: 'monthsAgo' },
    { unit: TIME_UNITS.WEEK, key: 'weeksAgo' },
    { unit: TIME_UNITS.DAY, key: 'daysAgo' },
    { unit: TIME_UNITS.HOUR, key: 'hoursAgo' },
    { unit: TIME_UNITS.MINUTE, key: 'minutesAgo' }
  ]

  for (const { unit, key } of timeUnits) {
    const value = Math.floor(diffTime / unit)
    if (value > 0) {
      return t(`g.relativeTime.${key}`, { count: value })
    }
  }

  return t('g.relativeTime.now')
}

// Event Handlers
const onMenuItemHover = async (
  key: string,
  event: MouseEvent
): Promise<void> => {
  if (key !== 'more' || !moreMenuItem.value?.items) return

  // Don't show submenu if all items are hidden
  const hasVisibleItems = moreMenuItem.value.items.some(
    (item) => item.visible !== false
  )
  if (!hasVisibleItems) return

  clearHoverTimeout()

  const moreButton = event.currentTarget as HTMLElement

  // Calculate initial position before showing submenu
  submenuStyle.value = calculateSubmenuPosition(moreButton)

  // Show submenu with correct position
  isSubmenuVisible.value = true

  // After submenu is rendered, refine position if needed
  await nextTick()
  if (submenuRef.value) {
    submenuStyle.value = calculateSubmenuPosition(moreButton)
  }
}

const onMenuItemLeave = (key: string): void => {
  if (key !== 'more') return

  hoverTimeout = window.setTimeout(() => {
    isSubmenuVisible.value = false
  }, SUBMENU_CONFIG.DELAY_MS)
}

const onSubmenuHover = (): void => {
  clearHoverTimeout()
}

const onSubmenuLeave = (): void => {
  isSubmenuVisible.value = false
}

const openDevTools = (): void => {
  if (isDesktop) {
    electronAPI().openDevTools()
  }
}

const onReinstall = (): void => {
  if (isDesktop) {
    void electronAPI().reinstall()
  }
}

const onUpdateComfyUI = async (): Promise<void> => {
  const { updateComfyUI, rebootComfyUI, error } = useComfyManagerService()

  toast.add({
    severity: 'info',
    summary: t('helpCenter.updateComfyUIStarted'),
    detail: t('helpCenter.updateComfyUIStartedDetail'),
    life: 3000
  })

  try {
    const result = await updateComfyUI({ is_stable: true })

    if (result === null || error.value) {
      toast.add({
        severity: 'error',
        summary: t('g.error'),
        detail: error.value || t('helpCenter.updateComfyUIFailed'),
        life: 5000
      })
      return
    }

    toast.add({
      severity: 'success',
      summary: t('helpCenter.updateComfyUISuccess'),
      detail: t('helpCenter.updateComfyUISuccessDetail'),
      life: 3000
    })

    await rebootComfyUI()
  } catch (err) {
    toast.add({
      severity: 'error',
      summary: t('g.error'),
      detail: err instanceof Error ? err.message : t('g.unknownError'),
      life: 5000
    })
  }
}

const onReleaseClick = (release: ReleaseNote): void => {
  trackResourceClick('release_notes', true)
  void releaseStore.handleShowChangelog(release.version)
  const versionAnchor = formatVersionAnchor(release.version)
  const changelogUrl = `${buildDocsUrl('/changelog', { includeLocale: true })}#${versionAnchor}`
  openExternalLink(changelogUrl)
  emit('close')
}

// Lifecycle
onMounted(async () => {
  telemetry?.trackHelpCenterOpened({ source: 'sidebar' })
  if (!hasReleases.value) {
    await releaseStore.fetchReleases()
  }
})

onBeforeUnmount(() => {
  const timeSpentSeconds = Math.round((Date.now() - openedAt.value) / 1000)
  telemetry?.trackHelpCenterClosed({ time_spent_seconds: timeSpentSeconds })
})
</script>

<style scoped>
.help-center-menu {
  width: 256px;
  max-height: 500px;
  overflow-y: auto;
  background: var(--interface-menu-surface);
  border-radius: 8px;
  box-shadow: 0 2px 12px 0 rgb(0 0 0 / 0.1);
  border: 1px solid var(--interface-menu-stroke);
  padding: 12px 8px;
  position: relative;
}

.help-menu-item {
  display: flex;
  align-items: center;
  width: 100%;
  height: 32px;
  min-height: 24px;
  padding: 8px;
  gap: 8px;
  background: transparent;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.2s;
  font-size: 0.9rem;
  color: var(--text-primary);
  text-align: left;
}

.help-menu-item:hover {
  background-color: var(--interface-menu-component-surface-hovered);
}

.help-menu-item:focus,
.help-menu-item:focus-visible {
  outline: none;
  box-shadow: none;
}

.help-menu-icon-container {
  position: relative;
  width: 16px;
  height: 16px;
  flex-shrink: 0;
}

.help-menu-icon {
  width: 16px;
  height: 16px;
  font-size: 16px;
  color: var(--text-primary);
  display: flex;
  justify-content: center;
  align-items: center;
  flex-shrink: 0;
}

.help-menu-icon svg {
  width: 16px;
  height: 16px;
  color: var(--text-primary);
}

.menu-red-dot {
  position: absolute;
  top: -2px;
  right: -2px;
  width: 8px;
  height: 8px;
  background: #ff3b30;
  border-radius: 50%;
  border: 1.5px solid var(--p-content-background);
  z-index: 1;
}

.menu-label {
  flex: 1;
}

.more-item {
  justify-content: space-between;
}

.section-description {
  color: var(--text-secondary);
  font-family: var(--font-inter);
  font-size: 12px;
  font-style: normal;
  font-weight: 700;
  line-height: normal;
  margin: 0;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.release-menu-item {
  position: relative;
}

.release-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}

.release-title {
  font-size: 0.9rem;
  line-height: 1.2;
  font-weight: 500;
  color: var(--text-primary);
}

.release-date {
  height: 16px;
  color: var(--text-secondary);
  font-family: var(--font-inter);
  font-size: 12px;
  font-style: normal;
  font-weight: 400;
  line-height: normal;
  overflow: hidden;
  text-overflow: ellipsis;
  display: -webkit-box;
  -webkit-line-clamp: 1;
  -webkit-box-orient: vertical;
}

.release-date .hover-state {
  display: none;
}

.release-menu-item:hover .release-date .normal-state,
.release-menu-item:focus-within .release-date .normal-state {
  display: none;
}

.release-menu-item:hover .release-date .hover-state,
.release-menu-item:focus-within .release-date .hover-state {
  display: inline;
}

/* Submenu Styles */
.more-submenu {
  width: 210px;
  padding: 12px 8px;
  background: var(--interface-menu-surface);
  border-radius: 8px;
  border: 1px solid var(--interface-menu-stroke);
  box-shadow: 0 2px 12px 0 rgb(0 0 0 / 0.1);
  overflow: hidden;
  transition: opacity 0.15s ease-out;
}

.submenu-item {
  padding: 8px;
  height: 32px;
  min-height: 24px;
  border-radius: 4px;
  color: var(--text-primary);
  font-size: 0.9rem;
  font-weight: inherit;
  line-height: inherit;
}

.submenu-item:hover {
  background-color: var(--interface-menu-component-surface-hovered);
}

.submenu-item:focus,
.submenu-item:focus-visible {
  outline: none;
  box-shadow: none;
}

.submenu-divider {
  height: 1px;
  background: var(--interface-menu-stroke);
  margin: 4px 0;
}

/* Scrollbar Styling */
.help-center-menu::-webkit-scrollbar {
  width: 6px;
}

.help-center-menu::-webkit-scrollbar-track {
  background: transparent;
}

.help-center-menu::-webkit-scrollbar-thumb {
  background: var(--interface-menu-stroke);
  border-radius: 3px;
}

.help-center-menu::-webkit-scrollbar-thumb:hover {
  background: var(--text-secondary);
}

/* Reduced Motion */
@media (prefers-reduced-motion: reduce) {
  .help-menu-item {
    transition: none;
  }
}
</style>
