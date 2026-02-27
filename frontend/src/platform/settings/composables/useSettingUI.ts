import { computed, defineAsyncComponent, onMounted, ref } from 'vue'
import type { Component } from 'vue'
import { useI18n } from 'vue-i18n'

import { useCurrentUser } from '@/composables/auth/useCurrentUser'
import { useBillingContext } from '@/composables/billing/useBillingContext'
import { useFeatureFlags } from '@/composables/useFeatureFlags'
import { useVueFeatureFlags } from '@/composables/useVueFeatureFlags'
import { isCloud, isDesktop } from '@/platform/distribution/types'
import {
  getSettingInfo,
  useSettingStore
} from '@/platform/settings/settingStore'
import type { SettingTreeNode } from '@/platform/settings/settingStore'
import type { SettingPanelType, SettingParams } from '@/platform/settings/types'
import type { NavGroupData } from '@/types/navTypes'
import { normalizeI18nKey } from '@/utils/formatUtil'
import { buildTree } from '@/utils/treeUtil'

const CATEGORY_ICONS: Record<string, string> = {
  '3D': 'icon-[lucide--box]',
  about: 'icon-[lucide--info]',
  Appearance: 'icon-[lucide--palette]',
  Comfy: 'icon-[lucide--settings]',
  credits: 'icon-[lucide--coins]',
  extension: 'icon-[lucide--puzzle]',
  keybinding: 'icon-[lucide--keyboard]',
  LiteGraph: 'icon-[lucide--workflow]',
  'Mask Editor': 'icon-[lucide--pen-tool]',
  Other: 'icon-[lucide--ellipsis]',
  PlanCredits: 'icon-[lucide--credit-card]',
  secrets: 'icon-[lucide--key-round]',
  'server-config': 'icon-[lucide--server]',
  subscription: 'icon-[lucide--credit-card]',
  user: 'icon-[lucide--user]',
  workspace: 'icon-[lucide--building-2]'
}

interface SettingPanelItem {
  node: SettingTreeNode
  component: Component
  props?: Record<string, unknown>
}

export function useSettingUI(
  defaultPanel?: SettingPanelType,
  scrollToSettingId?: string
) {
  const { t } = useI18n()
  const { isLoggedIn } = useCurrentUser()
  const settingStore = useSettingStore()
  const activeCategory = ref<SettingTreeNode | null>(null)

  const { flags } = useFeatureFlags()
  const { shouldRenderVueNodes } = useVueFeatureFlags()
  const { isActiveSubscription } = useBillingContext()

  const teamWorkspacesEnabled = computed(
    () => isCloud && flags.teamWorkspacesEnabled
  )

  const settingRoot = computed<SettingTreeNode>(() => {
    const root = buildTree(
      Object.values(settingStore.settingsById).filter(
        (setting: SettingParams) =>
          setting.type !== 'hidden' &&
          !(shouldRenderVueNodes.value && setting.hideInVueNodes)
      ),
      (setting: SettingParams) => setting.category || setting.id.split('.')
    )

    const floatingSettings = (root.children ?? []).filter((node) => node.leaf)
    if (floatingSettings.length) {
      root.children = (root.children ?? []).filter((node) => !node.leaf)
      root.children.push({
        key: 'Other',
        label: 'Other',
        leaf: false,
        children: floatingSettings
      })
    }

    return root
  })

  const settingCategories = computed<SettingTreeNode[]>(
    () => settingRoot.value.children ?? []
  )

  // Core setting categories (built-in to ComfyUI) in display order
  // 'Other' includes floating settings that don't have a specific category
  const CORE_CATEGORIES_ORDER = [
    'Comfy',
    'LiteGraph',
    'Appearance',
    '3D',
    'Mask Editor',
    'Other'
  ]
  const CORE_CATEGORIES = new Set(CORE_CATEGORIES_ORDER)

  const coreSettingCategories = computed<SettingTreeNode[]>(() => {
    const categories = settingCategories.value.filter((node) =>
      CORE_CATEGORIES.has(node.label)
    )
    return categories.sort(
      (a, b) =>
        CORE_CATEGORIES_ORDER.indexOf(a.label) -
        CORE_CATEGORIES_ORDER.indexOf(b.label)
    )
  })

  const customNodeSettingCategories = computed<SettingTreeNode[]>(() =>
    settingCategories.value.filter((node) => !CORE_CATEGORIES.has(node.label))
  )

  // Define panel items
  const aboutPanel: SettingPanelItem = {
    node: {
      key: 'about',
      label: 'About',
      children: []
    },
    component: defineAsyncComponent(
      () => import('@/components/dialog/content/setting/AboutPanel.vue')
    )
  }

  const creditsPanel: SettingPanelItem = {
    node: {
      key: 'credits',
      label: 'Credits',
      children: []
    },
    component: defineAsyncComponent(
      () => import('@/components/dialog/content/setting/LegacyCreditsPanel.vue')
    )
  }

  const subscriptionPanel: SettingPanelItem | null =
    !isCloud || !window.__CONFIG__?.subscription_required
      ? null
      : {
          node: {
            key: 'subscription',
            label: 'PlanCredits',
            children: []
          },
          component: defineAsyncComponent(
            () =>
              import('@/platform/cloud/subscription/components/SubscriptionPanel.vue')
          )
        }

  const shouldShowPlanCreditsPanel = computed(() => {
    if (!subscriptionPanel) return false
    return isActiveSubscription.value
  })

  const userPanel: SettingPanelItem = {
    node: {
      key: 'user',
      label: 'User',
      children: []
    },
    component: defineAsyncComponent(
      () => import('@/components/dialog/content/setting/UserPanel.vue')
    )
  }

  // Workspace panel: only available on cloud with team workspaces enabled
  const workspacePanel: SettingPanelItem = {
    node: {
      key: 'workspace',
      label: 'Workspace',
      children: []
    },
    component: defineAsyncComponent(
      () =>
        import('@/platform/workspace/components/dialogs/settings/WorkspacePanelContent.vue')
    )
  }

  const shouldShowWorkspacePanel = computed(
    () => teamWorkspacesEnabled.value && isLoggedIn.value
  )

  const secretsPanel: SettingPanelItem = {
    node: {
      key: 'secrets',
      label: 'Secrets',
      children: []
    },
    component: defineAsyncComponent(
      () => import('@/platform/secrets/components/SecretsPanel.vue')
    )
  }

  const shouldShowSecretsPanel = computed(
    () => flags.userSecretsEnabled && isLoggedIn.value
  )

  const keybindingPanel: SettingPanelItem = {
    node: {
      key: 'keybinding',
      label: 'Keybinding',
      children: []
    },
    component: defineAsyncComponent(
      () => import('@/components/dialog/content/setting/KeybindingPanel.vue')
    )
  }

  const extensionPanel: SettingPanelItem = {
    node: {
      key: 'extension',
      label: 'Extension',
      children: []
    },
    component: defineAsyncComponent(
      () => import('@/platform/settings/components/ExtensionPanel.vue')
    )
  }

  const serverConfigPanel: SettingPanelItem = {
    node: {
      key: 'server-config',
      label: 'Server-Config',
      children: []
    },
    component: defineAsyncComponent(
      () => import('@/platform/settings/components/ServerConfigPanel.vue')
    )
  }

  const panels = computed<SettingPanelItem[]>(() =>
    [
      aboutPanel,
      creditsPanel,
      userPanel,
      ...(shouldShowWorkspacePanel.value ? [workspacePanel] : []),
      keybindingPanel,
      extensionPanel,
      ...(isDesktop ? [serverConfigPanel] : []),
      ...(shouldShowPlanCreditsPanel.value && subscriptionPanel
        ? [subscriptionPanel]
        : []),
      ...(shouldShowSecretsPanel.value ? [secretsPanel] : [])
    ].filter((panel) => panel !== null && panel.component)
  )

  /**
   * The default category to show when the dialog is opened.
   */
  const defaultCategory = computed<SettingTreeNode>(() => {
    if (defaultPanel) {
      for (const group of groupedMenuTreeNodes.value) {
        const found = group.children?.find((node) => node.key === defaultPanel)
        if (found) return found
      }
      return settingCategories.value[0]
    }

    if (scrollToSettingId) {
      const setting = settingStore.settingsById[scrollToSettingId]
      if (setting) {
        const { category } = getSettingInfo(setting)
        const found = settingCategories.value.find((c) => c.label === category)
        if (found) return found
      }
    }

    return settingCategories.value[0]
  })

  const translateCategory = (node: SettingTreeNode) => ({
    ...node,
    translatedLabel: t(
      `settingsCategories.${normalizeI18nKey(node.label)}`,
      node.label
    )
  })

  // Sidebar structure when team workspaces is enabled
  const workspaceMenuTreeNodes = computed<SettingTreeNode[]>(() => [
    // Workspace settings
    translateCategory({
      key: 'workspace',
      label: 'Workspace',
      children: [
        ...(shouldShowWorkspacePanel.value ? [workspacePanel.node] : []),
        ...(isLoggedIn.value &&
        !(isCloud && window.__CONFIG__?.subscription_required)
          ? [creditsPanel.node]
          : [])
      ].map(translateCategory)
    }),
    // General settings - Profile + all core settings + special panels
    translateCategory({
      key: 'general',
      label: 'General',
      children: [
        translateCategory(userPanel.node),
        ...coreSettingCategories.value.slice(0, 1).map(translateCategory),
        ...(shouldShowSecretsPanel.value
          ? [translateCategory(secretsPanel.node)]
          : []),
        ...coreSettingCategories.value.slice(1).map(translateCategory),
        translateCategory(keybindingPanel.node),
        translateCategory(extensionPanel.node),
        translateCategory(aboutPanel.node),
        ...(isDesktop ? [translateCategory(serverConfigPanel.node)] : [])
      ]
    }),
    // Custom node settings (only shown if custom nodes have registered settings)
    ...(customNodeSettingCategories.value.length > 0
      ? [
          translateCategory({
            key: 'other',
            label: 'Other',
            children: customNodeSettingCategories.value.map(translateCategory)
          })
        ]
      : [])
  ])

  // Sidebar structure when team workspaces is disabled (legacy)
  const legacyMenuTreeNodes = computed<SettingTreeNode[]>(() => [
    // Account settings - show different panels based on distribution and auth state
    {
      key: 'account',
      label: 'Account',
      children: [
        userPanel.node,
        ...(isLoggedIn.value &&
        shouldShowPlanCreditsPanel.value &&
        subscriptionPanel
          ? [subscriptionPanel.node]
          : []),
        ...(shouldShowSecretsPanel.value ? [secretsPanel.node] : []),
        ...(isLoggedIn.value &&
        !(isCloud && window.__CONFIG__?.subscription_required)
          ? [creditsPanel.node]
          : [])
      ].map(translateCategory)
    },
    // Normal settings stored in the settingStore
    {
      key: 'settings',
      label: 'Application Settings',
      children: settingCategories.value.map(translateCategory)
    },
    // Special settings such as about, keybinding, extension, server-config
    {
      key: 'specialSettings',
      label: 'Special Settings',
      children: [
        keybindingPanel.node,
        extensionPanel.node,
        aboutPanel.node,
        ...(isDesktop ? [serverConfigPanel.node] : [])
      ].map(translateCategory)
    }
  ])

  const groupedMenuTreeNodes = computed<SettingTreeNode[]>(() =>
    teamWorkspacesEnabled.value
      ? workspaceMenuTreeNodes.value
      : legacyMenuTreeNodes.value
  )

  const navGroups = computed<NavGroupData[]>(() =>
    groupedMenuTreeNodes.value.map((group) => ({
      title:
        (group as SettingTreeNode & { translatedLabel?: string })
          .translatedLabel ?? group.label,
      items: (group.children ?? []).map((child) => ({
        id: child.key,
        label:
          (child as SettingTreeNode & { translatedLabel?: string })
            .translatedLabel ?? child.label,
        icon:
          CATEGORY_ICONS[child.key] ??
          CATEGORY_ICONS[child.label] ??
          'icon-[lucide--plug]'
      }))
    }))
  )

  function findCategoryByKey(key: string): SettingTreeNode | null {
    for (const group of groupedMenuTreeNodes.value) {
      const found = group.children?.find((node) => node.key === key)
      if (found) return found
    }
    return null
  }

  function findPanelByKey(key: string): SettingPanelItem | null {
    return panels.value.find((p) => p.node.key === key) ?? null
  }

  onMounted(() => {
    activeCategory.value = defaultCategory.value
  })

  return {
    panels,
    activeCategory,
    defaultCategory,
    groupedMenuTreeNodes,
    settingCategories,
    navGroups,
    teamWorkspacesEnabled,
    findCategoryByKey,
    findPanelByKey
  }
}
