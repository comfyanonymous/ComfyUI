<template>
  <div class="flex h-full w-full flex-col">
    <header class="mb-8 flex items-center gap-4">
      <WorkspaceProfilePic
        class="size-12 !text-3xl"
        :workspace-name="workspaceName"
      />
      <h1 class="text-3xl text-base-foreground">
        {{ workspaceName }}
      </h1>
    </header>
    <TabsRoot v-model="activeTab">
      <div class="flex w-full items-center justify-between">
        <TabsList class="flex items-center gap-2 pb-1">
          <TabsTrigger
            value="plan"
            :class="
              cn(
                tabTriggerBase,
                activeTab === 'plan' ? tabTriggerActive : tabTriggerInactive
              )
            "
          >
            {{ $t('workspacePanel.tabs.planCredits') }}
          </TabsTrigger>
          <TabsTrigger
            value="members"
            :class="
              cn(
                tabTriggerBase,
                activeTab === 'members' ? tabTriggerActive : tabTriggerInactive
              )
            "
          >
            {{
              $t('workspacePanel.tabs.membersCount', {
                count: members.length
              })
            }}
          </TabsTrigger>
        </TabsList>
        <div class="flex items-center gap-1">
          <Button
            v-if="permissions.canInviteMembers"
            v-tooltip="
              inviteTooltip
                ? { value: inviteTooltip, showDelay: 0 }
                : { value: $t('workspacePanel.inviteMember'), showDelay: 300 }
            "
            variant="secondary"
            size="lg"
            :disabled="!isSingleSeatPlan && isInviteLimitReached"
            :class="
              !isSingleSeatPlan &&
              isInviteLimitReached &&
              'opacity-50 cursor-not-allowed'
            "
            :aria-label="$t('workspacePanel.inviteMember')"
            @click="handleInviteMember"
          >
            <i class="pi pi-plus text-sm" />
          </Button>
          <template v-if="permissions.canAccessWorkspaceMenu">
            <Button
              v-tooltip="{ value: $t('g.moreOptions'), showDelay: 300 }"
              variant="muted-textonly"
              size="icon"
              :aria-label="$t('g.moreOptions')"
              @click="menu?.toggle($event)"
            >
              <i class="pi pi-ellipsis-h" />
            </Button>
            <Menu ref="menu" :model="menuItems" :popup="true">
              <template #item="{ item }">
                <button
                  v-tooltip="
                    item.disabled && deleteTooltip
                      ? { value: deleteTooltip, showDelay: 0 }
                      : null
                  "
                  type="button"
                  :disabled="!!item.disabled"
                  :class="
                    cn(
                      'flex w-full items-center gap-2 px-3 py-2 bg-transparent border-none cursor-pointer',
                      item.class,
                      item.disabled && 'pointer-events-auto cursor-not-allowed'
                    )
                  "
                  @click="
                    item.command?.({
                      originalEvent: $event,
                      item
                    })
                  "
                >
                  <i :class="item.icon" />
                  <span>{{ item.label }}</span>
                </button>
              </template>
            </Menu>
          </template>
        </div>
      </div>

      <TabsContent value="plan" class="mt-4">
        <SubscriptionPanelContentWorkspace />
      </TabsContent>
      <TabsContent value="members" class="mt-4">
        <MembersPanelContent :key="workspaceRole" />
      </TabsContent>
    </TabsRoot>
  </div>
</template>

<script setup lang="ts">
import { storeToRefs } from 'pinia'
import Menu from 'primevue/menu'
import { computed, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'

import { TabsContent, TabsList, TabsRoot, TabsTrigger } from 'reka-ui'

import WorkspaceProfilePic from '@/platform/workspace/components/WorkspaceProfilePic.vue'
import MembersPanelContent from '@/platform/workspace/components/dialogs/settings/MembersPanelContent.vue'
import Button from '@/components/ui/button/Button.vue'
import { useBillingContext } from '@/composables/billing/useBillingContext'
import { TIER_TO_KEY } from '@/platform/cloud/subscription/constants/tierPricing'
import SubscriptionPanelContentWorkspace from '@/platform/workspace/components/SubscriptionPanelContentWorkspace.vue'
import { useWorkspaceUI } from '@/platform/workspace/composables/useWorkspaceUI'
import { useTeamWorkspaceStore } from '@/platform/workspace/stores/teamWorkspaceStore'
import { useDialogService } from '@/services/dialogService'
import { cn } from '@/utils/tailwindUtil'

const tabTriggerBase =
  'flex items-center justify-center shrink-0 px-2.5 py-2 text-sm rounded-lg cursor-pointer transition-all duration-200 outline-hidden border-none'
const tabTriggerActive =
  'bg-interface-menu-component-surface-hovered text-text-primary font-bold'
const tabTriggerInactive =
  'bg-transparent text-text-secondary hover:bg-button-hover-surface focus:bg-button-hover-surface'

const { defaultTab = 'plan' } = defineProps<{
  defaultTab?: string
}>()

const { t } = useI18n()
const {
  showLeaveWorkspaceDialog,
  showDeleteWorkspaceDialog,
  showInviteMemberDialog,
  showInviteMemberUpsellDialog,
  showEditWorkspaceDialog
} = useDialogService()
const { isActiveSubscription, subscription, getMaxSeats } = useBillingContext()

const isSingleSeatPlan = computed(() => {
  if (!isActiveSubscription.value) return true
  const tier = subscription.value?.tier
  if (!tier) return true
  const tierKey = TIER_TO_KEY[tier]
  if (!tierKey) return true
  return getMaxSeats(tierKey) <= 1
})
const workspaceStore = useTeamWorkspaceStore()
const { workspaceName, members, isInviteLimitReached, isWorkspaceSubscribed } =
  storeToRefs(workspaceStore)
const { fetchMembers, fetchPendingInvites } = workspaceStore

const { workspaceRole, permissions, uiConfig } = useWorkspaceUI()
const activeTab = ref(defaultTab)

const menu = ref<InstanceType<typeof Menu> | null>(null)

function handleLeaveWorkspace() {
  showLeaveWorkspaceDialog()
}

function handleDeleteWorkspace() {
  showDeleteWorkspaceDialog()
}

function handleEditWorkspace() {
  showEditWorkspaceDialog()
}

// Disable delete when workspace has an active subscription (to prevent accidental deletion)
// Use workspace's own subscription status, not the global isActiveSubscription
const isDeleteDisabled = computed(
  () =>
    uiConfig.value.workspaceMenuAction === 'delete' &&
    isWorkspaceSubscribed.value
)

const deleteTooltip = computed(() => {
  if (!isDeleteDisabled.value) return null
  const tooltipKey = uiConfig.value.workspaceMenuDisabledTooltip
  return tooltipKey ? t(tooltipKey) : null
})

const inviteTooltip = computed(() => {
  if (isSingleSeatPlan.value) return null
  if (!isInviteLimitReached.value) return null
  return t('workspacePanel.inviteLimitReached')
})

function handleInviteMember() {
  if (isSingleSeatPlan.value) {
    showInviteMemberUpsellDialog()
    return
  }
  if (isInviteLimitReached.value) return
  showInviteMemberDialog()
}

const menuItems = computed(() => {
  const items = []

  // Add edit option for owners
  if (uiConfig.value.showEditWorkspaceMenuItem) {
    items.push({
      label: t('workspacePanel.menu.editWorkspace'),
      icon: 'pi pi-pencil',
      command: handleEditWorkspace
    })
  }

  const action = uiConfig.value.workspaceMenuAction
  if (action === 'delete') {
    items.push({
      label: t('workspacePanel.menu.deleteWorkspace'),
      icon: 'pi pi-trash',
      class: isDeleteDisabled.value
        ? 'text-danger/50 cursor-not-allowed'
        : 'text-danger',
      disabled: isDeleteDisabled.value,
      command: isDeleteDisabled.value ? undefined : handleDeleteWorkspace
    })
  } else if (action === 'leave') {
    items.push({
      label: t('workspacePanel.menu.leaveWorkspace'),
      icon: 'pi pi-sign-out',
      command: handleLeaveWorkspace
    })
  }

  return items
})

onMounted(() => {
  fetchMembers()
  fetchPendingInvites()
})
</script>
