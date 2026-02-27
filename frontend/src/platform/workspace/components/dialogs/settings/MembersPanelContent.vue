<template>
  <div class="grow overflow-auto pt-6">
    <div
      class="flex size-full flex-col gap-2 rounded-2xl border border-interface-stroke border-inter p-6"
    >
      <!-- Section Header -->
      <div class="flex w-full items-center gap-9">
        <div class="flex min-w-0 flex-1 items-baseline gap-2">
          <span class="text-base font-semibold text-base-foreground">
            <template v-if="activeView === 'active'">
              {{
                $t('workspacePanel.members.membersCount', {
                  count:
                    isSingleSeatPlan || isPersonalWorkspace
                      ? 1
                      : members.length,
                  maxSeats: maxSeats
                })
              }}
            </template>
            <template v-else-if="permissions.canViewPendingInvites">
              {{
                $t(
                  'workspacePanel.members.pendingInvitesCount',
                  pendingInvites.length
                )
              }}
            </template>
          </span>
        </div>
        <div
          v-if="uiConfig.showSearch && !isSingleSeatPlan"
          class="flex items-start gap-2"
        >
          <SearchBox
            v-model="searchQuery"
            :placeholder="$t('g.search')"
            size="lg"
            class="w-64"
          />
        </div>
      </div>

      <!-- Members Content -->
      <div class="flex min-h-0 flex-1 flex-col">
        <!-- Table Header with Tab Buttons and Column Headers -->
        <div
          v-if="uiConfig.showMembersList"
          :class="
            cn(
              'grid w-full items-center py-2',
              isSingleSeatPlan
                ? 'grid-cols-1 py-0'
                : activeView === 'pending'
                  ? uiConfig.pendingGridCols
                  : uiConfig.headerGridCols
            )
          "
        >
          <!-- Tab buttons in first column -->
          <div v-if="!isSingleSeatPlan" class="flex items-center gap-2">
            <Button
              :variant="
                activeView === 'active' ? 'secondary' : 'muted-textonly'
              "
              size="md"
              @click="activeView = 'active'"
            >
              {{ $t('workspacePanel.members.tabs.active') }}
            </Button>
            <Button
              v-if="uiConfig.showPendingTab"
              :variant="
                activeView === 'pending' ? 'secondary' : 'muted-textonly'
              "
              size="md"
              @click="activeView = 'pending'"
            >
              {{
                $t(
                  'workspacePanel.members.tabs.pendingCount',
                  pendingInvites.length
                )
              }}
            </Button>
          </div>
          <!-- Date column headers -->
          <template v-if="activeView === 'pending'">
            <Button
              variant="muted-textonly"
              size="sm"
              class="justify-start"
              @click="toggleSort('inviteDate')"
            >
              {{ $t('workspacePanel.members.columns.inviteDate') }}
              <i class="icon-[lucide--chevrons-up-down] size-4" />
            </Button>
            <Button
              variant="muted-textonly"
              size="sm"
              class="justify-start"
              @click="toggleSort('expiryDate')"
            >
              {{ $t('workspacePanel.members.columns.expiryDate') }}
              <i class="icon-[lucide--chevrons-up-down] size-4" />
            </Button>
            <div />
          </template>
          <template v-else>
            <template v-if="!isSingleSeatPlan">
              <Button
                variant="muted-textonly"
                size="sm"
                class="justify-end"
                @click="toggleSort('joinDate')"
              >
                {{ $t('workspacePanel.members.columns.joinDate') }}
                <i class="icon-[lucide--chevrons-up-down] size-4" />
              </Button>
              <!-- Empty cell for action column header (OWNER only) -->
              <div v-if="permissions.canRemoveMembers" />
            </template>
          </template>
        </div>

        <!-- Members List -->
        <div class="min-h-0 flex-1 overflow-y-auto">
          <!-- Active Members -->
          <template v-if="activeView === 'active'">
            <!-- Personal Workspace: show only current user -->
            <template v-if="isPersonalWorkspace">
              <div
                :class="
                  cn(
                    'grid w-full items-center rounded-lg p-2',
                    uiConfig.membersGridCols
                  )
                "
              >
                <div class="flex items-center gap-3">
                  <UserAvatar
                    class="size-8"
                    :photo-url="userPhotoUrl"
                    :pt:icon:class="{ 'text-xl!': !userPhotoUrl }"
                  />
                  <div class="flex min-w-0 flex-1 flex-col gap-1">
                    <div class="flex items-center gap-2">
                      <span class="text-sm text-base-foreground">
                        {{ userDisplayName }}
                        <span class="text-muted-foreground">
                          ({{ $t('g.you') }})
                        </span>
                      </span>
                      <span
                        v-if="uiConfig.showRoleBadge"
                        class="text-[10px] font-bold uppercase text-base-background bg-base-foreground px-1 py-0.5 rounded-full"
                      >
                        {{ $t('workspaceSwitcher.roleOwner') }}
                      </span>
                    </div>
                    <span class="text-sm text-muted-foreground">
                      {{ userEmail }}
                    </span>
                  </div>
                </div>
              </div>
            </template>

            <!-- Team Workspace: sorted list (owner first, current user second, then rest) -->
            <template v-else>
              <div
                v-for="(member, index) in filteredMembers"
                :key="member.id"
                :class="
                  cn(
                    'grid w-full items-center rounded-lg p-2',
                    isSingleSeatPlan ? 'grid-cols-1' : uiConfig.membersGridCols,
                    index % 2 === 1 && 'bg-secondary-background/50'
                  )
                "
              >
                <div class="flex items-center gap-3">
                  <UserAvatar
                    class="size-8"
                    :photo-url="
                      isCurrentUser(member) ? userPhotoUrl : undefined
                    "
                    :pt:icon:class="{
                      'text-xl!': !isCurrentUser(member) || !userPhotoUrl
                    }"
                  />
                  <div class="flex min-w-0 flex-1 flex-col gap-1">
                    <div class="flex items-center gap-2">
                      <span class="text-sm text-base-foreground">
                        {{ member.name }}
                        <span
                          v-if="isCurrentUser(member)"
                          class="text-muted-foreground"
                        >
                          ({{ $t('g.you') }})
                        </span>
                      </span>
                      <span
                        v-if="uiConfig.showRoleBadge"
                        class="text-[10px] font-bold uppercase text-base-background bg-base-foreground px-1 py-0.5 rounded-full"
                      >
                        {{ getRoleBadgeLabel(member.role) }}
                      </span>
                    </div>
                    <span class="text-sm text-muted-foreground">
                      {{ member.email }}
                    </span>
                  </div>
                </div>
                <!-- Join date -->
                <span
                  v-if="uiConfig.showDateColumn && !isSingleSeatPlan"
                  class="text-sm text-muted-foreground text-right"
                >
                  {{ formatDate(member.joinDate) }}
                </span>
                <!-- Remove member action (OWNER only, can't remove yourself) -->
                <div
                  v-if="permissions.canRemoveMembers && !isSingleSeatPlan"
                  class="flex items-center justify-end"
                >
                  <Button
                    v-if="!isCurrentUser(member)"
                    v-tooltip="{
                      value: $t('g.moreOptions'),
                      showDelay: 300
                    }"
                    variant="muted-textonly"
                    size="icon"
                    :aria-label="$t('g.moreOptions')"
                    @click="showMemberMenu($event, member)"
                  >
                    <i class="pi pi-ellipsis-h" />
                  </Button>
                </div>
              </div>

              <!-- Member actions menu (shared for all members) -->
              <Menu ref="memberMenu" :model="memberMenuItems" :popup="true" />
            </template>
          </template>

          <!-- Upsell Banner -->
          <div
            v-if="isSingleSeatPlan"
            class="flex items-center gap-2 rounded-xl border bg-secondary-background border-border-default px-4 py-3 mt-4 justify-center"
          >
            <p class="m-0 text-sm text-foreground">
              {{
                isActiveSubscription
                  ? $t('workspacePanel.members.upsellBannerUpgrade')
                  : $t('workspacePanel.members.upsellBannerSubscribe')
              }}
            </p>
            <Button
              variant="muted-textonly"
              class="cursor-pointer underline text-sm"
              @click="showSubscriptionDialog()"
            >
              {{ $t('workspacePanel.members.viewPlans') }}
            </Button>
          </div>

          <!-- Pending Invites -->
          <template v-if="activeView === 'pending'">
            <div
              v-for="(invite, index) in filteredPendingInvites"
              :key="invite.id"
              :class="
                cn(
                  'grid w-full items-center rounded-lg p-2',
                  uiConfig.pendingGridCols,
                  index % 2 === 1 && 'bg-secondary-background/50'
                )
              "
            >
              <!-- Invite info -->
              <div class="flex items-center gap-3">
                <div
                  class="flex size-8 shrink-0 items-center justify-center rounded-full bg-secondary-background"
                >
                  <span class="text-sm font-bold text-base-foreground">
                    {{ getInviteInitial(invite.email) }}
                  </span>
                </div>
                <div class="flex min-w-0 flex-1 flex-col gap-1">
                  <span class="text-sm text-base-foreground">
                    {{ getInviteDisplayName(invite.email) }}
                  </span>
                  <span class="text-sm text-muted-foreground">
                    {{ invite.email }}
                  </span>
                </div>
              </div>
              <!-- Invite date -->
              <span class="text-sm text-muted-foreground">
                {{ formatDate(invite.inviteDate) }}
              </span>
              <!-- Expiry date -->
              <span class="text-sm text-muted-foreground">
                {{ formatDate(invite.expiryDate) }}
              </span>
              <!-- Actions -->
              <div class="flex items-center justify-end gap-2">
                <Button
                  v-tooltip="{
                    value: $t('workspacePanel.members.actions.copyLink'),
                    showDelay: 300
                  }"
                  variant="secondary"
                  size="md"
                  :aria-label="$t('workspacePanel.members.actions.copyLink')"
                  @click="handleCopyInviteLink(invite)"
                >
                  <i class="icon-[lucide--link] size-4" />
                </Button>
                <Button
                  v-tooltip="{
                    value: $t('workspacePanel.members.actions.revokeInvite'),
                    showDelay: 300
                  }"
                  variant="secondary"
                  size="md"
                  :aria-label="
                    $t('workspacePanel.members.actions.revokeInvite')
                  "
                  @click="handleRevokeInvite(invite)"
                >
                  <i class="icon-[lucide--mail-x] size-4" />
                </Button>
              </div>
            </div>
            <div
              v-if="filteredPendingInvites.length === 0"
              class="flex w-full items-center justify-center py-8 text-sm text-muted-foreground"
            >
              {{ $t('workspacePanel.members.noInvites') }}
            </div>
          </template>
        </div>
      </div>
    </div>
    <!-- Personal Workspace Message -->
    <div v-if="isPersonalWorkspace" class="flex items-center">
      <p class="text-sm text-muted-foreground">
        {{ $t('workspacePanel.members.personalWorkspaceMessage') }}
      </p>
      <button
        class="underline bg-transparent border-none cursor-pointer"
        @click="handleCreateWorkspace"
      >
        {{ $t('workspacePanel.members.createNewWorkspace') }}
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { storeToRefs } from 'pinia'
import Menu from 'primevue/menu'
import { useToast } from 'primevue/usetoast'
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'

import SearchBox from '@/components/common/SearchBox.vue'
import UserAvatar from '@/components/common/UserAvatar.vue'
import Button from '@/components/ui/button/Button.vue'
import { useCurrentUser } from '@/composables/auth/useCurrentUser'
import { useBillingContext } from '@/composables/billing/useBillingContext'
import { TIER_TO_KEY } from '@/platform/cloud/subscription/constants/tierPricing'
import { useWorkspaceUI } from '@/platform/workspace/composables/useWorkspaceUI'
import type {
  PendingInvite,
  WorkspaceMember
} from '@/platform/workspace/stores/teamWorkspaceStore'
import { useTeamWorkspaceStore } from '@/platform/workspace/stores/teamWorkspaceStore'
import { useDialogService } from '@/services/dialogService'
import { cn } from '@/utils/tailwindUtil'

const { d, t } = useI18n()
const toast = useToast()
const { userPhotoUrl, userEmail, userDisplayName } = useCurrentUser()
const {
  showRemoveMemberDialog,
  showRevokeInviteDialog,
  showCreateWorkspaceDialog
} = useDialogService()
const workspaceStore = useTeamWorkspaceStore()
const {
  members,
  pendingInvites,
  isInPersonalWorkspace: isPersonalWorkspace
} = storeToRefs(workspaceStore)
const { copyInviteLink } = workspaceStore
const { permissions, uiConfig } = useWorkspaceUI()
const {
  isActiveSubscription,
  subscription,
  showSubscriptionDialog,
  getMaxSeats
} = useBillingContext()

const maxSeats = computed(() => {
  if (isPersonalWorkspace.value) return 1
  const tier = subscription.value?.tier
  if (!tier) return 1
  const tierKey = TIER_TO_KEY[tier]
  if (!tierKey) return 1
  return getMaxSeats(tierKey)
})

const isSingleSeatPlan = computed(() => {
  if (isPersonalWorkspace.value) return false
  if (!isActiveSubscription.value) return true
  return maxSeats.value <= 1
})

const searchQuery = ref('')
const activeView = ref<'active' | 'pending'>('active')
const sortField = ref<'inviteDate' | 'expiryDate' | 'joinDate'>('inviteDate')
const sortDirection = ref<'asc' | 'desc'>('desc')

const memberMenu = ref<InstanceType<typeof Menu> | null>(null)
const selectedMember = ref<WorkspaceMember | null>(null)

function getInviteDisplayName(email: string): string {
  return email.split('@')[0]
}

function getInviteInitial(email: string): string {
  return email.charAt(0).toUpperCase()
}

const memberMenuItems = computed(() => [
  {
    label: t('workspacePanel.members.actions.removeMember'),
    icon: 'pi pi-user-minus',
    command: () => {
      if (selectedMember.value) {
        handleRemoveMember(selectedMember.value)
      }
    }
  }
])

function showMemberMenu(event: Event, member: WorkspaceMember) {
  selectedMember.value = member
  memberMenu.value?.toggle(event)
}

function isCurrentUser(member: WorkspaceMember): boolean {
  return member.email.toLowerCase() === userEmail.value?.toLowerCase()
}

// All members sorted: owners first, current user second, then rest by join date
const filteredMembers = computed(() => {
  let result = [...members.value]

  if (searchQuery.value) {
    const query = searchQuery.value.toLowerCase()
    result = result.filter(
      (member) =>
        member.name.toLowerCase().includes(query) ||
        member.email.toLowerCase().includes(query)
    )
  }

  result.sort((a, b) => {
    // Owners always come first
    if (a.role === 'owner' && b.role !== 'owner') return -1
    if (a.role !== 'owner' && b.role === 'owner') return 1

    // Current user comes second (after owner)
    const aIsCurrentUser = isCurrentUser(a)
    const bIsCurrentUser = isCurrentUser(b)
    if (aIsCurrentUser && !bIsCurrentUser) return -1
    if (!aIsCurrentUser && bIsCurrentUser) return 1

    // Then sort by join date
    const aValue = a.joinDate.getTime()
    const bValue = b.joinDate.getTime()
    return sortDirection.value === 'asc' ? aValue - bValue : bValue - aValue
  })

  return result
})

function getRoleBadgeLabel(role: 'owner' | 'member'): string {
  return role === 'owner'
    ? t('workspaceSwitcher.roleOwner')
    : t('workspaceSwitcher.roleMember')
}

const filteredPendingInvites = computed(() => {
  let result = [...pendingInvites.value]

  if (searchQuery.value) {
    const query = searchQuery.value.toLowerCase()
    result = result.filter((invite) =>
      invite.email.toLowerCase().includes(query)
    )
  }

  const field = sortField.value === 'joinDate' ? 'inviteDate' : sortField.value
  result.sort((a, b) => {
    const aDate = a[field]
    const bDate = b[field]
    if (!aDate || !bDate) return 0
    const aValue = aDate.getTime()
    const bValue = bDate.getTime()
    return sortDirection.value === 'asc' ? aValue - bValue : bValue - aValue
  })

  return result
})

function toggleSort(field: 'inviteDate' | 'expiryDate' | 'joinDate') {
  if (sortField.value === field) {
    sortDirection.value = sortDirection.value === 'asc' ? 'desc' : 'asc'
  } else {
    sortField.value = field
    sortDirection.value = 'desc'
  }
}

function formatDate(date: Date): string {
  return d(date, { dateStyle: 'medium' })
}

async function handleCopyInviteLink(invite: PendingInvite) {
  try {
    await copyInviteLink(invite.id)
    toast.add({
      severity: 'success',
      summary: t('g.copied'),
      life: 2000
    })
  } catch {
    toast.add({
      severity: 'error',
      summary: t('g.error'),
      life: 3000
    })
  }
}

function handleRevokeInvite(invite: PendingInvite) {
  showRevokeInviteDialog(invite.id)
}

function handleCreateWorkspace() {
  showCreateWorkspaceDialog()
}

function handleRemoveMember(member: WorkspaceMember) {
  showRemoveMemberDialog(member.id)
}
</script>
