<!-- A button that shows workspace icon (Cloud) or user avatar -->
<template>
  <div>
    <Button
      v-if="isLoggedIn"
      class="p-1 hover:bg-transparent"
      variant="muted-textonly"
      :aria-label="$t('g.currentUser')"
      @click="popover?.toggle($event)"
    >
      <div
        :class="
          cn(
            'flex items-center gap-1 rounded-full hover:bg-interface-button-hover-surface justify-center',
            compact && 'size-full '
          )
        "
      >
        <Skeleton
          v-if="showWorkspaceSkeleton"
          shape="circle"
          width="32px"
          height="32px"
        />
        <WorkspaceProfilePic
          v-else-if="showWorkspaceIcon"
          :workspace-name="workspaceName"
          :class="compact && 'size-full'"
        />
        <UserAvatar
          v-else
          :photo-url="photoURL"
          :class="compact && 'size-full'"
        />

        <i v-if="showArrow" class="icon-[lucide--chevron-down] size-4 px-1" />
      </div>
    </Button>

    <Popover
      ref="popover"
      :show-arrow="false"
      :pt="{
        root: {
          class: 'rounded-lg w-80'
        }
      }"
      @show="onPopoverShow"
    >
      <!-- Workspace mode: workspace-aware popover (only when ready) -->
      <CurrentUserPopoverWorkspace
        v-if="teamWorkspacesEnabled && initState === 'ready'"
        ref="workspacePopoverContent"
        @close="closePopover"
      />
      <!-- Legacy mode: original popover -->
      <CurrentUserPopoverLegacy
        v-else-if="!teamWorkspacesEnabled"
        @close="closePopover"
      />
    </Popover>
  </div>
</template>

<script setup lang="ts">
import { storeToRefs } from 'pinia'
import Popover from 'primevue/popover'
import Skeleton from 'primevue/skeleton'
import { computed, defineAsyncComponent, ref } from 'vue'

import UserAvatar from '@/components/common/UserAvatar.vue'
import WorkspaceProfilePic from '@/platform/workspace/components/WorkspaceProfilePic.vue'
import Button from '@/components/ui/button/Button.vue'
import { useCurrentUser } from '@/composables/auth/useCurrentUser'
import { useFeatureFlags } from '@/composables/useFeatureFlags'
import { isCloud } from '@/platform/distribution/types'
import { useTeamWorkspaceStore } from '@/platform/workspace/stores/teamWorkspaceStore'
import { cn } from '@/utils/tailwindUtil'

import CurrentUserPopoverLegacy from './CurrentUserPopoverLegacy.vue'

const CurrentUserPopoverWorkspace = defineAsyncComponent(
  () =>
    import('../../platform/workspace/components/CurrentUserPopoverWorkspace.vue')
)

const { showArrow = true, compact = false } = defineProps<{
  showArrow?: boolean
  compact?: boolean
}>()

const { flags } = useFeatureFlags()
const teamWorkspacesEnabled = computed(() => flags.teamWorkspacesEnabled)

const { isLoggedIn, userPhotoUrl } = useCurrentUser()

const photoURL = computed<string | undefined>(
  () => userPhotoUrl.value ?? undefined
)

const { workspaceName: teamWorkspaceName, initState } = storeToRefs(
  useTeamWorkspaceStore()
)

const showWorkspaceSkeleton = computed(
  () => isCloud && teamWorkspacesEnabled.value && initState.value === 'loading'
)
const showWorkspaceIcon = computed(
  () => isCloud && teamWorkspacesEnabled.value && initState.value === 'ready'
)

const workspaceName = computed(() => {
  if (!showWorkspaceIcon.value) return ''
  return teamWorkspaceName.value
})

const popover = ref<InstanceType<typeof Popover> | null>(null)
const workspacePopoverContent = ref<{
  refreshBalance: () => void
} | null>(null)

const closePopover = () => {
  popover.value?.hide()
}

const onPopoverShow = () => {
  workspacePopoverContent.value?.refreshBalance()
}
</script>
