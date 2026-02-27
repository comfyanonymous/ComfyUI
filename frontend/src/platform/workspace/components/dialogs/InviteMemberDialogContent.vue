<template>
  <div
    class="flex w-full max-w-[512px] flex-col rounded-2xl border border-border-default bg-base-background"
  >
    <!-- Header -->
    <div
      class="flex h-12 items-center justify-between border-b border-border-default px-4"
    >
      <h2 class="m-0 text-sm font-normal text-base-foreground">
        {{
          step === 'email'
            ? $t('workspacePanel.inviteMemberDialog.title')
            : $t('workspacePanel.inviteMemberDialog.linkStep.title')
        }}
      </h2>
      <button
        class="cursor-pointer rounded border-none bg-transparent p-0 text-muted-foreground transition-colors hover:text-base-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-secondary-foreground"
        :aria-label="$t('g.close')"
        @click="onCancel"
      >
        <i class="pi pi-times size-4" />
      </button>
    </div>

    <!-- Body: Email Step -->
    <template v-if="step === 'email'">
      <div class="flex flex-col gap-4 px-4 py-4">
        <p class="m-0 text-sm text-muted-foreground">
          {{ $t('workspacePanel.inviteMemberDialog.message') }}
        </p>
        <input
          v-model="email"
          type="email"
          class="w-full rounded-lg border border-border-default bg-transparent px-3 py-2 text-sm text-base-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-secondary-foreground"
          :placeholder="$t('workspacePanel.inviteMemberDialog.placeholder')"
        />
      </div>

      <!-- Footer: Email Step -->
      <div class="flex items-center justify-end gap-4 px-4 py-4">
        <Button variant="muted-textonly" @click="onCancel">
          {{ $t('g.cancel') }}
        </Button>
        <Button
          variant="primary"
          size="lg"
          :loading
          :disabled="!isValidEmail"
          @click="onCreateLink"
        >
          {{ $t('workspacePanel.inviteMemberDialog.createLink') }}
        </Button>
      </div>
    </template>

    <!-- Body: Link Step -->
    <template v-else>
      <div class="flex flex-col gap-4 px-4 py-4">
        <p class="m-0 text-sm text-muted-foreground">
          {{ $t('workspacePanel.inviteMemberDialog.linkStep.message') }}
        </p>
        <p class="m-0 text-sm font-medium text-base-foreground">
          {{ email }}
        </p>
        <div class="relative">
          <input
            :value="generatedLink"
            readonly
            class="w-full cursor-pointer rounded-lg border border-border-default bg-transparent px-3 py-2 pr-10 text-sm text-base-foreground focus:outline-none"
            @click="onSelectLink"
          />
          <div
            class="absolute right-3 top-2.5 cursor-pointer"
            @click="onCopyLink"
          >
            <i
              :class="
                cn(
                  'pi size-4',
                  justCopied ? 'pi-check text-green-500' : 'pi-copy'
                )
              "
            />
          </div>
        </div>
      </div>

      <!-- Footer: Link Step -->
      <div class="flex items-center justify-end gap-4 px-4 py-4">
        <Button variant="muted-textonly" @click="onCancel">
          {{ $t('g.cancel') }}
        </Button>
        <Button variant="primary" size="lg" @click="onCopyLink">
          {{ $t('workspacePanel.inviteMemberDialog.linkStep.copyLink') }}
        </Button>
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import { useToast } from 'primevue/usetoast'
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'

import Button from '@/components/ui/button/Button.vue'
import { cn } from '@/utils/tailwindUtil'
import { useTeamWorkspaceStore } from '@/platform/workspace/stores/teamWorkspaceStore'
import { useDialogStore } from '@/stores/dialogStore'

const dialogStore = useDialogStore()
const toast = useToast()
const { t } = useI18n()
const workspaceStore = useTeamWorkspaceStore()

const loading = ref(false)
const email = ref('')
const step = ref<'email' | 'link'>('email')
const generatedLink = ref('')
const justCopied = ref(false)

const isValidEmail = computed(() => {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
  return emailRegex.test(email.value)
})

function onCancel() {
  dialogStore.closeDialog({ key: 'invite-member' })
}

async function onCreateLink() {
  if (!isValidEmail.value) return
  loading.value = true
  try {
    generatedLink.value = await workspaceStore.createInviteLink(email.value)
    step.value = 'link'
  } catch (error) {
    toast.add({
      severity: 'error',
      summary: t('workspacePanel.inviteMemberDialog.linkCopyFailed'),
      detail: error instanceof Error ? error.message : undefined,
      life: 3000
    })
  } finally {
    loading.value = false
  }
}

async function onCopyLink() {
  try {
    await navigator.clipboard.writeText(generatedLink.value)
    justCopied.value = true
    setTimeout(() => {
      justCopied.value = false
    }, 759)
    toast.add({
      severity: 'success',
      summary: t('workspacePanel.inviteMemberDialog.linkCopied'),
      life: 2000
    })
  } catch {
    toast.add({
      severity: 'error',
      summary: t('workspacePanel.inviteMemberDialog.linkCopyFailed'),
      life: 3000
    })
  }
}

function onSelectLink(event: Event) {
  const input = event.target as HTMLInputElement
  input.select()
}
</script>
