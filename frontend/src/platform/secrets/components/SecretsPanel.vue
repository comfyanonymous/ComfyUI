<template>
  <TabPanel value="Secrets" class="h-full">
    <div class="flex h-full flex-col">
      <div>
        <h2 class="text-2xl font-bold">{{ $t('secrets.title') }}</h2>
        <p class="mt-1 text-sm text-muted">{{ $t('secrets.description') }}</p>
        <p class="mt-1 text-sm text-muted">
          {{ $t('secrets.descriptionUsage') }}
        </p>
      </div>

      <Divider class="my-4" />

      <div class="flex items-center justify-between my-4">
        <h3 class="text-lg font-semibold my-0">
          {{ $t('secrets.modelProviders') }}
        </h3>
        <Button @click="openCreateDialog">
          <i class="pi pi-plus mr-1" />
          {{ $t('secrets.addSecret') }}
        </Button>
      </div>

      <div v-if="loading" class="flex items-center justify-center py-8">
        <ProgressSpinner class="h-8 w-8" />
      </div>

      <div
        v-else-if="secrets.length === 0"
        class="py-4 text-center text-sm text-muted"
      >
        {{ $t('secrets.noSecrets') }}
      </div>

      <div v-else class="flex flex-col gap-3">
        <SecretListItem
          v-for="secret in secrets"
          :key="secret.id"
          :secret="secret"
          :loading="operatingSecretId === secret.id"
          :disabled="operatingSecretId !== null"
          @edit="openEditDialog(secret)"
          @delete="confirmDelete(secret)"
        />
      </div>

      <SecretFormDialog
        v-model:visible="createDialogVisible"
        mode="create"
        :existing-providers="existingProviders"
        @saved="fetchSecrets"
      />

      <SecretFormDialog
        v-model:visible="editDialogVisible"
        mode="edit"
        :secret="selectedSecret"
        :existing-providers="existingProviders"
        @saved="fetchSecrets"
      />

      <ConfirmDialog group="secrets" />
    </div>
  </TabPanel>
</template>

<script setup lang="ts">
import ConfirmDialog from 'primevue/confirmdialog'
import Divider from 'primevue/divider'
import ProgressSpinner from 'primevue/progressspinner'
import TabPanel from 'primevue/tabpanel'
import { useConfirm } from 'primevue/useconfirm'
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'

import Button from '@/components/ui/button/Button.vue'

import { useSecrets } from '../composables/useSecrets'
import type { SecretMetadata } from '../types'
import SecretFormDialog from './SecretFormDialog.vue'
import SecretListItem from './SecretListItem.vue'

const { t } = useI18n()
const confirm = useConfirm()

const {
  loading,
  secrets,
  operatingSecretId,
  existingProviders,
  fetchSecrets,
  deleteSecret
} = useSecrets()

const createDialogVisible = ref(false)
const editDialogVisible = ref(false)
const selectedSecret = ref<SecretMetadata | undefined>()

function openCreateDialog() {
  createDialogVisible.value = true
}

function openEditDialog(secret: SecretMetadata) {
  selectedSecret.value = secret
  editDialogVisible.value = true
}

function confirmDelete(secret: SecretMetadata) {
  confirm.require({
    group: 'secrets',
    header: t('secrets.deleteConfirmTitle'),
    message: t('secrets.deleteConfirmMessage', { name: secret.name }),
    acceptClass: 'p-button-danger',
    accept: () => deleteSecret(secret)
  })
}

fetchSecrets()
</script>
