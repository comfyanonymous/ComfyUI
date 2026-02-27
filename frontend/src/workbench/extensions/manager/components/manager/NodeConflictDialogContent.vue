<template>
  <div class="flex w-138 flex-col">
    <ContentDivider :width="1" />
    <div class="flex h-full w-full flex-col gap-2 px-4 py-6">
      <!-- Description -->
      <div v-if="showAfterWhatsNew">
        <p class="m-0 mb-4 text-sm leading-4 text-base-foreground">
          {{ $t('manager.conflicts.description') }}
          <br /><br />
          {{ $t('manager.conflicts.info') }}
        </p>
      </div>

      <!-- Import Failed List Wrapper -->
      <div
        v-if="importFailedConflicts.length > 0"
        class="flex min-h-8 w-full flex-col rounded-lg bg-base-background"
      >
        <div
          data-testid="conflict-dialog-panel-toggle"
          class="flex h-8 w-full items-center justify-between gap-2 pl-4"
          @click="toggleImportFailedPanel"
        >
          <div class="flex flex-1">
            <span class="mr-2 text-xs font-bold text-warning-background">{{
              importFailedConflicts.length
            }}</span>
            <span class="text-xs font-bold text-base-foreground">{{
              $t('manager.conflicts.importFailedExtensions')
            }}</span>
          </div>
          <div>
            <Button variant="textonly" class="bg-transparent text-muted">
              <i
                :class="
                  importFailedExpanded
                    ? 'pi pi-chevron-down text-xs'
                    : 'pi pi-chevron-right text-xs'
                "
              />
            </Button>
          </div>
        </div>
        <!-- Import failed list -->
        <div
          v-if="importFailedExpanded"
          data-testid="conflict-dialog-panel-expanded"
          class="flex max-h-35.5 scrollbar-hide flex-col gap-2.5 overflow-y-auto px-4 py-2"
        >
          <div
            v-for="(packageName, i) in importFailedConflicts"
            :key="i"
            :aria-label="`Import failed package: ${packageName}`"
            class="flex min-h-6 shrink-0 hover:bg-node-component-surface-hovered items-center justify-between px-4 py-1"
          >
            <span class="text-xs text-muted">
              {{ packageName }}
            </span>
            <span class="pi pi-info-circle text-sm"></span>
          </div>
        </div>
      </div>
      <!-- Conflict List Wrapper -->
      <div
        v-if="allConflictDetails.length > 0"
        class="flex min-h-8 w-full flex-col rounded-lg bg-base-background"
      >
        <div
          data-testid="conflict-dialog-panel-toggle"
          class="flex h-8 w-full items-center justify-between gap-2 pl-4"
          @click="toggleConflictsPanel"
        >
          <div class="flex flex-1">
            <span class="mr-2 text-xs font-bold text-warning-background">{{
              allConflictDetails.length
            }}</span>
            <span class="text-xs font-bold text-base-foreground">{{
              $t('manager.conflicts.conflicts')
            }}</span>
          </div>
          <div>
            <Button variant="textonly" class="bg-transparent text-muted">
              <i
                :class="
                  conflictsExpanded
                    ? 'pi pi-chevron-down text-xs'
                    : 'pi pi-chevron-right text-xs'
                "
              />
            </Button>
          </div>
        </div>
        <!-- Conflicts list -->
        <div
          v-if="conflictsExpanded"
          data-testid="conflict-dialog-panel-expanded"
          class="flex max-h-[142px] scrollbar-hide flex-col gap-2.5 overflow-y-auto px-4 py-2"
        >
          <div
            v-for="(conflict, i) in allConflictDetails"
            :key="i"
            :aria-label="`Conflict: ${getConflictMessage(conflict, t)}`"
            class="flex min-h-6 shrink-0 hover:bg-node-component-surface-hovered items-center justify-between px-4 py-1"
          >
            <span class="text-xs text-muted">{{
              getConflictMessage(conflict, t)
            }}</span>
            <span class="pi pi-info-circle text-sm"></span>
          </div>
        </div>
      </div>
      <!-- Extension List Wrapper -->
      <div
        v-if="conflictData.length > 0"
        class="flex min-h-8 w-full flex-col rounded-lg bg-base-background"
      >
        <div
          data-testid="conflict-dialog-panel-toggle"
          class="flex h-8 w-full items-center justify-between gap-2 pl-4"
          @click="toggleExtensionsPanel"
        >
          <div class="flex flex-1">
            <span class="mr-2 text-xs font-bold text-warning-background">{{
              conflictData.length
            }}</span>
            <span class="text-xs font-bold text-base-foreground">{{
              $t('manager.conflicts.extensionAtRisk')
            }}</span>
          </div>
          <div>
            <Button variant="textonly" class="bg-transparent text-muted">
              <i
                :class="
                  extensionsExpanded
                    ? 'pi pi-chevron-down text-xs'
                    : 'pi pi-chevron-right text-xs'
                "
              />
            </Button>
          </div>
        </div>
        <!-- Extension list -->
        <div
          v-if="extensionsExpanded"
          data-testid="conflict-dialog-panel-expanded"
          class="flex max-h-35.5 scrollbar-hide flex-col gap-2.5 overflow-y-auto px-4 py-2"
        >
          <div
            v-for="conflictResult in conflictData"
            :key="conflictResult.package_id"
            class="flex min-h-6 shrink-0 hover:bg-node-component-surface-hovered items-center justify-between px-4 py-1"
          >
            <span class="text-xs text-muted">
              {{ conflictResult.package_name }}
            </span>
            <span class="pi pi-info-circle text-sm"></span>
          </div>
        </div>
      </div>
    </div>
    <ContentDivider :width="1" />
  </div>
</template>

<script setup lang="ts">
import { filter, flatMap, map, some } from 'es-toolkit/compat'
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'

import ContentDivider from '@/components/common/ContentDivider.vue'
import Button from '@/components/ui/button/Button.vue'
import { useConflictDetection } from '@/workbench/extensions/manager/composables/useConflictDetection'
import type {
  ConflictDetail,
  ConflictDetectionResult
} from '@/workbench/extensions/manager/types/conflictDetectionTypes'
import { getConflictMessage } from '@/workbench/extensions/manager/utils/conflictMessageUtil'

const { showAfterWhatsNew = false, conflictedPackages } = defineProps<{
  showAfterWhatsNew?: boolean
  conflictedPackages?: ConflictDetectionResult[]
}>()

const { t } = useI18n()
const { conflictedPackages: globalConflictPackages } = useConflictDetection()

const conflictsExpanded = ref<boolean>(false)
const extensionsExpanded = ref<boolean>(false)
const importFailedExpanded = ref<boolean>(false)

const conflictData = computed(
  () => conflictedPackages || globalConflictPackages.value
)

const allConflictDetails = computed(() => {
  const allConflicts = flatMap(
    conflictData.value,
    (result: ConflictDetectionResult) => result.conflicts
  )
  return filter(
    allConflicts,
    (conflict: ConflictDetail) => conflict.type !== 'import_failed'
  )
})

const packagesWithImportFailed = computed(() => {
  return filter(conflictData.value, (result: ConflictDetectionResult) =>
    some(
      result.conflicts,
      (conflict: ConflictDetail) => conflict.type === 'import_failed'
    )
  )
})

const importFailedConflicts = computed(() => {
  return map(
    packagesWithImportFailed.value,
    (result: ConflictDetectionResult) =>
      result.package_name || result.package_id
  )
})

const toggleImportFailedPanel = () => {
  importFailedExpanded.value = !importFailedExpanded.value
  conflictsExpanded.value = false
  extensionsExpanded.value = false
}

const toggleConflictsPanel = () => {
  conflictsExpanded.value = !conflictsExpanded.value
  extensionsExpanded.value = false
  importFailedExpanded.value = false
}

const toggleExtensionsPanel = () => {
  extensionsExpanded.value = !extensionsExpanded.value
  conflictsExpanded.value = false
  importFailedExpanded.value = false
}
</script>
