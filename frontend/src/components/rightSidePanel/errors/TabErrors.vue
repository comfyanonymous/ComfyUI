<template>
  <div class="flex flex-col h-full min-w-0">
    <!-- Search bar -->
    <div
      class="px-4 pt-1 pb-4 flex gap-2 border-b border-interface-stroke shrink-0 min-w-0"
    >
      <FormSearchInput v-model="searchQuery" />
    </div>

    <!-- Scrollable content -->
    <div class="flex-1 overflow-y-auto min-w-0">
      <div
        v-if="filteredGroups.length === 0"
        class="text-sm text-muted-foreground px-4 text-center pt-5 pb-15"
      >
        {{
          searchQuery.trim()
            ? t('rightSidePanel.noneSearchDesc')
            : t('rightSidePanel.noErrors')
        }}
      </div>

      <div v-else>
        <!-- Group by Class Type -->
        <PropertiesAccordionItem
          v-for="group in filteredGroups"
          :key="group.title"
          :collapse="collapseState[group.title] ?? false"
          class="border-b border-interface-stroke"
          :size="
            group.type === 'missing_node' || group.type === 'swap_nodes'
              ? 'lg'
              : 'default'
          "
          @update:collapse="collapseState[group.title] = $event"
        >
          <template #label>
            <div class="flex items-center gap-2 flex-1 min-w-0">
              <span class="flex-1 flex items-center gap-2 min-w-0">
                <i
                  class="icon-[lucide--octagon-alert] size-4 text-destructive-background-hover shrink-0"
                />
                <span class="text-destructive-background-hover truncate">
                  {{
                    group.type === 'missing_node'
                      ? `${group.title} (${missingPackGroups.length})`
                      : group.type === 'swap_nodes'
                        ? `${group.title} (${swapNodeGroups.length})`
                        : group.title
                  }}
                </span>
                <span
                  v-if="group.type === 'execution' && group.cards.length > 1"
                  class="text-destructive-background-hover"
                >
                  ({{ group.cards.length }})
                </span>
              </span>
              <Button
                v-if="
                  group.type === 'missing_node' &&
                  missingNodePacks.length > 0 &&
                  shouldShowInstallButton
                "
                variant="secondary"
                size="sm"
                class="shrink-0 mr-2 h-8 rounded-lg text-sm"
                :disabled="isInstallingAll"
                @click.stop="installAll"
              >
                <DotSpinner v-if="isInstallingAll" duration="1s" :size="12" />
                {{
                  isInstallingAll
                    ? t('rightSidePanel.missingNodePacks.installing')
                    : t('rightSidePanel.missingNodePacks.installAll')
                }}
              </Button>
              <Button
                v-else-if="group.type === 'swap_nodes'"
                v-tooltip.top="
                  t(
                    'nodeReplacement.replaceAllWarning',
                    'Replaces all available nodes in this group.'
                  )
                "
                variant="secondary"
                size="sm"
                class="shrink-0 mr-2 h-8 rounded-lg text-sm"
                @click.stop="handleReplaceAll()"
              >
                {{ t('nodeReplacement.replaceAll', 'Replace All') }}
              </Button>
            </div>
          </template>

          <!-- Missing Node Packs -->
          <MissingNodeCard
            v-if="group.type === 'missing_node'"
            :show-info-button="shouldShowManagerButtons"
            :show-node-id-badge="showNodeIdBadge"
            :missing-pack-groups="missingPackGroups"
            @locate-node="handleLocateMissingNode"
            @open-manager-info="handleOpenManagerInfo"
          />

          <!-- Swap Nodes -->
          <SwapNodesCard
            v-else-if="group.type === 'swap_nodes'"
            :swap-node-groups="swapNodeGroups"
            :show-node-id-badge="showNodeIdBadge"
            @locate-node="handleLocateMissingNode"
          />

          <!-- Execution Errors -->
          <div v-else-if="group.type === 'execution'" class="px-4 space-y-3">
            <ErrorNodeCard
              v-for="card in group.cards"
              :key="card.id"
              :card="card"
              :show-node-id-badge="showNodeIdBadge"
              :compact="isSingleNodeSelected"
              @locate-node="handleLocateNode"
              @enter-subgraph="handleEnterSubgraph"
              @copy-to-clipboard="copyToClipboard"
            />
          </div>
        </PropertiesAccordionItem>
      </div>
    </div>

    <!-- Fixed Footer: Help Links -->
    <div class="shrink-0 border-t border-interface-stroke p-4 min-w-0">
      <i18n-t
        keypath="rightSidePanel.errorHelp"
        tag="p"
        class="m-0 text-sm text-muted-foreground leading-tight break-words"
      >
        <template #github>
          <Button
            variant="textonly"
            size="unset"
            class="inline underline text-inherit text-sm whitespace-nowrap"
            @click="openGitHubIssues"
          >
            {{ t('rightSidePanel.errorHelpGithub') }}
          </Button>
        </template>
        <template #support>
          <Button
            variant="textonly"
            size="unset"
            class="inline underline text-inherit text-sm whitespace-nowrap"
            @click="contactSupport"
          >
            {{ t('rightSidePanel.errorHelpSupport') }}
          </Button>
        </template>
      </i18n-t>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'

import { useCommandStore } from '@/stores/commandStore'
import { useCopyToClipboard } from '@/composables/useCopyToClipboard'
import { useFocusNode } from '@/composables/canvas/useFocusNode'
import { useExternalLink } from '@/composables/useExternalLink'
import { useSettingStore } from '@/platform/settings/settingStore'
import { useTelemetry } from '@/platform/telemetry'
import { useRightSidePanelStore } from '@/stores/workspace/rightSidePanelStore'
import { useManagerState } from '@/workbench/extensions/manager/composables/useManagerState'
import { ManagerTab } from '@/workbench/extensions/manager/types/comfyManagerTypes'
import { NodeBadgeMode } from '@/types/nodeSource'

import PropertiesAccordionItem from '../layout/PropertiesAccordionItem.vue'
import FormSearchInput from '@/renderer/extensions/vueNodes/widgets/components/form/FormSearchInput.vue'
import ErrorNodeCard from './ErrorNodeCard.vue'
import MissingNodeCard from './MissingNodeCard.vue'
import SwapNodesCard from './SwapNodesCard.vue'
import Button from '@/components/ui/button/Button.vue'
import DotSpinner from '@/components/common/DotSpinner.vue'
import { usePackInstall } from '@/workbench/extensions/manager/composables/nodePack/usePackInstall'
import { useMissingNodes } from '@/workbench/extensions/manager/composables/nodePack/useMissingNodes'
import { useErrorGroups } from './useErrorGroups'
import { useNodeReplacement } from '@/platform/nodeReplacement/useNodeReplacement'
import { useExecutionErrorStore } from '@/stores/executionErrorStore'

const { t } = useI18n()
const { copyToClipboard } = useCopyToClipboard()
const { focusNode, enterSubgraph } = useFocusNode()
const { staticUrls } = useExternalLink()
const settingStore = useSettingStore()
const rightSidePanelStore = useRightSidePanelStore()
const { shouldShowManagerButtons, shouldShowInstallButton, openManager } =
  useManagerState()
const { missingNodePacks } = useMissingNodes()
const { isInstalling: isInstallingAll, installAllPacks: installAll } =
  usePackInstall(() => missingNodePacks.value)
const { replaceNodesInPlace } = useNodeReplacement()
const executionErrorStore = useExecutionErrorStore()

const searchQuery = ref('')

const showNodeIdBadge = computed(
  () =>
    (settingStore.get('Comfy.NodeBadge.NodeIdBadgeMode') as NodeBadgeMode) !==
    NodeBadgeMode.None
)

const {
  allErrorGroups,
  filteredGroups,
  collapseState,
  isSingleNodeSelected,
  errorNodeCache,
  missingNodeCache,
  missingPackGroups,
  swapNodeGroups
} = useErrorGroups(searchQuery, t)

/**
 * When an external trigger (e.g. "See Error" button in SectionWidgets)
 * sets focusedErrorNodeId, expand only the group containing the target
 * node and collapse all others so the user sees the relevant errors
 * immediately.
 */
watch(
  () => rightSidePanelStore.focusedErrorNodeId,
  (graphNodeId) => {
    if (!graphNodeId) return
    const prefix = `${graphNodeId}:`
    for (const group of allErrorGroups.value) {
      const hasMatch =
        group.type === 'execution' &&
        group.cards.some(
          (card) =>
            card.graphNodeId === graphNodeId ||
            (card.nodeId?.startsWith(prefix) ?? false)
        )
      collapseState[group.title] = !hasMatch
    }
    rightSidePanelStore.focusedErrorNodeId = null
  },
  { immediate: true }
)

function handleLocateNode(nodeId: string) {
  focusNode(nodeId, errorNodeCache.value)
}

function handleLocateMissingNode(nodeId: string) {
  focusNode(nodeId, missingNodeCache.value)
}

function handleOpenManagerInfo(packId: string) {
  const isKnownToRegistry = missingNodePacks.value.some((p) => p.id === packId)
  if (isKnownToRegistry) {
    openManager({ initialTab: ManagerTab.Missing, initialPackId: packId })
  } else {
    openManager({ initialTab: ManagerTab.All, initialPackId: packId })
  }
}

function handleReplaceAll() {
  const allNodeTypes = swapNodeGroups.value.flatMap((g) => g.nodeTypes)
  const replaced = replaceNodesInPlace(allNodeTypes)
  if (replaced.length > 0) {
    executionErrorStore.removeMissingNodesByType(replaced)
  }
}

function handleEnterSubgraph(nodeId: string) {
  enterSubgraph(nodeId, errorNodeCache.value)
}

function openGitHubIssues() {
  useTelemetry()?.trackUiButtonClicked({
    button_id: 'error_tab_github_issues_clicked'
  })
  window.open(staticUrls.githubIssues, '_blank', 'noopener,noreferrer')
}

async function contactSupport() {
  useTelemetry()?.trackHelpResourceClicked({
    resource_type: 'help_feedback',
    is_external: true,
    source: 'error_dialog'
  })
  useCommandStore().execute('Comfy.ContactSupport')
}
</script>
