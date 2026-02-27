import _ from 'es-toolkit/compat'
import { computed, onMounted, watch } from 'vue'

import { useNodePricing } from '@/composables/node/useNodePricing'
import { usePriceBadge } from '@/composables/node/usePriceBadge'
import { useComputedWithWidgetWatch } from '@/composables/node/useWatchWidget'
import { BadgePosition, LGraphBadge } from '@/lib/litegraph/src/litegraph'
import type { LGraphNode } from '@/lib/litegraph/src/litegraph'
import { useSettingStore } from '@/platform/settings/settingStore'
import { app } from '@/scripts/app'
import { useExtensionStore } from '@/stores/extensionStore'
import type { ComfyNodeDefImpl } from '@/stores/nodeDefStore'
import { useNodeDefStore } from '@/stores/nodeDefStore'
import { useColorPaletteStore } from '@/stores/workspace/colorPaletteStore'
import { NodeBadgeMode } from '@/types/nodeSource'

/**
 * Add LGraphBadge to LGraphNode based on settings.
 *
 * Following badges are added:
 * - Node ID badge
 * - Node source badge
 * - Node life cycle badge
 * - API node credits badge
 */
export const useNodeBadge = () => {
  const settingStore = useSettingStore()
  const extensionStore = useExtensionStore()
  const colorPaletteStore = useColorPaletteStore()
  const priceBadge = usePriceBadge()

  const nodeSourceBadgeMode = computed(
    () =>
      settingStore.get('Comfy.NodeBadge.NodeSourceBadgeMode') as NodeBadgeMode
  )
  const nodeIdBadgeMode = computed(
    () => settingStore.get('Comfy.NodeBadge.NodeIdBadgeMode') as NodeBadgeMode
  )
  const nodeLifeCycleBadgeMode = computed(
    () =>
      settingStore.get(
        'Comfy.NodeBadge.NodeLifeCycleBadgeMode'
      ) as NodeBadgeMode
  )

  const showApiPricingBadge = computed(() =>
    settingStore.get('Comfy.NodeBadge.ShowApiPricing')
  )

  watch(
    [
      nodeSourceBadgeMode,
      nodeIdBadgeMode,
      nodeLifeCycleBadgeMode,
      showApiPricingBadge
    ],
    () => {
      app.canvas?.setDirty(true, true)
    }
  )

  const nodeDefStore = useNodeDefStore()
  function badgeTextVisible(
    nodeDef: ComfyNodeDefImpl | null,
    badgeMode: NodeBadgeMode
  ): boolean {
    return !(
      badgeMode === NodeBadgeMode.None ||
      (nodeDef?.isCoreNode && badgeMode === NodeBadgeMode.HideBuiltIn)
    )
  }

  onMounted(() => {
    if (extensionStore.isExtensionInstalled('Comfy.NodeBadge')) return

    // TODO: Fix the composables and watchers being setup in onMounted
    const nodePricing = useNodePricing()

    watch(
      () => nodePricing.pricingRevision.value,
      () => {
        if (!showApiPricingBadge.value) return
        app.canvas?.setDirty(true, true)
      }
    )

    extensionStore.registerExtension({
      name: 'Comfy.NodeBadge',
      nodeCreated(node: LGraphNode) {
        node.badgePosition = BadgePosition.TopRight

        const badge = computed(() => {
          const nodeDef = nodeDefStore.fromLGraphNode(node)
          return new LGraphBadge({
            text: _.truncate(
              [
                badgeTextVisible(nodeDef, nodeIdBadgeMode.value)
                  ? `#${node.id}`
                  : '',
                badgeTextVisible(nodeDef, nodeLifeCycleBadgeMode.value)
                  ? (nodeDef?.nodeLifeCycleBadgeText ?? '')
                  : '',
                badgeTextVisible(nodeDef, nodeSourceBadgeMode.value)
                  ? (nodeDef?.nodeSource?.badgeText ?? '')
                  : ''
              ]
                .filter((s) => s.length > 0)
                .join(' '),
              {
                length: 31
              }
            ),
            fgColor:
              colorPaletteStore.completedActivePalette.colors.litegraph_base
                .BADGE_FG_COLOR,
            bgColor:
              colorPaletteStore.completedActivePalette.colors.litegraph_base
                .BADGE_BG_COLOR
          })
        })

        node.badges.push(() => badge.value)

        if (node.constructor.nodeData?.api_node && showApiPricingBadge.value) {
          // JSONata rules are dynamic if they depend on any widgets/inputs/input_groups
          const pricingConfig = nodePricing.getNodePricingConfig(node)
          const hasDynamicPricing =
            !!pricingConfig &&
            ((pricingConfig.depends_on?.widgets?.length ?? 0) > 0 ||
              (pricingConfig.depends_on?.inputs?.length ?? 0) > 0 ||
              (pricingConfig.depends_on?.input_groups?.length ?? 0) > 0)

          // Keep the existing widget-watch wiring ONLY to trigger redraws on widget change.
          // (We no longer rely on it to hold the current badge value.)
          if (hasDynamicPricing) {
            // For dynamic pricing nodes, use computed that watches widget changes
            const relevantWidgetNames = nodePricing.getRelevantWidgetNames(
              node.constructor.nodeData?.name
            )

            const computedWithWidgetWatch = useComputedWithWidgetWatch(node, {
              widgetNames: relevantWidgetNames,
              triggerCanvasRedraw: true
            })

            // Ensure watchers are installed; ignore the returned value.
            // (This call is what registers the widget listeners in most implementations.)
            computedWithWidgetWatch(() => 0)

            // Hook into connection changes to trigger price recalculation
            // This handles both connect and disconnect in VueNodes mode
            const relevantInputs = pricingConfig?.depends_on?.inputs ?? []
            const inputGroupPrefixes =
              pricingConfig?.depends_on?.input_groups ?? []
            const hasRelevantInputs =
              relevantInputs.length > 0 || inputGroupPrefixes.length > 0

            if (hasRelevantInputs) {
              const originalOnConnectionsChange = node.onConnectionsChange
              node.onConnectionsChange = function (
                type,
                slotIndex,
                isConnected,
                link,
                ioSlot
              ) {
                originalOnConnectionsChange?.call(
                  this,
                  type,
                  slotIndex,
                  isConnected,
                  link,
                  ioSlot
                )
                // Only trigger if this input affects pricing
                const inputName = ioSlot?.name
                if (!inputName) return
                const isRelevantInput =
                  relevantInputs.includes(inputName) ||
                  inputGroupPrefixes.some((prefix) =>
                    inputName.startsWith(prefix + '.')
                  )
                if (isRelevantInput) {
                  nodePricing.triggerPriceRecalculation(node)
                }
              }
            }
          }

          let lastLabel = nodePricing.getNodeDisplayPrice(node)
          let lastBadge = priceBadge.getCreditsBadge(lastLabel)

          const creditsBadgeGetter: () => LGraphBadge = () => {
            const label = nodePricing.getNodeDisplayPrice(node)
            if (label !== lastLabel) {
              lastLabel = label
              lastBadge = priceBadge.getCreditsBadge(label)
            }
            return lastBadge
          }

          node.badges.push(creditsBadgeGetter)
        }
      },
      init() {
        app.canvas.canvas.addEventListener<'litegraph:set-graph'>(
          'litegraph:set-graph',
          () => {
            for (const node of app.canvas.graph?.nodes ?? [])
              priceBadge.updateSubgraphCredits(node)
          }
        )
        app.canvas.canvas.addEventListener<'subgraph-converted'>(
          'subgraph-converted',
          (e) => priceBadge.updateSubgraphCredits(e.detail.subgraphNode)
        )
      },
      afterConfigureGraph() {
        for (const node of app.canvas.graph?.nodes ?? [])
          priceBadge.updateSubgraphCredits(node)
      }
    })
  })
}
