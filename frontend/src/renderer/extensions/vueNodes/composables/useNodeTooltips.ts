import type {
  TooltipOptions,
  TooltipPassThroughMethodOptions
} from 'primevue/tooltip'
import { computed, ref, unref } from 'vue'
import type { MaybeRef } from 'vue'

import type { SafeWidgetData } from '@/composables/graph/useGraphNodeManager'
import { st } from '@/i18n'
import { useSettingStore } from '@/platform/settings/settingStore'
import { useNodeDefStore } from '@/stores/nodeDefStore'
import { normalizeI18nKey } from '@/utils/formatUtil'
import { cn } from '@/utils/tailwindUtil'

// PrimeVue adds this internal property to elements with tooltips
interface PrimeVueTooltipElement extends Element {
  $_ptooltipId?: string
}

/**
 * Hide all visible tooltips by dispatching mouseleave events
 *
 *
 * IMPORTANT: this escape is needed for many reason due to primevue's directive tooltip system.
 * We cannot use PT to conditionally render the tooltips because the entire PT object only run
 * once during the initialization of the directive not every mount/unmount.
 * Once the directive is constructed its no longer reactive in the traditional sense.
 * We have to use something non destructive like mouseevents to dismiss the tooltip.
 *
 * TODO: use a better tooltip component like RekaUI for vue nodes specifically.
 */

const tooltipsTemporarilyDisabled = ref(false)

const hideTooltipsGlobally = () => {
  // Get all visible tooltip elements
  const tooltips = document.querySelectorAll('.p-tooltip')

  // Early return if no tooltips are visible
  if (tooltips.length === 0) return

  tooltips.forEach((tooltipEl) => {
    const tooltipId = tooltipEl.id
    if (!tooltipId) return

    // Find the target element that owns this tooltip
    const targetElements = document.querySelectorAll('[data-pd-tooltip="true"]')
    for (const targetEl of targetElements) {
      if ((targetEl as PrimeVueTooltipElement).$_ptooltipId === tooltipId) {
        ;(targetEl as HTMLElement).dispatchEvent(
          new MouseEvent('mouseleave', { bubbles: true })
        )
        break
      }
    }
  })

  // Disable tooltips temporarily after hiding (for drag operations)
  tooltipsTemporarilyDisabled.value = true
}

/**
 * Re-enable tooltips after pointer interaction ends
 */
const handlePointerUp = () => {
  tooltipsTemporarilyDisabled.value = false
}

// Global tooltip hiding system
const globalTooltipState = { listenersSetup: false }

function setupGlobalTooltipHiding() {
  if (globalTooltipState.listenersSetup) return

  document.addEventListener('pointerdown', hideTooltipsGlobally)
  document.addEventListener('pointerup', handlePointerUp)
  window.addEventListener('wheel', hideTooltipsGlobally, {
    capture: true, //Need this to bypass the event layer from Litegraph
    passive: true
  })

  globalTooltipState.listenersSetup = true
}

/**
 * Composable for managing Vue node tooltips
 * Provides tooltip text for node headers, slots, and widgets
 */
export function useNodeTooltips(nodeType: MaybeRef<string>) {
  const nodeDefStore = useNodeDefStore()
  const settingsStore = useSettingStore()

  // Setup global pointerdown listener once
  setupGlobalTooltipHiding()

  // Check if tooltips are globally enabled
  const tooltipsEnabled = computed(() =>
    settingsStore.get('Comfy.EnableTooltips')
  )

  // Get node definition for tooltip data
  const nodeDef = computed(() => nodeDefStore.nodeDefsByName[unref(nodeType)])

  /**
   * Get tooltip text for node description (header hover)
   */
  const getNodeDescription = computed(() => {
    if (!tooltipsEnabled.value || !nodeDef.value) return ''

    const key = `nodeDefs.${normalizeI18nKey(unref(nodeType))}.description`
    return st(key, nodeDef.value.description || '')
  })

  /**
   * Get tooltip text for input slots
   */
  const getInputSlotTooltip = (slotName: string) => {
    if (!tooltipsEnabled.value || !nodeDef.value) return ''

    const key = `nodeDefs.${normalizeI18nKey(unref(nodeType))}.inputs.${normalizeI18nKey(slotName)}.tooltip`
    const inputTooltip = nodeDef.value.inputs?.[slotName]?.tooltip ?? ''
    return st(key, inputTooltip)
  }

  /**
   * Get tooltip text for output slots
   */
  const getOutputSlotTooltip = (slotIndex: number) => {
    if (!tooltipsEnabled.value || !nodeDef.value) return ''

    const key = `nodeDefs.${normalizeI18nKey(unref(nodeType))}.outputs.${slotIndex}.tooltip`
    const outputTooltip = nodeDef.value.outputs?.[slotIndex]?.tooltip ?? ''
    return st(key, outputTooltip)
  }

  /**
   * Get tooltip text for widgets
   */
  const getWidgetTooltip = (widget: SafeWidgetData) => {
    if (!tooltipsEnabled.value || !nodeDef.value) return ''

    // First try widget-specific tooltip
    const widgetTooltip = (widget as { tooltip?: string }).tooltip
    if (widgetTooltip) return widgetTooltip

    // Then try input-based tooltip lookup
    const key = `nodeDefs.${normalizeI18nKey(unref(nodeType))}.inputs.${normalizeI18nKey(widget.name)}.tooltip`
    const inputTooltip = nodeDef.value.inputs?.[widget.name]?.tooltip ?? ''
    return st(key, inputTooltip)
  }

  /**
   * Create tooltip configuration object for v-tooltip directive
   * Components wrap this in computed() for reactivity
   */
  const createTooltipConfig = (text: string): TooltipOptions => {
    const tooltipDelay = settingsStore.get('LiteGraph.Node.TooltipDelay')
    const tooltipText = text || ''

    return {
      value: tooltipText,
      showDelay: tooltipDelay as number,
      hideDelay: 0, // Immediate hiding
      disabled:
        !tooltipsEnabled.value ||
        !tooltipText ||
        tooltipsTemporarilyDisabled.value, // this reactive value works but only on next mount,
      // so if the tooltip is already visible changing this will not hide it
      pt: {
        text: {
          class:
            'border-node-component-tooltip-border bg-node-component-tooltip-surface border rounded-md px-4 py-2 text-node-component-tooltip text-sm font-normal leading-tight max-w-75 shadow-none'
        },
        arrow: ({ context }: TooltipPassThroughMethodOptions) => ({
          class: cn(
            context.top && 'border-t-node-component-tooltip-border',
            context.bottom && 'border-b-node-component-tooltip-border',
            context.left && 'border-l-node-component-tooltip-border ',
            context.right && 'border-r-node-component-tooltip-border'
          )
        })
      }
    }
  }

  return {
    tooltipsEnabled,
    getNodeDescription,
    getInputSlotTooltip,
    getOutputSlotTooltip,
    getWidgetTooltip,
    createTooltipConfig
  }
}
