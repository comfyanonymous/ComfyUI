/**
 * Build a tooltip configuration object compatible with v-tooltip.
 * Consumers pass the translated text value.
 */
export const buildTooltipConfig = (value: string) => ({
  value,
  showDelay: 300,
  hideDelay: 0,
  pt: {
    text: {
      class:
        'border-node-component-tooltip-border bg-node-component-tooltip-surface text-node-component-tooltip border rounded-md px-2 py-1 text-xs leading-none shadow-none'
    },
    arrow: {
      class: 'border-t-node-component-tooltip-border'
    }
  }
})
