import { describe, expect, it } from 'vitest'

import { CREDITS_PER_USD, formatCredits } from '@/base/credits/comfyCredits'
import {
  evaluateNodeDefPricing,
  formatCreditsListValue,
  formatCreditsRangeValue,
  formatCreditsValue,
  formatPricingResult,
  useNodePricing
} from '@/composables/node/useNodePricing'
import type { LGraphNode } from '@/lib/litegraph/src/litegraph'
import type { ComfyNodeDef, PriceBadge } from '@/schemas/nodeDefSchema'
import { createMockLGraphNode } from '@/utils/__tests__/litegraphTestUtils'

// -----------------------------------------------------------------------------
// Test Types
// -----------------------------------------------------------------------------

interface MockNodeWidget {
  name: string
  value: unknown
  type: string
}

interface MockNodeInput {
  name: string
  link: number | null
}

interface MockNodeData {
  name: string
  api_node: boolean
  price_badge?: PriceBadge
}

// -----------------------------------------------------------------------------
// Test Helpers
// -----------------------------------------------------------------------------

/**
 * Determine if a number should display 1 decimal place.
 * Shows decimal only when the first decimal digit is non-zero.
 */
const shouldShowDecimal = (value: number): boolean => {
  const rounded = Math.round(value * 10) / 10
  return rounded % 1 !== 0
}

const creditValue = (usd: number): string => {
  const rawCredits = usd * CREDITS_PER_USD
  return formatCredits({
    value: rawCredits,
    numberOptions: {
      minimumFractionDigits: 0,
      maximumFractionDigits: shouldShowDecimal(rawCredits) ? 1 : 0
    }
  })
}

const creditsLabel = (usd: number, suffix = '/Run'): string =>
  `${creditValue(usd)} credits${suffix}`

/**
 * Create a mock node with price_badge for testing JSONata-based pricing.
 */
function createMockNodeWithPriceBadge(
  nodeTypeName: string,
  priceBadge: PriceBadge,
  widgets: Array<{ name: string; value: unknown }> = [],
  inputs: Array<{ name: string; connected?: boolean }> = []
): LGraphNode {
  const mockWidgets = widgets.map(({ name, value }) => ({
    name,
    value,
    type: 'combo'
  }))

  const mockInputs: MockNodeInput[] = inputs.map(({ name, connected }) => ({
    name,
    link: connected ? 1 : null
  }))

  const baseNode = createMockLGraphNode()
  return Object.assign(baseNode, {
    widgets: mockWidgets,
    inputs: mockInputs,
    constructor: {
      nodeData: {
        name: nodeTypeName,
        api_node: true,
        price_badge: priceBadge
      }
    }
  })
}

/** Helper to create a price badge with defaults */
const priceBadge = (
  expr: string,
  widgets: Array<{ name: string; type: string }> = [],
  inputs: string[] = [],
  inputGroups: string[] = []
): PriceBadge => ({
  engine: 'jsonata',
  expr,
  depends_on: { widgets, inputs, input_groups: inputGroups }
})

/** Helper to create a mock node for edge case testing */
function createMockNode(
  nodeData: MockNodeData,
  widgets: MockNodeWidget[] = [],
  inputs: MockNodeInput[] = []
): LGraphNode {
  const baseNode = createMockLGraphNode()
  return Object.assign(baseNode, {
    widgets,
    inputs,
    constructor: { nodeData }
  })
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

describe('useNodePricing', () => {
  describe('static expressions', () => {
    it('should evaluate simple static USD price', async () => {
      const { getNodeDisplayPrice } = useNodePricing()
      const node = createMockNodeWithPriceBadge(
        'TestStaticNode',
        priceBadge('{"type":"usd","usd":0.05}')
      )

      getNodeDisplayPrice(node)
      await new Promise((r) => setTimeout(r, 50))
      const price = getNodeDisplayPrice(node)
      expect(price).toBe(creditsLabel(0.05))
    })

    it('should evaluate static text result', async () => {
      const { getNodeDisplayPrice } = useNodePricing()
      const node = createMockNodeWithPriceBadge(
        'TestTextNode',
        priceBadge('{"type":"text","text":"Free"}')
      )

      getNodeDisplayPrice(node)
      await new Promise((r) => setTimeout(r, 50))
      const price = getNodeDisplayPrice(node)
      expect(price).toBe('Free')
    })
  })

  describe('widget value normalization', () => {
    it('should handle INT widget as number', async () => {
      const { getNodeDisplayPrice } = useNodePricing()
      const node = createMockNodeWithPriceBadge(
        'TestIntNode',
        priceBadge('{"type":"usd","usd": widgets.count * 0.01}', [
          { name: 'count', type: 'INT' }
        ]),
        [{ name: 'count', value: 5 }]
      )

      getNodeDisplayPrice(node)
      await new Promise((r) => setTimeout(r, 50))
      const price = getNodeDisplayPrice(node)
      expect(price).toBe(creditsLabel(0.05))
    })

    it('should handle FLOAT widget as number', async () => {
      const { getNodeDisplayPrice } = useNodePricing()
      const node = createMockNodeWithPriceBadge(
        'TestFloatNode',
        priceBadge('{"type":"usd","usd": widgets.rate * 10}', [
          { name: 'rate', type: 'FLOAT' }
        ]),
        [{ name: 'rate', value: 0.05 }]
      )

      getNodeDisplayPrice(node)
      await new Promise((r) => setTimeout(r, 50))
      const price = getNodeDisplayPrice(node)
      expect(price).toBe(creditsLabel(0.5))
    })

    it('should handle COMBO widget with numeric value', async () => {
      const { getNodeDisplayPrice } = useNodePricing()
      const node = createMockNodeWithPriceBadge(
        'TestComboNumericNode',
        priceBadge('{"type":"usd","usd": widgets.duration * 0.07}', [
          { name: 'duration', type: 'COMBO' }
        ]),
        [{ name: 'duration', value: 5 }]
      )

      getNodeDisplayPrice(node)
      await new Promise((r) => setTimeout(r, 50))
      const price = getNodeDisplayPrice(node)
      expect(price).toBe(creditsLabel(0.35))
    })

    it('should handle COMBO widget with string value', async () => {
      const { getNodeDisplayPrice } = useNodePricing()
      const node = createMockNodeWithPriceBadge(
        'TestComboStringNode',
        priceBadge(
          '(widgets.mode = "pro") ? {"type":"usd","usd":0.10} : {"type":"usd","usd":0.05}',
          [{ name: 'mode', type: 'COMBO' }]
        ),
        [{ name: 'mode', value: 'Pro' }] // Should be lowercased to "pro"
      )

      getNodeDisplayPrice(node)
      await new Promise((r) => setTimeout(r, 50))
      const price = getNodeDisplayPrice(node)
      expect(price).toBe(creditsLabel(0.1))
    })

    it('should handle BOOLEAN widget', async () => {
      const { getNodeDisplayPrice } = useNodePricing()
      const node = createMockNodeWithPriceBadge(
        'TestBooleanNode',
        priceBadge('{"type":"usd","usd": widgets.premium ? 0.10 : 0.05}', [
          { name: 'premium', type: 'BOOLEAN' }
        ]),
        [{ name: 'premium', value: true }]
      )

      getNodeDisplayPrice(node)
      await new Promise((r) => setTimeout(r, 50))
      const price = getNodeDisplayPrice(node)
      expect(price).toBe(creditsLabel(0.1))
    })

    it('should handle STRING widget (lowercased)', async () => {
      const { getNodeDisplayPrice } = useNodePricing()
      const node = createMockNodeWithPriceBadge(
        'TestStringNode',
        priceBadge(
          '$contains(widgets.model, "pro") ? {"type":"usd","usd":0.10} : {"type":"usd","usd":0.05}',
          [{ name: 'model', type: 'STRING' }]
        ),
        [{ name: 'model', value: 'ProModel' }] // Should be lowercased to "promodel"
      )

      getNodeDisplayPrice(node)
      await new Promise((r) => setTimeout(r, 50))
      const price = getNodeDisplayPrice(node)
      expect(price).toBe(creditsLabel(0.1))
    })
  })

  describe('complex expressions', () => {
    it('should handle lookup tables', async () => {
      const { getNodeDisplayPrice } = useNodePricing()
      const node = createMockNodeWithPriceBadge(
        'TestLookupNode',
        priceBadge(
          `(
            $rates := {"720p": 0.05, "1080p": 0.10};
            {"type":"usd","usd": $lookup($rates, widgets.resolution)}
          )`,
          [{ name: 'resolution', type: 'COMBO' }]
        ),
        [{ name: 'resolution', value: '1080p' }]
      )

      getNodeDisplayPrice(node)
      await new Promise((r) => setTimeout(r, 50))
      const price = getNodeDisplayPrice(node)
      expect(price).toBe(creditsLabel(0.1))
    })

    it('should handle multiple widgets', async () => {
      const { getNodeDisplayPrice } = useNodePricing()
      const node = createMockNodeWithPriceBadge(
        'TestMultiWidgetNode',
        priceBadge(
          `(
            $rate := (widgets.mode = "pro") ? 0.10 : 0.05;
            {"type":"usd","usd": $rate * widgets.duration}
          )`,
          [
            { name: 'mode', type: 'COMBO' },
            { name: 'duration', type: 'INT' }
          ]
        ),
        [
          { name: 'mode', value: 'pro' },
          { name: 'duration', value: 10 }
        ]
      )

      getNodeDisplayPrice(node)
      await new Promise((r) => setTimeout(r, 50))
      const price = getNodeDisplayPrice(node)
      expect(price).toBe(creditsLabel(1.0))
    })

    it('should handle conditional pricing based on widget values', async () => {
      const { getNodeDisplayPrice } = useNodePricing()
      const node = createMockNodeWithPriceBadge(
        'TestConditionalNode',
        priceBadge(
          `(
            $mode := (widgets.resolution = "720p") ? "std" : "pro";
            $rates := {"std": 0.084, "pro": 0.112};
            {"type":"usd","usd": $lookup($rates, $mode) * widgets.duration}
          )`,
          [
            { name: 'resolution', type: 'COMBO' },
            { name: 'duration', type: 'COMBO' }
          ]
        ),
        [
          { name: 'resolution', value: '1080p' },
          { name: 'duration', value: 5 }
        ]
      )

      getNodeDisplayPrice(node)
      await new Promise((r) => setTimeout(r, 50))
      const price = getNodeDisplayPrice(node)
      expect(price).toBe(creditsLabel(0.56))
    })
  })

  describe('range and list results', () => {
    it('should format range_usd result', async () => {
      const { getNodeDisplayPrice } = useNodePricing()
      const node = createMockNodeWithPriceBadge(
        'TestRangeNode',
        priceBadge('{"type":"range_usd","min_usd":0.05,"max_usd":0.10}')
      )

      getNodeDisplayPrice(node)
      await new Promise((r) => setTimeout(r, 50))
      const price = getNodeDisplayPrice(node)
      expect(price).toMatch(/\d+\.?\d*-\d+\.?\d* credits\/Run/)
    })

    it('should format list_usd result', async () => {
      const { getNodeDisplayPrice } = useNodePricing()
      const node = createMockNodeWithPriceBadge(
        'TestListNode',
        priceBadge('{"type":"list_usd","usd":[0.05, 0.10, 0.15]}')
      )

      getNodeDisplayPrice(node)
      await new Promise((r) => setTimeout(r, 50))
      const price = getNodeDisplayPrice(node)
      expect(price).toMatch(/\d+\.?\d*\/\d+\.?\d*\/\d+\.?\d* credits\/Run/)
    })

    it('should respect custom suffix in format options', async () => {
      const { getNodeDisplayPrice } = useNodePricing()
      const node = createMockNodeWithPriceBadge(
        'TestSuffixNode',
        priceBadge('{"type":"usd","usd":0.07,"format":{"suffix":"/second"}}')
      )

      getNodeDisplayPrice(node)
      await new Promise((r) => setTimeout(r, 50))
      const price = getNodeDisplayPrice(node)
      expect(price).toBe(creditsLabel(0.07, '/second'))
    })

    it('should add approximate prefix when specified', async () => {
      const { getNodeDisplayPrice } = useNodePricing()
      const node = createMockNodeWithPriceBadge(
        'TestApproximateNode',
        priceBadge('{"type":"usd","usd":0.05,"format":{"approximate":true}}')
      )

      getNodeDisplayPrice(node)
      await new Promise((r) => setTimeout(r, 50))
      const price = getNodeDisplayPrice(node)
      expect(price).toMatch(/^~\d+\.?\d* credits\/Run$/)
    })

    it('should add note suffix when specified', async () => {
      const { getNodeDisplayPrice } = useNodePricing()
      const node = createMockNodeWithPriceBadge(
        'TestNoteNode',
        priceBadge('{"type":"usd","usd":0.05,"format":{"note":"(estimated)"}}')
      )

      getNodeDisplayPrice(node)
      await new Promise((r) => setTimeout(r, 50))
      const price = getNodeDisplayPrice(node)
      expect(price).toMatch(/credits\/Run \(estimated\)$/)
    })

    it('should combine approximate prefix and note suffix', async () => {
      const { getNodeDisplayPrice } = useNodePricing()
      const node = createMockNodeWithPriceBadge(
        'TestCombinedFormatNode',
        priceBadge(
          '{"type":"usd","usd":0.05,"format":{"approximate":true,"note":"(beta)","suffix":"/image"}}'
        )
      )

      getNodeDisplayPrice(node)
      await new Promise((r) => setTimeout(r, 50))
      const price = getNodeDisplayPrice(node)
      expect(price).toMatch(/^~\d+\.?\d* credits\/image \(beta\)$/)
    })

    it('should use custom separator for list_usd', async () => {
      const { getNodeDisplayPrice } = useNodePricing()
      const node = createMockNodeWithPriceBadge(
        'TestListSeparatorNode',
        priceBadge(
          '{"type":"list_usd","usd":[0.05, 0.10],"format":{"separator":" or "}}'
        )
      )

      getNodeDisplayPrice(node)
      await new Promise((r) => setTimeout(r, 50))
      const price = getNodeDisplayPrice(node)
      expect(price).toMatch(/\d+\.?\d* or \d+\.?\d* credits\/Run/)
    })
  })

  describe('input connectivity', () => {
    it('should handle connected input check', async () => {
      const { getNodeDisplayPrice } = useNodePricing()
      const node = createMockNodeWithPriceBadge(
        'TestInputNode',
        priceBadge(
          'inputs.image.connected ? {"type":"usd","usd":0.10} : {"type":"usd","usd":0.05}',
          [],
          ['image']
        ),
        [],
        [{ name: 'image', connected: true }]
      )

      getNodeDisplayPrice(node)
      await new Promise((r) => setTimeout(r, 50))
      const price = getNodeDisplayPrice(node)
      expect(price).toBe(creditsLabel(0.1))
    })

    it('should handle disconnected input check', async () => {
      const { getNodeDisplayPrice } = useNodePricing()
      const node = createMockNodeWithPriceBadge(
        'TestInputDisconnectedNode',
        priceBadge(
          'inputs.image.connected ? {"type":"usd","usd":0.10} : {"type":"usd","usd":0.05}',
          [],
          ['image']
        ),
        [],
        [{ name: 'image', connected: false }]
      )

      getNodeDisplayPrice(node)
      await new Promise((r) => setTimeout(r, 50))
      const price = getNodeDisplayPrice(node)
      expect(price).toBe(creditsLabel(0.05))
    })
  })

  describe('edge cases', () => {
    it('should return empty string for non-API nodes', () => {
      const { getNodeDisplayPrice } = useNodePricing()
      const node = createMockNode({
        name: 'RegularNode',
        api_node: false
      })

      const price = getNodeDisplayPrice(node)
      expect(price).toBe('')
    })

    it('should return empty string for nodes without price_badge', () => {
      const { getNodeDisplayPrice } = useNodePricing()
      const node = createMockNode({
        name: 'ApiNodeNoPricing',
        api_node: true
      })

      const price = getNodeDisplayPrice(node)
      expect(price).toBe('')
    })

    it('should handle null widget value gracefully', async () => {
      const { getNodeDisplayPrice } = useNodePricing()
      const node = createMockNodeWithPriceBadge(
        'TestNullWidgetNode',
        priceBadge(
          '{"type":"usd","usd": (widgets.count != null) ? widgets.count * 0.01 : 0.05}',
          [{ name: 'count', type: 'INT' }]
        ),
        [{ name: 'count', value: null }]
      )

      getNodeDisplayPrice(node)
      await new Promise((r) => setTimeout(r, 50))
      const price = getNodeDisplayPrice(node)
      expect(price).toBe(creditsLabel(0.05))
    })

    it('should handle missing widget gracefully', async () => {
      const { getNodeDisplayPrice } = useNodePricing()
      const node = createMockNodeWithPriceBadge(
        'TestMissingWidgetNode',
        priceBadge(
          '{"type":"usd","usd": (widgets.count != null) ? widgets.count * 0.01 : 0.05}',
          [{ name: 'count', type: 'INT' }]
        ),
        []
      )

      getNodeDisplayPrice(node)
      await new Promise((r) => setTimeout(r, 50))
      const price = getNodeDisplayPrice(node)
      expect(price).toBe(creditsLabel(0.05))
    })

    it('should handle undefined widget value gracefully', async () => {
      const { getNodeDisplayPrice } = useNodePricing()
      const node = createMockNodeWithPriceBadge(
        'TestUndefinedWidgetNode',
        priceBadge(
          '{"type":"usd","usd": (widgets.count != null) ? widgets.count * 0.01 : 0.05}',
          [{ name: 'count', type: 'INT' }]
        ),
        [{ name: 'count', value: undefined }]
      )

      getNodeDisplayPrice(node)
      await new Promise((r) => setTimeout(r, 50))
      const price = getNodeDisplayPrice(node)
      expect(price).toBe(creditsLabel(0.05))
    })
  })

  describe('getNodePricingConfig', () => {
    it('should return pricing config for nodes with price_badge', () => {
      const { getNodePricingConfig } = useNodePricing()
      const node = createMockNodeWithPriceBadge(
        'TestConfigNode',
        priceBadge('{"type":"usd","usd":0.05}')
      )

      const config = getNodePricingConfig(node)
      expect(config).toBeDefined()
      expect(config?.engine).toBe('jsonata')
      expect(config?.expr).toBe('{"type":"usd","usd":0.05}')
      expect(config?.depends_on).toBeDefined()
    })

    it('should return undefined for nodes without price_badge', () => {
      const { getNodePricingConfig } = useNodePricing()
      const node = createMockNode({
        name: 'NoPricingNode',
        api_node: true
      })

      const config = getNodePricingConfig(node)
      expect(config).toBeUndefined()
    })

    it('should return undefined for non-API nodes', () => {
      const { getNodePricingConfig } = useNodePricing()
      const node = createMockNode({
        name: 'RegularNode',
        api_node: false
      })

      const config = getNodePricingConfig(node)
      expect(config).toBeUndefined()
    })
  })

  describe('getNodeRevisionRef', () => {
    it('should return a ref for a node ID', () => {
      const { getNodeRevisionRef } = useNodePricing()
      const ref = getNodeRevisionRef('node-1')

      expect(ref).toBeDefined()
      expect(ref.value).toBe(0)
    })

    it('should return the same ref for the same node ID', () => {
      const { getNodeRevisionRef } = useNodePricing()
      const ref1 = getNodeRevisionRef('node-same')
      const ref2 = getNodeRevisionRef('node-same')

      expect(ref1).toBe(ref2)
    })

    it('should return different refs for different node IDs', () => {
      const { getNodeRevisionRef } = useNodePricing()
      const ref1 = getNodeRevisionRef('node-a')
      const ref2 = getNodeRevisionRef('node-b')

      expect(ref1).not.toBe(ref2)
    })

    it('should handle both string and number node IDs', () => {
      const { getNodeRevisionRef } = useNodePricing()
      // Number ID gets stringified, so '123' and 123 should return the same ref
      const refFromNumber = getNodeRevisionRef(123)
      const refFromString = getNodeRevisionRef('123')

      expect(refFromNumber).toBe(refFromString)
    })
  })

  describe('triggerPriceRecalculation', () => {
    it('should not throw for API nodes with price_badge', () => {
      const { triggerPriceRecalculation } = useNodePricing()
      const node = createMockNodeWithPriceBadge(
        'TestTriggerNode',
        priceBadge('{"type":"usd","usd":0.05}')
      )

      expect(() => triggerPriceRecalculation(node)).not.toThrow()
    })

    it('should not throw for non-API nodes', () => {
      const { triggerPriceRecalculation } = useNodePricing()
      const node = createMockNode({
        name: 'RegularNode',
        api_node: false
      })

      expect(() => triggerPriceRecalculation(node)).not.toThrow()
    })
  })

  describe('error handling', () => {
    it('should return empty string for invalid JSONata expression', async () => {
      const { getNodeDisplayPrice } = useNodePricing()
      const node = createMockNodeWithPriceBadge(
        'TestInvalidExprNode',
        // Invalid JSONata syntax (unclosed parenthesis)
        priceBadge('{"type":"usd","usd": (widgets.count * 0.01')
      )

      getNodeDisplayPrice(node)
      await new Promise((r) => setTimeout(r, 50))
      const price = getNodeDisplayPrice(node)
      // Should not crash, just return empty
      expect(price).toBe('')
    })

    it('should return empty string for expression that throws at runtime', async () => {
      const { getNodeDisplayPrice } = useNodePricing()
      const node = createMockNodeWithPriceBadge(
        'TestRuntimeErrorNode',
        // Expression that will fail at runtime (calling function on undefined)
        priceBadge('$lookup(undefined, "key")')
      )

      getNodeDisplayPrice(node)
      await new Promise((r) => setTimeout(r, 50))
      const price = getNodeDisplayPrice(node)
      expect(price).toBe('')
    })

    it('should return empty string for invalid PricingResult type', async () => {
      const { getNodeDisplayPrice } = useNodePricing()
      const node = createMockNodeWithPriceBadge(
        'TestInvalidResultTypeNode',
        // Returns object with invalid type field
        priceBadge('{"type":"invalid_type","value":123}')
      )

      getNodeDisplayPrice(node)
      await new Promise((r) => setTimeout(r, 50))
      const price = getNodeDisplayPrice(node)
      expect(price).toBe('')
    })

    it('should handle legacy format without type field', async () => {
      const { getNodeDisplayPrice } = useNodePricing()
      const node = createMockNodeWithPriceBadge(
        'TestLegacyFormatNode',
        // Returns object without type field (legacy format)
        priceBadge('{"usd":0.05}')
      )

      getNodeDisplayPrice(node)
      await new Promise((r) => setTimeout(r, 50))
      const price = getNodeDisplayPrice(node)
      // Legacy format {usd: number} is supported
      expect(price).toBe(creditsLabel(0.05))
    })

    it('should return empty string for non-object result', async () => {
      const { getNodeDisplayPrice } = useNodePricing()
      const node = createMockNodeWithPriceBadge(
        'TestNonObjectNode',
        // Returns a plain number instead of PricingResult object
        priceBadge('0.05')
      )

      getNodeDisplayPrice(node)
      await new Promise((r) => setTimeout(r, 50))
      const price = getNodeDisplayPrice(node)
      expect(price).toBe('')
    })

    it('should return empty string for null result', async () => {
      const { getNodeDisplayPrice } = useNodePricing()
      const node = createMockNodeWithPriceBadge(
        'TestNullResultNode',
        priceBadge('null')
      )

      getNodeDisplayPrice(node)
      await new Promise((r) => setTimeout(r, 50))
      const price = getNodeDisplayPrice(node)
      expect(price).toBe('')
    })
  })

  describe('input_groups connectivity', () => {
    it('should count connected inputs in a group', async () => {
      const { getNodeDisplayPrice } = useNodePricing()

      // Create a node with autogrow-style inputs (group.input1, group.input2, etc.)
      const node = createMockNode(
        {
          name: 'TestInputGroupNode',
          api_node: true,
          price_badge: {
            engine: 'jsonata',
            expr: '{"type":"usd","usd": inputGroups.videos * 0.05}',
            depends_on: {
              widgets: [],
              inputs: [],
              input_groups: ['videos']
            }
          }
        },
        [],
        [
          { name: 'videos.clip1', link: 1 }, // connected
          { name: 'videos.clip2', link: 2 }, // connected
          { name: 'videos.clip3', link: null }, // disconnected
          { name: 'other_input', link: 3 } // connected but not in group
        ]
      )

      getNodeDisplayPrice(node)
      await new Promise((r) => setTimeout(r, 50))
      const price = getNodeDisplayPrice(node)
      // 2 connected inputs in 'videos' group * 0.05 = 0.10
      expect(price).toBe(creditsLabel(0.1))
    })
  })

  describe('decimal formatting', () => {
    describe('shouldShowDecimal helper', () => {
      it('should return true when first decimal digit is non-zero', () => {
        expect(shouldShowDecimal(10.5)).toBe(true)
        expect(shouldShowDecimal(10.1)).toBe(true)
        expect(shouldShowDecimal(10.9)).toBe(true)
        expect(shouldShowDecimal(1.5)).toBe(true)
      })

      it('should return false for whole numbers', () => {
        expect(shouldShowDecimal(10)).toBe(false)
        expect(shouldShowDecimal(10.0)).toBe(false)
        expect(shouldShowDecimal(1)).toBe(false)
        expect(shouldShowDecimal(100)).toBe(false)
      })

      it('should return false when decimal rounds to zero', () => {
        // 10.04 rounds to 10.0, so no decimal shown
        expect(shouldShowDecimal(10.04)).toBe(false)
        expect(shouldShowDecimal(10.049)).toBe(false)
      })

      it('should return true when decimal rounds to non-zero', () => {
        // 10.05 rounds to 10.1, so decimal shown
        expect(shouldShowDecimal(10.05)).toBe(true)
        expect(shouldShowDecimal(10.06)).toBe(true)
        // 10.45 rounds to 10.5
        expect(shouldShowDecimal(10.45)).toBe(true)
      })
    })

    describe('credit value formatting', () => {
      it('should show decimal for USD values that result in fractional credits', () => {
        // $0.05 * 211 = 10.55 credits → "10.6"
        const value1 = creditValue(0.05)
        expect(value1).toBe('10.6')

        // $0.10 * 211 = 21.1 credits → "21.1"
        const value2 = creditValue(0.1)
        expect(value2).toBe('21.1')
      })

      it('should not show decimal for USD values that result in whole credits', () => {
        // $1.00 * 211 = 211 credits → "211"
        const value = creditValue(1.0)
        expect(value).toBe('211')
      })

      it('should not show decimal when credits round to whole number', () => {
        // Find a USD value that results in credits close to a whole number
        // $0.0473933... * 211 ≈ 10.0 credits
        // Let's use a value that gives us ~10.02 credits which rounds to 10.0
        const usd = 10.02 / CREDITS_PER_USD // ~0.0475 USD → ~10.02 credits
        const value = creditValue(usd)
        expect(value).toBe('10')
      })
    })

    describe('integration with pricing display', () => {
      it('should display decimal in badge for fractional credits', async () => {
        const { getNodeDisplayPrice } = useNodePricing()
        // $0.05 * 211 = 10.55 credits → "10.6 credits/Run"
        const node = createMockNodeWithPriceBadge(
          'TestDecimalNode',
          priceBadge('{"type":"usd","usd":0.05}')
        )

        getNodeDisplayPrice(node)
        await new Promise((r) => setTimeout(r, 50))
        const price = getNodeDisplayPrice(node)
        expect(price).toBe('10.6 credits/Run')
      })

      it('should not display decimal in badge for whole credits', async () => {
        const { getNodeDisplayPrice } = useNodePricing()
        // $1.00 * 211 = 211 credits → "211 credits/Run"
        const node = createMockNodeWithPriceBadge(
          'TestWholeCreditsNode',
          priceBadge('{"type":"usd","usd":1.00}')
        )

        getNodeDisplayPrice(node)
        await new Promise((r) => setTimeout(r, 50))
        const price = getNodeDisplayPrice(node)
        expect(price).toBe('211 credits/Run')
      })

      it('should handle range with mixed decimal display', async () => {
        const { getNodeDisplayPrice } = useNodePricing()
        // min: $0.05 * 211 = 10.55 → 10.6
        // max: $1.00 * 211 = 211 → 211
        const node = createMockNodeWithPriceBadge(
          'TestMixedRangeNode',
          priceBadge('{"type":"range_usd","min_usd":0.05,"max_usd":1.00}')
        )

        getNodeDisplayPrice(node)
        await new Promise((r) => setTimeout(r, 50))
        const price = getNodeDisplayPrice(node)
        expect(price).toBe('10.6-211 credits/Run')
      })
    })
  })
})

// -----------------------------------------------------------------------------
// formatPricingResult Tests
// -----------------------------------------------------------------------------

describe('formatPricingResult', () => {
  describe('type: usd', () => {
    it('should format usd result', () => {
      const result = formatPricingResult({ type: 'usd', usd: 0.05 })
      expect(result).toBe('10.6 credits/Run')
    })

    it('should return valueOnly format', () => {
      const result = formatPricingResult(
        { type: 'usd', usd: 0.05 },
        { valueOnly: true }
      )
      expect(result).toBe('10.6')
    })

    it('should handle approximate prefix in valueOnly mode', () => {
      const result = formatPricingResult(
        { type: 'usd', usd: 0.05, format: { approximate: true } },
        { valueOnly: true }
      )
      expect(result).toBe('~10.6')
    })

    it('should return empty for null usd', () => {
      const result = formatPricingResult({ type: 'usd', usd: null as never })
      expect(result).toBe('')
    })
  })

  describe('type: range_usd', () => {
    it('should format range result', () => {
      const result = formatPricingResult({
        type: 'range_usd',
        min_usd: 0.05,
        max_usd: 0.1
      })
      expect(result).toBe('10.6-21.1 credits/Run')
    })

    it('should return valueOnly format', () => {
      const result = formatPricingResult(
        { type: 'range_usd', min_usd: 0.05, max_usd: 0.1 },
        { valueOnly: true }
      )
      expect(result).toBe('10.6-21.1')
    })

    it('should collapse range when min equals max', () => {
      const result = formatPricingResult(
        { type: 'range_usd', min_usd: 0.05, max_usd: 0.05 },
        { valueOnly: true }
      )
      expect(result).toBe('10.6')
    })
  })

  describe('type: list_usd', () => {
    it('should format list result', () => {
      const result = formatPricingResult({
        type: 'list_usd',
        usd: [0.05, 0.1, 0.15]
      })
      expect(result).toMatch(/\d+\.?\d*\/\d+\.?\d*\/\d+\.?\d* credits\/Run/)
    })

    it('should return valueOnly format', () => {
      const result = formatPricingResult(
        { type: 'list_usd', usd: [0.05, 0.1] },
        { valueOnly: true }
      )
      expect(result).toBe('10.6/21.1')
    })
  })

  describe('type: text', () => {
    it('should return text as-is', () => {
      const result = formatPricingResult({ type: 'text', text: 'Free' })
      expect(result).toBe('Free')
    })
  })

  describe('legacy format', () => {
    it('should handle {usd: number} without type field', () => {
      const result = formatPricingResult({ usd: 0.05 })
      expect(result).toBe('10.6 credits/Run')
    })

    it('should return valueOnly for legacy format', () => {
      const result = formatPricingResult({ usd: 0.05 }, { valueOnly: true })
      expect(result).toBe('10.6')
    })
  })

  describe('invalid inputs', () => {
    it('should return empty for invalid type', () => {
      const result = formatPricingResult({ type: 'invalid' })
      expect(result).toBe('')
    })

    it('should return empty for null', () => {
      const result = formatPricingResult(null)
      expect(result).toBe('')
    })

    it('should return empty for undefined', () => {
      const result = formatPricingResult(undefined)
      expect(result).toBe('')
    })
  })
})

// -----------------------------------------------------------------------------
// formatCreditsValue / Range / List Tests
// -----------------------------------------------------------------------------

describe('formatCreditsValue', () => {
  it('should format USD to credits', () => {
    expect(formatCreditsValue(0.05)).toBe('10.6')
    expect(formatCreditsValue(1.0)).toBe('211')
  })
})

describe('formatCreditsRangeValue', () => {
  it('should format min-max range', () => {
    expect(formatCreditsRangeValue(0.05, 0.1)).toBe('10.6-21.1')
  })

  it('should collapse when min equals max', () => {
    expect(formatCreditsRangeValue(0.05, 0.05)).toBe('10.6')
  })
})

describe('formatCreditsListValue', () => {
  it('should join values with separator', () => {
    expect(formatCreditsListValue([0.05, 0.1])).toBe('10.6/21.1')
  })

  it('should use custom separator', () => {
    expect(formatCreditsListValue([0.05, 0.1], ' | ')).toBe('10.6 | 21.1')
  })
})

// -----------------------------------------------------------------------------
// evaluateNodeDefPricing Tests
// -----------------------------------------------------------------------------

describe('evaluateNodeDefPricing', () => {
  const createMockNodeDef = (
    overrides: Partial<ComfyNodeDef> = {}
  ): ComfyNodeDef =>
    ({
      name: 'TestNode',
      display_name: 'Test Node',
      description: '',
      category: 'test',
      input: { required: {}, optional: {} },
      output: [],
      output_name: [],
      output_is_list: [],
      python_module: 'test',
      ...overrides
    }) as ComfyNodeDef

  it('should return empty for node without price_badge', async () => {
    const nodeDef = createMockNodeDef()
    const result = await evaluateNodeDefPricing(nodeDef)
    expect(result).toBe('')
  })

  it('should evaluate static expression', async () => {
    const nodeDef = createMockNodeDef({
      name: 'StaticPriceNode',
      price_badge: {
        engine: 'jsonata',
        expr: '{"type":"usd","usd":0.05}',
        depends_on: { widgets: [], inputs: [], input_groups: [] }
      }
    })
    const result = await evaluateNodeDefPricing(nodeDef)
    expect(result).toBe('10.6')
  })

  it('should use default value from input spec', async () => {
    const nodeDef = createMockNodeDef({
      name: 'DefaultValueNode',
      price_badge: {
        engine: 'jsonata',
        expr: '{"type":"usd","usd": widgets.count * 0.01}',
        depends_on: {
          widgets: [{ name: 'count', type: 'INT' }],
          inputs: [],
          input_groups: []
        }
      },
      input: {
        required: {
          count: ['INT', { default: 10 }]
        }
      }
    })
    const result = await evaluateNodeDefPricing(nodeDef)
    expect(result).toBe('21.1') // 10 * 0.01 = 0.1 USD = 21.1 credits
  })

  it('should use first option for COMBO without default', async () => {
    const nodeDef = createMockNodeDef({
      name: 'ComboNode',
      price_badge: {
        engine: 'jsonata',
        expr: '(widgets.mode = "pro") ? {"type":"usd","usd":0.10} : {"type":"usd","usd":0.05}',
        depends_on: {
          widgets: [{ name: 'mode', type: 'COMBO' }],
          inputs: [],
          input_groups: []
        }
      },
      input: {
        required: {
          mode: [['standard', 'pro'], {}]
        }
      }
    })
    const result = await evaluateNodeDefPricing(nodeDef)
    // First option is "standard", not "pro", so should be 0.05 USD
    expect(result).toBe('10.6')
  })

  it('should use "original" as fallback for dynamic COMBO without input', async () => {
    const nodeDef = createMockNodeDef({
      name: 'DynamicComboNode',
      price_badge: {
        engine: 'jsonata',
        expr: `(
          $prices := {"original": 0.05, "720p": 0.03};
          {"type":"usd","usd": $lookup($prices, widgets.resolution)}
        )`,
        depends_on: {
          widgets: [{ name: 'resolution', type: 'COMBO' }],
          inputs: [],
          input_groups: []
        }
      },
      input: {
        required: {
          // resolution widget is NOT in inputs (dynamically created)
        }
      }
    })
    const result = await evaluateNodeDefPricing(nodeDef)
    // Fallback to "original" = 0.05 USD
    expect(result).toBe('10.6')
  })

  it('should handle dynamic combo with options array', async () => {
    const nodeDef = createMockNodeDef({
      name: 'DynamicOptionsNode',
      price_badge: {
        engine: 'jsonata',
        expr: '{"type":"usd","usd": widgets.model = "model_a" ? 0.05 : 0.10}',
        depends_on: {
          widgets: [{ name: 'model', type: 'COMFY_DYNAMICCOMBO_V3' }],
          inputs: [],
          input_groups: []
        }
      },
      input: {
        required: {
          model: [
            'COMFY_DYNAMICCOMBO_V3',
            { options: [{ key: 'model_a' }, { key: 'model_b' }] }
          ]
        }
      }
    })
    const result = await evaluateNodeDefPricing(nodeDef)
    // First option key is "model_a" = 0.05 USD
    expect(result).toBe('10.6')
  })

  it('should assume inputs disconnected in preview', async () => {
    const nodeDef = createMockNodeDef({
      name: 'InputConnectedNode',
      price_badge: {
        engine: 'jsonata',
        expr: 'inputs.image.connected ? {"type":"usd","usd":0.10} : {"type":"usd","usd":0.05}',
        depends_on: {
          widgets: [],
          inputs: ['image'],
          input_groups: []
        }
      }
    })
    const result = await evaluateNodeDefPricing(nodeDef)
    // In preview, inputs are assumed disconnected
    expect(result).toBe('10.6')
  })

  it('should assume inputGroups have 0 count in preview', async () => {
    const nodeDef = createMockNodeDef({
      name: 'InputGroupNode',
      price_badge: {
        engine: 'jsonata',
        expr: '{"type":"usd","usd": 0.05 + inputGroups.videos * 0.02}',
        depends_on: {
          widgets: [],
          inputs: [],
          input_groups: ['videos']
        }
      }
    })
    const result = await evaluateNodeDefPricing(nodeDef)
    // 0.05 + 0 * 0.02 = 0.05 USD
    expect(result).toBe('10.6')
  })

  it('should return empty on JSONata error', async () => {
    const nodeDef = createMockNodeDef({
      name: 'ErrorNode',
      price_badge: {
        engine: 'jsonata',
        expr: '$lookup(undefined, "key")',
        depends_on: { widgets: [], inputs: [], input_groups: [] }
      }
    })
    const result = await evaluateNodeDefPricing(nodeDef)
    expect(result).toBe('')
  })

  it('should handle range_usd result', async () => {
    const nodeDef = createMockNodeDef({
      name: 'RangeNode',
      price_badge: {
        engine: 'jsonata',
        expr: '{"type":"range_usd","min_usd":0.05,"max_usd":0.10}',
        depends_on: { widgets: [], inputs: [], input_groups: [] }
      }
    })
    const result = await evaluateNodeDefPricing(nodeDef)
    expect(result).toBe('10.6-21.1')
  })

  it('should handle approximate format in valueOnly mode', async () => {
    const nodeDef = createMockNodeDef({
      name: 'ApproximateNode',
      price_badge: {
        engine: 'jsonata',
        expr: '{"type":"usd","usd":0.05,"format":{"approximate":true}}',
        depends_on: { widgets: [], inputs: [], input_groups: [] }
      }
    })
    const result = await evaluateNodeDefPricing(nodeDef)
    expect(result).toBe('~10.6')
  })
})
