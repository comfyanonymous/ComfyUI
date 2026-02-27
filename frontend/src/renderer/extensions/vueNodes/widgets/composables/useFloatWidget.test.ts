import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { INumericWidget } from '@/lib/litegraph/src/types/widgets'
import { _for_testing } from '@/renderer/extensions/vueNodes/widgets/composables/useFloatWidget'

vi.mock('@/scripts/widgets', () => ({
  addValueControlWidgets: vi.fn()
}))

vi.mock('@/platform/settings/settingStore', () => ({
  useSettingStore: () => ({
    settings: {}
  })
}))

const { onFloatValueChange } = _for_testing

describe('useFloatWidget', () => {
  describe('onFloatValueChange', () => {
    let widget: INumericWidget

    beforeEach(() => {
      // Reset the widget before each test
      widget = {
        type: 'number',
        name: 'test_widget',
        y: 0,
        options: {},
        value: 0
      } as Partial<INumericWidget> as INumericWidget
    })

    it('should not round values when round option is not set', () => {
      widget.options.round = undefined
      onFloatValueChange.call(widget, 5.7)
      expect(widget.value).toBe(5.7)
    })

    it('should round values based on round option', () => {
      widget.options.round = 0.5
      onFloatValueChange.call(widget, 5.7)
      expect(widget.value).toBe(5.5)

      widget.options.round = 0.1
      onFloatValueChange.call(widget, 5.74)
      expect(widget.value).toBe(5.7)

      widget.options.round = 1
      onFloatValueChange.call(widget, 5.7)
      expect(widget.value).toBe(6)
    })

    it('should respect min and max constraints after rounding', () => {
      widget.options.round = 0.5
      widget.options.min = 1
      widget.options.max = 5

      // Should round to 1 and respect min
      onFloatValueChange.call(widget, 0.7)
      expect(widget.value).toBe(1)

      // Should round to 5.5 but be clamped to max of 5
      onFloatValueChange.call(widget, 5.3)
      expect(widget.value).toBe(5)

      // Should round to 3.5 and be within bounds
      onFloatValueChange.call(widget, 3.6)
      expect(widget.value).toBe(3.5)
    })

    it('should handle Number.EPSILON for precision issues', () => {
      widget.options.round = 0.1

      // Without Number.EPSILON, 1.35 / 0.1 = 13.499999999999998
      // which would round to 13 * 0.1 = 1.3 instead of 1.4
      onFloatValueChange.call(widget, 1.35)
      expect(widget.value).toBeCloseTo(1.4, 10)

      // Test another edge case
      onFloatValueChange.call(widget, 2.95)
      expect(widget.value).toBeCloseTo(3, 10)
    })
  })
})
