import { ref } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { ComfyNodeDefImpl } from '@/stores/nodeDefStore'

import { useNodePreviewAndDrag } from './useNodePreviewAndDrag'

const mockStartDrag = vi.fn()
const mockHandleNativeDrop = vi.fn()

vi.mock('@/composables/node/useNodeDragToCanvas', () => ({
  useNodeDragToCanvas: () => ({
    startDrag: mockStartDrag,
    handleNativeDrop: mockHandleNativeDrop
  })
}))

vi.mock('@/platform/settings/settingStore', () => ({
  useSettingStore: () => ({
    get: vi.fn().mockReturnValue('left')
  })
}))

describe('useNodePreviewAndDrag', () => {
  const mockNodeDef = {
    name: 'TestNode',
    display_name: 'Test Node'
  } as ComfyNodeDefImpl

  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('initial state', () => {
    it('should initialize with correct default values', () => {
      const nodeDef = ref<ComfyNodeDefImpl | undefined>(mockNodeDef)
      const result = useNodePreviewAndDrag(nodeDef)

      expect(result.isHovered.value).toBe(false)
      expect(result.isDragging.value).toBe(false)
      expect(result.showPreview.value).toBe(false)
      expect(result.previewRef.value).toBeNull()
    })

    it('should compute showPreview based on hover and drag state', () => {
      const nodeDef = ref<ComfyNodeDefImpl | undefined>(mockNodeDef)
      const result = useNodePreviewAndDrag(nodeDef)

      result.isHovered.value = true
      expect(result.showPreview.value).toBe(true)

      result.isDragging.value = true
      expect(result.showPreview.value).toBe(false)
    })
  })

  describe('handleMouseEnter', () => {
    it('should set isHovered to true when nodeDef exists', () => {
      const nodeDef = ref<ComfyNodeDefImpl | undefined>(mockNodeDef)
      const result = useNodePreviewAndDrag(nodeDef)

      const mockElement = document.createElement('div')
      vi.spyOn(mockElement, 'getBoundingClientRect').mockReturnValue({
        top: 100,
        left: 50,
        right: 150,
        bottom: 200,
        width: 100,
        height: 100,
        x: 50,
        y: 100,
        toJSON: () => ({})
      })

      const mockEvent = {
        currentTarget: mockElement
      } as Partial<MouseEvent> as MouseEvent
      result.handleMouseEnter(mockEvent)

      expect(result.isHovered.value).toBe(true)
    })

    it('should not set isHovered when nodeDef is undefined', () => {
      const nodeDef = ref<ComfyNodeDefImpl | undefined>(undefined)
      const result = useNodePreviewAndDrag(nodeDef)

      const mockElement = document.createElement('div')
      const mockEvent = {
        currentTarget: mockElement
      } as Partial<MouseEvent> as MouseEvent
      result.handleMouseEnter(mockEvent)

      expect(result.isHovered.value).toBe(false)
    })
  })

  describe('handleMouseLeave', () => {
    it('should set isHovered to false', () => {
      const nodeDef = ref<ComfyNodeDefImpl | undefined>(mockNodeDef)
      const result = useNodePreviewAndDrag(nodeDef)

      result.isHovered.value = true
      result.handleMouseLeave()

      expect(result.isHovered.value).toBe(false)
    })
  })

  describe('handleDragStart', () => {
    it('should call startDrag with native mode when nodeDef exists', () => {
      const nodeDef = ref<ComfyNodeDefImpl | undefined>(mockNodeDef)
      const result = useNodePreviewAndDrag(nodeDef)

      const mockDataTransfer = {
        effectAllowed: '',
        setData: vi.fn(),
        setDragImage: vi.fn()
      }
      const mockEvent = {
        dataTransfer: mockDataTransfer
      } as unknown as DragEvent

      result.handleDragStart(mockEvent)

      expect(result.isDragging.value).toBe(true)
      expect(result.isHovered.value).toBe(false)
      expect(mockStartDrag).toHaveBeenCalledWith(mockNodeDef, 'native')
      expect(mockDataTransfer.effectAllowed).toBe('copy')
      expect(mockDataTransfer.setData).toHaveBeenCalledWith(
        'application/x-comfy-node',
        'TestNode'
      )
    })

    it('should not start drag when nodeDef is undefined', () => {
      const nodeDef = ref<ComfyNodeDefImpl | undefined>(undefined)
      const result = useNodePreviewAndDrag(nodeDef)

      const mockEvent = { dataTransfer: null } as DragEvent
      result.handleDragStart(mockEvent)

      expect(result.isDragging.value).toBe(false)
      expect(mockStartDrag).not.toHaveBeenCalled()
    })
  })

  describe('handleDragEnd', () => {
    it('should call handleNativeDrop with drop coordinates', () => {
      const nodeDef = ref<ComfyNodeDefImpl | undefined>(mockNodeDef)
      const result = useNodePreviewAndDrag(nodeDef)

      result.isDragging.value = true

      const mockEvent = {
        clientX: 100,
        clientY: 200
      } as Partial<DragEvent> as DragEvent

      result.handleDragEnd(mockEvent)

      expect(result.isDragging.value).toBe(false)
      expect(mockHandleNativeDrop).toHaveBeenCalledWith(100, 200)
    })

    it('should always call handleNativeDrop regardless of dropEffect', () => {
      const nodeDef = ref<ComfyNodeDefImpl | undefined>(mockNodeDef)
      const result = useNodePreviewAndDrag(nodeDef)

      result.isDragging.value = true

      const mockEvent = {
        dataTransfer: { dropEffect: 'none' },
        clientX: 300,
        clientY: 400
      } as Partial<DragEvent> as DragEvent

      result.handleDragEnd(mockEvent)

      expect(result.isDragging.value).toBe(false)
      expect(mockHandleNativeDrop).toHaveBeenCalledWith(300, 400)
    })
  })
})
