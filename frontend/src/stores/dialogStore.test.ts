import { createTestingPinia } from '@pinia/testing'
import { setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'
import { defineComponent } from 'vue'

import { useDialogStore } from '@/stores/dialogStore'

const MockComponent = defineComponent({
  name: 'MockComponent',
  template: '<div>Mock</div>'
})

describe('dialogStore', () => {
  beforeEach(() => {
    setActivePinia(createTestingPinia({ stubActions: false }))
  })

  describe('priority system', () => {
    it('should create dialogs in correct priority order', () => {
      const store = useDialogStore()

      // Create dialogs with different priorities
      store.showDialog({
        key: 'low-priority',
        component: MockComponent,
        priority: 0
      })

      store.showDialog({
        key: 'high-priority',
        component: MockComponent,
        priority: 10
      })

      store.showDialog({
        key: 'medium-priority',
        component: MockComponent,
        priority: 5
      })

      store.showDialog({
        key: 'no-priority',
        component: MockComponent
      })

      // Check order: high (2) -> medium (1) -> low (0)
      expect(store.dialogStack.map((d) => d.key)).toEqual([
        'high-priority',
        'medium-priority',
        'no-priority',
        'low-priority'
      ])
    })

    it('should maintain priority order when rising dialogs', () => {
      const store = useDialogStore()

      // Create dialogs with different priorities
      store.showDialog({
        key: 'priority-2',
        component: MockComponent,
        priority: 2
      })

      store.showDialog({
        key: 'priority-1',
        component: MockComponent,
        priority: 1
      })

      store.showDialog({
        key: 'priority-0',
        component: MockComponent,
        priority: 0
      })

      // Try to rise the lowest priority dialog
      store.riseDialog({ key: 'priority-0' })

      // Should still be at the bottom because of its priority
      expect(store.dialogStack.map((d) => d.key)).toEqual([
        'priority-2',
        'priority-1',
        'priority-0'
      ])

      // Rise the medium priority dialog
      store.riseDialog({ key: 'priority-1' })

      // Should be above priority-0 but below priority-2
      expect(store.dialogStack.map((d) => d.key)).toEqual([
        'priority-2',
        'priority-1',
        'priority-0'
      ])
    })

    it('should keep high priority dialogs on top when creating new lower priority dialogs', () => {
      const store = useDialogStore()

      // Create a high priority dialog (like manager progress)
      store.showDialog({
        key: 'manager-progress',
        component: MockComponent,
        priority: 10
      })

      store.showDialog({
        key: 'dialog-2',
        component: MockComponent,
        priority: 0
      })

      store.showDialog({
        key: 'dialog-3',
        component: MockComponent
        // Default priority is 1
      })

      // Manager progress should still be on top
      expect(store.dialogStack[0].key).toBe('manager-progress')

      // Check full order
      expect(store.dialogStack.map((d) => d.key)).toEqual([
        'manager-progress', // priority 2
        'dialog-3', // priority 1 (default)
        'dialog-2' // priority 0
      ])
    })
  })

  describe('basic dialog operations', () => {
    it('should show and close dialogs', () => {
      const store = useDialogStore()

      store.showDialog({
        key: 'test-dialog',
        component: MockComponent
      })

      expect(store.dialogStack).toHaveLength(1)
      expect(store.isDialogOpen('test-dialog')).toBe(true)

      store.closeDialog({ key: 'test-dialog' })

      expect(store.dialogStack).toHaveLength(0)
      expect(store.isDialogOpen('test-dialog')).toBe(false)
    })

    it('should reuse existing dialog when showing with same key', () => {
      const store = useDialogStore()

      store.showDialog({
        key: 'reusable-dialog',
        component: MockComponent,
        title: 'Original Title'
      })

      // First call should create the dialog
      expect(store.dialogStack).toHaveLength(1)
      expect(store.dialogStack[0].title).toBe('Original Title')

      // Second call with same key should reuse the dialog
      store.showDialog({
        key: 'reusable-dialog',
        component: MockComponent,
        title: 'New Title' // This should be ignored
      })

      // Should still have only one dialog with original title
      expect(store.dialogStack).toHaveLength(1)
      expect(store.dialogStack[0].key).toBe('reusable-dialog')
      expect(store.dialogStack[0].title).toBe('Original Title')
    })
  })

  describe('ESC key behavior with multiple dialogs', () => {
    it('should only allow the active dialog to close with ESC key', () => {
      const store = useDialogStore()

      // Create dialogs with different priorities
      store.showDialog({
        key: 'dialog-1',
        component: MockComponent,
        priority: 1
      })

      store.showDialog({
        key: 'dialog-2',
        component: MockComponent,
        priority: 2
      })

      store.showDialog({
        key: 'dialog-3',
        component: MockComponent,
        priority: 3
      })

      // Only the active dialog should be closable with ESC
      const activeDialog = store.dialogStack.find(
        (d) => d.key === store.activeKey
      )
      const inactiveDialogs = store.dialogStack.filter(
        (d) => d.key !== store.activeKey
      )

      expect(activeDialog?.dialogComponentProps.closeOnEscape).toBe(true)
      inactiveDialogs.forEach((dialog) => {
        expect(dialog.dialogComponentProps.closeOnEscape).toBe(false)
      })

      // Close the active dialog
      store.closeDialog({ key: store.activeKey! })

      // The new active dialog should now be closable with ESC
      const newActiveDialog = store.dialogStack.find(
        (d) => d.key === store.activeKey
      )
      const newInactiveDialogs = store.dialogStack.filter(
        (d) => d.key !== store.activeKey
      )

      expect(newActiveDialog?.dialogComponentProps.closeOnEscape).toBe(true)
      newInactiveDialogs.forEach((dialog) => {
        expect(dialog.dialogComponentProps.closeOnEscape).toBe(false)
      })
    })
  })
})
