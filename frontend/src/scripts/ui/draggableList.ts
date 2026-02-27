/*
  Original implementation:
    https://github.com/TahaSh/drag-to-reorder
    MIT License

    Copyright (c) 2023 Taha Shashtari

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/
const styleElement = document.createElement('style')
styleElement.textContent = `
        .draggable-item {
            position: relative;
            will-change: transform;
            user-select: none;
        }
        .draggable-item.is-idle {
            transition: 0.25s ease transform;
        }
        .draggable-item.is-draggable {
            z-index: 10;
        }
    `
document.head.append(styleElement)

export class DraggableList extends EventTarget {
  listContainer
  // @ts-expect-error fixme ts strict error
  draggableItem
  // @ts-expect-error fixme ts strict error
  pointerStartX
  // @ts-expect-error fixme ts strict error
  pointerStartY
  // @ts-expect-error fixme ts strict error
  scrollYMax
  itemsGap = 0
  items = []
  itemSelector
  handleClass = 'drag-handle'
  off = []
  offDrag = []

  // @ts-expect-error fixme ts strict error
  constructor(element, itemSelector) {
    super()
    this.listContainer = element
    this.itemSelector = itemSelector

    if (!this.listContainer) return

    // @ts-expect-error fixme ts strict error
    this.off.push(this.on(this.listContainer, 'mousedown', this.dragStart))
    // @ts-expect-error fixme ts strict error
    this.off.push(this.on(this.listContainer, 'touchstart', this.dragStart))
    // @ts-expect-error fixme ts strict error
    this.off.push(this.on(document, 'mouseup', this.dragEnd))
    // @ts-expect-error fixme ts strict error
    this.off.push(this.on(document, 'touchend', this.dragEnd))
    // @ts-expect-error fixme ts strict error
    this.off.push(this.on(document, 'pointercancel', this.dragEnd))
  }

  getAllItems() {
    if (!this.items?.length) {
      this.items = Array.from(
        this.listContainer.querySelectorAll(this.itemSelector)
      )
      this.items.forEach((element) => {
        // @ts-expect-error fixme ts strict error
        element.classList.add('is-idle')
      })
    }
    return this.items
  }

  getIdleItems() {
    return this.getAllItems().filter((item) =>
      // @ts-expect-error fixme ts strict error
      item.classList.contains('is-idle')
    )
  }

  // @ts-expect-error fixme ts strict error
  isItemAbove(item) {
    return item.hasAttribute('data-is-above')
  }

  // @ts-expect-error fixme ts strict error
  isItemToggled(item) {
    return item.hasAttribute('data-is-toggled')
  }

  // @ts-expect-error fixme ts strict error
  on(source, event, listener, options?) {
    listener = listener.bind(this)
    source.addEventListener(event, listener, options)
    return () => source.removeEventListener(event, listener)
  }

  // @ts-expect-error fixme ts strict error
  dragStart(e) {
    if (e.button > 0) return

    if (e.target.classList.contains(this.handleClass)) {
      this.draggableItem = e.target.closest(this.itemSelector)
    }

    if (!this.draggableItem) return

    this.pointerStartX = e.clientX || e.touches[0].clientX
    this.pointerStartY = e.clientY || e.touches[0].clientY
    this.scrollYMax =
      this.listContainer.scrollHeight - this.listContainer.clientHeight

    this.setItemsGap()
    this.initDraggableItem()
    this.initItemsState()

    // @ts-expect-error fixme ts strict error
    this.offDrag.push(this.on(document, 'mousemove', this.drag))
    this.offDrag.push(
      // @ts-expect-error fixme ts strict error
      this.on(document, 'touchmove', this.drag, { passive: false })
    )

    this.dispatchEvent(
      new CustomEvent('dragstart', {
        detail: {
          element: this.draggableItem,
          // @ts-expect-error fixme ts strict error
          position: this.getAllItems().indexOf(this.draggableItem)
        }
      })
    )
  }

  setItemsGap() {
    if (this.getIdleItems().length <= 1) {
      this.itemsGap = 0
      return
    }

    const item1 = this.getIdleItems()[0]
    const item2 = this.getIdleItems()[1]

    // @ts-expect-error fixme ts strict error
    const item1Rect = item1.getBoundingClientRect()
    // @ts-expect-error fixme ts strict error
    const item2Rect = item2.getBoundingClientRect()

    this.itemsGap = Math.abs(item1Rect.bottom - item2Rect.top)
  }

  initItemsState() {
    this.getIdleItems().forEach((item, i) => {
      // @ts-expect-error fixme ts strict error
      if (this.getAllItems().indexOf(this.draggableItem) > i) {
        // @ts-expect-error fixme ts strict error
        item.dataset.isAbove = ''
      }
    })
  }

  initDraggableItem() {
    this.draggableItem.classList.remove('is-idle')
    this.draggableItem.classList.add('is-draggable')
  }

  // @ts-expect-error fixme ts strict error
  drag(e) {
    if (!this.draggableItem) return

    e.preventDefault()

    const clientX = e.clientX || e.touches[0].clientX
    const clientY = e.clientY || e.touches[0].clientY

    const listRect = this.listContainer.getBoundingClientRect()

    if (clientY > listRect.bottom) {
      if (this.listContainer.scrollTop < this.scrollYMax) {
        this.listContainer.scrollBy(0, 10)
        this.pointerStartY -= 10
      }
    } else if (clientY < listRect.top && this.listContainer.scrollTop > 0) {
      this.pointerStartY += 10
      this.listContainer.scrollBy(0, -10)
    }

    const pointerOffsetX = clientX - this.pointerStartX
    const pointerOffsetY = clientY - this.pointerStartY

    this.updateIdleItemsStateAndPosition()
    this.draggableItem.style.transform = `translate(${pointerOffsetX}px, ${pointerOffsetY}px)`
  }

  updateIdleItemsStateAndPosition() {
    const draggableItemRect = this.draggableItem.getBoundingClientRect()
    const draggableItemY = draggableItemRect.top + draggableItemRect.height / 2

    // Update state
    this.getIdleItems().forEach((item) => {
      // @ts-expect-error fixme ts strict error
      const itemRect = item.getBoundingClientRect()
      const itemY = itemRect.top + itemRect.height / 2
      if (this.isItemAbove(item)) {
        if (draggableItemY <= itemY) {
          // @ts-expect-error fixme ts strict error
          item.dataset.isToggled = ''
        } else {
          // @ts-expect-error fixme ts strict error
          delete item.dataset.isToggled
        }
      } else {
        if (draggableItemY >= itemY) {
          // @ts-expect-error fixme ts strict error
          item.dataset.isToggled = ''
        } else {
          // @ts-expect-error fixme ts strict error
          delete item.dataset.isToggled
        }
      }
    })

    // Update position
    this.getIdleItems().forEach((item) => {
      if (this.isItemToggled(item)) {
        const direction = this.isItemAbove(item) ? 1 : -1
        // @ts-expect-error fixme ts strict error
        item.style.transform = `translateY(${direction * (draggableItemRect.height + this.itemsGap)}px)`
      } else {
        // @ts-expect-error fixme ts strict error
        item.style.transform = ''
      }
    })
  }

  dragEnd() {
    if (!this.draggableItem) return

    this.applyNewItemsOrder()
    this.cleanup()
  }

  applyNewItemsOrder() {
    const reorderedItems = []

    let oldPosition = -1
    this.getAllItems().forEach((item, index) => {
      if (item === this.draggableItem) {
        oldPosition = index
        return
      }
      if (!this.isItemToggled(item)) {
        reorderedItems[index] = item
        return
      }
      const newIndex = this.isItemAbove(item) ? index + 1 : index - 1
      reorderedItems[newIndex] = item
    })

    for (let index = 0; index < this.getAllItems().length; index++) {
      const item = reorderedItems[index]
      if (typeof item === 'undefined') {
        reorderedItems[index] = this.draggableItem
      }
    }

    reorderedItems.forEach((item) => {
      this.listContainer.appendChild(item)
    })

    // @ts-expect-error fixme ts strict error
    this.items = reorderedItems

    this.dispatchEvent(
      new CustomEvent('dragend', {
        detail: {
          element: this.draggableItem,
          oldPosition,
          newPosition: reorderedItems.indexOf(this.draggableItem)
        }
      })
    )
  }

  cleanup() {
    this.itemsGap = 0
    this.items = []
    this.unsetDraggableItem()
    this.unsetItemState()

    // @ts-expect-error fixme ts strict error
    this.offDrag.forEach((f) => f())
    this.offDrag = []
  }

  unsetDraggableItem() {
    this.draggableItem.style = null
    this.draggableItem.classList.remove('is-draggable')
    this.draggableItem.classList.add('is-idle')
    this.draggableItem = null
  }

  unsetItemState() {
    this.getIdleItems().forEach((item) => {
      // @ts-expect-error fixme ts strict error
      delete item.dataset.isAbove
      // @ts-expect-error fixme ts strict error
      delete item.dataset.isToggled
      // @ts-expect-error fixme ts strict error
      item.style.transform = ''
    })
  }

  dispose() {
    // @ts-expect-error fixme ts strict error
    this.off.forEach((f) => f())
  }
}
