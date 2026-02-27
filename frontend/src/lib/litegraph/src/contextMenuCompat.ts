import type { LGraphCanvas } from './LGraphCanvas'
import type { IContextMenuValue } from './interfaces'

/**
 * Simple compatibility layer for legacy getCanvasMenuOptions and getNodeMenuOptions monkey patches.
 * To disable legacy support, set ENABLE_LEGACY_SUPPORT = false
 */
const ENABLE_LEGACY_SUPPORT = true

type ContextMenuValueProvider = (
  ...args: unknown[]
) => (IContextMenuValue | null)[]

class LegacyMenuCompat {
  private originalMethods = new Map<string, ContextMenuValueProvider>()
  private hasWarned = new Set<string>()
  private currentExtension: string | null = null
  private isExtracting = false
  private readonly wrapperMethods = new Map<string, ContextMenuValueProvider>()
  private readonly preWrapperMethods = new Map<
    string,
    ContextMenuValueProvider
  >()
  private readonly wrapperInstalled = new Map<string, boolean>()

  /**
   * Set the name of the extension that is currently being set up.
   * This allows us to track which extension is monkey-patching.
   * @param extensionName The name of the extension
   */
  setCurrentExtension(extensionName: string | null) {
    this.currentExtension = extensionName
  }

  /**
   * Register a wrapper method that should NOT be treated as a legacy monkey-patch.
   * @param methodName The method name
   * @param wrapperFn The wrapper function
   * @param preWrapperFn The method that existed before the wrapper
   * @param prototype The prototype to verify wrapper installation
   */
  registerWrapper<K extends keyof LGraphCanvas>(
    methodName: K,
    wrapperFn: LGraphCanvas[K],
    preWrapperFn: LGraphCanvas[K],
    prototype?: LGraphCanvas
  ) {
    this.wrapperMethods.set(
      methodName as string,
      wrapperFn as unknown as ContextMenuValueProvider
    )
    this.preWrapperMethods.set(
      methodName as string,
      preWrapperFn as unknown as ContextMenuValueProvider
    )
    const isInstalled = prototype && prototype[methodName] === wrapperFn
    this.wrapperInstalled.set(methodName as string, !!isInstalled)
  }

  /**
   * Install compatibility layer to detect monkey-patching
   * @param prototype The prototype to install on
   * @param methodName The method name to track
   */
  install<K extends keyof LGraphCanvas>(
    prototype: LGraphCanvas,
    methodName: K
  ) {
    if (!ENABLE_LEGACY_SUPPORT) return

    const originalMethod = prototype[methodName]
    this.originalMethods.set(
      methodName as string,
      originalMethod as unknown as ContextMenuValueProvider
    )

    let currentImpl = originalMethod

    Object.defineProperty(prototype, methodName, {
      get() {
        return currentImpl
      },
      set: (newImpl: LGraphCanvas[K]) => {
        if (!newImpl) return
        const fnKey = `${methodName as string}:${newImpl.toString().slice(0, 100)}`
        if (!this.hasWarned.has(fnKey) && this.currentExtension) {
          this.hasWarned.add(fnKey)

          console.warn(
            `%c[DEPRECATED]%c Monkey-patching ${methodName as string} is deprecated. (Extension: "${this.currentExtension}")\n` +
              `Please use the new context menu API instead.\n\n` +
              `See: https://docs.comfy.org/custom-nodes/js/context-menu-migration`,
            'color: orange; font-weight: bold',
            'color: inherit'
          )
        }
        currentImpl = newImpl
      }
    })
  }

  /**
   * Extract items that were added by legacy monkey patches.
   *
   * Uses set-based diffing by reference to reliably detect additions regardless
   * of item reordering or replacement. Items present in patchedItems but not in
   * originalItems (by reference equality) are considered additions.
   *
   * Note: If a monkey patch removes items (patchedItems has fewer unique items
   * than originalItems), a warning is logged but we still return any new items.
   *
   * @param methodName The method name that was monkey-patched
   * @param context The context to call methods with
   * @param args Arguments to pass to the methods
   * @returns Array of menu items added by monkey patches
   */
  extractLegacyItems(
    methodName: keyof LGraphCanvas,
    context: LGraphCanvas,
    ...args: unknown[]
  ): (IContextMenuValue | null)[] {
    if (!ENABLE_LEGACY_SUPPORT) return []
    if (this.isExtracting) return []

    const originalMethod = this.originalMethods.get(methodName)
    if (!originalMethod) return []

    try {
      this.isExtracting = true

      const originalItems = originalMethod.apply(context, args) as
        | (IContextMenuValue | null)[]
        | undefined
      if (!originalItems) return []

      const currentMethod = context.constructor.prototype[methodName]
      if (!currentMethod || currentMethod === originalMethod) return []

      const registeredWrapper = this.wrapperMethods.get(methodName)
      if (registeredWrapper && currentMethod === registeredWrapper) return []

      const preWrapperMethod = this.preWrapperMethods.get(methodName)
      const wrapperWasInstalled = this.wrapperInstalled.get(methodName)

      const shouldSkipWrapper =
        preWrapperMethod &&
        wrapperWasInstalled &&
        currentMethod !== preWrapperMethod

      const methodToCall = shouldSkipWrapper ? preWrapperMethod : currentMethod

      const patchedItems = methodToCall.apply(context, args) as
        | (IContextMenuValue | null)[]
        | undefined
      if (!patchedItems) {
        return []
      }
      // Use content-based diff to detect additions (not reference-based)
      // Create composite keys from multiple properties to handle undefined content
      const createItemKey = (item: IContextMenuValue): string => {
        const parts = [
          item.content ?? '',
          item.title ?? '',
          item.className ?? '',
          item.property ?? '',
          item.type ?? ''
        ]
        return parts.join('|')
      }

      const originalKeys = new Set(
        originalItems
          .filter(
            (item): item is IContextMenuValue =>
              item !== null && typeof item === 'object' && 'content' in item
          )
          .map(createItemKey)
      )
      const addedItems = patchedItems.filter((item) => {
        if (item === null) return false
        if (typeof item !== 'object' || !('content' in item)) return false
        return !originalKeys.has(createItemKey(item))
      })

      // Warn if items were removed (patched has fewer original items than expected)
      const patchedKeys = new Set(
        patchedItems
          .filter(
            (item): item is IContextMenuValue =>
              item !== null && typeof item === 'object' && 'content' in item
          )
          .map(createItemKey)
      )
      const removedCount = [...originalKeys].filter(
        (key) => !patchedKeys.has(key)
      ).length
      if (removedCount > 0) {
        console.warn(
          `[Context Menu Compat] Monkey patch for ${methodName} removed ${removedCount} original menu item(s). ` +
            `This may cause unexpected behavior.`
        )
      }

      return addedItems
    } catch (e) {
      console.error('[Context Menu Compat] Failed to extract legacy items:', e)
      return []
    } finally {
      this.isExtracting = false
    }
  }
}

export const legacyMenuCompat = new LegacyMenuCompat()
