type PromotionEntry = {
  interiorNodeId: string
  widgetName: string
}

type CreateView<TView> = (entry: PromotionEntry) => TView

/**
 * Reconciles promoted widget entries to stable view instances.
 *
 * Keeps object identity stable by key while preserving the current
 * promotion order and deduplicating duplicate entries by first occurrence.
 */
export class PromotedWidgetViewManager<TView> {
  private viewCache = new Map<string, TView>()
  private cachedViews: TView[] | null = null
  private cachedEntriesRef: readonly PromotionEntry[] | null = null

  reconcile(
    entries: readonly PromotionEntry[],
    createView: CreateView<TView>
  ): TView[] {
    if (this.cachedViews && entries === this.cachedEntriesRef)
      return this.cachedViews

    const views: TView[] = []
    const seenKeys = new Set<string>()

    for (const entry of entries) {
      const key = this.makeKey(entry.interiorNodeId, entry.widgetName)
      if (seenKeys.has(key)) continue
      seenKeys.add(key)

      const existing = this.viewCache.get(key)
      if (existing) {
        views.push(existing)
        continue
      }

      const nextView = createView(entry)
      this.viewCache.set(key, nextView)
      views.push(nextView)
    }

    for (const key of this.viewCache.keys()) {
      if (!seenKeys.has(key)) this.viewCache.delete(key)
    }

    this.cachedViews = views
    this.cachedEntriesRef = entries
    return views
  }

  getOrCreate(
    interiorNodeId: string,
    widgetName: string,
    createView: () => TView
  ): TView {
    const key = this.makeKey(interiorNodeId, widgetName)
    const cached = this.viewCache.get(key)
    if (cached) return cached

    const view = createView()
    this.viewCache.set(key, view)
    return view
  }

  remove(interiorNodeId: string, widgetName: string): void {
    this.viewCache.delete(this.makeKey(interiorNodeId, widgetName))
    this.invalidateMemoizedList()
  }

  clear(): void {
    this.viewCache.clear()
    this.invalidateMemoizedList()
  }

  invalidateMemoizedList(): void {
    this.cachedViews = null
    this.cachedEntriesRef = null
  }

  private makeKey(interiorNodeId: string, widgetName: string): string {
    return `${interiorNodeId}:${widgetName}`
  }
}
