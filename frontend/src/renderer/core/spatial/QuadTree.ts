/**
 * QuadTree implementation for spatial indexing of nodes
 * Optimized for viewport culling in large node graphs
 */
import type {
  QuadNodeDebugInfo,
  SpatialIndexDebugInfo
} from '@/types/spatialIndex'

export interface Bounds {
  x: number
  y: number
  width: number
  height: number
}

interface QuadTreeItem<T> {
  id: string
  bounds: Bounds
  data: T
}

interface QuadTreeOptions {
  maxDepth?: number
  maxItemsPerNode?: number
  minNodeSize?: number
}

class QuadNode<T> {
  private bounds: Bounds
  private depth: number
  private maxDepth: number
  private maxItems: number
  private items: QuadTreeItem<T>[] = []
  private children: QuadNode<T>[] | null = null
  private divided = false

  constructor(
    bounds: Bounds,
    depth: number = 0,
    maxDepth: number = 5,
    maxItems: number = 4
  ) {
    this.bounds = bounds
    this.depth = depth
    this.maxDepth = maxDepth
    this.maxItems = maxItems
  }

  insert(item: QuadTreeItem<T>): boolean {
    // Check if item is within bounds
    if (!this.contains(item.bounds)) {
      return false
    }

    // If we have space and haven't divided, add to this node
    if (this.items.length < this.maxItems && !this.divided) {
      this.items.push(item)
      return true
    }

    // If we haven't reached max depth, subdivide
    if (!this.divided && this.depth < this.maxDepth) {
      this.subdivide()
    }

    // If divided, insert into children
    if (this.divided && this.children) {
      for (const child of this.children) {
        if (child.insert(item)) {
          return true
        }
      }
    }

    // If we can't subdivide further, add to this node anyway
    this.items.push(item)
    return true
  }

  remove(item: QuadTreeItem<T>): boolean {
    const index = this.items.findIndex((i) => i.id === item.id)
    if (index !== -1) {
      this.items.splice(index, 1)
      return true
    }

    if (this.divided && this.children) {
      for (const child of this.children) {
        if (child.remove(item)) {
          return true
        }
      }
    }

    return false
  }

  query(
    searchBounds: Bounds,
    found: QuadTreeItem<T>[] = []
  ): QuadTreeItem<T>[] {
    // Check if search area intersects with this node
    if (!this.intersects(searchBounds)) {
      return found
    }

    // Add items in this node that intersect with search bounds
    for (const item of this.items) {
      if (this.boundsIntersect(item.bounds, searchBounds)) {
        found.push(item)
      }
    }

    // Recursively search children
    if (this.divided && this.children) {
      for (const child of this.children) {
        child.query(searchBounds, found)
      }
    }

    return found
  }

  private subdivide() {
    const { x, y, width, height } = this.bounds
    const halfWidth = width / 2
    const halfHeight = height / 2

    this.children = [
      // Top-left
      new QuadNode<T>(
        { x, y, width: halfWidth, height: halfHeight },
        this.depth + 1,
        this.maxDepth,
        this.maxItems
      ),
      // Top-right
      new QuadNode<T>(
        { x: x + halfWidth, y, width: halfWidth, height: halfHeight },
        this.depth + 1,
        this.maxDepth,
        this.maxItems
      ),
      // Bottom-left
      new QuadNode<T>(
        { x, y: y + halfHeight, width: halfWidth, height: halfHeight },
        this.depth + 1,
        this.maxDepth,
        this.maxItems
      ),
      // Bottom-right
      new QuadNode<T>(
        {
          x: x + halfWidth,
          y: y + halfHeight,
          width: halfWidth,
          height: halfHeight
        },
        this.depth + 1,
        this.maxDepth,
        this.maxItems
      )
    ]

    this.divided = true

    // Redistribute existing items to children
    const itemsToRedistribute = [...this.items]
    this.items = []

    for (const item of itemsToRedistribute) {
      let inserted = false
      for (const child of this.children) {
        if (child.insert(item)) {
          inserted = true
          break
        }
      }
      // Keep in parent if it doesn't fit in any child
      if (!inserted) {
        this.items.push(item)
      }
    }
  }

  private contains(itemBounds: Bounds): boolean {
    return (
      itemBounds.x >= this.bounds.x &&
      itemBounds.y >= this.bounds.y &&
      itemBounds.x + itemBounds.width <= this.bounds.x + this.bounds.width &&
      itemBounds.y + itemBounds.height <= this.bounds.y + this.bounds.height
    )
  }

  private intersects(searchBounds: Bounds): boolean {
    return this.boundsIntersect(this.bounds, searchBounds)
  }

  private boundsIntersect(a: Bounds, b: Bounds): boolean {
    return !(
      a.x + a.width < b.x ||
      b.x + b.width < a.x ||
      a.y + a.height < b.y ||
      b.y + b.height < a.y
    )
  }

  // Debug helper to get tree structure
  getDebugInfo(): QuadNodeDebugInfo {
    return {
      bounds: this.bounds,
      depth: this.depth,
      itemCount: this.items.length,
      divided: this.divided,
      children: this.children?.map((child) => child.getDebugInfo())
    }
  }
}

export class QuadTree<T> {
  private root: QuadNode<T>
  private itemMap: Map<string, QuadTreeItem<T>> = new Map()
  private options: Required<QuadTreeOptions>

  constructor(bounds: Bounds, options: QuadTreeOptions = {}) {
    this.options = {
      maxDepth: options.maxDepth ?? 5,
      maxItemsPerNode: options.maxItemsPerNode ?? 4,
      minNodeSize: options.minNodeSize ?? 50
    }

    this.root = new QuadNode<T>(
      bounds,
      0,
      this.options.maxDepth,
      this.options.maxItemsPerNode
    )
  }

  insert(id: string, bounds: Bounds, data: T): boolean {
    const item: QuadTreeItem<T> = { id, bounds, data }

    // Remove old item if it exists
    if (this.itemMap.has(id)) {
      this.remove(id)
    }

    const success = this.root.insert(item)
    if (success) {
      this.itemMap.set(id, item)
    }
    return success
  }

  remove(id: string): boolean {
    const item = this.itemMap.get(id)
    if (!item) return false

    const success = this.root.remove(item)
    if (success) {
      this.itemMap.delete(id)
    }
    return success
  }

  update(id: string, newBounds: Bounds): boolean {
    const item = this.itemMap.get(id)
    if (!item) return false

    // Remove and re-insert with new bounds
    const data = item.data
    this.remove(id)
    return this.insert(id, newBounds, data)
  }

  query(searchBounds: Bounds): T[] {
    const items = this.root.query(searchBounds)
    return items.map((item) => item.data)
  }

  clear() {
    this.root = new QuadNode<T>(
      this.root['bounds'],
      0,
      this.options.maxDepth,
      this.options.maxItemsPerNode
    )
    this.itemMap.clear()
  }

  get size(): number {
    return this.itemMap.size
  }

  getDebugInfo(): SpatialIndexDebugInfo {
    return {
      size: this.size,
      tree: this.root.getDebugInfo()
    }
  }
}
