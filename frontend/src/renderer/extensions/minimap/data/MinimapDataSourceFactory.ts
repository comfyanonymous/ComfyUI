import type { LGraph } from '@/lib/litegraph/src/litegraph'
import { layoutStore } from '@/renderer/core/layout/store/layoutStore'

import type { IMinimapDataSource } from '../types'
import { LayoutStoreDataSource } from './LayoutStoreDataSource'
import { LiteGraphDataSource } from './LiteGraphDataSource'

/**
 * Factory for creating the appropriate data source
 */
export class MinimapDataSourceFactory {
  static create(graph: LGraph | null): IMinimapDataSource {
    // Check if LayoutStore has data
    const layoutStoreHasData = layoutStore.getAllNodes().value.size > 0

    if (layoutStoreHasData) {
      return new LayoutStoreDataSource(graph)
    }

    return new LiteGraphDataSource(graph)
  }
}
