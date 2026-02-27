import type { LGraph, LGraphNode } from '@/lib/litegraph/src/litegraph'
import { LGraphBadge } from '@/lib/litegraph/src/litegraph'

import { useColorPaletteStore } from '@/stores/workspace/colorPaletteStore'
import { adjustColor } from '@/utils/colorUtil'

const componentIconSvg = new Image()
componentIconSvg.src =
  "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24'%3E%3Cpath fill='none' stroke='oklch(83.01%25 0.163 83.16)' stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M15.536 11.293a1 1 0 0 0 0 1.414l2.376 2.377a1 1 0 0 0 1.414 0l2.377-2.377a1 1 0 0 0 0-1.414l-2.377-2.377a1 1 0 0 0-1.414 0zm-13.239 0a1 1 0 0 0 0 1.414l2.377 2.377a1 1 0 0 0 1.414 0l2.377-2.377a1 1 0 0 0 0-1.414L6.088 8.916a1 1 0 0 0-1.414 0zm6.619 6.619a1 1 0 0 0 0 1.415l2.377 2.376a1 1 0 0 0 1.414 0l2.377-2.376a1 1 0 0 0 0-1.415l-2.377-2.376a1 1 0 0 0-1.414 0zm0-13.238a1 1 0 0 0 0 1.414l2.377 2.376a1 1 0 0 0 1.414 0l2.377-2.376a1 1 0 0 0 0-1.414l-2.377-2.377a1 1 0 0 0-1.414 0z'/%3E%3C/svg%3E"

export const usePriceBadge = () => {
  function updateSubgraphCredits(node: LGraphNode) {
    if (!node.isSubgraphNode()) return
    node.badges = node.badges.filter((b) => !isCreditsBadge(b))
    const newBadges = collectCreditsBadges(node.subgraph)
    if (newBadges.length > 1) {
      node.badges.push(getCreditsBadge('Partner Nodes x ' + newBadges.length))
    } else {
      node.badges.push(...newBadges)
    }
  }
  function collectCreditsBadges(
    graph: LGraph,
    visited: Set<string> = new Set()
  ): (LGraphBadge | (() => LGraphBadge))[] {
    if (visited.has(graph.id)) return []
    visited.add(graph.id)
    const badges = []
    for (const node of graph.nodes) {
      badges.push(
        ...(node.isSubgraphNode()
          ? collectCreditsBadges(node.subgraph, visited)
          : node.badges.filter((b) => isCreditsBadge(b)))
      )
    }
    return badges
  }

  function isCreditsBadge(
    badge: Partial<LGraphBadge> | (() => Partial<LGraphBadge>)
  ): boolean {
    const badgeInstance = typeof badge === 'function' ? badge() : badge
    return badgeInstance.icon?.image === componentIconSvg
  }

  const colorPaletteStore = useColorPaletteStore()
  function getCreditsBadge(price: string): LGraphBadge {
    const isLightTheme = colorPaletteStore.completedActivePalette.light_theme

    return new LGraphBadge({
      text: price,
      iconOptions: {
        image: componentIconSvg,
        size: 8
      },
      fgColor:
        colorPaletteStore.completedActivePalette.colors.litegraph_base
          .BADGE_FG_COLOR,
      bgColor: isLightTheme
        ? adjustColor('#8D6932', { lightness: 0.5 })
        : '#8D6932'
    })
  }
  return {
    getCreditsBadge,
    isCreditsBadge,
    updateSubgraphCredits
  }
}
