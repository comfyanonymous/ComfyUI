import type { LGraphNode } from '@/lib/litegraph/src/LGraphNode'
import type { LLink } from '@/lib/litegraph/src/LLink'
import type { Reroute } from '@/lib/litegraph/src/Reroute'
import type { FloatingRenderLink } from '@/lib/litegraph/src/canvas/FloatingRenderLink'
import type { MovingInputLink } from '@/lib/litegraph/src/canvas/MovingInputLink'
import type { MovingOutputLink } from '@/lib/litegraph/src/canvas/MovingOutputLink'
import type { RenderLink } from '@/lib/litegraph/src/canvas/RenderLink'
import type { ToInputFromIoNodeLink } from '@/lib/litegraph/src/canvas/ToInputFromIoNodeLink'
import type { ToInputRenderLink } from '@/lib/litegraph/src/canvas/ToInputRenderLink'
import type { SubgraphInputNode } from '@/lib/litegraph/src/subgraph/SubgraphInputNode'
import type { SubgraphOutputNode } from '@/lib/litegraph/src/subgraph/SubgraphOutputNode'
import type { CanvasPointerEvent } from '@/lib/litegraph/src/types/events'
import type { IWidget } from '@/lib/litegraph/src/types/widgets'

export interface LinkConnectorEventMap {
  reset: boolean

  'before-drop-links': {
    renderLinks: RenderLink[]
    event: CanvasPointerEvent
  }
  'after-drop-links': {
    renderLinks: RenderLink[]
    event: CanvasPointerEvent
  }

  'before-move-input': MovingInputLink | FloatingRenderLink
  'before-move-output': MovingOutputLink | FloatingRenderLink

  'input-moved': MovingInputLink | FloatingRenderLink | ToInputFromIoNodeLink
  'output-moved': MovingOutputLink | FloatingRenderLink

  'link-created': LLink | null | undefined

  'dropped-on-reroute': {
    reroute: Reroute
    event: CanvasPointerEvent
  }
  'dropped-on-node': {
    node: LGraphNode
    event: CanvasPointerEvent
  }
  'dropped-on-io-node': {
    node: SubgraphInputNode | SubgraphOutputNode
    event: CanvasPointerEvent
  }
  'dropped-on-canvas': CanvasPointerEvent

  'dropped-on-widget': {
    link: ToInputRenderLink
    node: LGraphNode
    widget: IWidget
  }
}
