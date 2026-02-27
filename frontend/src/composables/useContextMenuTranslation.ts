import { st, te } from '@/i18n'
import { legacyMenuCompat } from '@/lib/litegraph/src/contextMenuCompat'
import type {
  IContextMenuOptions,
  IContextMenuValue,
  INodeInputSlot,
  IWidget
} from '@/lib/litegraph/src/litegraph'
import { LGraphCanvas, LiteGraph } from '@/lib/litegraph/src/litegraph'
import { app } from '@/scripts/app'
import { normalizeI18nKey } from '@/utils/formatUtil'

/**
 * Add translation for litegraph context menu.
 */
export const useContextMenuTranslation = () => {
  // Install compatibility layer BEFORE any extensions load
  legacyMenuCompat.install(LGraphCanvas.prototype, 'getCanvasMenuOptions')

  const { getCanvasMenuOptions } = LGraphCanvas.prototype
  const getCanvasCenterMenuOptions = function (
    this: LGraphCanvas,
    ...args: Parameters<typeof getCanvasMenuOptions>
  ) {
    const res: (IContextMenuValue | null)[] = getCanvasMenuOptions.apply(
      this,
      args
    )

    // Add items from new extension API
    const newApiItems = app.collectCanvasMenuItems(this)
    for (const item of newApiItems) {
      res.push(item)
    }

    // Add legacy monkey-patched items
    const legacyItems = legacyMenuCompat.extractLegacyItems(
      'getCanvasMenuOptions',
      this,
      ...args
    )
    for (const item of legacyItems) {
      res.push(item)
    }

    // Translate all items
    for (const item of res) {
      if (item?.content) {
        item.content = st(`contextMenu.${item.content}`, item.content)
      }
    }
    return res
  }

  LGraphCanvas.prototype.getCanvasMenuOptions = getCanvasCenterMenuOptions

  legacyMenuCompat.registerWrapper(
    'getCanvasMenuOptions',
    getCanvasCenterMenuOptions,
    getCanvasMenuOptions,
    LGraphCanvas.prototype
  )

  // Install compatibility layer for getNodeMenuOptions
  legacyMenuCompat.install(LGraphCanvas.prototype, 'getNodeMenuOptions')

  // Wrap getNodeMenuOptions to add new API items
  const nodeMenuFn = LGraphCanvas.prototype.getNodeMenuOptions
  const getNodeMenuOptionsWithExtensions = function (
    this: LGraphCanvas,
    ...args: Parameters<typeof nodeMenuFn>
  ) {
    const res = nodeMenuFn.apply(this, args) as (IContextMenuValue | null)[]

    // Add items from new extension API
    const node = args[0]
    const newApiItems = app.collectNodeMenuItems(node)
    for (const item of newApiItems) {
      res.push(item)
    }

    // Add legacy monkey-patched items
    const legacyItems = legacyMenuCompat.extractLegacyItems(
      'getNodeMenuOptions',
      this,
      ...args
    )
    for (const item of legacyItems) {
      res.push(item)
    }

    return res
  }

  LGraphCanvas.prototype.getNodeMenuOptions = getNodeMenuOptionsWithExtensions

  legacyMenuCompat.registerWrapper(
    'getNodeMenuOptions',
    getNodeMenuOptionsWithExtensions,
    nodeMenuFn,
    LGraphCanvas.prototype
  )

  function translateMenus(
    values: readonly (IContextMenuValue | string | null)[] | undefined,
    options: IContextMenuOptions
  ) {
    if (!values) return
    const reInput = /Convert (.*) to input/
    const reWidget = /Convert (.*) to widget/
    const cvt = st('contextMenu.Convert ', 'Convert ')
    const tinp = st('contextMenu. to input', ' to input')
    const twgt = st('contextMenu. to widget', ' to widget')
    for (const value of values) {
      if (typeof value === 'string') continue

      translateMenus(value?.submenu?.options, options)
      if (!value?.content) {
        continue
      }
      if (te(`contextMenu.${value.content}`)) {
        value.content = st(`contextMenu.${value.content}`, value.content)
      }

      // for capture translation text of input and widget
      const extraInfo = (options.extra || options.parentMenu?.options?.extra) as
        | { inputs?: INodeInputSlot[]; widgets?: IWidget[] }
        | undefined
      // widgets and inputs
      const matchInput = value.content?.match(reInput)
      if (matchInput) {
        let match = matchInput[1]
        extraInfo?.inputs?.find((i: INodeInputSlot) => {
          if (i.name != match) return false
          match = i.label ? i.label : i.name
        })
        extraInfo?.widgets?.find((i: IWidget) => {
          if (i.name != match) return false
          match = i.label ? i.label : i.name
        })
        value.content = cvt + match + tinp
        continue
      }
      const matchWidget = value.content?.match(reWidget)
      if (matchWidget) {
        let match = matchWidget[1]
        extraInfo?.inputs?.find((i: INodeInputSlot) => {
          if (i.name != match) return false
          match = i.label ? i.label : i.name
        })
        extraInfo?.widgets?.find((i: IWidget) => {
          if (i.name != match) return false
          match = i.label ? i.label : i.name
        })
        value.content = cvt + match + twgt
        continue
      }
    }
  }

  const OriginalContextMenu = LiteGraph.ContextMenu
  function ContextMenu(
    values: (IContextMenuValue | string)[],
    options: IContextMenuOptions
  ) {
    if (options.title) {
      options.title = st(
        `nodeDefs.${normalizeI18nKey(options.title)}.display_name`,
        options.title
      )
    }
    translateMenus(values, options)
    const ctx = new OriginalContextMenu(values, options)
    return ctx
  }

  LiteGraph.ContextMenu = ContextMenu as unknown as typeof LiteGraph.ContextMenu
  LiteGraph.ContextMenu.prototype = OriginalContextMenu.prototype
}
