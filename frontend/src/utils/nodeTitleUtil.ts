import { normalizeI18nKey } from '@/utils/formatUtil'

type NodeTitleInfo = {
  title?: string | number | null
  type?: string | number | null
}

type StaticTranslate = (key: string, fallbackMessage: string) => string

type ResolveNodeDisplayNameOptions = {
  emptyLabel: string
  untitledLabel: string
  st: StaticTranslate
}

export function resolveNodeDisplayName(
  node: NodeTitleInfo | null | undefined,
  options: ResolveNodeDisplayNameOptions
): string {
  if (!node) return options.emptyLabel

  const title = (node.title ?? '').toString().trim()
  if (title.length > 0) return title

  const nodeType = (node.type ?? '').toString().trim() || options.untitledLabel
  const key = `nodeDefs.${normalizeI18nKey(nodeType)}.display_name`
  return options.st(key, nodeType)
}
