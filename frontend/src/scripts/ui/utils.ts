export type ClassList = string | string[] | Record<string, boolean>

export function applyClasses(
  element: HTMLElement,
  classList: ClassList,
  ...requiredClasses: string[]
) {
  classList ??= ''

  let str: string
  if (typeof classList === 'string') {
    str = classList
  } else if (classList instanceof Array) {
    str = classList.join(' ')
  } else {
    str = Object.entries(classList).reduce((p, c) => {
      if (c[1]) {
        p += (p.length ? ' ' : '') + c[0]
      }
      return p
    }, '')
  }
  element.className = str
  if (requiredClasses) {
    element.classList.add(...requiredClasses)
  }
}

export function toggleElement(
  element: HTMLElement,
  {
    onHide,
    onShow
  }: {
    onHide?: (el: HTMLElement) => void
    // @ts-expect-error fixme ts strict error
    onShow?: (el: HTMLElement, value) => void
  } = {}
) {
  let placeholder: HTMLElement | Comment
  let hidden: boolean
  // @ts-expect-error fixme ts strict error
  return (value) => {
    if (value) {
      if (hidden) {
        hidden = false
        placeholder.replaceWith(element)
      }
      onShow?.(element, value)
    } else {
      if (!placeholder) {
        placeholder = document.createComment('')
      }
      hidden = true
      element.replaceWith(placeholder)
      onHide?.(element)
    }
  }
}
