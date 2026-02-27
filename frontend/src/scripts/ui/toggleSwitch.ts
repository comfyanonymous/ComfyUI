import { $el } from '../ui'

/**
 * @typedef { { text: string, value?: string, tooltip?: string } } ToggleSwitchItem
 */
/**
 * Creates a toggle switch element
 * @param { string } name
 * @param { Array<string> | ToggleSwitchItem } items
 * @param { Object } [opts]
 * @param { (e: { item: ToggleSwitchItem, prev?: ToggleSwitchItem }) => void } [opts.onChange]
 */
// @ts-expect-error fixme ts strict error
export function toggleSwitch(name, items, e?) {
  const onChange = e?.onChange

  // @ts-expect-error fixme ts strict error
  let selectedIndex
  // @ts-expect-error fixme ts strict error
  let elements

  // @ts-expect-error fixme ts strict error
  function updateSelected(index) {
    // @ts-expect-error fixme ts strict error
    if (selectedIndex != null) {
      // @ts-expect-error fixme ts strict error
      elements[selectedIndex].classList.remove('comfy-toggle-selected')
    }
    onChange?.({
      item: items[index],
      // @ts-expect-error fixme ts strict error
      prev: selectedIndex == null ? undefined : items[selectedIndex]
    })
    selectedIndex = index
    // @ts-expect-error fixme ts strict error
    elements[selectedIndex].classList.add('comfy-toggle-selected')
  }

  // @ts-expect-error fixme ts strict error
  elements = items.map((item, i) => {
    if (typeof item === 'string') item = { text: item }
    if (!item.value) item.value = item.text

    const toggle = $el(
      'label',
      {
        textContent: item.text,
        title: item.tooltip ?? ''
      },
      $el('input', {
        name,
        type: 'radio',
        value: item.value ?? item.text,
        checked: item.selected,
        onchange: () => {
          updateSelected(i)
        }
      })
    )
    if (item.selected) {
      updateSelected(i)
    }
    return toggle
  })

  const container = $el('div.comfy-toggle-switch', elements)

  if (selectedIndex == null) {
    elements[0].children[0].checked = true
    updateSelected(0)
  }

  return container
}
