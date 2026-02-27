import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import { h, nextTick } from 'vue'
import { createI18n } from 'vue-i18n'

import TagsInput from './TagsInput.vue'
import TagsInputInput from './TagsInputInput.vue'
import TagsInputItem from './TagsInputItem.vue'
import TagsInputItemDelete from './TagsInputItemDelete.vue'
import TagsInputItemText from './TagsInputItemText.vue'

const i18n = createI18n({
  legacy: false,
  locale: 'en',
  messages: { en: { g: { removeTag: 'Remove tag' } } }
})

describe('TagsInput', () => {
  function mountTagsInput(props = {}, slots = {}) {
    return mount(TagsInput, {
      props: {
        modelValue: [],
        ...props
      },
      slots
    })
  }

  it('renders slot content', () => {
    const wrapper = mountTagsInput({}, { default: '<span>Slot Content</span>' })

    expect(wrapper.text()).toContain('Slot Content')
  })
})

describe('TagsInput with child components', () => {
  function mountFullTagsInput(tags: string[] = ['tag1', 'tag2']) {
    return mount(TagsInput, {
      global: { plugins: [i18n] },
      props: {
        modelValue: tags
      },
      slots: {
        default: () => [
          ...tags.map((tag) =>
            h(TagsInputItem, { key: tag, value: tag }, () => [
              h(TagsInputItemText),
              h(TagsInputItemDelete)
            ])
          ),
          h(TagsInputInput, { placeholder: 'Add tag...' })
        ]
      }
    })
  }

  it('renders tags structure and content', () => {
    const tags = ['tag1', 'tag2']
    const wrapper = mountFullTagsInput(tags)

    const items = wrapper.findAllComponents(TagsInputItem)
    const textElements = wrapper.findAllComponents(TagsInputItemText)
    const deleteButtons = wrapper.findAllComponents(TagsInputItemDelete)

    expect(items).toHaveLength(tags.length)
    expect(textElements).toHaveLength(tags.length)
    expect(deleteButtons).toHaveLength(tags.length)

    textElements.forEach((el, i) => {
      expect(el.text()).toBe(tags[i])
    })

    expect(wrapper.findComponent(TagsInputInput).exists()).toBe(true)
  })

  it('updates model value when adding a tag', async () => {
    let currentTags = ['existing']

    const wrapper = mount<typeof TagsInput<string>>(TagsInput, {
      props: {
        modelValue: currentTags,
        'onUpdate:modelValue': (payload) => {
          currentTags = payload
        }
      },
      slots: {
        default: () => h(TagsInputInput, { placeholder: 'Add tag...' })
      }
    })

    await wrapper.trigger('click')
    await nextTick()

    const input = wrapper.find('input')
    await input.setValue('newTag')
    await input.trigger('keydown', { key: 'Enter' })
    await nextTick()

    expect(currentTags).toContain('newTag')
  })

  it('does not enter edit mode when disabled', async () => {
    const wrapper = mount<typeof TagsInput<string>>(TagsInput, {
      props: {
        modelValue: ['tag1'],
        disabled: true
      },
      slots: {
        default: () => h(TagsInputInput, { placeholder: 'Add tag...' })
      }
    })

    expect(wrapper.find('input').exists()).toBe(false)

    await wrapper.trigger('click')
    await nextTick()

    expect(wrapper.find('input').exists()).toBe(false)
  })

  it('exits edit mode when clicking outside', async () => {
    const outsideElement = document.createElement('div')
    document.body.appendChild(outsideElement)
    const wrapper = mount<typeof TagsInput<string>>(TagsInput, {
      props: {
        modelValue: ['tag1']
      },
      slots: {
        default: () => h(TagsInputInput, { placeholder: 'Add tag...' })
      }
    })

    await wrapper.trigger('click')
    await nextTick()
    expect(wrapper.find('input').exists()).toBe(true)

    outsideElement.dispatchEvent(new PointerEvent('click', { bubbles: true }))
    await nextTick()

    expect(wrapper.find('input').exists()).toBe(false)

    wrapper.unmount()
    outsideElement.remove()
  })

  it('shows placeholder when modelValue is empty', async () => {
    const wrapper = mount<typeof TagsInput<string>>(TagsInput, {
      props: {
        modelValue: []
      },
      slots: {
        default: ({ isEmpty }: { isEmpty: boolean }) =>
          h(TagsInputInput, { placeholder: 'Add tag...', isEmpty })
      }
    })

    await nextTick()

    const input = wrapper.find('input')
    expect(input.exists()).toBe(true)
    expect(input.attributes('placeholder')).toBe('Add tag...')
  })
})
