# Component Testing Guide

This guide covers patterns and examples for testing Vue components in the ComfyUI Frontend codebase.

## Table of Contents

1. [Basic Component Testing](#basic-component-testing)
2. [PrimeVue Components Testing](#primevue-components-testing)
3. [Tooltip Directives](#tooltip-directives)
4. [Component Events Testing](#component-events-testing)
5. [User Interaction Testing](#user-interaction-testing)
6. [Asynchronous Component Testing](#asynchronous-component-testing)
7. [Working with Vue Reactivity](#working-with-vue-reactivity)

## Basic Component Testing

Basic approach to testing a component's rendering and structure:

```typescript
// Example from: src/components/sidebar/SidebarIcon.spec.ts
import { mount } from '@vue/test-utils'
import SidebarIcon from './SidebarIcon.vue'

describe('SidebarIcon', () => {
  const exampleProps = {
    icon: 'pi pi-cog',
    selected: false
  }

  const mountSidebarIcon = (props, options = {}) => {
    return mount(SidebarIcon, {
      props: { ...exampleProps, ...props },
      ...options
    })
  }

  it('renders label', () => {
    const wrapper = mountSidebarIcon({})
    expect(wrapper.find('.p-button.p-component').exists()).toBe(true)
    expect(wrapper.find('.p-button-label').exists()).toBe(true)
  })
})
```

## PrimeVue Components Testing

Setting up and testing PrimeVue components:

```typescript
// Example from: src/components/common/ColorCustomizationSelector.spec.ts
import { mount } from '@vue/test-utils'
import ColorPicker from 'primevue/colorpicker'
import PrimeVue from 'primevue/config'
import SelectButton from 'primevue/selectbutton'
import { createApp } from 'vue'

import ColorCustomizationSelector from './ColorCustomizationSelector.vue'

describe('ColorCustomizationSelector', () => {
  beforeEach(() => {
    // Setup PrimeVue
    const app = createApp({})
    app.use(PrimeVue)
  })

  const mountComponent = (props = {}) => {
    return mount(ColorCustomizationSelector, {
      global: {
        plugins: [PrimeVue],
        components: { SelectButton, ColorPicker }
      },
      props: {
        modelValue: null,
        colorOptions: [
          { name: 'Blue', value: '#0d6efd' },
          { name: 'Green', value: '#28a745' }
        ],
        ...props
      }
    })
  }

  it('initializes with predefined color when provided', async () => {
    const wrapper = mountComponent({
      modelValue: '#0d6efd'
    })

    await nextTick()
    const selectButton = wrapper.findComponent(SelectButton)
    expect(selectButton.props('modelValue')).toEqual({
      name: 'Blue',
      value: '#0d6efd'
    })
  })
})
```

## Tooltip Directives

Testing components with tooltip directives:

```typescript
// Example from: src/components/sidebar/SidebarIcon.spec.ts
import { mount } from '@vue/test-utils'
import PrimeVue from 'primevue/config'
import Tooltip from 'primevue/tooltip'

describe('SidebarIcon with tooltip', () => {
  it('shows tooltip on hover', async () => {
    const tooltipShowDelay = 300
    const tooltipText = 'Settings'

    const wrapper = mount(SidebarIcon, {
      global: {
        plugins: [PrimeVue],
        directives: { tooltip: Tooltip }
      },
      props: {
        icon: 'pi pi-cog',
        selected: false,
        tooltip: tooltipText
      }
    })

    // Hover over the icon
    await wrapper.trigger('mouseenter')
    await new Promise((resolve) => setTimeout(resolve, tooltipShowDelay + 16))

    const tooltipElAfterHover = document.querySelector('[role="tooltip"]')
    expect(tooltipElAfterHover).not.toBeNull()
  })

  it('sets aria-label attribute when tooltip is provided', () => {
    const tooltipText = 'Settings'
    const wrapper = mount(SidebarIcon, {
      global: {
        plugins: [PrimeVue],
        directives: { tooltip: Tooltip }
      },
      props: {
        icon: 'pi pi-cog',
        selected: false,
        tooltip: tooltipText
      }
    })

    expect(wrapper.attributes('aria-label')).toEqual(tooltipText)
  })
})
```

## Component Events Testing

Testing component events:

```typescript
// Example from: src/components/common/ColorCustomizationSelector.spec.ts
it('emits update when predefined color is selected', async () => {
  const wrapper = mountComponent()
  const selectButton = wrapper.findComponent(SelectButton)

  await selectButton.setValue(colorOptions[0])

  expect(wrapper.emitted('update:modelValue')?.[0]).toEqual(['#0d6efd'])
})

it('emits update when custom color is changed', async () => {
  const wrapper = mountComponent()
  const selectButton = wrapper.findComponent(SelectButton)

  // Select custom option
  await selectButton.setValue({ name: '_custom', value: '' })

  // Change custom color
  const colorPicker = wrapper.findComponent(ColorPicker)
  await colorPicker.setValue('ff0000')

  expect(wrapper.emitted('update:modelValue')?.[0]).toEqual(['#ff0000'])
})
```

## User Interaction Testing

Testing user interactions:

```typescript
// Example from: src/components/common/EditableText.spec.ts
describe('EditableText', () => {
  it('switches to edit mode on click', async () => {
    const wrapper = mount(EditableText, {
      props: {
        modelValue: 'Initial Text',
        editable: true
      }
    })

    // Initially in view mode
    expect(wrapper.find('input').exists()).toBe(false)

    // Click to edit
    await wrapper.find('.editable-text').trigger('click')

    // Should switch to edit mode
    expect(wrapper.find('input').exists()).toBe(true)
    expect(wrapper.find('input').element.value).toBe('Initial Text')
  })

  it('saves changes on enter key press', async () => {
    const wrapper = mount(EditableText, {
      props: {
        modelValue: 'Initial Text',
        editable: true
      }
    })

    // Switch to edit mode
    await wrapper.find('.editable-text').trigger('click')

    // Change input value
    const input = wrapper.find('input')
    await input.setValue('New Text')

    // Press enter to save
    await input.trigger('keydown.enter')

    // Check if event was emitted with new value
    expect(wrapper.emitted('update:modelValue')[0]).toEqual(['New Text'])

    // Should switch back to view mode
    expect(wrapper.find('input').exists()).toBe(false)
  })
})
```

## Asynchronous Component Testing

Testing components with async behavior:

```typescript
// Example from: src/components/dialog/content/manager/PackVersionSelectorPopover.test.ts
import { nextTick } from 'vue'

it('shows dropdown options when clicked', async () => {
  const wrapper = mount(PackVersionSelectorPopover, {
    props: {
      versions: ['1.0.0', '1.1.0', '2.0.0'],
      selectedVersion: '1.1.0'
    }
  })

  // Initially dropdown should be hidden
  expect(wrapper.find('.p-dropdown-panel').isVisible()).toBe(false)

  // Click dropdown
  await wrapper.find('.p-dropdown').trigger('click')
  await nextTick() // Wait for Vue to update the DOM

  // Dropdown should be visible now
  expect(wrapper.find('.p-dropdown-panel').isVisible()).toBe(true)

  // Options should match the provided versions
  const options = wrapper.findAll('.p-dropdown-item')
  expect(options.length).toBe(3)
  expect(options[0].text()).toBe('1.0.0')
  expect(options[1].text()).toBe('1.1.0')
  expect(options[2].text()).toBe('2.0.0')
})
```

## Working with Vue Reactivity

Testing components with complex reactive behavior can be challenging. Here are patterns to help manage reactivity issues in tests:

### Helper Function for Waiting on Reactivity

Use a helper function to wait for both promises and the Vue reactivity cycle:

```typescript
// Example from: src/components/dialog/content/manager/PackVersionSelectorPopover.test.ts
const waitForPromises = async () => {
  // Wait for any promises in the microtask queue
  await new Promise((resolve) => setTimeout(resolve, 16))
  // Wait for Vue to update the DOM
  await nextTick()
}

it('fetches versions on mount', async () => {
  mockGetPackVersions.mockResolvedValueOnce(defaultMockVersions)

  mountComponent()
  await waitForPromises() // Wait for async operations and reactivity

  expect(mockGetPackVersions).toHaveBeenCalledWith(mockNodePack.id)
})
```

### Testing Components with Async Lifecycle Hooks

When components use `onMounted` or other lifecycle hooks with async operations:

```typescript
it('shows loading state while fetching versions', async () => {
  // Delay the promise resolution
  mockGetPackVersions.mockImplementationOnce(
    () =>
      new Promise((resolve) =>
        setTimeout(() => resolve(defaultMockVersions), 1000)
      )
  )

  const wrapper = mountComponent()

  // Check loading state before promises resolve
  expect(wrapper.text()).toContain('Loading versions...')
})
```

### Testing Prop Changes

Test components' reactivity to prop changes:

```typescript
// Example from: src/components/dialog/content/manager/PackVersionSelectorPopover.test.ts
it('is reactive to nodePack prop changes', async () => {
  // Set up the mock for the initial fetch
  mockGetPackVersions.mockResolvedValueOnce(defaultMockVersions)

  const wrapper = mountComponent()
  await waitForPromises()

  // Set up the mock for the second fetch after prop change
  mockGetPackVersions.mockResolvedValueOnce(defaultMockVersions)

  // Update the nodePack prop
  const newNodePack = { ...mockNodePack, id: 'new-test-pack' }
  await wrapper.setProps({ nodePack: newNodePack })
  await waitForPromises()

  // Should fetch versions for the new nodePack
  expect(mockGetPackVersions).toHaveBeenCalledWith(newNodePack.id)
})
```

### Handling Computed Properties

Testing components with computed properties that depend on async data:

```typescript
it('displays special options and version options in the listbox', async () => {
  mockGetPackVersions.mockResolvedValueOnce(defaultMockVersions)

  const wrapper = mountComponent()
  await waitForPromises() // Wait for data fetching and computed property updates

  const listbox = wrapper.findComponent(Listbox)
  const options = listbox.props('options')!

  // Now options should be populated through computed properties
  expect(options.length).toBe(defaultMockVersions.length + 2)
})
```

### Common Reactivity Pitfalls

1. **Not waiting for all promises**: Ensure you wait for both component promises and Vue's reactivity system
2. **Timing issues with component mounting**: Components might not be fully mounted when assertions run
3. **Async lifecycle hooks**: Components using async `onMounted` require careful handling
4. **PrimeVue components**: PrimeVue components often have their own internal state and reactivity that needs time to update
5. **Computed properties depending on async data**: Always ensure async data is loaded before testing computed properties

By using the `waitForPromises` helper and being mindful of these patterns, you can write more robust tests for components with complex reactivity.
