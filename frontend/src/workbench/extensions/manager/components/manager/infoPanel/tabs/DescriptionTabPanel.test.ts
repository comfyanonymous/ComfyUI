import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import { createI18n } from 'vue-i18n'

import enMessages from '@/locales/en/main.json' with { type: 'json' }
import type { components } from '@/types/comfyRegistryTypes'

import DescriptionTabPanel from './DescriptionTabPanel.vue'

const i18n = createI18n({
  legacy: false,
  locale: 'en',
  messages: {
    en: enMessages
  }
})

describe('DescriptionTabPanel', () => {
  const mountComponent = (props: {
    nodePack: Partial<components['schemas']['Node']>
  }) => {
    return mount(DescriptionTabPanel, {
      props,
      global: {
        plugins: [i18n]
      }
    })
  }

  const createNodePack = (
    overrides: Partial<components['schemas']['Node']> = {}
  ) => ({
    description: 'Test description',
    ...overrides
  })

  const licenseTests = [
    {
      name: 'handles plain text license',
      nodePack: createNodePack({
        license: 'MIT License',
        repository: 'https://github.com/user/repo'
      }),
      expected: {
        text: 'MIT License',
        isUrl: false
      }
    },
    {
      name: 'handles license file names',
      nodePack: createNodePack({
        license: 'LICENSE',
        repository: 'https://github.com/user/repo'
      }),
      expected: {
        text: 'https://github.com/user/repo/blob/main/LICENSE',
        isUrl: true
      }
    },
    {
      name: 'handles license.md file names',
      nodePack: createNodePack({
        license: 'license.md',
        repository: 'https://github.com/user/repo'
      }),
      expected: {
        text: 'https://github.com/user/repo/blob/main/license.md',
        isUrl: true
      }
    },
    {
      name: 'handles JSON license objects with text property',
      nodePack: createNodePack({
        license: JSON.stringify({ text: 'GPL-3.0' }),
        repository: 'https://github.com/user/repo'
      }),
      expected: {
        text: 'GPL-3.0',
        isUrl: false
      }
    },
    {
      name: 'handles JSON license objects with file property',
      nodePack: createNodePack({
        license: JSON.stringify({ file: 'LICENSE.md' }),
        repository: 'https://github.com/user/repo'
      }),
      expected: {
        text: 'https://github.com/user/repo/blob/main/LICENSE.md',
        isUrl: true
      }
    },
    {
      name: 'handles missing repository URL',
      nodePack: createNodePack({
        license: 'LICENSE'
      }),
      expected: {
        text: 'LICENSE',
        isUrl: false
      }
    },
    {
      name: 'handles non-GitHub repository URLs',
      nodePack: createNodePack({
        license: 'LICENSE',
        repository: 'https://gitlab.com/user/repo'
      }),
      expected: {
        text: 'https://gitlab.com/user/repo/blob/main/LICENSE',
        isUrl: true
      }
    }
  ]

  describe('license formatting', () => {
    licenseTests.forEach((test) => {
      it(test.name, () => {
        const wrapper = mountComponent({ nodePack: test.nodePack })
        if (test.expected.isUrl) {
          const link = wrapper
            .findAll('a')
            .find((a) => a.text().includes(test.expected.text))
          expect(link).toBeDefined()
          expect(link!.attributes('href')).toBe(test.expected.text)
        } else {
          expect(wrapper.text()).toContain(test.expected.text)
        }
      })
    })
  })

  describe('description sections', () => {
    it('shows description text', () => {
      const wrapper = mountComponent({
        nodePack: createNodePack()
      })
      expect(wrapper.text()).toContain('Test description')
    })

    it('shows repository link when available', () => {
      const wrapper = mountComponent({
        nodePack: createNodePack({
          repository: 'https://github.com/user/repo'
        })
      })
      const repoLink = wrapper.find('a[href="https://github.com/user/repo"]')
      expect(repoLink.exists()).toBe(true)
      expect(repoLink.attributes('target')).toBe('_blank')
    })

    it('shows fallback text when description is missing', () => {
      const wrapper = mountComponent({
        nodePack: {
          description: undefined
        }
      })
      expect(wrapper.text()).toContain('No description available')
    })
  })
})
