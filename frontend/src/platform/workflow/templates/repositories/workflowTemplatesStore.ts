import { defineStore } from 'pinia'
import { computed, ref, shallowRef, watch } from 'vue'

import { i18n, st } from '@/i18n'
import { isCloud } from '@/platform/distribution/types'
import { api } from '@/scripts/api'
import type { NavGroupData, NavItemData } from '@/types/navTypes'
import { generateCategoryId, getCategoryIcon } from '@/utils/categoryUtil'
import { normalizeI18nKey } from '@/utils/formatUtil'

import { zLogoIndex } from '../schemas/templateSchema'
import type { LogoIndex } from '../schemas/templateSchema'
import type {
  TemplateGroup,
  TemplateInfo,
  WorkflowTemplates
} from '../types/template'

// Enhanced template interface for easier filtering
interface EnhancedTemplate extends TemplateInfo {
  sourceModule: string
  category?: string
  categoryType?: string
  categoryGroup?: string // 'GENERATION TYPE' or 'CLOSED SOURCE MODELS'
  isEssential?: boolean
  isPartnerNode?: boolean // Computed from OpenSource === false
  searchableText?: string
}

export const useWorkflowTemplatesStore = defineStore(
  'workflowTemplates',
  () => {
    const customTemplates = shallowRef<{ [moduleName: string]: string[] }>({})
    const coreTemplates = shallowRef<WorkflowTemplates[]>([])
    const englishTemplates = shallowRef<WorkflowTemplates[]>([])
    const logoIndex = shallowRef<LogoIndex>({})
    const isLoaded = ref(false)
    const knownTemplateNames = ref(new Set<string>())

    const getTemplateByName = (name: string): EnhancedTemplate | undefined => {
      return enhancedTemplates.value.find((template) => template.name === name)
    }

    // Store filter mappings for dynamic categories
    type FilterData = {
      category?: string
      categoryGroup?: string
    }

    const categoryFilters = ref(new Map<string, FilterData>())

    /**
     * Add localization fields to a template.
     */
    const addLocalizedFieldsToTemplate = (
      template: TemplateInfo,
      categoryTitle: string
    ) => ({
      ...template,
      localizedTitle: st(
        `templateWorkflows.template.${normalizeI18nKey(categoryTitle)}.${normalizeI18nKey(template.name)}`,
        template.title ?? template.name
      ),
      localizedDescription: st(
        `templateWorkflows.templateDescription.${normalizeI18nKey(categoryTitle)}.${normalizeI18nKey(template.name)}`,
        template.description
      )
    })

    /**
     * Add localization fields to all templates in a list of templates.
     */
    const localizeTemplateList = (
      templates: TemplateInfo[],
      categoryTitle: string
    ) =>
      templates.map((template) =>
        addLocalizedFieldsToTemplate(template, categoryTitle)
      )

    /**
     * Add localization fields to a template category and all its constituent templates.
     */
    const localizeTemplateCategory = (templateCategory: WorkflowTemplates) => ({
      ...templateCategory,
      localizedTitle: st(
        `templateWorkflows.category.${normalizeI18nKey(templateCategory.title)}`,
        templateCategory.title ?? templateCategory.moduleName
      ),
      templates: localizeTemplateList(
        templateCategory.templates,
        templateCategory.title
      )
    })

    // Create an "All" category that combines all templates
    const createAllCategory = () => {
      // First, get core templates with source module added
      const coreTemplatesWithSourceModule = coreTemplates.value.flatMap(
        (category) =>
          // For each template in each category, add the sourceModule and pass through any localized fields
          category.templates.map((template) => {
            // Get localized template with its original category title for i18n lookup
            const localizedTemplate = addLocalizedFieldsToTemplate(
              template,
              category.title
            )
            return {
              ...localizedTemplate,
              sourceModule: category.moduleName
            }
          })
      )

      // Now handle custom templates
      const customTemplatesWithSourceModule = Object.entries(
        customTemplates.value
      ).flatMap(([moduleName, templates]) =>
        templates.map((name) => ({
          name,
          mediaType: 'image',
          mediaSubtype: 'jpg',
          description: name,
          sourceModule: moduleName
        }))
      )

      return {
        moduleName: 'all',
        title: 'All',
        localizedTitle: st('templateWorkflows.category.All', 'All Templates'),
        templates: [
          ...coreTemplatesWithSourceModule,
          ...customTemplatesWithSourceModule
        ]
      }
    }

    /**
     * Original grouped templates for backward compatibility
     */
    const groupedTemplates = computed<TemplateGroup[]>(() => {
      // Get regular categories
      const allTemplates = [
        ...coreTemplates.value.map(localizeTemplateCategory),
        ...Object.entries(customTemplates.value).map(
          ([moduleName, templates]) => ({
            moduleName,
            title: moduleName,
            localizedTitle: st(
              `templateWorkflows.category.${normalizeI18nKey(moduleName)}`,
              moduleName
            ),
            templates: templates.map((name) => ({
              name,
              mediaType: 'image',
              mediaSubtype: 'jpg',
              description: name
            }))
          })
        )
      ]

      // Group templates by their main category
      const groupedByCategory = [
        {
          label: st(
            'templateWorkflows.category.ComfyUI Examples',
            'ComfyUI Examples'
          ),
          modules: [
            createAllCategory(),
            ...allTemplates.filter((t) => t.moduleName === 'default')
          ]
        },
        ...(Object.keys(customTemplates.value).length > 0
          ? [
              {
                label: st(
                  'templateWorkflows.category.Custom Nodes',
                  'Custom Nodes'
                ),
                modules: allTemplates.filter((t) => t.moduleName !== 'default')
              }
            ]
          : [])
      ]

      return groupedByCategory
    })

    /**
     * Enhanced templates with proper categorization for filtering
     */
    const enhancedTemplates = computed<EnhancedTemplate[]>(() => {
      const allTemplates: EnhancedTemplate[] = []

      // Process core templates
      coreTemplates.value.forEach((category) => {
        category.templates.forEach((template) => {
          const enhancedTemplate: EnhancedTemplate = {
            ...template,
            sourceModule: category.moduleName,
            category: category.title,
            categoryType: category.type,
            categoryGroup: category.category,
            isEssential: category.isEssential,
            isPartnerNode: template.openSource === false,
            searchableText: [
              template.title || template.name,
              template.description || '',
              category.title,
              ...(template.tags || []),
              ...(template.models || [])
            ].join(' ')
          }

          allTemplates.push(enhancedTemplate)
        })
      })

      // Process custom templates
      Object.entries(customTemplates.value).forEach(
        ([moduleName, templates]) => {
          templates.forEach((name) => {
            const enhancedTemplate: EnhancedTemplate = {
              name,
              title: name,
              description: name,
              mediaType: 'image',
              mediaSubtype: 'jpg',
              sourceModule: moduleName,
              category: 'Extensions',
              categoryType: 'extension',
              searchableText: `${name} ${moduleName} extension`
            }
            allTemplates.push(enhancedTemplate)
          })
        }
      )

      // TODO: Temporary filtering of custom node templates on local installations
      // Future: Add UX that allows local users to opt-in to templates with custom nodes,
      // potentially conditional on whether they have those specific custom nodes installed.
      // This would provide better template discovery while respecting local user workflows.
      const filteredTemplates = isCloud
        ? allTemplates
        : allTemplates.filter(
            (template) => !template.requiresCustomNodes?.length
          )

      return filteredTemplates
    })

    /**
     * Filter templates by category ID using stored filter mappings
     */
    const filterTemplatesByCategory = (categoryId: string) => {
      if (categoryId === 'all') {
        return enhancedTemplates.value
      }

      if (categoryId.startsWith('basics-')) {
        // Filter for templates from categories marked as essential
        return enhancedTemplates.value.filter(
          (t) =>
            t.isEssential &&
            t.category?.toLowerCase().replace(/\s+/g, '-') ===
              categoryId.replace('basics-', '')
        )
      }

      if (categoryId === 'popular') {
        return enhancedTemplates.value
      }

      if (categoryId === 'partner-nodes') {
        // Filter for templates where OpenSource === false
        return enhancedTemplates.value.filter((t) => t.isPartnerNode)
      }

      // Handle extension-specific filters
      if (categoryId.startsWith('extension-')) {
        const moduleName = categoryId.replace('extension-', '')
        return enhancedTemplates.value.filter(
          (t) => t.sourceModule === moduleName
        )
      }

      // Look up the filter from our stored mappings
      const filter = categoryFilters.value.get(categoryId)
      if (!filter) {
        return enhancedTemplates.value
      }

      // Apply the filter
      return enhancedTemplates.value.filter((template) => {
        if (filter.category && template.category !== filter.category) {
          return false
        }
        if (
          filter.categoryGroup &&
          template.categoryGroup !== filter.categoryGroup
        ) {
          return false
        }
        return true
      })
    }

    /**
     * New navigation structure dynamically built from JSON categories
     */
    const navGroupedTemplates = computed<(NavItemData | NavGroupData)[]>(() => {
      if (!isLoaded.value) return []

      const items: (NavItemData | NavGroupData)[] = []

      // Clear and rebuild filter mappings
      categoryFilters.value.clear()

      // 1. All Templates - always first
      items.push({
        id: 'all',
        label: st('templateWorkflows.category.All', 'All Templates'),
        icon: getCategoryIcon('all')
      })

      // 1.5. Popular categories

      items.push({
        id: 'popular',
        label: st('templateWorkflows.category.Popular', 'Popular'),
        icon: 'icon-[lucide--flame]'
      })

      // 2. Basics (isEssential categories) - always beneath All Templates if they exist
      const essentialCats = coreTemplates.value.filter(
        (cat) => cat.isEssential && cat.templates.length > 0
      )

      if (essentialCats.length > 0) {
        essentialCats.forEach((essentialCat) => {
          const categoryIcon = essentialCat.icon
          const categoryTitle = essentialCat.title ?? 'Getting Started'
          const categoryId = generateCategoryId('basics', essentialCat.title)
          items.push({
            id: categoryId,
            label: st(
              `templateWorkflows.category.${normalizeI18nKey(categoryTitle)}`,
              categoryTitle
            ),
            icon:
              categoryIcon ||
              getCategoryIcon(essentialCat.type || 'getting-started')
          })
        })
      }

      // 3. Group categories from JSON dynamically
      const categoryGroups = new Map<
        string,
        { title: string; items: NavItemData[] }
      >()

      // Process all categories from JSON
      coreTemplates.value.forEach((category) => {
        // Skip essential categories as they're handled as Basics
        if (category.isEssential) return

        const categoryGroup = category.category
        const categoryIcon = category.icon

        if (categoryGroup) {
          if (!categoryGroups.has(categoryGroup)) {
            categoryGroups.set(categoryGroup, {
              title: categoryGroup,
              items: []
            })
          }

          const group = categoryGroups.get(categoryGroup)!

          // Generate unique ID for this category
          const categoryId = generateCategoryId(categoryGroup, category.title)

          // Store the filter mapping
          categoryFilters.value.set(categoryId, {
            category: category.title,
            categoryGroup: categoryGroup
          })

          group.items.push({
            id: categoryId,
            label: st(
              `templateWorkflows.category.${normalizeI18nKey(category.title)}`,
              category.title
            ),
            icon: categoryIcon || getCategoryIcon(category.type || 'default')
          })
        }
      })

      // Add grouped categories
      categoryGroups.forEach((group, groupName) => {
        if (group.items.length > 0) {
          items.push({
            title: st(
              `templateWorkflows.category.${normalizeI18nKey(groupName)}`,
              groupName
                .split(' ')
                .map(
                  (word) =>
                    word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()
                )
                .join(' ')
            ),
            items: group.items
          })
        }
      })

      // 3.5. Partner Nodes - virtual category for OpenSource === false templates
      const partnerNodeCount = enhancedTemplates.value.filter(
        (t) => t.isPartnerNode
      ).length

      if (partnerNodeCount > 0) {
        items.push({
          id: 'partner-nodes',
          label: st(
            'templateWorkflows.category.Partner Nodes',
            'Partner Nodes'
          ),
          icon: 'icon-[lucide--handshake]'
        })
      }

      // 4. Extensions - always last
      const extensionCounts = enhancedTemplates.value.filter(
        (t) => t.sourceModule !== 'default'
      ).length

      if (extensionCounts > 0) {
        // Get unique extension modules
        const extensionModules = Array.from(
          new Set(
            enhancedTemplates.value
              .filter((t) => t.sourceModule !== 'default')
              .map((t) => t.sourceModule)
          )
        ).sort()

        const extensionItems: NavItemData[] = extensionModules.map(
          (moduleName) => ({
            id: `extension-${moduleName}`,
            label: st(
              `templateWorkflows.category.${normalizeI18nKey(moduleName)}`,
              moduleName
            ),
            icon: getCategoryIcon('extensions')
          })
        )

        items.push({
          title: st('templateWorkflows.category.Extensions', 'Extensions'),
          items: extensionItems,
          collapsible: true
        })
      }

      return items
    })

    async function fetchCoreTemplates() {
      const locale = i18n.global.locale.value
      const [coreResult, englishResult, logoIndexResult] = await Promise.all([
        api.getCoreWorkflowTemplates(locale),
        isCloud && locale !== 'en'
          ? api.getCoreWorkflowTemplates('en')
          : Promise.resolve([]),
        fetchLogoIndex()
      ])

      coreTemplates.value = coreResult
      englishTemplates.value = englishResult
      logoIndex.value = logoIndexResult

      const coreNames = coreTemplates.value.flatMap((category) =>
        category.templates.map((template) => template.name)
      )
      const customNames = Object.values(customTemplates.value).flat()
      knownTemplateNames.value = new Set([...coreNames, ...customNames])
    }

    async function loadWorkflowTemplates() {
      try {
        if (!isLoaded.value) {
          customTemplates.value = await api.getWorkflowTemplates()
          await fetchCoreTemplates()
          isLoaded.value = true
        }
      } catch (error) {
        console.error('Error fetching workflow templates:', error)
      }
    }

    watch(
      () => i18n.global.locale.value,
      async () => {
        if (!isLoaded.value) return
        try {
          await fetchCoreTemplates()
        } catch (error) {
          console.error('Error reloading templates for new locale:', error)
        }
      }
    )

    async function fetchLogoIndex(): Promise<LogoIndex> {
      try {
        const response = await fetch(api.fileURL('/templates/index_logo.json'))
        const contentType = response.headers.get('content-type')
        if (!contentType?.includes('application/json')) return {}
        const data = await response.json()
        const result = zLogoIndex.safeParse(data)
        return result.success ? result.data : {}
      } catch {
        return {}
      }
    }

    function getLogoUrl(provider: string): string {
      const logoPath = logoIndex.value[provider]
      if (!logoPath) return ''

      // Validate path to prevent directory traversal and ensure safe file extensions
      const safePathPattern = /^[a-zA-Z0-9_\-./]+\.(png|jpg|jpeg|svg|webp)$/i
      if (
        !safePathPattern.test(logoPath) ||
        logoPath.includes('..') ||
        logoPath.startsWith('/')
      ) {
        return ''
      }

      return api.fileURL(`/templates/${logoPath}`)
    }

    function getEnglishMetadata(templateName: string): {
      tags?: string[]
      category?: string
      useCase?: string
      models?: string[]
      license?: string
    } | null {
      if (englishTemplates.value.length === 0) {
        return null
      }

      for (const category of englishTemplates.value) {
        const template = category.templates.find((t) => t.name === templateName)
        if (template) {
          return {
            tags: template.tags,
            category: category.title,
            useCase: template.useCase,
            models: template.models,
            license: template.license
          }
        }
      }

      return null
    }

    return {
      groupedTemplates,
      navGroupedTemplates,
      enhancedTemplates,
      filterTemplatesByCategory,
      isLoaded,
      loadWorkflowTemplates,
      knownTemplateNames,
      getTemplateByName,
      getEnglishMetadata,
      getLogoUrl
    }
  }
)
