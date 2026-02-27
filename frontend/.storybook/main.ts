import type { StorybookConfig } from '@storybook/vue3-vite'
import { FileSystemIconLoader } from 'unplugin-icons/loaders'
import IconsResolver from 'unplugin-icons/resolver'
import Icons from 'unplugin-icons/vite'
import Components from 'unplugin-vue-components/vite'
import type { InlineConfig } from 'vite'

const config: StorybookConfig = {
  stories: ['../src/**/*.stories.@(js|jsx|mjs|ts|tsx)'],
  addons: ['@storybook/addon-docs', '@storybook/addon-mcp'],
  framework: {
    name: '@storybook/vue3-vite',
    options: {}
  },
  async viteFinal(config) {
    // Use dynamic import to avoid CJS deprecation warning
    const { mergeConfig } = await import('vite')
    const { default: tailwindcss } = await import('@tailwindcss/vite')

    // Filter out any plugins that might generate import maps
    if (config.plugins) {
      config.plugins = config.plugins
        // Type guard: ensure we have valid plugin objects with names
        .filter(
          (plugin): plugin is NonNullable<typeof plugin> & { name: string } => {
            return (
              plugin !== null &&
              plugin !== undefined &&
              typeof plugin === 'object' &&
              'name' in plugin &&
              typeof plugin.name === 'string'
            )
          }
        )
        // Business logic: filter out import-map plugins
        .filter((plugin) => !plugin.name.includes('import-map'))
    }

    return mergeConfig(config, {
      // Replace plugins entirely to avoid inheritance issues
      plugins: [
        // Only include plugins we explicitly need for Storybook
        tailwindcss(),
        Icons({
          compiler: 'vue3',
          customCollections: {
            comfy: FileSystemIconLoader(
              process.cwd() + '/packages/design-system/src/icons'
            )
          }
        }),
        Components({
          dts: false, // Disable dts generation in Storybook
          resolvers: [
            IconsResolver({
              customCollections: ['comfy']
            })
          ],
          dirs: [
            process.cwd() + '/src/components',
            process.cwd() + '/src/layout',
            process.cwd() + '/src/views'
          ],
          deep: true,
          extensions: ['vue']
        })
      ],
      server: {
        allowedHosts: true
      },
      resolve: {
        alias: [
          {
            find: '@/composables/queue/useJobList',
            replacement: process.cwd() + '/src/storybook/mocks/useJobList.ts'
          },
          {
            find: '@/composables/queue/useJobActions',
            replacement: process.cwd() + '/src/storybook/mocks/useJobActions.ts'
          },
          {
            find: '@/utils/formatUtil',
            replacement:
              process.cwd() +
              '/packages/shared-frontend-utils/src/formatUtil.ts'
          },
          {
            find: '@/utils/networkUtil',
            replacement:
              process.cwd() +
              '/packages/shared-frontend-utils/src/networkUtil.ts'
          },
          {
            find: '@',
            replacement: process.cwd() + '/src'
          }
        ]
      },
      build: {
        rolldownOptions: {
          treeshake: false,
          output: {
            keepNames: true,
            strictExecutionOrder: true
          },
          onwarn: (warning, warn) => {
            // Suppress specific warnings
            if (
              warning.code === 'UNUSED_EXTERNAL_IMPORT' &&
              warning.message?.includes('resolveComponent')
            ) {
              return
            }
            // Suppress Storybook font asset warnings
            if (
              warning.code === 'UNRESOLVED_IMPORT' &&
              warning.message?.includes('nunito-sans')
            ) {
              return
            }
            warn(warning)
          }
        },
        chunkSizeWarningLimit: 1000
      }
    } satisfies InlineConfig)
  }
}
export default config
