import tailwindcss from '@tailwindcss/vite'
import vue from '@vitejs/plugin-vue'
import dotenv from 'dotenv'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { FileSystemIconLoader } from 'unplugin-icons/loaders'
import IconsResolver from 'unplugin-icons/resolver'
import Icons from 'unplugin-icons/vite'
import Components from 'unplugin-vue-components/vite'
import { defineConfig } from 'vite'
import { createHtmlPlugin } from 'vite-plugin-html'
import vueDevTools from 'vite-plugin-vue-devtools'

dotenv.config()

const projectRoot = fileURLToPath(new URL('.', import.meta.url))

const SHOULD_MINIFY = process.env.ENABLE_MINIFY === 'true'
const VITE_REMOTE_DEV = process.env.VITE_REMOTE_DEV === 'true'
const DISABLE_VUE_PLUGINS = process.env.DISABLE_VUE_PLUGINS === 'true'

export default defineConfig(() => {
  return {
    root: projectRoot,
    base: '',
    publicDir: path.resolve(projectRoot, 'public'),
    server: {
      port: 5174,
      host: VITE_REMOTE_DEV ? '0.0.0.0' : undefined
    },
    resolve: {
      alias: {
        '@': path.resolve(projectRoot, 'src'),
        '@frontend-locales': path.resolve(projectRoot, '../../src/locales')
      }
    },
    plugins: [
      ...(!DISABLE_VUE_PLUGINS
        ? [vueDevTools(), vue(), createHtmlPlugin({})]
        : [vue()]),
      tailwindcss(),
      Icons({
        compiler: 'vue3',
        customCollections: {
          comfy: FileSystemIconLoader(
            path.resolve(projectRoot, '../../packages/design-system/src/icons')
          )
        }
      }),
      Components({
        dts: path.resolve(projectRoot, 'components.d.ts'),
        resolvers: [
          IconsResolver({
            customCollections: ['comfy']
          })
        ],
        dirs: [
          path.resolve(projectRoot, 'src/components'),
          path.resolve(projectRoot, 'src/views')
        ],
        deep: true,
        extensions: ['vue'],
        directoryAsNamespace: true
      })
    ],
    build: {
      minify: SHOULD_MINIFY,
      target: 'es2022',
      sourcemap: true
    }
  }
})
