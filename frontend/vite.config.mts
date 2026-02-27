import { sentryVitePlugin } from '@sentry/vite-plugin'
import tailwindcss from '@tailwindcss/vite'
import vue from '@vitejs/plugin-vue'
import { config as dotenvConfig } from 'dotenv'
import type { IncomingMessage, ServerResponse } from 'http'
import { Readable } from 'stream'
import type { ReadableStream as NodeReadableStream } from 'stream/web'
import { visualizer } from 'rollup-plugin-visualizer'
import { FileSystemIconLoader } from 'unplugin-icons/loaders'
import IconsResolver from 'unplugin-icons/resolver'
import Icons from 'unplugin-icons/vite'
import Components from 'unplugin-vue-components/vite'
import typegpuPlugin from 'unplugin-typegpu/vite'
import { defineConfig } from 'vitest/config'
import type { ProxyOptions } from 'vite'
import { createHtmlPlugin } from 'vite-plugin-html'
import vueDevTools from 'vite-plugin-vue-devtools'

import { comfyAPIPlugin } from './build/plugins'

dotenvConfig()

const IS_DEV = process.env.NODE_ENV === 'development'
const SHOULD_MINIFY = process.env.ENABLE_MINIFY === 'true'
const ANALYZE_BUNDLE = process.env.ANALYZE_BUNDLE === 'true'
// vite dev server will listen on all addresses, including LAN and public addresses
const VITE_REMOTE_DEV = process.env.VITE_REMOTE_DEV === 'true'
const DISABLE_TEMPLATES_PROXY = process.env.DISABLE_TEMPLATES_PROXY === 'true'
const GENERATE_SOURCEMAP = process.env.GENERATE_SOURCEMAP !== 'false'
const IS_STORYBOOK = process.env.npm_lifecycle_event === 'storybook'

// Open Graph / Twitter Meta Tags Constants
const VITE_OG_URL = 'https://cloud.comfy.org'
const VITE_OG_TITLE =
  'Comfy Cloud: Run ComfyUI online | Zero Setup, Powerful GPUs, Create anywhere'
const VITE_OG_DESC =
  'Bring your creative ideas to life with Comfy Cloud. Build and run your workflows to generate stunning images and videos instantly using powerful GPUs — all from your browser, no installation required.'
const VITE_OG_IMAGE = `${VITE_OG_URL}/assets/images/og-image.png`
const VITE_OG_KEYWORDS = 'ComfyUI, Comfy Cloud, ComfyUI online'

// Auto-detect cloud mode from DEV_SERVER_COMFYUI_URL
const DEV_SERVER_COMFYUI_ENV_URL = process.env.DEV_SERVER_COMFYUI_URL
const IS_CLOUD_URL = DEV_SERVER_COMFYUI_ENV_URL?.includes('.comfy.org')

const DISTRIBUTION: 'desktop' | 'localhost' | 'cloud' =
  process.env.DISTRIBUTION === 'desktop' ||
  process.env.DISTRIBUTION === 'localhost' ||
  process.env.DISTRIBUTION === 'cloud'
    ? process.env.DISTRIBUTION
    : IS_CLOUD_URL
      ? 'cloud'
      : 'localhost'

// Nightly builds are from main branch; RC/stable builds are from core/* branches
// Can be overridden via IS_NIGHTLY env var for testing
const IS_NIGHTLY = process.env.IS_NIGHTLY === 'true'

// Disable Vue DevTools for production cloud distribution
const DISABLE_VUE_PLUGINS =
  process.env.DISABLE_VUE_PLUGINS === 'true' ||
  (DISTRIBUTION === 'cloud' && !IS_DEV) ||
  IS_STORYBOOK

const DEV_SEVER_FALLBACK_URL =
  DISTRIBUTION === 'cloud'
    ? 'https://stagingcloud.comfy.org'
    : 'http://127.0.0.1:8188'

const DEV_SERVER_COMFYUI_URL =
  DEV_SERVER_COMFYUI_ENV_URL || DEV_SEVER_FALLBACK_URL

const cloudProxyConfig =
  DISTRIBUTION === 'cloud' ? { secure: false, changeOrigin: true } : {}

function handleGcsRedirect(
  proxyRes: IncomingMessage,
  _req: IncomingMessage,
  res: ServerResponse
) {
  const location = proxyRes.headers.location
  const isGcsRedirect =
    proxyRes.statusCode === 302 &&
    location?.includes('storage.googleapis.com') &&
    proxyRes.headers.via?.includes('google')

  // Not a GCS redirect - pass through normally
  if (!isGcsRedirect || !location) {
    Object.keys(proxyRes.headers).forEach((key) => {
      const value = proxyRes.headers[key]
      if (value !== undefined) {
        res.setHeader(key, value)
      }
    })
    res.writeHead(proxyRes.statusCode || 200)
    proxyRes.pipe(res)
    return
  }

  // GCS redirect detected - fetch server-side to avoid CORS
  fetch(location)
    .then(async (gcsResponse) => {
      if (!gcsResponse.body) {
        res.statusCode = 500
        res.end('Empty response from GCS')
        return
      }

      // Set response headers from GCS
      res.statusCode = 200
      res.setHeader(
        'Content-Type',
        gcsResponse.headers.get('content-type') || 'application/octet-stream'
      )

      const contentLength = gcsResponse.headers.get('content-length')
      if (contentLength) {
        res.setHeader('Content-Length', contentLength)
      }

      // Convert Web ReadableStream to Node.js stream and pipe to client
      const readable = Readable.fromWeb(gcsResponse.body as NodeReadableStream)
      readable.pipe(res)
    })
    .catch((error) => {
      console.error('Error fetching from GCS:', error)
      res.statusCode = 500
      res.end('Error fetching media')
    })
}

const gcsRedirectProxyConfig: ProxyOptions = {
  target: DEV_SERVER_COMFYUI_URL,
  ...cloudProxyConfig,
  selfHandleResponse: true,
  configure: (proxy) => {
    proxy.on('proxyRes', handleGcsRedirect)
  }
}

export default defineConfig({
  base: DISTRIBUTION === 'cloud' ? '/' : '',
  server: {
    host: VITE_REMOTE_DEV ? '0.0.0.0' : undefined,
    watch: {
      ignored: [
        './browser_tests/**',
        './node_modules/**',
        './tests-ui/**',
        '.eslintcache',
        '.oxlintrc.json',
        '*.config.{ts,mts}',
        '**/.git/**',
        '**/.github/**',
        '**/.nx/**',
        '**/*.{test,spec,stories}.ts',
        '**/coverage/**',
        '**/dist/**',
        '**/playwright-report/**',
        '**/test-results/**'
      ]
    },
    proxy: {
      '/internal': {
        target: DEV_SERVER_COMFYUI_URL,
        ...cloudProxyConfig
      },

      ...(DISTRIBUTION === 'cloud'
        ? {
            '/api/view': gcsRedirectProxyConfig,
            '/api/viewvideo': gcsRedirectProxyConfig
          }
        : {}),

      '/api': {
        target: DEV_SERVER_COMFYUI_URL,
        ...cloudProxyConfig,
        bypass: (req, res, _options) => {
          if (!res) return null

          // Return empty array for extensions API as these modules
          // are not on vite's dev server.
          if (req.url === '/api/extensions') {
            res.end(JSON.stringify([]))
            return false
          }

          // Bypass multi-user auth check from staging (cloud only)
          if (DISTRIBUTION === 'cloud' && req.url === '/api/users') {
            res.setHeader('Content-Type', 'application/json')
            res.end(JSON.stringify({})) // Return empty object to simulate single-user mode
            return false
          }

          return null
        }
      },

      '/ws': {
        target: DEV_SERVER_COMFYUI_URL,
        ws: true,
        ...cloudProxyConfig
      },

      '/workflow_templates': {
        target: DEV_SERVER_COMFYUI_URL,
        ...cloudProxyConfig
      },

      '/extensions': {
        target: DEV_SERVER_COMFYUI_URL,
        changeOrigin: true,
        ...cloudProxyConfig
      },

      '/docs': {
        target: DEV_SERVER_COMFYUI_URL,
        changeOrigin: true,
        ...cloudProxyConfig
      },

      ...(!DISABLE_TEMPLATES_PROXY
        ? {
            '/templates': {
              target: DEV_SERVER_COMFYUI_URL,
              ...cloudProxyConfig
            }
          }
        : {}),

      '/testsubrouteindex': {
        target: 'http://localhost:5173',
        rewrite: (path) => path.substring('/testsubrouteindex'.length)
      }
    }
  },

  plugins: [
    ...(!DISABLE_VUE_PLUGINS
      ? [vueDevTools(), vue(), createHtmlPlugin({})]
      : [vue()]),
    tailwindcss(),
    typegpuPlugin({}),
    comfyAPIPlugin(IS_DEV),
    // Exclude proprietary ABCROM fonts from non-cloud builds
    {
      name: 'exclude-proprietary-fonts',
      generateBundle(_options, bundle) {
        if (DISTRIBUTION !== 'cloud') {
          // Remove ABCROM font files from bundle
          for (const [fileName] of Object.entries(bundle)) {
            if (/ABCROM.*\.(woff2?|ttf|otf)$/i.test(fileName)) {
              delete bundle[fileName]
            }
          }
        }
      }
    },
    // Inject legacy user stylesheet links for desktop/localhost only
    {
      name: 'inject-user-stylesheet-links',
      enforce: 'post',
      transformIndexHtml(html) {
        if (DISTRIBUTION === 'cloud') return html

        return {
          html,
          tags: [
            {
              tag: 'link',
              attrs: {
                rel: 'stylesheet',
                type: 'text/css',
                href: 'user.css'
              },
              injectTo: 'head-prepend'
            },
            {
              tag: 'link',
              attrs: {
                rel: 'stylesheet',
                type: 'text/css',
                href: 'api/userdata/user.css'
              },
              injectTo: 'head-prepend'
            }
          ]
        }
      }
    },

    // Twitter/Open Graph meta tags plugin (cloud distribution only)
    {
      name: 'inject-twitter-meta',
      transformIndexHtml(html) {
        if (DISTRIBUTION !== 'cloud') return html

        return {
          html,
          tags: [
            // Basic SEO
            { tag: 'title', children: VITE_OG_TITLE, injectTo: 'head' },
            {
              tag: 'meta',
              attrs: { name: 'description', content: VITE_OG_DESC },
              injectTo: 'head'
            },
            {
              tag: 'meta',
              attrs: { name: 'keywords', content: VITE_OG_KEYWORDS },
              injectTo: 'head'
            },

            // Twitter Card tags
            {
              tag: 'meta',
              attrs: { name: 'twitter:card', content: 'summary_large_image' },
              injectTo: 'head'
            },
            {
              tag: 'meta',
              attrs: { name: 'twitter:title', content: VITE_OG_TITLE },
              injectTo: 'head'
            },
            {
              tag: 'meta',
              attrs: { name: 'twitter:description', content: VITE_OG_DESC },
              injectTo: 'head'
            },
            {
              tag: 'meta',
              attrs: { name: 'twitter:image', content: VITE_OG_IMAGE },
              injectTo: 'head'
            },

            // Open Graph tags (Twitter fallback & other platforms)
            {
              tag: 'meta',
              attrs: { property: 'og:title', content: VITE_OG_TITLE },
              injectTo: 'head'
            },
            {
              tag: 'meta',
              attrs: { property: 'og:description', content: VITE_OG_DESC },
              injectTo: 'head'
            },
            {
              tag: 'meta',
              attrs: { property: 'og:image', content: VITE_OG_IMAGE },
              injectTo: 'head'
            },
            {
              tag: 'meta',
              attrs: { property: 'og:url', content: VITE_OG_URL },
              injectTo: 'head'
            },
            {
              tag: 'meta',
              attrs: { property: 'og:type', content: 'website' },
              injectTo: 'head'
            },
            {
              tag: 'meta',
              attrs: { property: 'og:site_name', content: 'Comfy Cloud' },
              injectTo: 'head'
            },
            {
              tag: 'meta',
              attrs: { property: 'og:locale', content: 'en_US' },
              injectTo: 'head'
            }
          ]
        }
      }
    },
    Icons({
      compiler: 'vue3',
      customCollections: {
        comfy: FileSystemIconLoader('packages/design-system/src/icons')
      }
    }),

    Components({
      dts: true,
      resolvers: [
        IconsResolver({
          customCollections: ['comfy']
        })
      ],
      dirs: ['src/components', 'src/layout', 'src/views'],
      deep: true,
      extensions: ['vue'],
      directoryAsNamespace: true
    }),

    // Bundle analyzer - generates dist/stats.html after build
    // Only enabled when ANALYZE_BUNDLE=true
    ...(ANALYZE_BUNDLE
      ? [
          visualizer({
            filename: 'dist/stats.html',
            open: true,
            gzipSize: true,
            brotliSize: true,
            template: 'treemap' // or 'sunburst', 'network'
          })
        ]
      : []),

    // Sentry sourcemap upload plugin
    // Only runs during cloud production builds when all Sentry env vars are present
    // Requires: SENTRY_AUTH_TOKEN, SENTRY_ORG, SENTRY_PROJECT env vars
    ...(DISTRIBUTION === 'cloud' &&
    process.env.SENTRY_AUTH_TOKEN &&
    process.env.SENTRY_ORG &&
    process.env.SENTRY_PROJECT &&
    !IS_DEV
      ? [
          sentryVitePlugin({
            org: process.env.SENTRY_ORG,
            project: process.env.SENTRY_PROJECT,
            authToken: process.env.SENTRY_AUTH_TOKEN,
            sourcemaps: {
              // Delete source maps after upload to prevent public access
              filesToDeleteAfterUpload: ['**/*.map']
            }
          })
        ]
      : [])
  ],

  build: {
    minify: SHOULD_MINIFY,
    target: 'es2022',
    sourcemap: GENERATE_SOURCEMAP,
    // Exclude heavy optional vendor chunks from initial module preload
    // These chunks are only needed when their features are used (3D, terminal, etc.)
    modulePreload: {
      resolveDependencies: (_filename, deps, { hostType }) => {
        // Only filter for HTML entry points, not for dynamic imports
        if (hostType !== 'html') return deps

        // Exclude heavy vendor chunks that should be lazy-loaded
        // - vendor-three: 3D preview (Load3D nodes)
        // - vendor-xterm: Terminal emulator (logs panel)
        // - vendor-tiptap: Rich text editor (markdown widgets)
        // - vendor-chart: Chart.js (stats/monitoring)
        // - vendor-yjs: CRDT library (layout store, loaded on first graph)
        const lazyVendors = [
          'vendor-three',
          'vendor-xterm',
          'vendor-tiptap',
          'vendor-chart',
          'vendor-yjs'
        ]
        return deps.filter(
          (dep) => !lazyVendors.some((vendor) => dep.includes(vendor))
        )
      }
    },
    rolldownOptions: {
      treeshake: {
        manualPureFunctions: [
          'console.clear',
          'console.count',
          'console.countReset',
          'console.debug',
          'console.dir',
          'console.dirxml',
          'console.group',
          'console.groupCollapsed',
          'console.groupEnd',
          'console.info',
          'console.log',
          'console.profile',
          'console.profileEnd',
          'console.table',
          'console.time',
          'console.timeEnd',
          'console.timeLog',
          'console.trace'
        ]
      },
      output: {
        keepNames: true,
        codeSplitting: {
          groups: [
            // Framework core - highest priority, very stable
            {
              name: 'vendor-vue-core',
              test: /[\\/]node_modules[\\/](vue|@vue|pinia|vue-router)[\\/]/,
              priority: 20
            },

            {
              name: 'vendor-firebase',
              test: /[\\/]node_modules[\\/](@?firebase|@firebase)[\\/]/,
              priority: 15
            },
            {
              name: 'vendor-sentry',
              test: /[\\/]node_modules[\\/]@sentry[\\/]/,
              priority: 15
            },

            // UI component libraries
            {
              name: 'vendor-primevue',
              test: /[\\/]node_modules[\\/](@?primevue|@primeuix)[\\/]/,
              priority: 15
            },
            {
              name: 'vendor-reka-ui',
              test: /[\\/]node_modules[\\/]reka-ui[\\/]/,
              priority: 15
            },

            // Heavy optional features
            {
              name: 'vendor-three',
              test: /[\\/]node_modules[\\/](three|@sparkjsdev)[\\/]/,
              priority: 15
            },
            {
              name: 'vendor-tiptap',
              test: /[\\/]node_modules[\\/]@tiptap[\\/]/,
              priority: 15
            },
            {
              name: 'vendor-chart',
              test: /[\\/]node_modules[\\/]chart\.js[\\/]/,
              priority: 15
            },
            {
              name: 'vendor-xterm',
              test: /[\\/]node_modules[\\/]@xterm[\\/]/,
              priority: 15
            },
            {
              name: 'vendor-yjs',
              test: /[\\/]node_modules[\\/](yjs|lib0)[\\/]/,
              priority: 15
            },

            // Utilities and validation
            {
              name: 'vendor-vueuse',
              test: /[\\/]node_modules[\\/]@vueuse[\\/]/,
              priority: 12
            },
            {
              name: 'vendor-i18n',
              test: /[\\/]node_modules[\\/](vue-i18n|@intlify)[\\/]/,
              priority: 12
            },
            {
              name: 'vendor-zod',
              test: /[\\/]node_modules[\\/](zod|zod-validation-error)[\\/]/,
              priority: 12
            },
            {
              name: 'vendor-axios',
              test: /[\\/]node_modules[\\/]axios[\\/]/,
              priority: 12
            },
            {
              name: 'vendor-markdown',
              test: /[\\/]node_modules[\\/](marked|dompurify)[\\/]/,
              priority: 12
            },

            // Catch-all for remaining node_modules
            {
              name: 'vendor-other',
              test: /[\\/]node_modules[\\/]/,
              priority: 0,
              minSize: 10000
            }
          ]
        }
      }
    }
  },

  define: {
    __COMFYUI_FRONTEND_VERSION__: JSON.stringify(
      process.env.npm_package_version
    ),
    __SENTRY_ENABLED__: JSON.stringify(
      !(process.env.NODE_ENV === 'development' || !process.env.SENTRY_DSN)
    ),
    __SENTRY_DSN__: JSON.stringify(process.env.SENTRY_DSN || ''),
    __ALGOLIA_APP_ID__: JSON.stringify(process.env.ALGOLIA_APP_ID || ''),
    __ALGOLIA_API_KEY__: JSON.stringify(process.env.ALGOLIA_API_KEY || ''),
    __USE_PROD_CONFIG__: process.env.USE_PROD_CONFIG === 'true',
    __DISTRIBUTION__: JSON.stringify(DISTRIBUTION),
    __IS_NIGHTLY__: JSON.stringify(IS_NIGHTLY)
  },

  resolve: {
    alias: {
      '@/utils/formatUtil': '/packages/shared-frontend-utils/src/formatUtil.ts',
      '@/utils/networkUtil':
        '/packages/shared-frontend-utils/src/networkUtil.ts',
      '@': '/src'
    }
  },

  optimizeDeps: {
    exclude: ['@comfyorg/comfyui-electron-types'],
    entries: ['index.html']
  },

  test: {
    globals: true,
    environment: 'happy-dom',
    setupFiles: ['./vitest.setup.ts'],
    retry: process.env.CI ? 2 : 0,
    include: [
      'src/**/*.{test,spec}.{js,mjs,cjs,ts,mts,cts,jsx,tsx}',
      'packages/**/*.{test,spec}.{js,mjs,cjs,ts,mts,cts,jsx,tsx}'
    ],
    coverage: {
      reporter: ['text', 'json', 'html']
    },
    exclude: [
      '**/node_modules/**',
      '**/dist/**',
      '**/cypress/**',
      '**/.{idea,git,cache,output,temp}/**',
      '**/{karma,rollup,webpack,vite,vitest,jest,ava,babel,nyc,cypress,tsup,build,eslint}.config.*',
      '**/.{oxlintrc,oxfmtrc}.json'
    ],
    silent: 'passed-only'
  }
})
