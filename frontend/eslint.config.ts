// For more info, see https://github.com/storybookjs/eslint-plugin-storybook#configuration-flat-config-format
import pluginJs from '@eslint/js'
import pluginI18n from '@intlify/eslint-plugin-vue-i18n'
import { createTypeScriptImportResolver } from 'eslint-import-resolver-typescript'
import { importX } from 'eslint-plugin-import-x'
import oxlint from 'eslint-plugin-oxlint'
// eslint-config-prettier disables ESLint rules that conflict with formatters (oxfmt)
import eslintConfigPrettier from 'eslint-config-prettier'
import { configs as storybookConfigs } from 'eslint-plugin-storybook'
import unusedImports from 'eslint-plugin-unused-imports'
import pluginVue from 'eslint-plugin-vue'
import { defineConfig } from 'eslint/config'
import globals from 'globals'
import {
  configs as tseslintConfigs,
  parser as tseslintParser
} from 'typescript-eslint'
import vueParser from 'vue-eslint-parser'
import path from 'node:path'

const extraFileExtensions = ['.vue']

const commonGlobals = {
  ...globals.browser,
  __COMFYUI_FRONTEND_VERSION__: 'readonly',
  __DISTRIBUTION__: 'readonly',
  __IS_NIGHTLY__: 'readonly'
} as const

const settings = {
  'import-x/resolver-next': [
    createTypeScriptImportResolver({
      alwaysTryTypes: true,
      project: [
        './tsconfig.json',
        './apps/*/tsconfig.json',
        './packages/*/tsconfig.json'
      ],
      noWarnOnMultipleProjects: true
    })
  ],
  'vue-i18n': {
    localeDir: [
      {
        pattern: './src/locales/**/*.json',
        localeKey: 'path',
        localePattern:
          /^\.?\/?src\/locales\/(?<locale>[A-Za-z0-9-]+)\/.+\.json$/
      }
    ],
    messageSyntaxVersion: '^9.0.0'
  }
} as const

const commonParserOptions = {
  parser: tseslintParser,
  projectService: true,
  tsConfigRootDir: import.meta.dirname,
  ecmaVersion: 2020,
  sourceType: 'module',
  extraFileExtensions
} as const

export default defineConfig([
  {
    ignores: [
      '.i18nrc.cjs',
      '.nx/*',
      '**/vite.config.*.timestamp*',
      '**/vitest.config.*.timestamp*',
      'components.d.ts',
      'coverage/*',
      'dist/*',
      'packages/registry-types/src/comfyRegistryTypes.ts',
      'playwright-report/*',
      'src/extensions/core/*',
      'src/scripts/*',
      'src/types/generatedManagerTypes.ts',
      'src/types/vue-shim.d.ts',
      'test-results/*',
      'vitest.setup.ts'
    ]
  },
  {
    files: ['./**/*.{ts,mts}'],
    settings,
    languageOptions: {
      globals: commonGlobals,
      parserOptions: {
        ...commonParserOptions,
        projectService: {
          allowDefaultProject: [
            'vite.electron.config.mts',
            'vite.types.config.mts'
          ]
        }
      }
    }
  },
  {
    files: ['./**/*.vue'],
    settings,
    languageOptions: {
      globals: commonGlobals,
      parser: vueParser,
      parserOptions: commonParserOptions
    }
  },
  pluginJs.configs.recommended,

  tseslintConfigs.recommended,
  // Difference in typecheck on CI vs Local
  pluginVue.configs['flat/recommended'],
  // Disables ESLint rules that conflict with formatters
  eslintConfigPrettier,
  // @ts-expect-error Type incompatibility between storybook plugin and ESLint config types
  storybookConfigs['flat/recommended'],
  // @ts-expect-error Type incompatibility between import-x plugin and ESLint config types
  importX.flatConfigs.recommended,
  // @ts-expect-error Type incompatibility between import-x plugin and ESLint config types
  importX.flatConfigs.typescript,
  {
    plugins: {
      'unused-imports': unusedImports,
      // @ts-expect-error Type incompatibility in i18n plugin
      '@intlify/vue-i18n': pluginI18n
    },
    rules: {
      '@typescript-eslint/no-explicit-any': 'off',
      '@typescript-eslint/no-unused-vars': 'off',
      '@typescript-eslint/prefer-as-const': 'off',
      '@typescript-eslint/consistent-type-imports': 'error',
      '@typescript-eslint/no-import-type-side-effects': 'error',
      '@typescript-eslint/no-empty-object-type': [
        'error',
        {
          allowInterfaces: 'always'
        }
      ],
      'import-x/no-useless-path-segments': 'error',
      'import-x/no-relative-packages': 'error',
      'unused-imports/no-unused-imports': 'error',
      'vue/no-v-html': 'off',
      // Prohibit dark-theme: and dark: prefixes
      'vue/no-restricted-class': ['error', '/^dark(-theme)?:/'],
      'vue/multi-word-component-names': 'off', // TODO: fix
      'vue/no-template-shadow': 'off', // TODO: fix
      'vue/match-component-import-name': 'error',
      'vue/no-unused-properties': 'error',
      'vue/no-unused-refs': 'error',
      'vue/no-useless-mustaches': 'error',
      'vue/no-useless-v-bind': 'error',
      'vue/no-unused-emit-declarations': 'error',
      'vue/no-use-v-else-with-v-for': 'error',
      'vue/one-component-per-file': 'error',
      'vue/require-default-prop': 'off', // TODO: fix -- this one is very worthwhile

      // i18n rules
      '@intlify/vue-i18n/no-raw-text': [
        'error',
        {
          attributes: {
            '/.+/': [
              'aria-label',
              'aria-placeholder',
              'aria-roledescription',
              'aria-valuetext',
              'label',
              'placeholder',
              'title',
              'v-tooltip'
            ],
            img: ['alt']
          },
          // Ignore strings that are:
          // 1. Less than 2 characters
          // 2. Only symbols/numbers/whitespace (no letters)
          // 3. Match specific patterns
          ignorePattern:
            '^[^a-zA-Z]*$|^.{0,1}$|^[\\w._%+-]+@[\\w.-]+\\.[A-Za-z]{2,}$',
          ignoreNodes: ['md-icon', 'v-icon', 'pre', 'code', 'script', 'style'],
          // Brand names and technical terms that shouldn't be translated
          ignoreText: [
            'API',
            'App Data:',
            'App Path:',
            'ComfyUI',
            'CPU',
            'fps',
            'GB',
            'GitHub',
            'GPU',
            'JSON',
            'KB',
            'LoRA',
            'MB',
            'ms',
            'OpenAI',
            'png',
            'px',
            'RAM',
            'URL',
            'YAML',
            '1.2 MB'
          ]
        }
      ]
    }
  },
  {
    files: ['tests-ui/**/*'],
    rules: {
      '@typescript-eslint/consistent-type-imports': [
        'error',
        { disallowTypeAnnotations: false }
      ]
    }
  },
  {
    files: ['**/*.spec.ts'],
    ignores: ['browser_tests/tests/**/*.spec.ts'],
    rules: {
      'no-restricted-syntax': [
        'error',
        {
          selector: 'Program',
          message: '.spec.ts files are only allowed under browser_tests/tests/'
        }
      ]
    }
  },
  {
    files: ['browser_tests/tests/**/*.test.ts'],
    rules: {
      'no-restricted-syntax': [
        'error',
        {
          selector: 'Program',
          message:
            '.test.ts files are not allowed in browser_tests/tests/; use .spec.ts instead'
        }
      ]
    }
  },
  {
    files: ['**/*.test.ts'],
    rules: {
      'no-restricted-properties': [
        'error',
        {
          object: 'vi',
          property: 'doMock',
          message:
            'Use vi.mock() with vi.hoisted() instead of vi.doMock(). See docs/testing/vitest-patterns.md'
        }
      ]
    }
  },
  {
    files: ['scripts/**/*.js'],
    languageOptions: {
      globals: {
        ...globals.node
      }
    },
    rules: {
      '@typescript-eslint/no-floating-promises': 'off',
      'no-console': 'off'
    }
  },

  // Turn off ESLint rules that are already handled by oxlint
  ...oxlint.buildFromOxlintConfigFile(
    path.resolve(import.meta.dirname, '.oxlintrc.json')
  ),
  {
    rules: {
      'import-x/default': 'off',
      'import-x/export': 'off',
      'import-x/namespace': 'off',
      'import-x/no-duplicates': 'off',
      'import-x/consistent-type-specifier-style': 'off'
    }
  },

  // i18n import enforcement
  // Vue components must use the useI18n() composable, not the global t/d/st/te
  {
    files: ['**/*.vue'],
    rules: {
      'no-restricted-imports': [
        'error',
        {
          paths: [
            {
              name: '@/i18n',
              importNames: ['t', 'd', 'te'],
              message:
                "In Vue components, use `const { t } = useI18n()` instead of importing from '@/i18n'."
            }
          ]
        }
      ]
    }
  },
  // Non-composable .ts files must use the global t/d/te, not useI18n()
  {
    files: ['**/*.ts'],
    ignores: ['**/use[A-Z]*.ts', '**/*.test.ts', 'src/i18n.ts'],
    rules: {
      'no-restricted-imports': [
        'error',
        {
          paths: [
            {
              name: 'vue-i18n',
              importNames: ['useI18n'],
              message:
                "useI18n() requires Vue setup context. Use `import { t } from '@/i18n'` instead."
            }
          ]
        }
      ]
    }
  }
])
