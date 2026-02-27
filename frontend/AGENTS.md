# Repository Guidelines

See @docs/guidance/\*.md for file-type-specific conventions (auto-loaded by glob).

## Project Structure & Module Organization

- Source: `src/`
  - Vue 3.5+
  - TypeScript
  - Tailwind 4
  - Key areas:
    - `components/`
    - `views/`
    - `stores/` (Pinia)
    - `composables/`
    - `services/`
    - `utils/`
    - `assets/`
    - `locales/`
- Routing: `src/router.ts`,
- i18n: `src/i18n.ts`,
- Entry Point: `src/main.ts`.
- Tests:
  - unit/component in `src/**/*.test.ts`
  - E2E (Playwright) in `browser_tests/**/*.spec.ts`
- Public assets: `public/`
- Build output: `dist/`
- Configs
  - `vite.config.mts`
  - `playwright.config.ts`
  - `eslint.config.ts`
  - `.oxfmtrc.json`
  - `.oxlintrc.json`
  - etc.

## Monorepo Architecture

The project uses **Nx** for build orchestration and task management

## Package Manager

This project uses **pnpm**. Always prefer scripts defined in `package.json` (e.g., `pnpm test:unit`, `pnpm lint`). To run arbitrary packages not in scripts, use `pnpx` or `pnpm dlx` — never `npx`.

## Build, Test, and Development Commands

- `pnpm dev`: Start Vite dev server.
- `pnpm dev:electron`: Dev server with Electron API mocks
- `pnpm build`: Type-check then production build to `dist/`
- `pnpm preview`: Preview the production build locally
- `pnpm test:unit`: Run Vitest unit tests
- `pnpm test:browser:local`: Run Playwright E2E tests (`browser_tests/`)
- `pnpm lint` / `pnpm lint:fix`: Lint (ESLint)
- `pnpm format` / `pnpm format:check`: oxfmt
- `pnpm typecheck`: Vue TSC type checking
- `pnpm storybook`: Start Storybook development server

## Development Workflow

1. Make code changes
2. Run relevant tests
3. Run `pnpm typecheck`, `pnpm lint`, `pnpm format`
4. Check if README updates are needed
5. Suggest docs.comfy.org updates for user-facing changes

## Git Conventions

- Use `prefix:` format: `feat:`, `fix:`, `test:`
- Add "Fixes #n" to PR descriptions
- Never mention Claude/AI in commits

## Coding Style & Naming Conventions

- Language:
  - TypeScript (exclusive, no new JavaScript)
  - Vue 3 SFCs (`.vue`)
    - Composition API only
  - Tailwind 4 styling
    - Avoid `<style>` blocks
- Style: (see `.oxfmtrc.json`)
  - Indent 2 spaces
  - single quotes
  - no trailing semicolons
  - width 80
- Imports:
  - sorted/grouped by plugin
  - run `pnpm format` before committing
  - use separate `import type` statements, not inline `type` in mixed imports
    - ✅ `import type { Foo } from './foo'` + `import { bar } from './foo'`
    - ❌ `import { bar, type Foo } from './foo'`
- ESLint:
  - Vue + TS rules
  - no floating promises
  - unused imports disallowed
  - i18n raw text restrictions in templates
- Naming:
  - Vue components in PascalCase (e.g., `MenuHamburger.vue`)
  - composables `useXyz.ts`
  - Pinia stores `*Store.ts`

## Commit & Pull Request Guidelines

- PRs:
  - Include clear description
  - Reference linked issues (e.g. `- Fixes #123`)
  - Keep it extremely concise and information-dense
  - Don't use emojis or add excessive headers/sections
  - Follow the PR description template in the `.github/` folder.
- Quality gates:
  - `pnpm lint`
  - `pnpm typecheck`
  - `pnpm knip`
  - Relevant tests must pass
- Never use `--no-verify` to bypass failing tests
  - Identify the issue and present root cause analysis and possible solutions if you are unable to solve quickly yourself
- Keep PRs focused and small
  - If it looks like the current changes will have 300+ lines of non-test code, suggest ways it could be broken into multiple PRs

## Security & Configuration Tips

- Secrets: Use `.env` (see `.env_example`); do not commit secrets.

## Vue 3 Composition API Best Practices

- Use `<script setup lang="ts">` for component logic
- Utilize `ref` for reactive state
- Implement computed properties with computed()
- Use watch and watchEffect for side effects
  - Avoid using a `ref` and a `watch` if a `computed` would work instead
- Implement lifecycle hooks with onMounted, onUpdated, etc.
- Utilize provide/inject for dependency injection
  - Do not use dependency injection if a Store or a shared composable would be simpler
- Use Vue 3.5 TypeScript style of default prop declaration
  - Example:

    ```typescript
    const { nodes, showTotal = true } = defineProps<{
      nodes: ApiNodeCost[]
      showTotal?: boolean
    }>()
    ```

  - Prefer reactive props destructuring to `const props = defineProps<...>`
  - Do not use `withDefaults` or runtime props declaration
  - Do not import Vue macros unnecessarily
  - Prefer `defineModel` to separately defining a prop and emit for v-model bindings
  - Define slots via template usage, not `defineSlots`
  - Use same-name shorthand for slot prop bindings: `:isExpanded` instead of `:is-expanded="isExpanded"`
  - Derive component types using `vue-component-type-helpers` (`ComponentProps`, `ComponentSlots`) instead of separate type files
  - Be judicious with addition of new refs or other state
    - If it's possible to accomplish the design goals with just a prop, don't add a `ref`
    - If it's possible to use the `ref` or prop directly, don't add a `computed`
    - If it's possible to use a `computed` to name and reuse a derived value, don't use a `watch`

## Development Guidelines

1. Leverage VueUse functions for performance-enhancing styles
2. Use es-toolkit for utility functions
3. Use TypeScript for type safety
4. If a complex type definition is inlined in multiple related places, extract and name it for reuse
5. In Vue Components, implement proper props and emits definitions
6. Utilize Vue 3's Teleport component when needed
7. Use Suspense for async components
8. Implement proper error handling
9. Follow Vue 3 style guide and naming conventions
10. Use Vite for fast development and building
11. Use vue-i18n in composition API for any string literals. Place new translation entries in src/locales/en/main.json. Use the plurals system in i18n instead of hardcoding pluralization in templates.
12. Avoid new usage of PrimeVue components
13. Write tests for all changes, especially bug fixes to catch future regressions
14. Write code that is expressive and self-documenting to the furthest degree possible. This reduces the need for code comments which can get out of sync with the code itself. Try to avoid comments unless absolutely necessary
15. Do not add or retain redundant comments, clean as you go
16. Whenever a new piece of code is written, the author should ask themselves 'is there a simpler way to introduce the same functionality?'. If the answer is yes, the simpler course should be chosen
17. [Refactoring](https://refactoring.com/catalog/) should be used to make complex code simpler
18. Try to minimize the surface area (exported values) of each module and composable
19. Don't use barrel files, e.g. `/some/package/index.ts` to re-export within `/src`
20. Keep functions short and functional
21. Minimize [nesting](https://wiki.c2.com/?ArrowAntiPattern), e.g. `if () { ... }` or `for () { ... }`
22. Avoid mutable state, prefer immutability and assignment at point of declaration
23. Favor pure functions (especially testable ones)
24. Do not use function expressions if it's possible to use function declarations instead
25. Watch out for [Code Smells](https://wiki.c2.com/?CodeSmell) and refactor to avoid them

## Testing Guidelines

See @docs/testing/\*.md for detailed patterns.

- Frameworks:
  - Vitest (unit/component, happy-dom)
  - Playwright (E2E)
- Test files:
  - Unit/Component: `**/*.test.ts`
  - E2E: `browser_tests/**/*.spec.ts`
  - Litegraph Specific: `src/lib/litegraph/test/`

### General

1. Do not write change detector tests  
   e.g. a test that just asserts that the defaults are certain values
2. Do not write tests that are dependent on non-behavioral features like utility classes or styles
3. Be parsimonious in testing, do not write redundant tests  
   See <https://tidyfirst.substack.com/p/composable-tests>
4. [Don’t Mock What You Don’t Own](https://hynek.me/articles/what-to-mock-in-5-mins/)

### Vitest / Unit Tests

1. Do not write tests that just test the mocks  
   Ensure that the tests fail when the code itself would behave in a way that was not expected or desired
2. For mocking, leverage [Vitest's utilities](https://vitest.dev/guide/mocking.html) where possible
3. Keep your module mocks contained  
   Do not use global mutable state within the test file  
   Use `vi.hoisted()` if necessary to allow for per-test Arrange phase manipulation of deeper mock state
4. For Component testing, use [Vue Test Utils](https://test-utils.vuejs.org/) and especially follow the advice [about making components easy to test](https://test-utils.vuejs.org/guide/essentials/easy-to-test.html)
5. Aim for behavioral coverage of critical and new features

### Playwright / Browser / E2E Tests

1. Follow the Best Practices described [in the Playwright documentation](https://playwright.dev/docs/best-practices)
2. Do not use waitForTimeout, use Locator actions and [retrying assertions](https://playwright.dev/docs/test-assertions#auto-retrying-assertions)
3. Tags like `@mobile`, `@2x` are respected by config and should be used for relevant tests

## External Resources

- Vue: <https://vuejs.org/api/>
- Tailwind: <https://tailwindcss.com/docs/styling-with-utility-classes>
- VueUse: <https://vueuse.org/functions.html>
- shadcn/vue: <https://www.shadcn-vue.com/>
- Reka UI: <https://reka-ui.com/>
- PrimeVue: <https://primevue.org>
- ComfyUI: <https://docs.comfy.org>
- Electron: <https://www.electronjs.org/docs/latest/>
- Wiki: <https://deepwiki.com/Comfy-Org/ComfyUI_frontend/1-overview>
- Nx: <https://nx.dev/docs/reference/nx-commands>
- [Practical Test Pyramid](https://martinfowler.com/articles/practical-test-pyramid.html)

## Project Philosophy

- Follow good software engineering principles
  - YAGNI
  - AHA
  - DRY
  - SOLID
- Clean, stable public APIs
- Domain-driven design
- Thousands of users and extensions
- Prioritize clean interfaces that restrict extension access

### Code Review

In doing a code review, you should make sure that:

- The code is well-designed.
- The functionality is good for the users of the code.
- Any UI changes are sensible and look good.
- Any parallel programming is done safely.
- The code isn’t more complex than it needs to be.
- The developer isn’t implementing things they might need in the future but don’t know they need now.
- Code has appropriate unit tests.
- Tests are well-designed.
- The developer used clear names for everything.
- Comments are clear and useful, and mostly explain why instead of what.
- Code is appropriately documented (generally in g3doc).
- The code conforms to our style guides.

#### [Complexity](https://google.github.io/eng-practices/review/reviewer/looking-for.html#complexity)

Is the CL more complex than it should be? Check this at every level of the CL—are individual lines too complex? Are functions too complex? Are classes too complex? “Too complex” usually means “can’t be understood quickly by code readers.” It can also mean “developers are likely to introduce bugs when they try to call or modify this code.”

A particular type of complexity is over-engineering, where developers have made the code more generic than it needs to be, or added functionality that isn’t presently needed by the system. Reviewers should be especially vigilant about over-engineering. Encourage developers to solve the problem they know needs to be solved now, not the problem that the developer speculates might need to be solved in the future. The future problem should be solved once it arrives and you can see its actual shape and requirements in the physical universe.

## Repository Navigation

- Check README files in key folders (browser_tests, composables, etc.)
- Prefer running single tests for performance
- Use --help for unfamiliar CLI tools

## GitHub Integration

When referencing Comfy-Org repos:

1. Check for local copy
2. Use GitHub API for branches/PRs/metadata
3. Curl GitHub website if needed

## Common Pitfalls

- NEVER use `any` type - use proper TypeScript types
- NEVER use `as any` type assertions - fix the underlying type issue
- NEVER use `--no-verify` flag when committing
- NEVER delete or disable tests to make them pass
- NEVER circumvent quality checks
- NEVER use the `dark:` tailwind variant
  - Instead use a semantic value from the `style.css` theme
    - e.g. `bg-node-component-surface`
- NEVER use `:class="[]"` to merge class names
  - Always use `import { cn } from '@/utils/tailwindUtil'`
    - e.g. `<div :class="cn('text-node-component-header-icon', hasError && 'text-danger')" />`
  - Use `cn()` inline in the template when feasible instead of creating a `computed` to hold the value
- NEVER use `!important` or the `!` important prefix for tailwind classes
  - Find existing `!important` classes that are interfering with the styling and propose corrections of those instead.
- NEVER use arbitrary percentage values like `w-[80%]` when a Tailwind fraction utility exists
  - Use `w-4/5` instead of `w-[80%]`, `w-1/2` instead of `w-[50%]`, etc.

## Agent-only rules

Rules for agent-based coding tasks.

### Chrome DevTools MCP

When using `take_snapshot` to inspect dropdowns, listboxes, or other components with dynamic options:

- Use `verbose: true` to see the full accessibility tree including list items
- Non-verbose snapshots often omit nested options in comboboxes/listboxes

### Temporary Files

- Put planning documents under `/temp/plans/`
- Put scripts used under `/temp/scripts/`
- Put summaries of work performed under `/temp/summaries/`
- Put TODOs and status updates under `/temp/in_progress/`
