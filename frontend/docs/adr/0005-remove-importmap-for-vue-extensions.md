# 5. Remove Import Map for Vue Extensions

Date: 2025-12-13

## Status

Accepted

## Context

ComfyUI frontend previously used a Vite plugin (`generateImportMapPlugin`) to inject an HTML import map exposing shared modules to extensions. This allowed Vue-based extensions to mark dependencies as external in their Vite configs:

```typescript
// Extension vite.config.ts (old pattern)
rollupOptions: {
  external: ['vue', 'vue-i18n', 'pinia', /^primevue\/?.*/, ...]
}
```

The import map resolved bare specifiers like `import { ref } from 'vue'` at runtime by mapping them to pre-built ESM files served from `/assets/lib/`.

**Modules exposed via import map:**

- `vue` (vue.esm-browser.prod.js)
- `vue-i18n` (vue-i18n.esm-browser.prod.js)
- `primevue/*` (all PrimeVue components)
- `@primevue/themes/*`
- `@primevue/forms/*`

**Problems with import map approach:**

1. **Blocked tree shaking**: Vue and PrimeVue loaded as remote modules at runtime, preventing bundler optimizations. The entire Vue runtime was loaded even if only a few APIs were used.

2. **Poor code splitting**: PrimeVue's component library split into hundreds of small chunks, each requiring a separate network request on mount. This significantly impacted initial page load.

3. **Cold start performance**: Each externalized module required a separate HTTP request and browser module resolution step. This compounded on lower-end systems and slower networks.

4. **Version alignment complexity**: Extensions relied on the frontend's Vue version at runtime. Subtle version mismatches between build-time types and runtime code caused debugging difficulties.

5. **Incompatible with Cloud distribution**: The Cloud deployment model requires fully bundled, optimized assets. Import maps added a layer of indirection incompatible with our CDN and caching strategy.

## Decision

Remove the `generateImportMapPlugin` and require Vue-based extensions to bundle their own Vue instance.

**Implementation (PR #6899):**

- Deleted `build/plugins/generateImportMapPlugin.ts`
- Removed plugin configuration from `vite.config.mts`
- Removed `fast-glob` dependency used by the plugin

**Extension migration path:**

1. Remove `external: ['vue', ...]` from Vite rollup options
2. Vue and related dependencies will be bundled into the extension output
3. No code changes required in extension source files

The import map was already disabled for Cloud builds (PR #6559) before complete removal. Removal aligns all distribution channels on the same bundling strategy.

## Consequences

### Positive

- **Improved page load**: Full tree shaking and optimal code splitting now apply to Vue and PrimeVue
- **Faster development**: No import map generation step; simplified build pipeline
- **Better debugging**: Extension's bundled Vue matches build-time expectations exactly
- **Cloud compatibility**: All assets fully bundled and CDN-optimizable
- **Consistent behavior**: Same bundling strategy across desktop, localhost, and cloud distributions
- **Reduced network requests**: Fewer module fetches on initial page load

### Negative

- **Breaking change for existing extensions**: Extensions using `external: ['vue']` pattern fail with "Failed to resolve module specifier 'vue'" error
- **Larger extension bundles**: Each extension now includes its own Vue instance (~30KB gzipped)
- **Potential version fragmentation**: Different extensions may bundle different Vue versions (mitigated by Vue's stable API)

### Migration Impact

Extensions affected must update their build configuration. The migration is straightforward:

```diff
// vite.config.ts
rollupOptions: {
-  external: ['vue', 'vue-i18n', 'primevue', ...]
}
```

Affected versions:

- **v1.32.x - v1.33.8**: Import map present, external pattern works
- **v1.33.9+**: Import map removed, bundling required

## Notes

- [ComfyUI_frontend_vue_basic](https://github.com/jtydhr88/ComfyUI_frontend_vue_basic) has been updated to demonstrate the new bundled pattern
- Issue #7267 documents the user-facing impact and migration discussion
- Future Extension API v2 (Issue #4668) may provide alternative mechanisms for shared dependencies
