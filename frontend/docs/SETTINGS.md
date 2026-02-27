# Settings System

## Overview

ComfyUI frontend uses a comprehensive settings system for user preferences with support for dynamic defaults, version-based rollouts, and environment-aware configuration.

### Settings Architecture

- Settings are defined as `SettingParams` in `src/constants/coreSettings.ts`
- Registered at app startup, loaded/saved via `useSettingStore` (Pinia)
- Persisted per user via backend `/settings` endpoint
- If a value hasn't been set by the user, the store returns the computed default

```typescript
// From src/stores/settingStore.ts:105-122
function getDefaultValue<K extends keyof Settings>(
  key: K
): Settings[K] | undefined {
  const param = getSettingById(key)
  if (param === undefined) return

  const versionedDefault = getVersionedDefaultValue(key, param)
  if (versionedDefault) {
    return versionedDefault
  }

  return typeof param.defaultValue === 'function'
    ? param.defaultValue()
    : param.defaultValue
}
```

### Settings Registration Process

Settings are registered after server values are loaded:

```typescript
// From src/components/graph/GraphCanvas.vue:311-315
CORE_SETTINGS.forEach((setting) => {
  settingStore.addSetting(setting)
})

await newUserService().initializeIfNewUser(settingStore)
```

## Dynamic and Environment-Based Defaults

### Computed Defaults

You can compute defaults dynamically using function defaults that access runtime context:

```typescript
// From src/constants/coreSettings.ts:94-101
{
  id: 'Comfy.Sidebar.Size',
  // Default to small if the window is less than 1536px(2xl) wide
  defaultValue: () => (window.innerWidth < 1536 ? 'small' : 'normal')
}
```

```typescript
// From src/constants/coreSettings.ts:306
{
  id: 'Comfy.Locale',
  defaultValue: () => navigator.language.split('-')[0] || 'en'
}
```

### Version-Based Defaults

You can vary defaults by installed frontend version using `defaultsByInstallVersion`:

```typescript
// From src/stores/settingStore.ts:129-150
function getVersionedDefaultValue<
  K extends keyof Settings,
  TValue = Settings[K]
>(key: K, param: SettingParams<TValue> | undefined): TValue | null {
  const defaultsByInstallVersion = param?.defaultsByInstallVersion
  if (defaultsByInstallVersion && key !== 'Comfy.InstalledVersion') {
    const installedVersion = get('Comfy.InstalledVersion')
    if (installedVersion) {
      const sortedVersions = Object.keys(defaultsByInstallVersion).sort(
        (a, b) => compareVersions(b, a)
      )
      for (const version of sortedVersions) {
        if (!isSemVer(version)) continue
        if (compareVersions(installedVersion, version) >= 0) {
          const versionedDefault = defaultsByInstallVersion[version]
          return typeof versionedDefault === 'function'
            ? versionedDefault()
            : versionedDefault
        }
      }
    }
  }
  return null
}
```

Example versioned defaults from codebase:

```typescript
// From src/constants/coreSettings.ts:38-40
{
  id: 'Comfy.Graph.LinkReleaseAction',
  defaultValue: LinkReleaseTriggerAction.CONTEXT_MENU,
  defaultsByInstallVersion: {
    '1.24.1': LinkReleaseTriggerAction.SEARCH_BOX
  }
}

// Another versioned default example
{
  id: 'Comfy.Graph.LinkReleaseAction.Shift',
  defaultValue: LinkReleaseTriggerAction.SEARCH_BOX,
  defaultsByInstallVersion: {
    '1.24.1': LinkReleaseTriggerAction.CONTEXT_MENU
  }
}
```

### Real Examples from Codebase

Here are actual settings showing different patterns:

```typescript
// Number setting with validation
{
  id: 'LiteGraph.Node.TooltipDelay',
  name: 'Tooltip Delay',
  type: 'number',
  attrs: {
    min: 100,
    max: 3000,
    step: 50
  },
  defaultValue: 500,
  versionAdded: '1.9.0'
}

// Hidden system setting for tracking
{
  id: 'Comfy.InstalledVersion',
  name: 'The frontend version that was running when the user first installed ComfyUI',
  type: 'hidden',
  defaultValue: null,
  versionAdded: '1.24.0'
}

// Slider with complex tooltip
{
  id: 'LiteGraph.Canvas.LowQualityRenderingZoomThreshold',
  name: 'Low quality rendering zoom threshold',
  tooltip: 'Zoom level threshold for performance mode. Lower values (0.1) = quality at all zoom levels. Higher values (1.0) = performance mode even when zoomed in.',
  type: 'slider',
  attrs: {
    min: 0.1,
    max: 1.0,
    step: 0.05
  },
  defaultValue: 0.5
}
```

### New User Version Capture

The initial installed version is captured for new users to ensure versioned defaults remain stable:

```typescript
// From src/services/newUserService.ts:49-53
await settingStore.set('Comfy.InstalledVersion', __COMFYUI_FRONTEND_VERSION__)
```

## Practical Patterns for Environment-Based Defaults

### Dynamic Default Patterns

```typescript
// Device-based default
{
  id: 'Comfy.Example.MobileDefault',
  type: 'boolean',
  defaultValue: () => /Mobile/i.test(navigator.userAgent)
}

// Environment-based default
{
  id: 'Comfy.Example.DevMode',
  type: 'boolean',
  defaultValue: () => import.meta.env.DEV
}

// Window size based
{
  id: 'Comfy.Example.CompactUI',
  type: 'boolean',
  defaultValue: () => window.innerWidth < 1024
}
```

### Version-Based Rollout Pattern

```typescript
{
  id: 'Comfy.Example.NewFeature',
  type: 'combo',
  options: ['legacy', 'enhanced'],
  defaultValue: 'legacy',
  defaultsByInstallVersion: {
    '1.25.0': 'enhanced'
  }
}
```

## Settings Persistence and Access

### API Interaction

Values are stored per user via the backend. The store writes through API and falls back to defaults when not set:

```typescript
// From src/stores/settingStore.ts:73-75
onChange(settingsById.value[key], newValue, oldValue)
settingValues.value[key] = newValue
await api.storeSetting(key, newValue)
```

### Usage in Components

```typescript
const settingStore = useSettingStore()

// Get setting value (returns computed default if not set by user)
const value = settingStore.get('Comfy.SomeSetting')

// Update setting value
await settingStore.set('Comfy.SomeSetting', newValue)
```

## Advanced Settings Features

### Migration and Backward Compatibility

Settings support migration from deprecated values:

```typescript
// From src/stores/settingStore.ts:68-69, 172-175
const newValue = tryMigrateDeprecatedValue(settingsById.value[key], clonedValue)

// Migration happens during addSetting for existing values:
if (settingValues.value[setting.id] !== undefined) {
  settingValues.value[setting.id] = tryMigrateDeprecatedValue(
    setting,
    settingValues.value[setting.id]
  )
}
```

### onChange Callbacks

Settings can define onChange callbacks that receive the setting definition, new value, and old value:

```typescript
// From src/stores/settingStore.ts:73, 177
onChange(settingsById.value[key], newValue, oldValue) // During set()
onChange(setting, get(setting.id), undefined) // During addSetting()
```

### Settings UI and Categories

Settings are automatically grouped for UI based on their `category` or derived from `id`:

```typescript
{
  id: 'Comfy.Sidebar.Size',
  category: ['Appearance', 'Sidebar', 'Size'],
  // UI will group this under Appearance > Sidebar > Size
}
```

## Related Documentation

- Feature flag system: `docs/FEATURE_FLAGS.md`
- Settings schema for backend: `src/schemas/apiSchema.ts` (zSettings)
- Server configuration (separate from user settings): `src/constants/serverConfig.ts`

## Summary

- **Settings**: User preferences with dynamic/versioned defaults, persisted per user
- **Environment Defaults**: Use function defaults to read runtime context (window, navigator, env)
- **Version Rollouts**: Use `defaultsByInstallVersion` for gradual feature releases
- **API Interaction**: Settings persist to `/settings` endpoint via `storeSetting()`
