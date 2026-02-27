# Settings and Feature Flags Sequence Diagram

This diagram shows the flow of settings initialization, default resolution, persistence, and feature flags exchange.

This diagram accurately reflects the actual implementation in the ComfyUI frontend codebase.

```mermaid
sequenceDiagram
    participant User as User
    participant Vue as Vue Component
    participant Store as SettingStore (Pinia)
    participant API as ComfyApi (WebSocket/REST)
    participant Backend as Backend
    participant NewUserSvc as NewUserService

    Note over Vue,Store: App startup (GraphCanvas.vue)
    Vue->>Store: loadSettingValues()
    Store->>API: getSettings()
    API->>Backend: GET /settings
    Backend-->>API: settings map (per-user)
    API-->>Store: settings map
    Store-->>Vue: loaded

    Vue->>Store: register CORE_SETTINGS (addSetting for each)
    loop For each setting registration
      Store->>Store: tryMigrateDeprecatedValue(existing value)
      Store->>Store: onChange(setting, currentValue, undefined)
    end

    Note over Vue,NewUserSvc: New user detection
    Vue->>NewUserSvc: initializeIfNewUser(settingStore)
    NewUserSvc->>NewUserSvc: checkIsNewUser(settingStore)
    alt New user detected
      NewUserSvc->>Store: set("Comfy.InstalledVersion", __COMFYUI_FRONTEND_VERSION__)
      Store->>Store: tryMigrateDeprecatedValue(newValue)
      Store->>Store: onChange(setting, newValue, oldValue)
      Store->>API: storeSetting(key, newValue)
      API->>Backend: POST /settings/{id}
    else Existing user
      Note over NewUserSvc: Skip setting installed version
    end

    Note over Vue,Store: Component reads a setting
    Vue->>Store: get(key)
    Store->>Store: exists(key)?
    alt User value exists
      Store-->>Vue: return stored user value
    else Not set by user
      Store->>Store: getVersionedDefaultValue(key)
      alt Versioned default matched (defaultsByInstallVersion)
        Store-->>Vue: return versioned default
      else No version match
        Store->>Store: evaluate defaultValue (function or constant)
        Note over Store: defaultValue can use window size,<br/>locale, env, etc.
        Store-->>Vue: return computed default
      end
    end

    Note over User,Store: User updates a setting
    User->>Vue: changes setting in UI
    Vue->>Store: set(key, newValue)
    Store->>Store: tryMigrateDeprecatedValue(newValue)
    Store->>Store: check if newValue === oldValue (early return if same)
    Store->>Store: onChange(setting, newValue, oldValue)
    Store->>Store: update settingValues[key]
    Store->>API: storeSetting(key, newValue)
    API->>Backend: POST /settings/{id}
    Backend-->>API: 200 OK
    API-->>Store: ack

    Note over API,Backend: Feature Flags WebSocket Exchange
    API->>Backend: WS connect
    API->>Backend: send { type: "feature_flags", data: clientFeatureFlags.json }
    Backend-->>API: WS send { type: "feature_flags", data: server flags }
    API->>API: store serverFeatureFlags = data

    Note over Vue,API: Feature flag consumption in UI/logic
    Vue->>API: serverSupportsFeature(name)
    API-->>Vue: boolean (true only if flag === true)
    Vue->>API: getServerFeature(name, default)
    API-->>Vue: value or default
```
