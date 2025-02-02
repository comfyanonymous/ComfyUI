var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });
import { I as defineStore, U as ref, c as computed } from "./index-4Hb32CNk.js";
const useServerConfigStore = defineStore("serverConfig", () => {
  const serverConfigById = ref({});
  const serverConfigs = computed(() => {
    return Object.values(serverConfigById.value);
  });
  const modifiedConfigs = computed(
    () => {
      return serverConfigs.value.filter((config) => {
        return config.initialValue !== config.value;
      });
    }
  );
  const revertChanges = /* @__PURE__ */ __name(() => {
    for (const config of modifiedConfigs.value) {
      config.value = config.initialValue;
    }
  }, "revertChanges");
  const serverConfigsByCategory = computed(() => {
    return serverConfigs.value.reduce(
      (acc, config) => {
        const category = config.category?.[0] ?? "General";
        acc[category] = acc[category] || [];
        acc[category].push(config);
        return acc;
      },
      {}
    );
  });
  const serverConfigValues = computed(() => {
    return Object.fromEntries(
      serverConfigs.value.map((config) => {
        return [
          config.id,
          config.value === config.defaultValue || config.value === null || config.value === void 0 ? void 0 : config.value
        ];
      })
    );
  });
  const launchArgs = computed(() => {
    const args = Object.assign(
      {},
      ...serverConfigs.value.map((config) => {
        if (config.value === config.defaultValue || config.value === null || config.value === void 0) {
          return {};
        }
        return config.getValue ? config.getValue(config.value) : { [config.id]: config.value };
      })
    );
    return Object.fromEntries(
      Object.entries(args).map(([key, value]) => {
        if (value === true) {
          return [key, ""];
        }
        return [key, value.toString()];
      })
    );
  });
  const commandLineArgs = computed(() => {
    return Object.entries(launchArgs.value).map(([key, value]) => [`--${key}`, value]).flat().filter((arg) => arg !== "").join(" ");
  });
  function loadServerConfig(configs, values) {
    for (const config of configs) {
      const value = values[config.id] ?? config.defaultValue;
      serverConfigById.value[config.id] = {
        ...config,
        value,
        initialValue: value
      };
    }
  }
  __name(loadServerConfig, "loadServerConfig");
  return {
    serverConfigById,
    serverConfigs,
    modifiedConfigs,
    serverConfigsByCategory,
    serverConfigValues,
    launchArgs,
    commandLineArgs,
    revertChanges,
    loadServerConfig
  };
});
export {
  useServerConfigStore as u
};
//# sourceMappingURL=serverConfigStore-BYbZcbWj.js.map
