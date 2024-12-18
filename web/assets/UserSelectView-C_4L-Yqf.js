var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });
import { a as defineComponent, J as useUserStore, bU as useRouter, r as ref, q as computed, o as onMounted, f as openBlock, g as createElementBlock, A as createBaseVNode, a8 as toDisplayString, h as createVNode, z as unref, aq as script, bN as script$1, bV as script$2, x as createBlock, y as withCtx, ay as createTextVNode, bW as script$3, i as createCommentVNode, D as script$4 } from "./index-DIU5yZe9.js";
const _hoisted_1 = {
  id: "comfy-user-selection",
  class: "font-sans flex flex-col items-center h-screen m-0 text-neutral-300 bg-neutral-900 dark-theme pointer-events-auto"
};
const _hoisted_2 = { class: "mt-[5vh] 2xl:mt-[20vh] min-w-84 relative rounded-lg bg-[var(--comfy-menu-bg)] p-5 px-10 shadow-lg" };
const _hoisted_3 = /* @__PURE__ */ createBaseVNode("h1", { class: "my-2.5 mb-7 font-normal" }, "ComfyUI", -1);
const _hoisted_4 = { class: "flex w-full flex-col items-center" };
const _hoisted_5 = { class: "flex w-full flex-col gap-2" };
const _hoisted_6 = { for: "new-user-input" };
const _hoisted_7 = { class: "flex w-full flex-col gap-2" };
const _hoisted_8 = { for: "existing-user-select" };
const _hoisted_9 = { class: "mt-5" };
const _sfc_main = /* @__PURE__ */ defineComponent({
  __name: "UserSelectView",
  setup(__props) {
    const userStore = useUserStore();
    const router = useRouter();
    const selectedUser = ref(null);
    const newUsername = ref("");
    const loginError = ref("");
    const createNewUser = computed(() => newUsername.value.trim() !== "");
    const newUserExistsError = computed(() => {
      return userStore.users.find((user) => user.username === newUsername.value) ? `User "${newUsername.value}" already exists` : "";
    });
    const error = computed(() => newUserExistsError.value || loginError.value);
    const login = /* @__PURE__ */ __name(async () => {
      try {
        const user = createNewUser.value ? await userStore.createUser(newUsername.value) : selectedUser.value;
        if (!user) {
          throw new Error("No user selected");
        }
        userStore.login(user);
        router.push("/");
      } catch (err) {
        loginError.value = err.message ?? JSON.stringify(err);
      }
    }, "login");
    onMounted(async () => {
      if (!userStore.initialized) {
        await userStore.initialize();
      }
    });
    return (_ctx, _cache) => {
      return openBlock(), createElementBlock("div", _hoisted_1, [
        createBaseVNode("main", _hoisted_2, [
          _hoisted_3,
          createBaseVNode("form", _hoisted_4, [
            createBaseVNode("div", _hoisted_5, [
              createBaseVNode("label", _hoisted_6, toDisplayString(_ctx.$t("userSelect.newUser")) + ":", 1),
              createVNode(unref(script), {
                id: "new-user-input",
                modelValue: newUsername.value,
                "onUpdate:modelValue": _cache[0] || (_cache[0] = ($event) => newUsername.value = $event),
                placeholder: _ctx.$t("userSelect.enterUsername")
              }, null, 8, ["modelValue", "placeholder"])
            ]),
            createVNode(unref(script$1)),
            createBaseVNode("div", _hoisted_7, [
              createBaseVNode("label", _hoisted_8, toDisplayString(_ctx.$t("userSelect.existingUser")) + ":", 1),
              createVNode(unref(script$2), {
                modelValue: selectedUser.value,
                "onUpdate:modelValue": _cache[1] || (_cache[1] = ($event) => selectedUser.value = $event),
                class: "w-full",
                inputId: "existing-user-select",
                options: unref(userStore).users,
                "option-label": "username",
                placeholder: _ctx.$t("userSelect.selectUser"),
                disabled: createNewUser.value
              }, null, 8, ["modelValue", "options", "placeholder", "disabled"]),
              error.value ? (openBlock(), createBlock(unref(script$3), {
                key: 0,
                severity: "error"
              }, {
                default: withCtx(() => [
                  createTextVNode(toDisplayString(error.value), 1)
                ]),
                _: 1
              })) : createCommentVNode("", true)
            ]),
            createBaseVNode("footer", _hoisted_9, [
              createVNode(unref(script$4), {
                label: _ctx.$t("userSelect.next"),
                onClick: login
              }, null, 8, ["label"])
            ])
          ])
        ])
      ]);
    };
  }
});
export {
  _sfc_main as default
};
//# sourceMappingURL=UserSelectView-C_4L-Yqf.js.map
