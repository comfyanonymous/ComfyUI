var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });
var __async = (__this, __arguments, generator) => {
  return new Promise((resolve, reject) => {
    var fulfilled = (value) => {
      try {
        step(generator.next(value));
      } catch (e) {
        reject(e);
      }
    };
    var rejected = (value) => {
      try {
        step(generator.throw(value));
      } catch (e) {
        reject(e);
      }
    };
    var step = (x) => x.done ? resolve(x.value) : Promise.resolve(x.value).then(fulfilled, rejected);
    step((generator = generator.apply(__this, __arguments)).next());
  });
};
import { j as createSpinner, g as api, $ as $el } from "./index-CaD4RONs.js";
const _UserSelectionScreen = class _UserSelectionScreen {
  show(users, user) {
    return __async(this, null, function* () {
      const userSelection = document.getElementById("comfy-user-selection");
      userSelection.style.display = "";
      return new Promise((resolve) => {
        const input = userSelection.getElementsByTagName("input")[0];
        const select = userSelection.getElementsByTagName("select")[0];
        const inputSection = input.closest("section");
        const selectSection = select.closest("section");
        const form = userSelection.getElementsByTagName("form")[0];
        const error = userSelection.getElementsByClassName("comfy-user-error")[0];
        const button = userSelection.getElementsByClassName(
          "comfy-user-button-next"
        )[0];
        let inputActive = null;
        input.addEventListener("focus", () => {
          inputSection.classList.add("selected");
          selectSection.classList.remove("selected");
          inputActive = true;
        });
        select.addEventListener("focus", () => {
          inputSection.classList.remove("selected");
          selectSection.classList.add("selected");
          inputActive = false;
          select.style.color = "";
        });
        select.addEventListener("blur", () => {
          if (!select.value) {
            select.style.color = "var(--descrip-text)";
          }
        });
        form.addEventListener("submit", (e) => __async(this, null, function* () {
          var _a, _b, _c;
          e.preventDefault();
          if (inputActive == null) {
            error.textContent = "Please enter a username or select an existing user.";
          } else if (inputActive) {
            const username = input.value.trim();
            if (!username) {
              error.textContent = "Please enter a username.";
              return;
            }
            input.disabled = select.disabled = // @ts-expect-error
            input.readonly = // @ts-expect-error
            select.readonly = true;
            const spinner = createSpinner();
            button.prepend(spinner);
            try {
              const resp = yield api.createUser(username);
              if (resp.status >= 300) {
                let message = "Error creating user: " + resp.status + " " + resp.statusText;
                try {
                  const res = yield resp.json();
                  if (res.error) {
                    message = res.error;
                  }
                } catch (error2) {
                }
                throw new Error(message);
              }
              resolve({ username, userId: yield resp.json(), created: true });
            } catch (err) {
              spinner.remove();
              error.textContent = (_c = (_b = (_a = err.message) != null ? _a : err.statusText) != null ? _b : err) != null ? _c : "An unknown error occurred.";
              input.disabled = select.disabled = // @ts-expect-error
              input.readonly = // @ts-expect-error
              select.readonly = false;
              return;
            }
          } else if (!select.value) {
            error.textContent = "Please select an existing user.";
            return;
          } else {
            resolve({
              username: users[select.value],
              userId: select.value,
              created: false
            });
          }
        }));
        if (user) {
          const name = localStorage["Comfy.userName"];
          if (name) {
            input.value = name;
          }
        }
        if (input.value) {
          input.focus();
        }
        const userIds = Object.keys(users != null ? users : {});
        if (userIds.length) {
          for (const u of userIds) {
            $el("option", { textContent: users[u], value: u, parent: select });
          }
          select.style.color = "var(--descrip-text)";
          if (select.value) {
            select.focus();
          }
        } else {
          userSelection.classList.add("no-users");
          input.focus();
        }
      }).then((r) => {
        userSelection.remove();
        return r;
      });
    });
  }
};
__name(_UserSelectionScreen, "UserSelectionScreen");
let UserSelectionScreen = _UserSelectionScreen;
window.comfyAPI = window.comfyAPI || {};
window.comfyAPI.userSelection = window.comfyAPI.userSelection || {};
window.comfyAPI.userSelection.UserSelectionScreen = UserSelectionScreen;
export {
  UserSelectionScreen
};
//# sourceMappingURL=userSelection-GRU1gtOt.js.map
