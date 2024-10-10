var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });
import { b4 as api, ca as $el } from "./index-DGAbdBYF.js";
function createSpinner() {
  const div = document.createElement("div");
  div.innerHTML = `<div class="lds-ring"><div></div><div></div><div></div><div></div></div>`;
  return div.firstElementChild;
}
__name(createSpinner, "createSpinner");
window.comfyAPI = window.comfyAPI || {};
window.comfyAPI.spinner = window.comfyAPI.spinner || {};
window.comfyAPI.spinner.createSpinner = createSpinner;
class UserSelectionScreen {
  static {
    __name(this, "UserSelectionScreen");
  }
  async show(users, user) {
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
      form.addEventListener("submit", async (e) => {
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
            const resp = await api.createUser(username);
            if (resp.status >= 300) {
              let message = "Error creating user: " + resp.status + " " + resp.statusText;
              try {
                const res = await resp.json();
                if (res.error) {
                  message = res.error;
                }
              } catch (error2) {
              }
              throw new Error(message);
            }
            resolve({ username, userId: await resp.json(), created: true });
          } catch (err) {
            spinner.remove();
            error.textContent = err.message ?? err.statusText ?? err ?? "An unknown error occurred.";
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
      });
      if (user) {
        const name = localStorage["Comfy.userName"];
        if (name) {
          input.value = name;
        }
      }
      if (input.value) {
        input.focus();
      }
      const userIds = Object.keys(users ?? {});
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
  }
}
window.comfyAPI = window.comfyAPI || {};
window.comfyAPI.userSelection = window.comfyAPI.userSelection || {};
window.comfyAPI.userSelection.UserSelectionScreen = UserSelectionScreen;
export {
  UserSelectionScreen
};
//# sourceMappingURL=userSelection-Duxc-t_S.js.map
