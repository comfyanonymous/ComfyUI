import {sleep} from "./utils.js";
import {$t} from "./i18n.js";

class Toast{

    constructor() {
        this.info_icon = `<svg focusable="false" data-icon="info-circle" width="1em" height="1em" fill="currentColor" aria-hidden="true" viewBox="64 64 896 896"><path d="M512 64C264.6 64 64 264.6 64 512s200.6 448 448 448 448-200.6 448-448S759.4 64 512 64zm32 664c0 4.4-3.6 8-8 8h-48c-4.4 0-8-3.6-8-8V456c0-4.4 3.6-8 8-8h48c4.4 0 8 3.6 8 8v272zm-32-344a48.01 48.01 0 010-96 48.01 48.01 0 010 96z"></path></svg>`
        this.success_icon = `<svg focusable="false" data-icon="check-circle" width="1em" height="1em" fill="currentColor" aria-hidden="true" viewBox="64 64 896 896"><path d="M512 64C264.6 64 64 264.6 64 512s200.6 448 448 448 448-200.6 448-448S759.4 64 512 64zm193.5 301.7l-210.6 292a31.8 31.8 0 01-51.7 0L318.5 484.9c-3.8-5.3 0-12.7 6.5-12.7h46.9c10.2 0 19.9 4.9 25.9 13.3l71.2 98.8 157.2-218c6-8.3 15.6-13.3 25.9-13.3H699c6.5 0 10.3 7.4 6.5 12.7z"></path></svg>`
        this.error_icon = `<svg focusable="false" data-icon="close-circle" width="1em" height="1em" fill="currentColor" aria-hidden="true" fill-rule="evenodd" viewBox="64 64 896 896"><path d="M512 64c247.4 0 448 200.6 448 448S759.4 960 512 960 64 759.4 64 512 264.6 64 512 64zm127.98 274.82h-.04l-.08.06L512 466.75 384.14 338.88c-.04-.05-.06-.06-.08-.06a.12.12 0 00-.07 0c-.03 0-.05.01-.09.05l-45.02 45.02a.2.2 0 00-.05.09.12.12 0 000 .07v.02a.27.27 0 00.06.06L466.75 512 338.88 639.86c-.05.04-.06.06-.06.08a.12.12 0 000 .07c0 .03.01.05.05.09l45.02 45.02a.2.2 0 00.09.05.12.12 0 00.07 0c.02 0 .04-.01.08-.05L512 557.25l127.86 127.87c.04.04.06.05.08.05a.12.12 0 00.07 0c.03 0 .05-.01.09-.05l45.02-45.02a.2.2 0 00.05-.09.12.12 0 000-.07v-.02a.27.27 0 00-.05-.06L557.25 512l127.87-127.86c.04-.04.05-.06.05-.08a.12.12 0 000-.07c0-.03-.01-.05-.05-.09l-45.02-45.02a.2.2 0 00-.09-.05.12.12 0 00-.07 0z"></path></svg>`
        this.warn_icon = `<svg focusable="false" data-icon="exclamation-circle" width="1em" height="1em" fill="currentColor" aria-hidden="true" viewBox="64 64 896 896"><path d="M512 64C264.6 64 64 264.6 64 512s200.6 448 448 448 448-200.6 448-448S759.4 64 512 64zm-32 232c0-4.4 3.6-8 8-8h48c4.4 0 8 3.6 8 8v272c0 4.4-3.6 8-8 8h-48c-4.4 0-8-3.6-8-8V296zm32 440a48.01 48.01 0 010-96 48.01 48.01 0 010 96z"></path></svg>`
        this.loading_icon = `<svg focusable="false" data-icon="loading" width="1em" height="1em" fill="currentColor" aria-hidden="true" viewBox="0 0 1024 1024"><path d="M988 548c-19.9 0-36-16.1-36-36 0-59.4-11.6-117-34.6-171.3a440.45 440.45 0 00-94.3-139.9 437.71 437.71 0 00-139.9-94.3C629 83.6 571.4 72 512 72c-19.9 0-36-16.1-36-36s16.1-36 36-36c69.1 0 136.2 13.5 199.3 40.3C772.3 66 827 103 874 150c47 47 83.9 101.8 109.7 162.7 26.7 63.1 40.2 130.2 40.2 199.3.1 19.9-16 36-35.9 36z"></path></svg>`
    }

    async showToast(data){
        let container = document.querySelector(".easyuse-toast-container");
        if (!container) {
            container = document.createElement("div");
            container.classList.add("easyuse-toast-container");
            document.body.appendChild(container);
        }
        await this.hideToast(data.id);
        const toastContainer = document.createElement("div");
        const content = document.createElement("span");
        content.innerHTML = data.content;
        toastContainer.appendChild(content);
        for (let a = 0; a < (data.actions || []).length; a++) {
            const action = data.actions[a];
            if (a > 0) {
                const sep = document.createElement("span");
                sep.innerHTML = "&nbsp;|&nbsp;";
                toastContainer.appendChild(sep);
            }
            const actionEl = document.createElement("a");
            actionEl.innerText = action.label;
            if (action.href) {
                actionEl.target = "_blank";
                actionEl.href = action.href;
            }
            if (action.callback) {
                actionEl.onclick = (e) => {
                    return action.callback(e);
                };
            }
            toastContainer.appendChild(actionEl);
        }
        const animContainer = document.createElement("div");
        animContainer.setAttribute("toast-id", data.id);
        animContainer.appendChild(toastContainer);
        container.appendChild(animContainer);
        await sleep(64);
        animContainer.style.marginTop = `-${animContainer.offsetHeight}px`;
        await sleep(64);
        animContainer.classList.add("-show");
        if (data.duration) {
            await sleep(data.duration);
            this.hideToast(data.id);
        }
    }
    async hideToast(id) {
        const msg = document.querySelector(`.easyuse-toast-container > [toast-id="${id}"]`);
        if (msg === null || msg === void 0 ? void 0 : msg.classList.contains("-show")) {
            msg.classList.remove("-show");
            await sleep(750);
        }
        msg && msg.remove();
    }
    async clearAllMessages() {
        let container = document.querySelector(".easyuse-toast-container");
        container && (container.innerHTML = "");
    }

    async copyright(duration = 5000, actions = []) {
        this.showToast({
            id: `toast-info`,
            content: `${this.info_icon} ${$t('Workflow created by')} <a href="https://github.com/yolain/">Yolain</a> , ${$t('Watch more video content')} <a href="https://space.bilibili.com/1840885116">B站乱乱呀</a>`,
            duration,
            actions
        });
    }
    async info(content, duration = 3000, actions = []) {
        this.showToast({
            id: `toast-info`,
            content: `${this.info_icon} ${content}`,
            duration,
            actions
        });
    }
    async success(content, duration = 3000, actions = []) {
        this.showToast({
            id: `toast-success`,
            content: `${this.success_icon} ${content}`,
            duration,
            actions
        });
    }
    async error(content, duration = 3000, actions = []) {
        this.showToast({
            id: `toast-error`,
            content: `${this.error_icon} ${content}`,
            duration,
            actions
        });
    }
    async warn(content, duration = 3000, actions = []) {
        this.showToast({
            id: `toast-warn`,
            content: `${this.warn_icon} ${content}`,
            duration,
            actions
        });
    }
    async showLoading(content, duration = 0, actions = []) {
        this.showToast({
            id: `toast-loading`,
            content: `${this.loading_icon} ${content}`,
            duration,
            actions
        });
    }

    async hideLoading() {
        this.hideToast("toast-loading");
    }

}

export const toast = new Toast();
