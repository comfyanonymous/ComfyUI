import { app } from "../../scripts/app.js";
import { ComfyButtonGroup } from "../../scripts/ui/components/buttonGroup.js";
import { ComfyButton } from "../../scripts/ui/components/button.js";
import { iconGear, iconStarFilled, logoRgthree } from "../../rgthree/common/media/svgs.js";
import { createElement, empty } from "../../rgthree/common/utils_dom.js";
import { SERVICE as BOOKMARKS_SERVICE } from "./services/bookmarks_services.js";
import { SERVICE as CONFIG_SERVICE } from "./services/config_service.js";
import { ComfyPopup } from "../../scripts/ui/components/popup.js";
import { RgthreeConfigDialog } from "./config.js";
let rgthreeButtonGroup = null;
function addRgthreeTopBarButtons() {
    var _a, _b, _c;
    if (!CONFIG_SERVICE.getFeatureValue("comfy_top_bar_menu.enabled")) {
        if ((_a = rgthreeButtonGroup === null || rgthreeButtonGroup === void 0 ? void 0 : rgthreeButtonGroup.element) === null || _a === void 0 ? void 0 : _a.parentElement) {
            rgthreeButtonGroup.element.parentElement.removeChild(rgthreeButtonGroup.element);
        }
        return;
    }
    else if (rgthreeButtonGroup) {
        (_b = app.menu) === null || _b === void 0 ? void 0 : _b.settingsGroup.element.before(rgthreeButtonGroup.element);
        return;
    }
    const buttons = [];
    const rgthreeButton = new ComfyButton({
        icon: "rgthree",
        tooltip: "rgthree-comfy",
        app,
        enabled: true,
        classList: "comfyui-button comfyui-menu-mobile-collapse primary",
    });
    buttons.push(rgthreeButton);
    rgthreeButton.iconElement.style.width = "1.2rem";
    rgthreeButton.iconElement.innerHTML = logoRgthree;
    rgthreeButton.withPopup(new ComfyPopup({ target: rgthreeButton.element, classList: "rgthree-top-menu" }, createElement("menu", {
        children: [
            createElement("li", {
                child: createElement("button.rgthree-button-reset", {
                    html: iconGear + "Settings (rgthree-comfy)",
                    onclick: () => new RgthreeConfigDialog().show(),
                }),
            }),
            createElement("li", {
                child: createElement("button.rgthree-button-reset", {
                    html: iconStarFilled + "Star on Github",
                    onclick: () => window.open("https://github.com/rgthree/rgthree-comfy", "_blank"),
                }),
            }),
        ],
    })), "click");
    if (CONFIG_SERVICE.getFeatureValue("comfy_top_bar_menu.button_bookmarks.enabled")) {
        const bookmarksListEl = createElement("menu");
        bookmarksListEl.appendChild(createElement("li.rgthree-message", {
            child: createElement("span", { text: "No bookmarks in current workflow." }),
        }));
        const bookmarksButton = new ComfyButton({
            icon: "bookmark",
            tooltip: "Workflow Bookmarks (rgthree-comfy)",
            app,
        });
        const bookmarksPopup = new ComfyPopup({ target: bookmarksButton.element, classList: "rgthree-top-menu" }, bookmarksListEl);
        bookmarksPopup.addEventListener("open", () => {
            const bookmarks = BOOKMARKS_SERVICE.getCurrentBookmarks();
            empty(bookmarksListEl);
            if (bookmarks.length) {
                for (const b of bookmarks) {
                    bookmarksListEl.appendChild(createElement("li", {
                        child: createElement("button.rgthree-button-reset", {
                            text: `[${b.shortcutKey}] ${b.title}`,
                            onclick: () => {
                                b.canvasToBookmark();
                            },
                        }),
                    }));
                }
            }
            else {
                bookmarksListEl.appendChild(createElement("li.rgthree-message", {
                    child: createElement("span", { text: "No bookmarks in current workflow." }),
                }));
            }
            bookmarksPopup.update();
        });
        bookmarksButton.withPopup(bookmarksPopup, "hover");
        buttons.push(bookmarksButton);
    }
    rgthreeButtonGroup = new ComfyButtonGroup(...buttons);
    (_c = app.menu) === null || _c === void 0 ? void 0 : _c.settingsGroup.element.before(rgthreeButtonGroup.element);
}
app.registerExtension({
    name: "rgthree.TopMenu",
    async setup() {
        addRgthreeTopBarButtons();
        CONFIG_SERVICE.addEventListener("config-change", ((e) => {
            var _a, _b;
            if ((_b = (_a = e.detail) === null || _a === void 0 ? void 0 : _a.key) === null || _b === void 0 ? void 0 : _b.includes("features.comfy_top_bar_menu")) {
                addRgthreeTopBarButtons();
            }
        }));
    },
});
