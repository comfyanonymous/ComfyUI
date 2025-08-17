import { app } from "../../../scripts/app.js";
import { NodeTypesString } from "../constants.js";
import { reduceNodesDepthFirst } from "../utils.js";
const SHORTCUT_DEFAULTS = "1234567890abcdefghijklmnopqrstuvwxyz".split("");
class BookmarksService {
    getCurrentBookmarks() {
        return reduceNodesDepthFirst(app.graph.nodes, (n, acc) => {
            if (n.type === NodeTypesString.BOOKMARK) {
                acc.push(n);
            }
        }, []).sort((a, b) => a.title.localeCompare(b.title));
    }
    getExistingShortcuts() {
        const bookmarkNodes = this.getCurrentBookmarks();
        const usedShortcuts = new Set(bookmarkNodes.map((n) => n.shortcutKey));
        return usedShortcuts;
    }
    getNextShortcut() {
        var _a;
        const existingShortcuts = this.getExistingShortcuts();
        return (_a = SHORTCUT_DEFAULTS.find((char) => !existingShortcuts.has(char))) !== null && _a !== void 0 ? _a : "1";
    }
}
export const SERVICE = new BookmarksService();
