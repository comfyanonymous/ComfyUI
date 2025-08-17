import type {Bookmark} from "../bookmark.js";

import {app} from "scripts/app.js";
import {NodeTypesString} from "../constants.js";
import {reduceNodesDepthFirst} from "../utils.js";

const SHORTCUT_DEFAULTS = "1234567890abcdefghijklmnopqrstuvwxyz".split("");

class BookmarksService {
  /**
   * Gets a list of the current bookmarks within the current workflow.
   */
  getCurrentBookmarks(): Bookmark[] {
    return reduceNodesDepthFirst<Bookmark[]>(app.graph.nodes, (n, acc) => {
      if (n.type === NodeTypesString.BOOKMARK) {
        acc.push(n as Bookmark);
      }
    }, []).sort((a, b) => a.title.localeCompare(b.title));
  }

  getExistingShortcuts() {
    const bookmarkNodes = this.getCurrentBookmarks();
    const usedShortcuts = new Set(bookmarkNodes.map((n) => n.shortcutKey));
    return usedShortcuts;
  }

  getNextShortcut() {
    const existingShortcuts = this.getExistingShortcuts();
    return SHORTCUT_DEFAULTS.find((char) => !existingShortcuts.has(char)) ?? "1";
  }
}

/** The BookmarksService singleton. */
export const SERVICE = new BookmarksService();
