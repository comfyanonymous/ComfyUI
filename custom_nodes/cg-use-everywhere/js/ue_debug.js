import { app } from "../../scripts/app.js";
import { defineProperty } from "./use_everywhere_utilities.js";

/*
Things that can be useful (generally as breakpoints) when debugging
*/
export function add_debug() {
    var dirty_canvas = true;
    defineProperty(app.canvas, 'dirty_canvas', {
        get : () => { return dirty_canvas },
        set : (v) => { dirty_canvas = v;}  // a breakpoint here catches the calls that mark the canvas as dirty
    })

    var dirty_bg_canvas = true;
    defineProperty(app.canvas, 'dirty_bg_canvas', {
        get : () => { return dirty_bg_canvas },
        set : (v) => { dirty_bg_canvas = v;}  // a breakpoint here catches the calls that mark the background canvas as dirty
    })
}

export const version = 500006