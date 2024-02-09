import { LiteGraph } from 'litegraph.js';
import { ComfyNode } from '../litegraph/comfyNode';

// ========= Helper functions for onDrawbackground =========

export function calculateGrid(w: number, h: number, n: number) {
    let columns, rows, cellsize;

    if (w > h) {
        cellsize = h;
        columns = Math.ceil(w / cellsize);
        rows = Math.ceil(n / columns);
    } else {
        cellsize = w;
        rows = Math.ceil(h / cellsize);
        columns = Math.ceil(n / rows);
    }

    while (columns * rows < n) {
        cellsize++;
        if (w >= h) {
            columns = Math.ceil(w / cellsize);
            rows = Math.ceil(n / columns);
        } else {
            rows = Math.ceil(h / cellsize);
            columns = Math.ceil(n / rows);
        }
    }

    const cell_size = Math.min(w / columns, h / rows);
    return { cell_size, columns, rows };
}

export function getImageTop(node: ComfyNode) {
    let shiftY: number;
    if (node.imageOffset != null) {
        shiftY = node.imageOffset;
    } else {
        if (node.widgets?.length) {
            const w = node.widgets[node.widgets.length - 1];
            if (!w.last_y) throw '';

            shiftY = w.last_y;
            if (w.computeSize) {
                shiftY += w.computeSize(node.size[0])[1] + 4;
            } else if (w.computedHeight) {
                shiftY += w.computedHeight;
            } else {
                shiftY += LiteGraph.NODE_WIDGET_HEIGHT + 4;
            }
        } else {
            shiftY = node.computeSize()[1];
        }
    }
    return shiftY;
}

export function is_all_same_aspect_ratio(imgs: HTMLImageElement[]) {
    // assume: imgs.length >= 2
    const ratio = imgs[0].naturalWidth / imgs[0].naturalHeight;

    for (let i = 1; i < imgs.length; i++) {
        const this_ratio = imgs[i].naturalWidth / imgs[i].naturalHeight;
        if (ratio != this_ratio) return false;
    }

    return true;
}
