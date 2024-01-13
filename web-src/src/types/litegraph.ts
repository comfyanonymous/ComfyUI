// Litegraph.d.js file is incomplete and missing several properties, so we add them here

import { Vector4 } from 'litegraph.js';

export interface LiteGraphCorrected {
    DEFAULT_GROUP_FONT_SIZE: number;
    GRID_SHAPE: number;

    overlapBounding: (a: Vector4, b: Vector4) => boolean;

    release_link_on_empty_shows_menu: boolean;
    alt_drag_do_clone_nodes: boolean;
}
