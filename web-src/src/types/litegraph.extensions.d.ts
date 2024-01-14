// Litegraph.d.js file is incomplete and missing some properties, so we add them here
import 'litegraph.js';

declare module 'litegraph.js' {
    interface LiteGraph {
        release_link_on_empty_shows_menu: boolean;
        alt_drag_do_clone_nodes: boolean;
    }
}
