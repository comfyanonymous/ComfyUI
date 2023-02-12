var COMFY_NODES = [];

COMFY_NODES["EmptyLatentImage"] = {
    onExecute: function() {
        for (var idx of this.outputs[0].links) {
            let link = this.graph.links[idx];
            link.data = this.widgets.map(function(w) { return w.value; });
        }
    }
};
