(function() {
    var LGraphCanvas = LiteGraph.LGraphCanvas;
    var LGraph = LiteGraph.LGraph;

    // Save the original renderLink function
    var originalRenderLink = LGraphCanvas.prototype.renderLink;

    // Save the original connect function
    var originalConnect = LGraph.prototype.connect;

    // Override the connect function
    LGraph.prototype.connect = function (
        origin_slot,
        target_slot,
        options
    ) {
        var origin_id = origin_slot[0];
        var target_id = target_slot[0];

        var origin_node = this.getNodeById(origin_id);
        var target_node = this.getNodeById(target_id);


        if (origin_node && target_node) {
            var output_slot = origin_slot[1];
            var output_slot_info = origin_node.getOutputInfo(output_slot);


            console.log(output_slot_info)
            if (output_slot_info) {
                options = options || {};
                options.color = output_slot_info.label_color || null;
            }
        }

        return originalConnect.call(this, origin_slot, target_slot, options);
    };

    // Override the renderLink function
    LGraphCanvas.prototype.renderLink = function (
        ctx,
        a,
        b,
        link,
        skip_border,
        flow,
        color,
        start_dir,
        end_dir,
        num_sublines
    ) {
        if (link && link.color) {
            color = link.color;
        }

        // Call the original renderLink function with the new color
        originalRenderLink.call(
            this,
            ctx,
            a,
            b,
            link,
            skip_border,
            flow,
            color,
            start_dir,
            end_dir,
            num_sublines
        );
    };
})();
