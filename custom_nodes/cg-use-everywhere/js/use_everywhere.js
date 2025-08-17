import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

import { is_UEnode, is_helper, inject, Logger, get_real_node, defineProperty, graphConverter, fix_inputs, create } from "./use_everywhere_utilities.js";
import { update_input_label, indicate_restriction } from "./use_everywhere_ui.js";
import { LinkRenderController } from "./use_everywhere_ui.js";
import { GraphAnalyser } from "./use_everywhere_graph_analysis.js";
import { canvas_menu_settings, SETTINGS, add_extra_menu_items } from "./use_everywhere_settings.js";
import { add_debug } from "./ue_debug.js";
import { settingsCache } from "./use_everywhere_cache.js";
import { convert_to_links } from "./use_everywhere_apply.js";
import { get_subgraph_input_type, link_is_from_subgraph_input, node_graph, visible_graph } from "./use_everywhere_subgraph_utils.js";
import { any_restrictions, setup_ue_properties_oncreate, setup_ue_properties_onload } from "./ue_properties.js";
import { edit_restrictions } from "./ue_properties_editor.js";

/*
The ui component that looks after the link rendering
*/
var linkRenderController;
var graphAnalyser;

/*
Inject a call to linkRenderController.mark_list_link_outdated into a method with name methodname on all objects in the array
If object is undefined, do nothing.
The injection is added at the end of the existing method (if the method didn't exist, it is created).
*/
function inject_outdating_into_objects(array, methodname, tracetext) {
    if (array) {
        array.forEach((object) => { inject_outdating_into_object_method(object, methodname, tracetext); })
    }
}
function inject_outdating_into_object_method(object, methodname, tracetext) {
    if (object) inject(object, methodname, tracetext, linkRenderController.mark_link_list_outdated, linkRenderController);
}

class Deferred {
    constructor() { this.deferred_actions = [] }
    push(x) { this.deferred_actions.push(x) } // add action of the form: { fn:function, args:array }
    execute() {
        while (this.deferred_actions.length>0) {
            const action = this.deferred_actions.pop()
            try { action?.fn(...action?.args) } 
            catch (e) { Logger.log_error(e) }
        }
    }
}

const deferred_actions = new Deferred()

app.registerExtension({
	name: "cg.customnodes.use_everywhere",
    settings: SETTINGS, 

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        /*
        When a node is connected or unconnected, the link list is dirty.
        If it is a UE node, we need to update it as well
        */
        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (side,slot,connect,link_info,output) {        
            if (this.IS_UE && side==1) { // side 1 is input
                var type = '*'
                if (connect && link_info) {
                    var connected_type
                    if (link_is_from_subgraph_input(link_info)) { // input slot of subgraph
                        connected_type = get_subgraph_input_type(node_graph(this), link_info.origin_slot)
                    } else {
                        connected_type = get_real_node(link_info.origin_id, node_graph(this))?.outputs[link_info.origin_slot]?.type
                    }
                    if (connected_type) type = connected_type;
                };
                this.inputs[slot].type = type;
                if (link_info) link_info.type = type
                update_input_label(this, slot, app);
                if (!graphConverter.graph_being_configured) {
                    // do the fix at the end of graph change
                    deferred_actions.push( { fn:fix_inputs, args:[this,]} )
                    // disconnecting doesn't trigger graphChange call?
                    setTimeout(deferred_actions.execute.bind(deferred_actions), 100)
                }
            }
            linkRenderController.mark_link_list_outdated();
            onConnectionsChange?.apply(this, arguments);
        };

        /*
        Reject duplicated inputs for now
        */
        if (is_UEnode(nodeType)) {
            const onConnectInput = nodeType.prototype.onConnectInput
            nodeType.prototype.onConnectInput = function (index, type) {
                if (!this.properties.ue_properties.fixed_inputs) {
                    if (this.inputs.find((i, j)=>(i.type==type && j!=index))) return false
                }
                return onConnectInput?.apply(this, arguments)
            }
        }
        

        /*
        Extra menu options are the node right click menu.
        We add to this list, and also insert a link list outdate to everything.
        */
        add_extra_menu_items(nodeType.prototype, inject_outdating_into_object_method)

        if (is_UEnode(nodeType)) {
            const original_onDrawTitleBar = nodeType.prototype.onDrawTitleBar;
            nodeType.prototype.onDrawTitleBar = function(ctx, title_height) {
                original_onDrawTitleBar?.apply(this, arguments);
                if (any_restrictions(this)) indicate_restriction(ctx, title_height);
            }
        }

        if (is_UEnode(nodeType)) {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                /*if (!this.properties) this.properties = {}
                if (this.inputs) {
                    if (!this.widgets) this.widgets = [];
                    for (const input of this.inputs) {
                        if (input.widget && !this.widgets.find((w) => w.name === input.widget.name)) this.widgets.push(input.widget)
                    }
                }*/
                Logger.log_detail(`Node ${this.id} created`)
                setup_ue_properties_oncreate(this)
                return r;
            }
        }
    },

    async nodeCreated(node) {
        // TODO see if we still need this
        if (!node.__mode) {
            node.__mode = node.mode
            defineProperty(node, "mode", {
                get: ( )=>{return node.__mode},
                set: (v)=>{node.__mode = v; node.afterChangeMade?.('mode', v);}            
            })
        }

        const original_afterChangeMade = node.afterChangeMade
        node.afterChangeMade = (p, v) => {
            original_afterChangeMade?.(p,v)
            if (p==='mode') {
                linkRenderController.mark_link_list_outdated();
                node.widgets?.forEach((widget) => {widget.onModeChange?.(v)}); // no idea why I have this?
            }
        }

        if (is_helper(node)) { // editing a helper node makes the list dirty
            inject_outdating_into_objects(node.widgets,'callback',`widget callback on ${this.id}`);
        }

        // removing a node makes the list dirty
        inject_outdating_into_object_method(node, 'onRemoved', `node ${node.id} removed`)

        // check if the extra menu_items have been added (catch subgraph niode creation)
        add_extra_menu_items(node, inject_outdating_into_object_method)

        // creating a node makes the link list dirty - but give the system a moment to finish
        setTimeout( ()=>{linkRenderController.mark_link_list_outdated()}, 100 );
    }, 

    // When a graph node is loaded convert it if needed
    loadedGraphNode(node) { 
        if (graphConverter.running_116_plus()) { 
            graphConverter.convert_if_pre_116(node);
            if (node.isSubgraphNode?.()) {
                node.subgraph.nodes.forEach((n) => {
                    graphConverter.convert_if_pre_116(n);
                })
            }
         }
         setup_ue_properties_onload(node)
    },

	async setup() {

        create('link', null, document.getElementsByTagName('HEAD')[0], 
            {'rel':'stylesheet', 'type':'text/css', 'href': new URL("./ue.css", import.meta.url).href } )

        api.addEventListener("status", ({detail}) => {
            if (linkRenderController) linkRenderController.note_queue_size(detail ? detail.exec_info.queue_remaining : 0)
        });

        /* if we are on version 1.16 or later, stash input data to convert nodes when they are loaded */
        if (graphConverter.running_116_plus()) {
            const original_loadGraphData = app.loadGraphData;
            app.loadGraphData = function (data) {
                try {
                    graphConverter.store_node_input_map(data);
                } catch (e) { Logger.log_error(e); }
                const cvw_was = settingsCache.getSettingValue("Comfy.Validation.Workflows")
                if (settingsCache.getSettingValue("Use Everywhere.Options.block_graph_validation")) {
                    app.ui.settings.setSettingValue("Comfy.Validation.Workflows", false);
                }
                original_loadGraphData.apply(this, arguments);
                app.ui.settings.setSettingValue("Comfy.Validation.Workflows", cvw_was);
            }
        }
        
        /* 
        When we draw a node, render the virtual connection points
        */
        const original_drawNode = LGraphCanvas.prototype.drawNode;
        LGraphCanvas.prototype.drawNode = function(node, ctx) {
            try {
                linkRenderController.pause('drawNode')
                const v = original_drawNode.apply(this, arguments);
                linkRenderController.highlight_ue_connections(node, ctx);
                if (node._last_seen_bg !== node.bgcolor) linkRenderController.mark_link_list_outdated();
                node._last_seen_bg = node.bgcolor
                return v
            } catch (e) {
                Logger.log_error(e)
            } finally {          
                linkRenderController.unpause()
            }
        }

        const original_drawFrontCanvas = LGraphCanvas.prototype.drawFrontCanvas
        LGraphCanvas.prototype.drawFrontCanvas = function() {
            try {
                linkRenderController.disable_all_connected_widgets(true)
                return original_drawFrontCanvas.apply(this, arguments);
            }  catch (e) {
                Logger.log_error(e)
            } finally {
                try {
                    linkRenderController.disable_all_connected_widgets(false)
                } catch (e) {
                    Logger.log_error(e)
                } 
            }
        }

        /*
        When we draw connections, do the ue ones as well (logic for on/off is in lrc)
        */
        const drawConnections = LGraphCanvas.prototype.drawConnections;
        LGraphCanvas.prototype.drawConnections = function(ctx) {
            drawConnections?.apply(this, arguments);
            try {
                linkRenderController.render_all_ue_links(ctx);
            } catch (e) {
                Logger.log_error(e)
            }
        }
        
        /* 
        Canvas menu is the right click on backdrop.
        We need to add our options, and hijack the others to mark link list dirty
        */
        const original_getCanvasMenuOptions = LGraphCanvas.prototype.getCanvasMenuOptions;
        LGraphCanvas.prototype.getCanvasMenuOptions = function () {
            // Add our items to the canvas menu 
            const options = original_getCanvasMenuOptions.apply(this, arguments);
            canvas_menu_settings(options);
            
            //  every menu item makes our list dirty
            inject_outdating_into_objects(options,'callback',`menu option on canvas`);

            return options;
        }

                        /*
        Finding a widget by it's name is something done a lot of times in rendering, 
        so add a method that caches the names that can be used deep in the rendering code.

        TODO: Ought to delete this._widgetNameMap when widgets are added or removed.
        */
        LGraphNode.prototype._getWidgetByName = function(nm) {
            if (this._widgetNameMap === undefined || !this._widgetNameMap[nm]) {
                this._widgetNameMap = {}
                this.widgets?.forEach((w)=>{this._widgetNameMap[w.name] = w})
            }
            if (!this._widgetNameMap[nm]) {
                let breakpoint_be_here; // someone is asking for a widget that doesn't exist
            }
            return this._widgetNameMap[nm]
        }


	},

    init() {
        graphAnalyser = GraphAnalyser.instance();
        linkRenderController = LinkRenderController.instance(graphAnalyser);

        const original_afterChange = app.graph.afterChange
        app.graph.afterChange = function () {
            deferred_actions.execute()
            original_afterChange?.apply(this, arguments)
        }


        var prompt_being_queued = false;

        const original_graphToPrompt = app.graphToPrompt;
        app.graphToPrompt = async function () {
            if (prompt_being_queued) {
                return await graphAnalyser.graph_to_prompt( );
            } else {
                return await original_graphToPrompt.apply(app, arguments);
            }
        }
        
        app.ue_modified_prompt = async function () {
            return await graphAnalyser.graph_to_prompt();
        }

        const original_queuePrompt = app.queuePrompt;
        app.queuePrompt = async function () {
            prompt_being_queued = true;
            try {
                return await original_queuePrompt.apply(app, arguments);
            } finally {
                prompt_being_queued = false;
            }
        }
        
        app.canvas.__node_over = app.canvas.node_over;
        defineProperty(app.canvas, 'node_over', {
            get: ( )=>{return app.canvas.__node_over },
            set: (v)=>{app.canvas.__node_over = v; linkRenderController.node_over_changed(v)}   
        } )

        app.canvas.canvas.addEventListener('litegraph:set-graph', ()=>{
            linkRenderController.mark_link_list_outdated()
            setTimeout(()=>{app.canvas.setDirty(true,true)},200)
        })

        app.canvas.canvas.addEventListener('litegraph:canvas', (e)=>{
            if (e?.detail?.subType=='node-double-click') {
                const node = e.detail.node
                if (node.IS_UE) {
                    if (app.ui.settings.getSettingValue('Comfy.Node.DoubleClickTitleToEdit') && e.detail.originalEvent.canvasY<node.pos[1]) return
                    edit_restrictions(null, null, null, null, node)
                }
            }
        })

        if (false) add_debug();

        const export_api_label = Array.from(document.getElementsByClassName('p-menubar-item-label')).find((e)=>e.innerText=='Export (API)')
        if (export_api_label) {
            export_api_label.addEventListener('click', (e)=>{
                const ue_links = app.graph.extra['ue_links'];
                if (ue_links.length>0) {
                    if (!confirm("This model contains links added by Use Everywhere which won't work with the API. " + 
                        "You probably want to use 'Convert all UEs to real links' on the canvas right click menu before saving.\n\n" + 
                        "Save anyway?")) 
                    {
                        e.stopImmediatePropagation()
                        e.stopPropagation()
                        e.preventDefault()
                    }
                }
            })
        }

        const original_subgraph = app.graph.convertToSubgraph
        app.graph.convertToSubgraph = function () {
            const ctb_was = graphAnalyser.connect_to_bypassed
            graphAnalyser.connect_to_bypassed = true
            try {
                
                const cur_list = graphAnalyser.wait_to_analyse_visible_graph()
                const mods = convert_to_links(cur_list, null, visible_graph());
                const r = original_subgraph.apply(this, arguments);
                mods.restorer()
                return r
            } finally {
                graphAnalyser.connect_to_bypassed = ctb_was
            }
        }
    },

    beforeConfigureGraph() {
        linkRenderController.pause("before configure", 1000)
        graphAnalyser.pause("before configure", 1000)
        graphConverter.graph_being_configured = true
    },

    afterConfigureGraph() {
        graphConverter.remove_saved_ue_links_recursively(app.graph)
        //convert_old_nodes(app.graph)
        graphConverter.graph_being_configured = false
    }

});
