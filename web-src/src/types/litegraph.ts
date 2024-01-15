// Litegraph.d.js file is incomplete and missing several properties on several of its objects
// We correct this by reclaring them here
import {
    ContextMenuEventListener,
    ContextMenuItem,
    DragAndScale,
    Vector2,
    Vector4,
    LGraphNode,
    SerializedLGraphGroup,
    serializedLGraph,
    LLink,
    LGraphNodeConstructor,
    LiteGraph,
    IWidget,
    SlotShape,
} from 'litegraph.js';

// Added several missing properties
export interface LiteGraphCorrected {
    DEFAULT_GROUP_FONT_SIZE: number;
    GRID_SHAPE: number;

    overlapBounding: (a: Vector4, b: Vector4) => boolean;

    release_link_on_empty_shows_menu: boolean;
    alt_drag_do_clone_nodes: boolean;
}

// Added several missing propertyes
export declare class LGraphGroup {
    title: string;
    color: string;
    font: string;
    font_size: number;
    graph: LGraph | null;
    private _bounding: Vector4;
    private _nodes: LGraphNode[];
    private _pos: Float32Array;
    private _size: Float32Array;

    constructor(title?: string);

    configure(o: SerializedLGraphGroup): void;
    serialize(): SerializedLGraphGroup;
    move(deltaX: number, deltaY: number, ignoreNodes?: boolean): void;
    recomputeInsideNodes(): void;
    isPointInside(x: number, y: number, margin?: number): boolean;
    setDirtyCanvas(dirtyForeground: boolean, dirtyBackground?: boolean): void;

    // Define getters and setters for pos and size
    get bounding(): Vector4;
    get pos(): Float32Array;
    set pos(v: Float32Array);
    get size(): Float32Array;
    set size(v: Float32Array);
}

// identical to the original, but with our own LGraphGroup class
export declare class LGraph {
    static supported_types: string[];
    static STATUS_STOPPED: 1;
    static STATUS_RUNNING: 2;

    constructor(o?: object);

    filter: string;
    catch_errors: boolean;
    /** custom data */
    config: object;
    elapsed_time: number;
    fixedtime: number;
    fixedtime_lapse: number;
    globaltime: number;
    inputs: any;
    iteration: number;
    last_link_id: number;
    last_node_id: number;
    last_update_time: number;
    links: Record<number, LLink>;
    list_of_graphcanvas: LGraphCanvas[];
    outputs: any;
    runningtime: number;
    starttime: number;
    status: typeof LGraph.STATUS_RUNNING | typeof LGraph.STATUS_STOPPED;

    private _nodes: LGraphNode[];
    private _groups: LGraphGroup[];
    private _nodes_by_id: Record<number, LGraphNode>;
    /** nodes that are executable sorted in execution order */
    private _nodes_executable: (LGraphNode & { onExecute: NonNullable<LGraphNode['onExecute']> }[]) | null;
    /** nodes that contain onExecute */
    private _nodes_in_order: LGraphNode[];
    private _version: number;

    getSupportedTypes(): string[];
    /** Removes all nodes from this graph */
    clear(): void;
    /** Attach Canvas to this graph */
    attachCanvas(graphCanvas: LGraphCanvas): void;
    /** Detach Canvas to this graph */
    detachCanvas(graphCanvas: LGraphCanvas): void;
    /**
     * Starts running this graph every interval milliseconds.
     * @param interval amount of milliseconds between executions, if 0 then it renders to the monitor refresh rate
     */
    start(interval?: number): void;
    /** Stops the execution loop of the graph */
    stop(): void;
    /**
     * Run N steps (cycles) of the graph
     * @param num number of steps to run, default is 1
     */
    runStep(num?: number, do_not_catch_errors?: boolean): void;
    /**
     * Updates the graph execution order according to relevance of the nodes (nodes with only outputs have more relevance than
     * nodes with only inputs.
     */
    updateExecutionOrder(): void;
    /** This is more internal, it computes the executable nodes in order and returns it */
    computeExecutionOrder<T = any>(only_onExecute: boolean, set_level: any): T;
    /**
     * Returns all the nodes that could affect this one (ancestors) by crawling all the inputs recursively.
     * It doesn't include the node itself
     * @return an array with all the LGraphNodes that affect this node, in order of execution
     */
    getAncestors(node: LGraphNode): LGraphNode[];
    /**
     * Positions every node in a more readable manner
     */
    arrange(margin?: number, layout?: string): void;
    /**
     * Returns the amount of time the graph has been running in milliseconds
     * @return number of milliseconds the graph has been running
     */
    getTime(): number;

    /**
     * Returns the amount of time accumulated using the fixedtime_lapse var. This is used in context where the time increments should be constant
     * @return number of milliseconds the graph has been running
     */
    getFixedTime(): number;

    /**
     * Returns the amount of time it took to compute the latest iteration. Take into account that this number could be not correct
     * if the nodes are using graphical actions
     * @return number of milliseconds it took the last cycle
     */
    getElapsedTime(): number;
    /**
     * Sends an event to all the nodes, useful to trigger stuff
     * @param eventName the name of the event (function to be called)
     * @param params parameters in array format
     */
    sendEventToAllNodes(eventName: string, params: any[], mode?: any): void;

    sendActionToCanvas(action: any, params: any[]): void;
    /**
     * Adds a new node instance to this graph
     * @param node the instance of the node
     */
    add(node: LGraphNode, skip_compute_order?: boolean): void;
    /**
     * Called when a new node is added
     * @param node the instance of the node
     */
    onNodeAdded(node: LGraphNode): void;
    /** Removes a node from the graph */
    remove(node: LGraphNode): void;
    /** Returns a node by its id. */
    getNodeById(id: number): LGraphNode | undefined;
    /**
     * Returns a list of nodes that matches a class
     * @param classObject the class itself (not an string)
     * @return a list with all the nodes of this type
     */
    findNodesByClass<T extends LGraphNode>(classObject: LGraphNodeConstructor<T>): T[];
    /**
     * Returns a list of nodes that matches a type
     * @param type the name of the node type
     * @return a list with all the nodes of this type
     */
    findNodesByType<T extends LGraphNode = LGraphNode>(type: string): T[];
    /**
     * Returns the first node that matches a name in its title
     * @param title the name of the node to search
     * @return the node or null
     */
    findNodeByTitle<T extends LGraphNode = LGraphNode>(title: string): T | null;
    /**
     * Returns a list of nodes that matches a name
     * @param title the name of the node to search
     * @return a list with all the nodes with this name
     */
    findNodesByTitle<T extends LGraphNode = LGraphNode>(title: string): T[];
    /**
     * Returns the top-most node in this position of the canvas
     * @param x the x coordinate in canvas space
     * @param y the y coordinate in canvas space
     * @param nodes_list a list with all the nodes to search from, by default is all the nodes in the graph
     * @return the node at this position or null
     */
    getNodeOnPos<T extends LGraphNode = LGraphNode>(
        x: number,
        y: number,
        node_list?: LGraphNode[],
        margin?: number
    ): T | null;
    /**
     * Returns the top-most group in that position
     * @param x the x coordinate in canvas space
     * @param y the y coordinate in canvas space
     * @return the group or null
     */
    getGroupOnPos(x: number, y: number): LGraphGroup | null;

    onAction(action: any, param: any): void;
    trigger(action: any, param: any): void;
    /** Tell this graph it has a global graph input of this type */
    addInput(name: string, type: string, value?: any): void;
    /** Assign a data to the global graph input */
    setInputData(name: string, data: any): void;
    /** Returns the current value of a global graph input */
    getInputData<T = any>(name: string): T;
    /** Changes the name of a global graph input */
    renameInput(old_name: string, name: string): false | undefined;
    /** Changes the type of a global graph input */
    changeInputType(name: string, type: string): false | undefined;
    /** Removes a global graph input */
    removeInput(name: string): boolean;
    /** Creates a global graph output */
    addOutput(name: string, type: string, value: any): void;
    /** Assign a data to the global output */
    setOutputData(name: string, value: string): void;
    /** Returns the current value of a global graph output */
    getOutputData<T = any>(name: string): T;

    /** Renames a global graph output */
    renameOutput(old_name: string, name: string): false | undefined;
    /** Changes the type of a global graph output */
    changeOutputType(name: string, type: string): false | undefined;
    /** Removes a global graph output */
    removeOutput(name: string): boolean;
    triggerInput(name: string, value: any): void;
    setCallback(name: string, func: (...args: any[]) => any): void;
    beforeChange(info?: LGraphNode): void;
    afterChange(info?: LGraphNode): void;
    connectionChange(node: LGraphNode): void;
    /** returns if the graph is in live mode */
    isLive(): boolean;
    /** clears the triggered slot animation in all links (stop visual animation) */
    clearTriggeredSlots(): void;
    /* Called when something visually changed (not the graph!) */
    change(): void;
    setDirtyCanvas(fg: boolean, bg: boolean): void;
    /** Destroys a link */
    removeLink(link_id: number): void;
    /** Creates a Object containing all the info about this graph, it can be serialized */
    serialize<T extends serializedLGraph>(): T;
    /**
     * Configure a graph from a JSON string
     * @param data configure a graph from a JSON string
     * @returns if there was any error parsing
     */
    configure(data: object, keep_old?: boolean): boolean | undefined;
    load(url: string): void;
}

// Same as original, but with our new version of LGraph -> LGraphGroup
export declare class LGraphCanvas {
    static node_colors: Record<
        string,
        {
            color: string;
            bgcolor: string;
            groupcolor: string;
        }
    >;
    static link_type_colors: Record<string, string>;
    static gradients: object;
    static search_limit: number;

    static getFileExtension(url: string): string;
    static decodeHTML(str: string): string;

    static onMenuCollapseAll(): void;
    static onMenuNodeEdit(): void;
    static onShowPropertyEditor(item: any, options: any, e: any, menu: any, node: any): void;
    /** Create menu for `Add Group` */
    static onGroupAdd: ContextMenuEventListener;
    /** Create menu for `Add Node` */
    static onMenuAdd: ContextMenuEventListener;
    static showMenuNodeOptionalInputs: ContextMenuEventListener;
    static showMenuNodeOptionalOutputs: ContextMenuEventListener;
    static onShowMenuNodeProperties: ContextMenuEventListener;
    static onResizeNode: ContextMenuEventListener;
    static onMenuNodeCollapse: ContextMenuEventListener;
    static onMenuNodePin: ContextMenuEventListener;
    static onMenuNodeMode: ContextMenuEventListener;
    static onMenuNodeColors: ContextMenuEventListener;
    static onMenuNodeShapes: ContextMenuEventListener;
    static onMenuNodeRemove: ContextMenuEventListener;
    static onMenuNodeClone: ContextMenuEventListener;

    constructor(
        canvas: HTMLCanvasElement | string,
        graph?: LGraph,
        options?: {
            skip_render?: boolean;
            autoresize?: boolean;
        }
    );

    static active_canvas: HTMLCanvasElement;

    allow_dragcanvas: boolean;
    allow_dragnodes: boolean;
    /** allow to control widgets, buttons, collapse, etc */
    allow_interaction: boolean;
    /** allows to change a connection with having to redo it again */
    allow_reconnect_links: boolean;
    /** allow selecting multi nodes without pressing extra keys */
    multi_select: boolean;
    /** No effect */
    allow_searchbox: boolean;
    always_render_background: boolean;
    autoresize?: boolean;
    background_image: string;
    bgcanvas: HTMLCanvasElement;
    bgctx: CanvasRenderingContext2D;
    canvas: HTMLCanvasElement;
    canvas_mouse: Vector2;
    clear_background: boolean;
    connecting_node: LGraphNode | null;
    connections_width: number;
    ctx: CanvasRenderingContext2D;
    current_node: LGraphNode | null;
    default_connection_color: {
        input_off: string;
        input_on: string;
        output_off: string;
        output_on: string;
    };
    default_link_color: string;
    dirty_area: Vector4 | null;
    dirty_bgcanvas?: boolean;
    dirty_canvas?: boolean;
    drag_mode: boolean;
    dragging_canvas: boolean;
    dragging_rectangle: Vector4 | null;
    ds: DragAndScale;
    /** used for transition */
    editor_alpha: number;
    filter: any;
    fps: number;
    frame: number;
    graph: LGraph;
    highlighted_links: Record<number, boolean>;
    highquality_render: boolean;
    inner_text_font: string;
    is_rendering: boolean;
    last_draw_time: number;
    last_mouse: Vector2;
    /**
     * Possible duplicated with `last_mouse`
     * https://github.com/jagenjo/litegraph.js/issues/70
     */
    last_mouse_position: Vector2;
    /** Timestamp of last mouse click, defaults to 0 */
    last_mouseclick: number;
    links_render_mode: typeof LiteGraph.STRAIGHT_LINK | typeof LiteGraph.LINEAR_LINK | typeof LiteGraph.SPLINE_LINK;
    live_mode: boolean;
    node_capturing_input: LGraphNode | null;
    node_dragged: LGraphNode | null;
    node_in_panel: LGraphNode | null;
    node_over: LGraphNode | null;
    node_title_color: string;
    node_widget: [LGraphNode, IWidget] | null;
    /** Called by `LGraphCanvas.drawBackCanvas` */
    onDrawBackground: ((ctx: CanvasRenderingContext2D, visibleArea: Vector4) => void) | null;
    /** Called by `LGraphCanvas.drawFrontCanvas` */
    onDrawForeground: ((ctx: CanvasRenderingContext2D, visibleArea: Vector4) => void) | null;
    onDrawOverlay: ((ctx: CanvasRenderingContext2D) => void) | null;
    /** Called by `LGraphCanvas.processMouseDown` */
    onMouse: ((event: MouseEvent) => boolean) | null;
    /** Called by `LGraphCanvas.drawFrontCanvas` and `LGraphCanvas.drawLinkTooltip` */
    onDrawLinkTooltip: ((ctx: CanvasRenderingContext2D, link: LLink, _this: this) => void) | null;
    /** Called by `LGraphCanvas.selectNodes` */
    onNodeMoved: ((node: LGraphNode) => void) | null;
    /** Called by `LGraphCanvas.processNodeSelected` */
    onNodeSelected: ((node: LGraphNode) => void) | null;
    /** Called by `LGraphCanvas.deselectNode` */
    onNodeDeselected: ((node: LGraphNode) => void) | null;
    /** Called by `LGraphCanvas.processNodeDblClicked` */
    onShowNodePanel: ((node: LGraphNode) => void) | null;
    /** Called by `LGraphCanvas.processNodeDblClicked` */
    onNodeDblClicked: ((node: LGraphNode) => void) | null;
    /** Called by `LGraphCanvas.selectNodes` */
    onSelectionChange: ((nodes: Record<number, LGraphNode>) => void) | null;
    /** Called by `LGraphCanvas.showSearchBox` */
    onSearchBox: ((helper: Element, value: string, graphCanvas: LGraphCanvas) => string[]) | null;
    onSearchBoxSelection: ((name: string, event: MouseEvent, graphCanvas: LGraphCanvas) => void) | null;
    pause_rendering: boolean;
    render_canvas_border: boolean;
    render_collapsed_slots: boolean;
    render_connection_arrows: boolean;
    render_connections_border: boolean;
    render_connections_shadows: boolean;
    render_curved_connections: boolean;
    render_execution_order: boolean;
    render_only_selected: boolean;
    render_shadows: boolean;
    render_title_colored: boolean;
    round_radius: number;
    selected_group: null | LGraphGroup;
    selected_group_resizing: boolean;
    selected_nodes: Record<number, LGraphNode>;
    show_info: boolean;
    title_text_font: string;
    /** set to true to render title bar with gradients */
    use_gradients: boolean;
    visible_area: DragAndScale['visible_area'];
    visible_links: LLink[];
    visible_nodes: LGraphNode[];
    zoom_modify_alpha: boolean;

    /** clears all the data inside */
    clear(): void;
    /** assigns a graph, you can reassign graphs to the same canvas */
    setGraph(graph: LGraph, skipClear?: boolean): void;
    /** opens a graph contained inside a node in the current graph */
    openSubgraph(graph: LGraph): void;
    /** closes a subgraph contained inside a node */
    closeSubgraph(): void;
    /** assigns a canvas */
    setCanvas(canvas: HTMLCanvasElement, skipEvents?: boolean): void;
    /** binds mouse, keyboard, touch and drag events to the canvas */
    bindEvents(): void;
    /** unbinds mouse events from the canvas */
    unbindEvents(): void;

    /**
     * this function allows to render the canvas using WebGL instead of Canvas2D
     * this is useful if you plant to render 3D objects inside your nodes, it uses litegl.js for webgl and canvas2DtoWebGL to emulate the Canvas2D calls in webGL
     **/
    enableWebGL(): void;

    /**
     * marks as dirty the canvas, this way it will be rendered again
     * @param fg if the foreground canvas is dirty (the one containing the nodes)
     * @param bg if the background canvas is dirty (the one containing the wires)
     */
    setDirty(fg: boolean, bg: boolean): void;

    /**
     * Used to attach the canvas in a popup
     * @return the window where the canvas is attached (the DOM root node)
     */
    getCanvasWindow(): Window;
    /** starts rendering the content of the canvas when needed */
    startRendering(): void;
    /** stops rendering the content of the canvas (to save resources) */
    stopRendering(): void;

    processMouseDown(e: MouseEvent): boolean | undefined;
    processMouseMove(e: MouseEvent): boolean | undefined;
    processMouseUp(e: MouseEvent): boolean | undefined;
    processMouseWheel(e: MouseEvent): boolean | undefined;

    /** returns true if a position (in graph space) is on top of a node little corner box */
    isOverNodeBox(node: LGraphNode, canvasX: number, canvasY: number): boolean;
    /** returns true if a position (in graph space) is on top of a node input slot */
    isOverNodeInput(node: LGraphNode, canvasX: number, canvasY: number, slotPos: Vector2): boolean;

    /** process a key event */
    processKey(e: KeyboardEvent): boolean | undefined;

    copyToClipboard(): void;
    pasteFromClipboard(): void;
    processDrop(e: DragEvent): void;
    checkDropItem(e: DragEvent): void;
    processNodeDblClicked(n: LGraphNode): void;
    processNodeSelected(n: LGraphNode, e: MouseEvent): void;
    processNodeDeselected(node: LGraphNode): void;

    /** selects a given node (or adds it to the current selection) */
    selectNode(node: LGraphNode, add?: boolean): void;
    /** selects several nodes (or adds them to the current selection) */
    selectNodes(nodes?: LGraphNode[], add?: boolean): void;
    /** removes a node from the current selection */
    deselectNode(node: LGraphNode): void;
    /** removes all nodes from the current selection */
    deselectAllNodes(): void;
    /** deletes all nodes in the current selection from the graph */
    deleteSelectedNodes(): void;

    /** centers the camera on a given node */
    centerOnNode(node: LGraphNode): void;
    /** changes the zoom level of the graph (default is 1), you can pass also a place used to pivot the zoom */
    setZoom(value: number, center: Vector2): void;
    /** brings a node to front (above all other nodes) */
    bringToFront(node: LGraphNode): void;
    /** sends a node to the back (below all other nodes) */
    sendToBack(node: LGraphNode): void;
    /** checks which nodes are visible (inside the camera area) */
    computeVisibleNodes(nodes: LGraphNode[]): LGraphNode[];
    /** renders the whole canvas content, by rendering in two separated canvas, one containing the background grid and the connections, and one containing the nodes) */
    draw(forceFG?: boolean, forceBG?: boolean): void;
    /** draws the front canvas (the one containing all the nodes) */
    drawFrontCanvas(): void;
    /** draws some useful stats in the corner of the canvas */
    renderInfo(ctx: CanvasRenderingContext2D, x: number, y: number): void;
    /** draws the back canvas (the one containing the background and the connections) */
    drawBackCanvas(): void;
    /** draws the given node inside the canvas */
    drawNode(node: LGraphNode, ctx: CanvasRenderingContext2D): void;
    /** draws graphic for node's slot */
    drawSlotGraphic(ctx: CanvasRenderingContext2D, pos: number[], shape: SlotShape, horizontal: boolean): void;
    /** draws the shape of the given node in the canvas */
    drawNodeShape(
        node: LGraphNode,
        ctx: CanvasRenderingContext2D,
        size: [number, number],
        fgColor: string,
        bgColor: string,
        selected: boolean,
        mouseOver: boolean
    ): void;
    /** draws every connection visible in the canvas */
    drawConnections(ctx: CanvasRenderingContext2D): void;
    /**
     * draws a link between two points
     * @param a start pos
     * @param b end pos
     * @param link the link object with all the link info
     * @param skipBorder ignore the shadow of the link
     * @param flow show flow animation (for events)
     * @param color the color for the link
     * @param startDir the direction enum
     * @param endDir the direction enum
     * @param numSublines number of sublines (useful to represent vec3 or rgb)
     **/
    renderLink(
        a: Vector2,
        b: Vector2,
        link: object,
        skipBorder: boolean,
        flow: boolean,
        color?: string,
        startDir?: number,
        endDir?: number,
        numSublines?: number
    ): void;

    computeConnectionPoint(a: Vector2, b: Vector2, t: number, startDir?: number, endDir?: number): void;

    drawExecutionOrder(ctx: CanvasRenderingContext2D): void;
    /** draws the widgets stored inside a node */
    drawNodeWidgets(node: LGraphNode, posY: number, ctx: CanvasRenderingContext2D, activeWidget: object): void;
    /** process an event on widgets */
    processNodeWidgets(node: LGraphNode, pos: Vector2, event: Event, activeWidget: object): void;
    /** draws every group area in the background */
    drawGroups(canvas: any, ctx: CanvasRenderingContext2D): void;
    adjustNodesSize(): void;
    /** resizes the canvas to a given size, if no size is passed, then it tries to fill the parentNode */
    resize(width?: number, height?: number): void;
    /**
     * switches to live mode (node shapes are not rendered, only the content)
     * this feature was designed when graphs where meant to create user interfaces
     **/
    switchLiveMode(transition?: boolean): void;
    onNodeSelectionChange(): void;
    touchHandler(event: TouchEvent): void;

    showLinkMenu(link: LLink, e: any): false;
    prompt(title: string, value: any, callback: Function, event: any): HTMLDivElement;
    showSearchBox(event?: MouseEvent): void;
    showEditPropertyValue(node: LGraphNode, property: any, options: any): void;
    createDialog(html: string, options?: { position?: Vector2; event?: MouseEvent }): void;

    convertOffsetToCanvas: DragAndScale['convertOffsetToCanvas'];
    convertCanvasToOffset: DragAndScale['convertCanvasToOffset'];
    /** converts event coordinates from canvas2D to graph coordinates */
    convertEventToCanvasOffset(e: MouseEvent): Vector2;
    /** adds some useful properties to a mouse event, like the position in graph coordinates */
    adjustMouseEvent(e: MouseEvent): void;

    getCanvasMenuOptions(): ContextMenuItem[];
    getNodeMenuOptions(node: LGraphNode): ContextMenuItem[];
    getGroupMenuOptions(): ContextMenuItem[];
    /** Called by `getCanvasMenuOptions`, replace default options */
    getMenuOptions?(): ContextMenuItem[];
    /** Called by `getCanvasMenuOptions`, append to default options */
    getExtraMenuOptions?(): ContextMenuItem[];
    /** Called when mouse right click */
    processContextMenu(node: LGraphNode, event: Event): void;
}
