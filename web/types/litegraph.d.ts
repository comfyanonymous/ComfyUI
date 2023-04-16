// Type definitions for litegraph.js 0.7.0
// Project: litegraph.js
// Definitions by: NateScarlet <https://github.com/NateScarlet>

export type Vector2 = [number, number];
export type Vector4 = [number, number, number, number];
export type widgetTypes =
    | "number"
    | "slider"
    | "combo"
    | "text"
    | "toggle"
    | "button";
export type SlotShape =
    | typeof LiteGraph.BOX_SHAPE
    | typeof LiteGraph.CIRCLE_SHAPE
    | typeof LiteGraph.ARROW_SHAPE
    | typeof LiteGraph.SQUARE_SHAPE
    | number; // For custom shapes

/** https://github.com/jagenjo/litegraph.js/tree/master/guides#node-slots */
export interface INodeSlot {
    name: string;
    type: string | -1;
    label?: string;
    dir?:
        | typeof LiteGraph.UP
        | typeof LiteGraph.RIGHT
        | typeof LiteGraph.DOWN
        | typeof LiteGraph.LEFT;
    color_on?: string;
    color_off?: string;
    shape?: SlotShape;
    locked?: boolean;
    nameLocked?: boolean;
}

export interface INodeInputSlot extends INodeSlot {
    link: LLink["id"] | null;
}
export interface INodeOutputSlot extends INodeSlot {
    links: LLink["id"][] | null;
}

export type WidgetCallback<T extends IWidget = IWidget> = (
    this: T,
    value: T["value"],
    graphCanvas: LGraphCanvas,
    node: LGraphNode,
    pos: Vector2,
    event?: MouseEvent
) => void;

export interface IWidget<TValue = any, TOptions = any> {
    name: string | null;
    value: TValue;
    options?: TOptions;
    type?: widgetTypes;
    y?: number;
    property?: string;
    last_y?: number;
    clicked?: boolean;
    marker?: boolean;
    callback?: WidgetCallback<this>;
    /** Called by `LGraphCanvas.drawNodeWidgets` */
    draw?(
        ctx: CanvasRenderingContext2D,
        node: LGraphNode,
        width: number,
        posY: number,
        height: number
    ): void;
    /**
     * Called by `LGraphCanvas.processNodeWidgets`
     * https://github.com/jagenjo/litegraph.js/issues/76
     */
    mouse?(
        event: MouseEvent,
        pos: Vector2,
        node: LGraphNode
    ): boolean;
    /** Called by `LGraphNode.computeSize` */
    computeSize?(width: number): [number, number];
}
export interface IButtonWidget extends IWidget<null, {}> {
    type: "button";
}
export interface IToggleWidget
    extends IWidget<boolean, { on?: string; off?: string }> {
    type: "toggle";
}
export interface ISliderWidget
    extends IWidget<number, { max: number; min: number }> {
    type: "slider";
}
export interface INumberWidget extends IWidget<number, { precision: number }> {
    type: "number";
}
export interface IComboWidget
    extends IWidget<
        string[],
        {
            values:
                | string[]
                | ((widget: IComboWidget, node: LGraphNode) => string[]);
        }
    > {
    type: "combo";
}

export interface ITextWidget extends IWidget<string, {}> {
    type: "text";
}

export interface IContextMenuItem {
    content: string;
    callback?: ContextMenuEventListener;
    /** Used as innerHTML for extra child element */
    title?: string;
    disabled?: boolean;
    has_submenu?: boolean;
    submenu?: {
        options: ContextMenuItem[];
    } & IContextMenuOptions;
    className?: string;
}
export interface IContextMenuOptions {
    callback?: ContextMenuEventListener;
    ignore_item_callbacks?: Boolean;
    event?: MouseEvent | CustomEvent;
    parentMenu?: ContextMenu;
    autoopen?: boolean;
    title?: string;
    extra?: any;
}

export type ContextMenuItem = IContextMenuItem | null;
export type ContextMenuEventListener = (
    value: ContextMenuItem,
    options: IContextMenuOptions,
    event: MouseEvent,
    parentMenu: ContextMenu | undefined,
    node: LGraphNode
) => boolean | void;

export const LiteGraph: {
    VERSION: number;

    CANVAS_GRID_SIZE: number;

    NODE_TITLE_HEIGHT: number;
    NODE_TITLE_TEXT_Y: number;
    NODE_SLOT_HEIGHT: number;
    NODE_WIDGET_HEIGHT: number;
    NODE_WIDTH: number;
    NODE_MIN_WIDTH: number;
    NODE_COLLAPSED_RADIUS: number;
    NODE_COLLAPSED_WIDTH: number;
    NODE_TITLE_COLOR: string;
    NODE_TEXT_SIZE: number;
    NODE_TEXT_COLOR: string;
    NODE_SUBTEXT_SIZE: number;
    NODE_DEFAULT_COLOR: string;
    NODE_DEFAULT_BGCOLOR: string;
    NODE_DEFAULT_BOXCOLOR: string;
    NODE_DEFAULT_SHAPE: string;
    DEFAULT_SHADOW_COLOR: string;
    DEFAULT_GROUP_FONT: number;

    LINK_COLOR: string;
    EVENT_LINK_COLOR: string;
    CONNECTING_LINK_COLOR: string;

    MAX_NUMBER_OF_NODES: number; //avoid infinite loops
    DEFAULT_POSITION: Vector2; //default node position
    VALID_SHAPES: ["default", "box", "round", "card"]; //,"circle"

    //shapes are used for nodes but also for slots
    BOX_SHAPE: 1;
    ROUND_SHAPE: 2;
    CIRCLE_SHAPE: 3;
    CARD_SHAPE: 4;
    ARROW_SHAPE: 5;
    SQUARE_SHAPE: 6;

    //enums
    INPUT: 1;
    OUTPUT: 2;

    EVENT: -1; //for outputs
    ACTION: -1; //for inputs

    ALWAYS: 0;
    ON_EVENT: 1;
    NEVER: 2;
    ON_TRIGGER: 3;

    UP: 1;
    DOWN: 2;
    LEFT: 3;
    RIGHT: 4;
    CENTER: 5;

    STRAIGHT_LINK: 0;
    LINEAR_LINK: 1;
    SPLINE_LINK: 2;

    NORMAL_TITLE: 0;
    NO_TITLE: 1;
    TRANSPARENT_TITLE: 2;
    AUTOHIDE_TITLE: 3;

    node_images_path: string;

    debug: boolean;
    catch_exceptions: boolean;
    throw_errors: boolean;
    /** if set to true some nodes like Formula would be allowed to evaluate code that comes from unsafe sources (like node configuration), which could lead to exploits */
    allow_scripts: boolean;
    /** node types by string */
    registered_node_types: Record<string, LGraphNodeConstructor>;
    /** used for dropping files in the canvas */
    node_types_by_file_extension: Record<string, LGraphNodeConstructor>;
    /** node types by class name */
    Nodes: Record<string, LGraphNodeConstructor>;

    /** used to add extra features to the search box */
    searchbox_extras: Record<
        string,
        {
            data: { outputs: string[][]; title: string };
            desc: string;
            type: string;
        }
    >;

    createNode<T extends LGraphNode = LGraphNode>(type: string): T;
    /** Register a node class so it can be listed when the user wants to create a new one */
    registerNodeType(type: string, base: { new (): LGraphNode }): void;
    /** removes a node type from the system */
    unregisterNodeType(type: string): void;
    /** Removes all previously registered node's types. */
    clearRegisteredTypes(): void;
    /**
     * Create a new node type by passing a function, it wraps it with a proper class and generates inputs according to the parameters of the function.
     * Useful to wrap simple methods that do not require properties, and that only process some input to generate an output.
     * @param name node name with namespace (p.e.: 'math/sum')
     * @param func
     * @param param_types an array containing the type of every parameter, otherwise parameters will accept any type
     * @param return_type string with the return type, otherwise it will be generic
     * @param properties properties to be configurable
     */
    wrapFunctionAsNode(
        name: string,
        func: (...args: any[]) => any,
        param_types?: string[],
        return_type?: string,
        properties?: object
    ): void;

    /**
     * Adds this method to all node types, existing and to be created
     * (You can add it to LGraphNode.prototype but then existing node types wont have it)
     */
    addNodeMethod(name: string, func: (...args: any[]) => any): void;

    /**
     * Create a node of a given type with a name. The node is not attached to any graph yet.
     * @param type full name of the node class. p.e. "math/sin"
     * @param name a name to distinguish from other nodes
     * @param options to set options
     */
    createNode<T extends LGraphNode>(
        type: string,
        title: string,
        options: object
    ): T;

    /**
     * Returns a registered node type with a given name
     * @param type full name of the node class. p.e. "math/sin"
     */
    getNodeType<T extends LGraphNode>(type: string): LGraphNodeConstructor<T>;

    /**
     * Returns a list of node types matching one category
     * @method getNodeTypesInCategory
     * @param {String} category category name
     * @param {String} filter only nodes with ctor.filter equal can be shown
     * @return {Array} array with all the node classes
     */
    getNodeTypesInCategory(
        category: string,
        filter: string
    ): LGraphNodeConstructor[];

    /**
     * Returns a list with all the node type categories
     * @method getNodeTypesCategories
     * @param {String} filter only nodes with ctor.filter equal can be shown
     * @return {Array} array with all the names of the categories
     */                           
    getNodeTypesCategories(filter: string): string[];

    /** debug purposes: reloads all the js scripts that matches a wildcard */
    reloadNodes(folder_wildcard: string): void;

    getTime(): number;
    LLink: typeof LLink;
    LGraph: typeof LGraph;
    DragAndScale: typeof DragAndScale;
    compareObjects(a: object, b: object): boolean;
    distance(a: Vector2, b: Vector2): number;
    colorToString(c: string): string;
    isInsideRectangle(
        x: number,
        y: number,
        left: number,
        top: number,
        width: number,
        height: number
    ): boolean;
    growBounding(bounding: Vector4, x: number, y: number): Vector4;
    isInsideBounding(p: Vector2, bb: Vector4): boolean;
    hex2num(hex: string): [number, number, number];
    num2hex(triplet: [number, number, number]): string;
    ContextMenu: typeof ContextMenu;
    extendClass<A, B>(target: A, origin: B): A & B;
    getParameterNames(func: string): string[];
};

export type serializedLGraph<
    TNode = ReturnType<LGraphNode["serialize"]>,
    // https://github.com/jagenjo/litegraph.js/issues/74
    TLink = [number, number, number, number, number, string],
    TGroup = ReturnType<LGraphGroup["serialize"]>
> = {
    last_node_id: LGraph["last_node_id"];
    last_link_id: LGraph["last_link_id"];
    nodes: TNode[];
    links: TLink[];
    groups: TGroup[];
    config: LGraph["config"];
    version: typeof LiteGraph.VERSION;
};

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
    private _nodes_executable:
        | (LGraphNode & { onExecute: NonNullable<LGraphNode["onExecute"]> }[])
        | null;
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
    arrange(margin?: number,layout?: string): void;
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
    findNodesByClass<T extends LGraphNode>(
        classObject: LGraphNodeConstructor<T>
    ): T[];
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

export type SerializedLLink = [number, string, number, number, number, number];
export declare class LLink {
    id: number;
    type: string;
    origin_id: number;
    origin_slot: number;
    target_id: number;
    target_slot: number;
    constructor(
        id: number,
        type: string,
        origin_id: number,
        origin_slot: number,
        target_id: number,
        target_slot: number
    );
    configure(o: LLink | SerializedLLink): void;
    serialize(): SerializedLLink;
}

export type SerializedLGraphNode<T extends LGraphNode = LGraphNode> = {
    id: T["id"];
    type: T["type"];
    pos: T["pos"];
    size: T["size"];
    flags: T["flags"];
    mode: T["mode"];
    inputs: T["inputs"];
    outputs: T["outputs"];
    title: T["title"];
    properties: T["properties"];
    widgets_values?: IWidget["value"][];
};

/** https://github.com/jagenjo/litegraph.js/blob/master/guides/README.md#lgraphnode */
export declare class LGraphNode {
    static title_color: string;
    static title: string;
    static type: null | string;
    static widgets_up: boolean;
    constructor(title?: string);

    title: string;
    type: null | string;
    size: Vector2;
    graph: null | LGraph;
    graph_version: number;
    pos: Vector2;
    is_selected: boolean;
    mouseOver: boolean;

    id: number;

    //inputs available: array of inputs
    inputs: INodeInputSlot[];
    outputs: INodeOutputSlot[];
    connections: any[];

    //local data
    properties: Record<string, any>;
    properties_info: any[];

    flags: Partial<{
        collapsed: boolean
    }>;

    color: string;
    bgcolor: string;
    boxcolor: string;
    shape:
        | typeof LiteGraph.BOX_SHAPE
        | typeof LiteGraph.ROUND_SHAPE
        | typeof LiteGraph.CIRCLE_SHAPE
        | typeof LiteGraph.CARD_SHAPE
        | typeof LiteGraph.ARROW_SHAPE;

    serialize_widgets: boolean;
    skip_list: boolean;

    /** Used in `LGraphCanvas.onMenuNodeMode` */
    mode?:
        | typeof LiteGraph.ON_EVENT
        | typeof LiteGraph.ON_TRIGGER
        | typeof LiteGraph.NEVER
        | typeof LiteGraph.ALWAYS;

    /** If set to true widgets do not start after the slots */
    widgets_up: boolean;
    /** widgets start at y distance from the top of the node */
    widgets_start_y: number;
    /** if you render outside the node, it will be clipped */
    clip_area: boolean;
    /** if set to false it wont be resizable with the mouse */
    resizable: boolean;
    /** slots are distributed horizontally */
    horizontal: boolean;
    /** if true, the node will show the bgcolor as 'red'  */
    has_errors?: boolean;

    /** configure a node from an object containing the serialized info */
    configure(info: SerializedLGraphNode): void;
    /** serialize the content */
    serialize(): SerializedLGraphNode;
    /** Creates a clone of this node  */
    clone(): this;
    /** serialize and stringify */
    toString(): string;
    /** get the title string */
    getTitle(): string;
    /** sets the value of a property */
    setProperty(name: string, value: any): void;
    /** sets the output data */
    setOutputData(slot: number, data: any): void;
    /** sets the output data */
    setOutputDataType(slot: number, type: string): void;
    /**
     * Retrieves the input data (data traveling through the connection) from one slot
     * @param slot
     * @param force_update if set to true it will force the connected node of this slot to output data into this link
     * @return data or if it is not connected returns undefined
     */
    getInputData<T = any>(slot: number, force_update?: boolean): T;
    /**
     * Retrieves the input data type (in case this supports multiple input types)
     * @param slot
     * @return datatype in string format
     */
    getInputDataType(slot: number): string;
    /**
     * Retrieves the input data from one slot using its name instead of slot number
     * @param slot_name
     * @param force_update if set to true it will force the connected node of this slot to output data into this link
     * @return data or if it is not connected returns null
     */
    getInputDataByName<T = any>(slot_name: string, force_update?: boolean): T;
    /** tells you if there is a connection in one input slot */
    isInputConnected(slot: number): boolean;
    /** tells you info about an input connection (which node, type, etc) */
    getInputInfo(
        slot: number
    ): { link: number; name: string; type: string | 0 } | null;
    /** returns the node connected in the input slot */
    getInputNode(slot: number): LGraphNode | null;
    /** returns the value of an input with this name, otherwise checks if there is a property with that name */
    getInputOrProperty<T = any>(name: string): T;
    /** tells you the last output data that went in that slot */
    getOutputData<T = any>(slot: number): T | null;
    /** tells you info about an output connection (which node, type, etc) */
    getOutputInfo(
        slot: number
    ): { name: string; type: string; links: number[] } | null;
    /** tells you if there is a connection in one output slot */
    isOutputConnected(slot: number): boolean;
    /** tells you if there is any connection in the output slots */
    isAnyOutputConnected(): boolean;
    /** retrieves all the nodes connected to this output slot */
    getOutputNodes(slot: number): LGraphNode[];
    /**  Triggers an event in this node, this will trigger any output with the same name */
    trigger(action: string, param: any): void;
    /**
     * Triggers an slot event in this node
     * @param slot the index of the output slot
     * @param param
     * @param link_id in case you want to trigger and specific output link in a slot
     */
    triggerSlot(slot: number, param: any, link_id?: number): void;
    /**
     * clears the trigger slot animation
     * @param slot the index of the output slot
     * @param link_id in case you want to trigger and specific output link in a slot
     */
    clearTriggeredSlot(slot: number, link_id?: number): void;
    /**
     * add a new property to this node
     * @param name
     * @param default_value
     * @param type string defining the output type ("vec3","number",...)
     * @param extra_info this can be used to have special properties of the property (like values, etc)
     */
    addProperty<T = any>(
        name: string,
        default_value: any,
        type: string,
        extra_info?: object
    ): T;
    /**
     * add a new output slot to use in this node
     * @param name
     * @param type string defining the output type ("vec3","number",...)
     * @param extra_info this can be used to have special properties of an output (label, special color, position, etc)
     */
    addOutput(
        name: string,
        type: string | -1,
        extra_info?: Partial<INodeOutputSlot>
    ): INodeOutputSlot;
    /**
     * add a new output slot to use in this node
     * @param array of triplets like [[name,type,extra_info],[...]]
     */
    addOutputs(
        array: [string, string | -1, Partial<INodeOutputSlot> | undefined][]
    ): void;
    /** remove an existing output slot */
    removeOutput(slot: number): void;
    /**
     * add a new input slot to use in this node
     * @param name
     * @param type string defining the input type ("vec3","number",...), it its a generic one use 0
     * @param extra_info this can be used to have special properties of an input (label, color, position, etc)
     */
    addInput(
        name: string,
        type: string | -1,
        extra_info?: Partial<INodeInputSlot>
    ): INodeInputSlot;
    /**
     * add several new input slots in this node
     * @param array of triplets like [[name,type,extra_info],[...]]
     */
    addInputs(
        array: [string, string | -1, Partial<INodeInputSlot> | undefined][]
    ): void;
    /** remove an existing input slot */
    removeInput(slot: number): void;
    /**
     * add an special connection to this node (used for special kinds of graphs)
     * @param name
     * @param type string defining the input type ("vec3","number",...)
     * @param pos position of the connection inside the node
     * @param direction if is input or output
     */
    addConnection(
        name: string,
        type: string,
        pos: Vector2,
        direction: string
    ): {
        name: string;
        type: string;
        pos: Vector2;
        direction: string;
        links: null;
    };
    setValue(v: any): void;
    /** computes the size of a node according to its inputs and output slots */
    computeSize(): [number, number];
    /**
     * https://github.com/jagenjo/litegraph.js/blob/master/guides/README.md#node-widgets
     * @return created widget
     */
    addWidget<T extends IWidget>(
        type: T["type"],
        name: string,
        value: T["value"],
        callback?: WidgetCallback<T> | string,
        options?: T["options"]
    ): T;

    addCustomWidget<T extends IWidget>(customWidget: T): T;

    /**
     * returns the bounding of the object, used for rendering purposes
     * @return [x, y, width, height]
     */
    getBounding(): Vector4;
    /** checks if a point is inside the shape of a node */
    isPointInside(
        x: number,
        y: number,
        margin?: number,
        skipTitle?: boolean
    ): boolean;
    /** checks if a point is inside a node slot, and returns info about which slot */
    getSlotInPosition(
        x: number,
        y: number
    ): {
        input?: INodeInputSlot;
        output?: INodeOutputSlot;
        slot: number;
        link_pos: Vector2;
    };
    /**
     * returns the input slot with a given name (used for dynamic slots), -1 if not found
     * @param name the name of the slot
     * @return the slot (-1 if not found)
     */
    findInputSlot(name: string): number;
    /**
     * returns the output slot with a given name (used for dynamic slots), -1 if not found
     * @param name the name of the slot
     * @return  the slot (-1 if not found)
     */
    findOutputSlot(name: string): number;
    /**
     * connect this node output to the input of another node
     * @param slot (could be the number of the slot or the string with the name of the slot)
     * @param  targetNode the target node
     * @param  targetSlot the input slot of the target node (could be the number of the slot or the string with the name of the slot, or -1 to connect a trigger)
     * @return {Object} the link_info is created, otherwise null
     */
    connect<T = any>(
        slot: number | string,
        targetNode: LGraphNode,
        targetSlot: number | string
    ): T | null;
    /**
     * disconnect one output to an specific node
     * @param slot (could be the number of the slot or the string with the name of the slot)
     * @param target_node the target node to which this slot is connected [Optional, if not target_node is specified all nodes will be disconnected]
     * @return if it was disconnected successfully
     */
    disconnectOutput(slot: number | string, targetNode?: LGraphNode): boolean;
    /**
     * disconnect one input
     * @param slot (could be the number of the slot or the string with the name of the slot)
     * @return if it was disconnected successfully
     */
    disconnectInput(slot: number | string): boolean;
    /**
     * returns the center of a connection point in canvas coords
     * @param is_input true if if a input slot, false if it is an output
     * @param slot (could be the number of the slot or the string with the name of the slot)
     * @param out a place to store the output, to free garbage
     * @return the position
     **/
    getConnectionPos(
        is_input: boolean,
        slot: number | string,
        out?: Vector2
    ): Vector2;
    /** Force align to grid */
    alignToGrid(): void;
    /** Console output */
    trace(msg: string): void;
    /** Forces to redraw or the main canvas (LGraphNode) or the bg canvas (links) */
    setDirtyCanvas(fg: boolean, bg: boolean): void;
    loadImage(url: string): void;
    /** Allows to get onMouseMove and onMouseUp events even if the mouse is out of focus */
    captureInput(v: any): void;
    /** Collapse the node to make it smaller on the canvas */
    collapse(force: boolean): void;
    /** Forces the node to do not move or realign on Z */
    pin(v?: boolean): void;
    localToScreen(x: number, y: number, graphCanvas: LGraphCanvas): Vector2;

    // https://github.com/jagenjo/litegraph.js/blob/master/guides/README.md#custom-node-appearance
    onDrawBackground?(
        ctx: CanvasRenderingContext2D,
        canvas: HTMLCanvasElement
    ): void;
    onDrawForeground?(
        ctx: CanvasRenderingContext2D,
        canvas: HTMLCanvasElement
    ): void;

    // https://github.com/jagenjo/litegraph.js/blob/master/guides/README.md#custom-node-behaviour
    onMouseDown?(
        event: MouseEvent,
        pos: Vector2,
        graphCanvas: LGraphCanvas
    ): void;
    onMouseMove?(
        event: MouseEvent,
        pos: Vector2,
        graphCanvas: LGraphCanvas
    ): void;
    onMouseUp?(
        event: MouseEvent,
        pos: Vector2,
        graphCanvas: LGraphCanvas
    ): void;
    onMouseEnter?(
        event: MouseEvent,
        pos: Vector2,
        graphCanvas: LGraphCanvas
    ): void;
    onMouseLeave?(
        event: MouseEvent,
        pos: Vector2,
        graphCanvas: LGraphCanvas
    ): void;
    onKey?(event: KeyboardEvent, pos: Vector2, graphCanvas: LGraphCanvas): void;

    /** Called by `LGraphCanvas.selectNodes` */
    onSelected?(): void;
    /** Called by `LGraphCanvas.deselectNode` */
    onDeselected?(): void;
    /** Called by `LGraph.runStep` `LGraphNode.getInputData` */
    onExecute?(): void;
    /** Called by `LGraph.serialize` */
    onSerialize?(o: SerializedLGraphNode): void;
    /** Called by `LGraph.configure` */
    onConfigure?(o: SerializedLGraphNode): void;
    /**
     * when added to graph (warning: this is called BEFORE the node is configured when loading)
     * Called by `LGraph.add`
     */
    onAdded?(graph: LGraph): void;
    /**
     * when removed from graph
     * Called by `LGraph.remove` `LGraph.clear`
     */
    onRemoved?(): void;
    /**
     * if returns false the incoming connection will be canceled
     * Called by `LGraph.connect`
     * @param inputIndex target input slot number
     * @param outputType type of output slot
     * @param outputSlot output slot object
     * @param outputNode node containing the output
     * @param outputIndex index of output slot
     */
    onConnectInput?(
        inputIndex: number,
        outputType: INodeOutputSlot["type"],
        outputSlot: INodeOutputSlot,
        outputNode: LGraphNode,
        outputIndex: number
    ): boolean;
    /**
     * if returns false the incoming connection will be canceled
     * Called by `LGraph.connect`
     * @param outputIndex target output slot number
     * @param inputType type of input slot
     * @param inputSlot input slot object
     * @param inputNode node containing the input
     * @param inputIndex index of input slot
     */
    onConnectOutput?(
        outputIndex: number,
        inputType: INodeInputSlot["type"],
        inputSlot: INodeInputSlot,
        inputNode: LGraphNode,
        inputIndex: number
    ): boolean;

    /**
     * Called just before connection (or disconnect - if input is linked).
     * A convenient place to switch to another input, or create new one.
     * This allow for ability to automatically add slots if needed
     * @param inputIndex
     * @return selected input slot index, can differ from parameter value
     */
    onBeforeConnectInput?(
        inputIndex: number
    ): number;
    
    /** a connection changed (new one or removed) (LiteGraph.INPUT or LiteGraph.OUTPUT, slot, true if connected, link_info, input_info or output_info ) */
    onConnectionsChange(
        type: number,
        slotIndex: number,
        isConnected: boolean,
        link: LLink,
        ioSlot: (INodeOutputSlot | INodeInputSlot)
    ): void;                           

    /**
     * if returns false, will abort the `LGraphNode.setProperty`
     * Called when a property is changed
     * @param property
     * @param value
     * @param prevValue
     */
    onPropertyChanged?(property: string, value: any, prevValue: any): void | boolean;

    /** Called by `LGraphCanvas.processContextMenu` */
    getMenuOptions?(graphCanvas: LGraphCanvas): ContextMenuItem[];
    getSlotMenuOptions?(slot: INodeSlot): ContextMenuItem[];
}

export type LGraphNodeConstructor<T extends LGraphNode = LGraphNode> = {
    new (): T;
};

export type SerializedLGraphGroup = {
    title: LGraphGroup["title"];
    bounding: LGraphGroup["_bounding"];
    color: LGraphGroup["color"];
    font: LGraphGroup["font"];
};
export declare class LGraphGroup {
    title: string;
    private _bounding: Vector4;
    color: string;
    font: string;

    configure(o: SerializedLGraphGroup): void;
    serialize(): SerializedLGraphGroup;
    move(deltaX: number, deltaY: number, ignoreNodes?: boolean): void;
    recomputeInsideNodes(): void;
    isPointInside: LGraphNode["isPointInside"];
    setDirtyCanvas: LGraphNode["setDirtyCanvas"];
}

export declare class DragAndScale {
    constructor(element?: HTMLElement, skipEvents?: boolean);
    offset: [number, number];
    scale: number;
    max_scale: number;
    min_scale: number;
    onredraw: Function | null;
    enabled: boolean;
    last_mouse: Vector2;
    element: HTMLElement | null;
    visible_area: Vector4;
    bindEvents(element: HTMLElement): void;
    computeVisibleArea(): void;
    onMouse(e: MouseEvent): void;
    toCanvasContext(ctx: CanvasRenderingContext2D): void;
    convertOffsetToCanvas(pos: Vector2): Vector2;
    convertCanvasToOffset(pos: Vector2): Vector2;
    mouseDrag(x: number, y: number): void;
    changeScale(value: number, zooming_center?: Vector2): void;
    changeDeltaScale(value: number, zooming_center?: Vector2): void;
    reset(): void;
}

/**
 * This class is in charge of rendering one graph inside a canvas. And provides all the interaction required.
 * Valid callbacks are: onNodeSelected, onNodeDeselected, onShowNodePanel, onNodeDblClicked
 *
 * @param canvas the canvas where you want to render (it accepts a selector in string format or the canvas element itself)
 * @param graph
 * @param options { skip_rendering, autoresize }
 */
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
    static onShowPropertyEditor(
        item: any,
        options: any,
        e: any,
        menu: any,
        node: any
    ): void;
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
    links_render_mode:
        | typeof LiteGraph.STRAIGHT_LINK
        | typeof LiteGraph.LINEAR_LINK
        | typeof LiteGraph.SPLINE_LINK;
    live_mode: boolean;
    node_capturing_input: LGraphNode | null;
    node_dragged: LGraphNode | null;
    node_in_panel: LGraphNode | null;
    node_over: LGraphNode | null;
    node_title_color: string;
    node_widget: [LGraphNode, IWidget] | null;
    /** Called by `LGraphCanvas.drawBackCanvas` */
    onDrawBackground:
        | ((ctx: CanvasRenderingContext2D, visibleArea: Vector4) => void)
        | null;
    /** Called by `LGraphCanvas.drawFrontCanvas` */
    onDrawForeground:
        | ((ctx: CanvasRenderingContext2D, visibleArea: Vector4) => void)
        | null;
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
    onSearchBox:
        | ((
              helper: Element,
              value: string,
              graphCanvas: LGraphCanvas
          ) => string[])
        | null;
    onSearchBoxSelection:
        | ((name: string, event: MouseEvent, graphCanvas: LGraphCanvas) => void)
        | null;
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
    visible_area: DragAndScale["visible_area"];
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
    isOverNodeInput(
        node: LGraphNode,
        canvasX: number,
        canvasY: number,
        slotPos: Vector2
    ): boolean;

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

    computeConnectionPoint(
        a: Vector2,
        b: Vector2,
        t: number,
        startDir?: number,
        endDir?: number
    ): void;

    drawExecutionOrder(ctx: CanvasRenderingContext2D): void;
    /** draws the widgets stored inside a node */
    drawNodeWidgets(
        node: LGraphNode,
        posY: number,
        ctx: CanvasRenderingContext2D,
        activeWidget: object
    ): void;
    /** process an event on widgets */
    processNodeWidgets(
        node: LGraphNode,
        pos: Vector2,
        event: Event,
        activeWidget: object
    ): void;
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
    prompt(
        title: string,
        value: any,
        callback: Function,
        event: any
    ): HTMLDivElement;
    showSearchBox(event?: MouseEvent): void;
    showEditPropertyValue(node: LGraphNode, property: any, options: any): void;
    createDialog(
        html: string,
        options?: { position?: Vector2; event?: MouseEvent }
    ): void;

    convertOffsetToCanvas: DragAndScale["convertOffsetToCanvas"];
    convertCanvasToOffset: DragAndScale["convertCanvasToOffset"];
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

declare class ContextMenu {
    static trigger(
        element: HTMLElement,
        event_name: string,
        params: any,
        origin: any
    ): void;
    static isCursorOverElement(event: MouseEvent, element: HTMLElement): void;
    static closeAllContextMenus(window: Window): void;
    constructor(values: ContextMenuItem[], options?: IContextMenuOptions, window?: Window);
    options: IContextMenuOptions;
    parentMenu?: ContextMenu;
    lock: boolean;
    current_submenu?: ContextMenu;
    addItem(
        name: string,
        value: ContextMenuItem,
        options?: IContextMenuOptions
    ): void;
    close(e?: MouseEvent, ignore_parent_menu?: boolean): void;
    getTopMenu(): void;
    getFirstEvent(): void;
}

declare global {
    interface CanvasRenderingContext2D {
        /** like rect but rounded corners */
        roundRect(
            x: number,
            y: number,
            width: number,
            height: number,
            radius: number,
            radiusLow: number
        ): void;
    }

    interface Math {
        clamp(v: number, min: number, max: number): number;
    }
}
