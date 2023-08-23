var w = /* @__PURE__ */ ((e) => (e[e.UP = 1] = "UP", e[e.DOWN = 2] = "DOWN", e[e.LEFT = 3] = "LEFT", e[e.RIGHT = 4] = "RIGHT", e[e.CENTER = 5] = "CENTER", e))(w || {}), Z = /* @__PURE__ */ ((e) => (e[e.ALWAYS = 0] = "ALWAYS", e[e.ON_EVENT = 1] = "ON_EVENT", e[e.NEVER = 2] = "NEVER", e[e.ON_TRIGGER = 3] = "ON_TRIGGER", e[e.ON_REQUEST = 4] = "ON_REQUEST", e))(Z || {});
const re = ["Always", "On Event", "Never", "On Trigger"], Oe = ["#666", "#422", "#333", "#224", "#626"];
var k = /* @__PURE__ */ ((e) => (e[e.DEFAULT = 0] = "DEFAULT", e[e.BOX_SHAPE = 1] = "BOX_SHAPE", e[e.ROUND_SHAPE = 2] = "ROUND_SHAPE", e[e.CIRCLE_SHAPE = 3] = "CIRCLE_SHAPE", e[e.CARD_SHAPE = 4] = "CARD_SHAPE", e[e.ARROW_SHAPE = 5] = "ARROW_SHAPE", e[e.GRID_SHAPE = 6] = "GRID_SHAPE", e))(k || {});
const Ie = ["default", "box", "round", "circle", "card", "arrow", "square"];
var W = /* @__PURE__ */ ((e) => (e[e.INPUT = 0] = "INPUT", e[e.OUTPUT = 1] = "OUTPUT", e))(W || {}), de = /* @__PURE__ */ ((e) => (e[e.STRAIGHT_LINK = 0] = "STRAIGHT_LINK", e[e.LINEAR_LINK = 1] = "LINEAR_LINK", e[e.SPLINE_LINK = 2] = "SPLINE_LINK", e))(de || {});
const Xe = ["Straight", "Linear", "Spline"];
var se = /* @__PURE__ */ ((e) => (e[e.NORMAL_TITLE = 0] = "NORMAL_TITLE", e[e.NO_TITLE = 1] = "NO_TITLE", e[e.TRANSPARENT_TITLE = 2] = "TRANSPARENT_TITLE", e[e.AUTOHIDE_TITLE = 3] = "AUTOHIDE_TITLE", e))(se || {}), I = /* @__PURE__ */ ((e) => (e[e.EVENT = -2] = "EVENT", e[e.ACTION = -1] = "ACTION", e[e.DEFAULT = 0] = "DEFAULT", e))(I || {});
const Ae = ["*", "array", "object", "number", "string", "enum", "boolean", "table"];
var ue = /* @__PURE__ */ ((e) => (e.VERTICAL_LAYOUT = "vertical", e.HORIZONTAL_LAYOUT = "horizontal", e))(ue || {});
function Te(e, t, i) {
  return t > e ? t : i < e ? i : e;
}
function ve(e, t) {
  return e.reduce((i, n) => {
    const s = t(n);
    return i[s] = n, i;
  }, {});
}
function Ce(e, t) {
  return t in e ? e[t] : null;
}
function ye(e, t) {
  return t in e.constructor ? e.constructor[t] : null;
}
function Ge(e, t) {
  if (e.target !== t)
    return;
  let i = e.clientX - parseInt(window.getComputedStyle(t).left), n = e.clientY - parseInt(window.getComputedStyle(t).top);
  const s = (o) => {
    if (o.buttons === 0) {
      r();
      return;
    }
    t.style.top = o.clientY - n + "px", t.style.left = o.clientX - i + "px";
  }, r = () => {
    window.removeEventListener("mousemove", s), window.removeEventListener("mouseup", r);
  };
  window.addEventListener("mousemove", s), window.addEventListener("mouseup", r);
}
function Ee(e) {
  return e.addEventListener("mousedown", (t) => Ge(t, e)), e.classList.add("draggable"), e;
}
function J(e) {
  return e === I.EVENT ? "Event" : e === I.ACTION ? "Action" : e === I.DEFAULT ? "Default" : e;
}
function Se(e) {
  return e === I.EVENT || e === I.ACTION || e === I.DEFAULT || typeof e == "string";
}
const S = class {
  /** Register a node class so it can be listed when the user wants to create a new one */
  static registerNodeType(e) {
    S.debug && console.log("Node registered: " + e.type);
    const t = e.name, i = e.type;
    if (!i)
      throw console.error(e), new Error("Config has no type: " + e);
    if (S.debug && console.debug(t, i), e.category == null || e.category === "") {
      const s = i.lastIndexOf("/");
      e.category = i.substring(0, s);
    }
    e.title || (e.title = t);
    const n = S.registered_node_types[i];
    if (n && console.warn("replacing node type: " + i), e.supported_extensions)
      for (let s in e.supported_extensions) {
        const r = e.supported_extensions[s];
        r && r.constructor === String && (S.node_types_by_file_extension[r.toLowerCase()] = e);
      }
    e.class.__LITEGRAPH_TYPE__ = i, S.registered_node_types[i] = e, e.class.name && (S.Nodes[t] = e), S.onNodeTypeRegistered && S.onNodeTypeRegistered(i, e), n && S.onNodeTypeReplaced && S.onNodeTypeReplaced(i, e, n);
  }
  /** removes a node type from the system */
  static unregisterNodeType(e) {
    let t;
    if (typeof e == "string" ? t = S.registered_node_types[e] : t = e, !t)
      throw "node type not found: " + e;
    delete S.registered_node_types[t.type], t.constructor.name && delete S.Nodes[t.constructor.name];
  }
  /**
   * Save a slot type and his node
   * @method registerSlotType
   * @param {String|Object} type name of the node or the node constructor itself
   * @param {String} slot_type name of the slot type (variable type), eg. string, number, array, boolean, ..
   */
  static registerNodeAndSlotType(e, t, i = !1) {
    let n;
    if (typeof e == "string" ? n = S.registered_node_types[e] : "type" in e ? n = S.registered_node_types[e.type] : n = e, !n)
      throw new Error("Node not registered!" + e);
    var s = n.class.__litegraph_type__;
    if (typeof t == "string")
      var r = t.split(",");
    else if (t == I.EVENT || t == I.ACTION)
      var r = ["_event_"];
    else
      var r = ["*"];
    for (var o = 0; o < r.length; ++o) {
      var a = r[o];
      a === "" && (a = "*");
      var l = i ? "registered_slot_out_types" : "registered_slot_in_types";
      typeof this[l][a] > "u" && (this[l][a] = { nodes: [] }), this[l][a].nodes.push(s), a !== "_event_" && a !== "*" && (i ? S.slot_types_out.includes(a.toLowerCase()) || (S.slot_types_out.push(a.toLowerCase()), S.slot_types_out.sort()) : S.slot_types_in.includes(a.toLowerCase()) || (S.slot_types_in.push(a.toLowerCase()), S.slot_types_in.sort()));
    }
  }
  /** Removes all previously registered node's types. */
  static clearRegisteredTypes() {
    S.registered_node_types = {}, S.node_types_by_file_extension = {}, S.Nodes = {}, S.searchbox_extras = {};
  }
  /**
   * Create a new node type by passing a function, it wraps it with a proper class and generates inputs according to the parameters of the function.
   * Useful to wrap simple methods that do not require properties, and that only process some input to generate an output.
   * @param name node name with namespace (p.e.: 'math/sum')
   * @param func
   * @param param_types an array containing the type of every parameter, otherwise parameters will accept any type
   * @param return_type string with the return type, otherwise it will be generic
   * @param properties properties to be configurable
   */
  // static wrapFunctionAsNode(
  //     name: string,
  //     func: (...args: any[]) => any,
  //     param_types?: string[],
  //     return_type?: string,
  //     properties?: object
  // ): void {
  //     var params = Array(func.length);
  //     var code = "";
  //     var names = LiteGraph.getParameterNames(func);
  //     for (var i = 0; i < names.length; ++i) {
  //         code +=
  //         "this.addInput('" +
  //             names[i] +
  //             "'," +
  //             (param_types && param_types[i]
  //                 ? "'" + param_types[i] + "'"
  //                 : "0") +
  //             ");\n";
  //     }
  //     code +=
  //     "this.addOutput('out'," +
  //         (return_type ? "'" + return_type + "'" : 0) +
  //         ");\n";
  //     if (properties) {
  //         code +=
  //         "this.properties = " + JSON.stringify(properties) + ";\n";
  //     }
  //     var classobj = Function(code) as any;
  //     classobj.title = name.split("/").pop();
  //     classobj.desc = "Generated from " + func.name;
  //     classobj.prototype.onExecute = function onExecute() {
  //         for (var i = 0; i < params.length; ++i) {
  //             params[i] = this.getInputData(i);
  //         }
  //         var r = func.apply(this, params);
  //         this.setOutputData(0, r);
  //     };
  //     LiteGraph.registerNodeType(name, classobj);
  // }
  /**
   * Adds this method to all node types, existing and to be created
   * (You can add it to LGraphNode.prototype but then existing node types wont have it)
   */
  // static addNodeMethod(name: string, func: (...args: any[]) => any): void {
  //     LGraphNode.prototype[name] = func;
  //     for (var i in LiteGraph.registered_node_types) {
  //         var type = LiteGraph.registered_node_types[i];
  //         if (type.prototype[name]) {
  //             type.prototype["_" + name] = type.prototype[name];
  //         } //keep old in case of replacing
  //         type.prototype[name] = func;
  //     }
  // }
  /**
   * Create a node of a given type with a name. The node is not attached to any graph yet.
   * @param type full name of the node class. p.e. "math/sin"
   * @param name a name to distinguish from other nodes
   * @param options to set options
   */
  static createNode(e, t, i = {}) {
    let n = null, s;
    if (typeof e == "string")
      s = e;
    else if (s = e.__LITEGRAPH_TYPE__, !s)
      throw console.error(e), "Node was not registered yet!";
    if (n = S.registered_node_types[s], !n)
      return console.warn(
        'GraphNode type "' + e + '" not registered.'
      ), null;
    t = t || n.title || s;
    var r = null;
    const o = i.constructorArgs || [];
    if (S.catch_exceptions)
      try {
        r = new n.class(t, ...o);
      } catch (p) {
        return console.error("Error creating node!", p), null;
      }
    else
      r = new n.class(t, ...o);
    if (r.class = n.class, r.type = s, !r.title && t && (r.title = t), r.properties || (r.properties = {}), r.properties_info || (r.properties_info = []), r.flags || (r.flags = {}), r.size || (r.size = r.computeSize()), r.pos || (r.pos = [S.DEFAULT_POSITION[0], S.DEFAULT_POSITION[1]]), r.mode || (r.mode = Z.ALWAYS), i.instanceProps)
      for (var a in i.instanceProps)
        r[a] = i.instanceProps[a];
    const l = Ce(n.class, "propertyLayout");
    if (l) {
      S.debug && console.debug("Found property layout!", l);
      for (const p of l) {
        const { name: f, defaultValue: c, type: v, options: g } = p;
        r.addProperty(f, c, v, g);
      }
    }
    const h = Ce(n.class, "slotLayout");
    if (h) {
      if (S.debug && console.debug("Found slot layout!", h), h.inputs)
        for (const p of h.inputs) {
          const { name: f, type: c, options: v } = p;
          r.addInput(f, c, v);
        }
      if (h.outputs)
        for (const p of h.outputs) {
          const { name: f, type: c, options: v } = p;
          r.addOutput(f, c, v);
        }
    }
    return r.onNodeCreated && r.onNodeCreated(), r;
  }
  /**
   * Returns a registered node type with a given name
   * @param type full name of the node class. p.e. "math/sin"
   */
  static getNodeType(e) {
    return S.registered_node_types[e];
  }
  /**
   * Returns a list of node types matching one category
   * @method getNodeTypesInCategory
   * @param {String} category category name
   * @param {String} filter only nodes with ctor.filter equal can be shown
   * @return {Array} array with all the node classes
   */
  static getNodeTypesInCategory(e, t) {
    var i = [];
    for (var n in S.registered_node_types) {
      var s = S.registered_node_types[n];
      s.filter == t && (e == "" ? s.category == null && i.push(s) : s.category == e && i.push(s));
    }
    return S.auto_sort_node_types && i.sort(function(r, o) {
      return r.title.localeCompare(o.title);
    }), i;
  }
  /**
   * Returns a list with all the node type categories
   * @method getNodeTypesCategories
   * @param {String} filter only nodes with ctor.filter equal can be shown
   * @return {Array} array with all the names of the categories
   */
  static getNodeTypesCategories(e) {
    var t = { "": 1 };
    for (var i in S.registered_node_types) {
      var n = S.registered_node_types[i];
      if (n.category && !n.hide_in_node_lists) {
        if (n.filter != e)
          continue;
        t[n.category] = 1;
      }
    }
    var s = [];
    for (var i in t)
      s.push(i);
    return S.auto_sort_node_types ? s.sort() : s;
  }
  /** debug purposes: reloads all the js scripts that matches a wildcard */
  static reloadNodes(e) {
    for (var t = document.getElementsByTagName("script"), i = [], n = 0; n < t.length; n++)
      i.push(t[n]);
    var s = document.getElementsByTagName("head")[0];
    e = document.location.href + e;
    for (var n = 0; n < i.length; n++) {
      var r = i[n].src;
      if (!(!r || r.substr(0, e.length) != e))
        try {
          S.debug && console.log("Reloading: " + r);
          var o = document.createElement("script");
          o.type = "text/javascript", o.src = r, s.appendChild(o), s.removeChild(i[n]);
        } catch (l) {
          if (S.throw_errors)
            throw l;
          S.debug && console.log("Error while reloading " + r);
        }
    }
    S.debug && console.log("Nodes reloaded");
  }
  // TODO move
  //separated just to improve if it doesn't work
  static cloneObject(e, t) {
    if (e == null)
      return null;
    var i = JSON.parse(JSON.stringify(e));
    if (!t)
      return i;
    for (var n in i)
      t[n] = i[n];
    return t;
  }
  /**
   * Returns if the types of two slots are compatible (taking into account wildcards, etc)
   * @method isValidConnection
   * @param {String} type_a
   * @param {String} type_b
   * @return {Boolean} true if they can be connected
   */
  static isValidConnection(e, t) {
    if ((e == "" || e === "*") && (e = I.DEFAULT), (t == "" || t === "*") && (t = I.DEFAULT), !e || !t || e == t || e == I.EVENT && t == I.ACTION || e == I.ACTION && t == I.EVENT)
      return !0;
    if (e = String(e), t = String(t), e = e.toLowerCase(), t = t.toLowerCase(), e.indexOf(",") == -1 && t.indexOf(",") == -1)
      return e == t;
    for (var i = e.split(","), n = t.split(","), s = 0; s < i.length; ++s)
      for (var r = 0; r < n.length; ++r)
        if (this.isValidConnection(i[s], n[r]))
          return !0;
    return !1;
  }
  static getTime() {
    return Date.now();
  }
  // static LLink: typeof LLink;
  // static LGraph: typeof LGraph;
  // static DragAndScale: typeof DragAndScale;
  static compareObjects(e, t) {
    for (var i in e)
      if (e[i] != t[i])
        return !1;
    return !0;
  }
  static distance(e, t) {
    return Math.sqrt(
      (t[0] - e[0]) * (t[0] - e[0]) + (t[1] - e[1]) * (t[1] - e[1])
    );
  }
  static colorToString(e) {
    return "rgba(" + Math.round(e[0] * 255).toFixed() + "," + Math.round(e[1] * 255).toFixed() + "," + Math.round(e[2] * 255).toFixed() + "," + (e.length == 4 ? e[3].toFixed(2) : "1.0") + ")";
  }
  static isInsideRectangle(e, t, i, n, s, r) {
    return i < e && i + s > e && n < t && n + r > t;
  }
  // [minx,miny,maxx,maxy]
  static growBounding(e, t, i) {
    return t < e[0] ? e[0] = t : t > e[2] && (e[2] = t), i < e[1] ? e[1] = i : i > e[3] && (e[3] = i), e;
  }
  static isInsideBounding(e, t) {
    return !(e[0] < t[0][0] || e[1] < t[0][1] || e[0] > t[1][0] || e[1] > t[1][1]);
  }
  // bounding overlap, format: [ startx, starty, width, height ]
  static overlapBounding(e, t) {
    var i = e[0] + e[2], n = e[1] + e[3], s = t[0] + t[2], r = t[1] + t[3];
    return !(e[0] > s || e[1] > r || i < t[0] || n < t[1]);
  }
  // Convert a hex value to its decimal value - the inputted hex must be in the
  // format of a hex triplet - the kind we use for HTML colours. The function
  // will return an array with three values.
  static hex2num(e) {
    e.charAt(0) == "#" && (e = e.slice(1)), e = e.toUpperCase();
    var t = "0123456789ABCDEF";
    let i;
    for (var n = 0, s, r, o = 0; o < 6; o += 2)
      s = t.indexOf(e.charAt(o)), r = t.indexOf(e.charAt(o + 1)), i[n] = s * 16 + r, n++;
    return i;
  }
  //Give a array with three values as the argument and the function will return
  //	the corresponding hex triplet.
  static num2hex(e) {
    for (var t = "0123456789ABCDEF", i = "#", n, s, r = 0; r < 3; r++)
      n = e[r] / 16, s = e[r] % 16, i += t.charAt(n) + t.charAt(s);
    return i;
  }
  // ContextMenu: typeof ContextMenu;
  // static extendClass<A, B>(target: A, origin: B): A & B;
  // static getParameterNames(func: string | Function): string[];
  /* helper for interaction: pointer, touch, mouse Listeners
     used by LGraphCanvas DragAndScale ContextMenu*/
  static pointerListenerAdd(e, t, i, n = !1) {
    if (!(!e || !e.addEventListener || !t || typeof i != "function")) {
      var s = S.pointerevents_method, r = t;
      if (s == "pointer" && !window.PointerEvent)
        switch (console.warn("sMethod=='pointer' && !window.PointerEvent"), console.log("Converting pointer[" + r + "] : down move up cancel enter TO touchstart touchmove touchend, etc .."), r) {
          case "down": {
            s = "touch", r = "start";
            break;
          }
          case "move": {
            s = "touch";
            break;
          }
          case "up": {
            s = "touch", r = "end";
            break;
          }
          case "cancel": {
            s = "touch";
            break;
          }
          case "enter": {
            console.log("debug: Should I send a move event?");
            break;
          }
          default:
            console.warn("PointerEvent not available in this browser ? The event " + r + " would not be called");
        }
      switch (r) {
        case "down":
        case "up":
        case "move":
        case "over":
        case "out":
        case "enter":
          e.addEventListener(s + r, i, n);
        case "leave":
        case "cancel":
        case "gotpointercapture":
        case "lostpointercapture":
          if (s != "mouse")
            return e.addEventListener(s + r, i, n);
        default:
          return e.addEventListener(r, i, n);
      }
    }
  }
  static pointerListenerRemove(e, t, i, n = !1) {
    if (!(!e || !e.removeEventListener || !t || typeof i != "function"))
      switch (t) {
        case "down":
        case "up":
        case "move":
        case "over":
        case "out":
        case "enter":
          (S.pointerevents_method == "pointer" || S.pointerevents_method == "mouse") && e.removeEventListener(S.pointerevents_method + t, i, n);
        case "leave":
        case "cancel":
        case "gotpointercapture":
        case "lostpointercapture":
          if (S.pointerevents_method == "pointer")
            return e.removeEventListener(S.pointerevents_method + t, i, n);
        default:
          return e.removeEventListener(t, i, n);
      }
  }
};
let u = S;
u.VERSION = 10;
u.CANVAS_GRID_SIZE = 10;
u.NODE_TITLE_HEIGHT = 20;
u.NODE_TITLE_TEXT_Y = 15;
u.NODE_SLOT_HEIGHT = 20;
u.NODE_WIDGET_HEIGHT = 20;
u.NODE_WIDTH = 140;
u.NODE_MIN_WIDTH = 50;
u.NODE_COLLAPSED_RADIUS = 10;
u.NODE_COLLAPSED_WIDTH = 80;
u.NODE_TITLE_COLOR = "#999";
u.NODE_SELECTED_TITLE_COLOR = "#FFF";
u.NODE_TEXT_SIZE = 14;
u.NODE_TEXT_COLOR = "#AAA";
u.NODE_SUBTEXT_SIZE = 12;
u.NODE_DEFAULT_COLOR = "#333";
u.NODE_DEFAULT_BGCOLOR = "#353535";
u.NODE_DEFAULT_BOXCOLOR = "#666";
u.NODE_DEFAULT_SHAPE = "box";
u.NODE_BOX_OUTLINE_COLOR = "#FFF";
u.DEFAULT_SHADOW_COLOR = "rgba(0,0,0,0.5)";
u.DEFAULT_GROUP_FONT_SIZE = 24;
u.WIDGET_BGCOLOR = "#222";
u.WIDGET_OUTLINE_COLOR = "#666";
u.WIDGET_TEXT_COLOR = "#DDD";
u.WIDGET_SECONDARY_TEXT_COLOR = "#999";
u.LINK_COLOR = "#9A9";
u.EVENT_LINK_COLOR = "#A86";
u.ACTION_LINK_COLOR = "#86A";
u.CONNECTING_LINK_COLOR = "#AFA";
u.MAX_NUMBER_OF_NODES = 1e3;
u.DEFAULT_POSITION = [100, 100];
u.proxy = null;
u.node_images_path = "";
u.debug = !1;
u.catch_exceptions = !0;
u.throw_errors = !0;
u.allow_scripts = !1;
u.registered_node_types = {};
u.node_types_by_file_extension = {};
u.Nodes = {};
u.Globals = {};
u.searchbox_extras = {};
u.auto_sort_node_types = !1;
u.node_box_coloured_when_on = !1;
u.node_box_coloured_by_mode = !1;
u.dialog_close_on_mouse_leave = !0;
u.dialog_close_on_mouse_leave_delay = 500;
u.shift_click_do_break_link_from = !1;
u.click_do_break_link_to = !1;
u.search_hide_on_mouse_leave = !0;
u.search_filter_enabled = !1;
u.search_show_all_on_open = !0;
u.auto_load_slot_types = !1;
u.registered_slot_in_types = {};
u.registered_slot_out_types = {};
u.slot_types_in = [];
u.slot_types_out = [];
u.slot_types_default_in = {};
u.slot_types_default_out = {};
u.alt_drag_do_clone_nodes = !1;
u.do_add_triggers_slots = !1;
u.allow_multi_output_for_events = !0;
u.middle_click_slot_add_default_node = !1;
u.release_link_on_empty_shows_menu = !1;
u.ignore_all_widget_events = !1;
u.pointerevents_method = "mouse";
u.use_uuids = !1;
u.search_box_refresh_interval_ms = 250;
u.graph_inputs_outputs_use_combo_widget = !1;
u.serialize_slot_data = !1;
class Be {
  constructor(t, i = !1) {
    this.offset = [0, 0], this.scale = 1, this.max_scale = 10, this.min_scale = 0.1, this.onredraw = null, this.enabled = !0, this.last_mouse = [0, 0], this.element = null, this.visible_area = new Float32Array([0, 0, 0, 0]), this.viewport = null, this.dragging = !1, this._binded_mouse_callback = null, t && (this.element = t, i || this.bindEvents(t));
  }
  bindEvents(t) {
    this.last_mouse = [0, 0], this._binded_mouse_callback = this.onMouse.bind(this), u.pointerListenerAdd(t, "down", this._binded_mouse_callback), u.pointerListenerAdd(t, "move", this._binded_mouse_callback), u.pointerListenerAdd(t, "up", this._binded_mouse_callback), t.addEventListener(
      "mousewheel",
      this._binded_mouse_callback,
      !1
    ), t.addEventListener("wheel", this._binded_mouse_callback, !1);
  }
  computeVisibleArea(t) {
    if (!this.element) {
      this.visible_area[0] = this.visible_area[1] = this.visible_area[2] = this.visible_area[3] = 0;
      return;
    }
    var i = this.element.width, n = this.element.height, s = -this.offset[0], r = -this.offset[1];
    t && (s += t[0] / this.scale, r += t[1] / this.scale, i = t[2], n = t[3]);
    var o = s + i / this.scale, a = r + n / this.scale;
    this.visible_area[0] = s, this.visible_area[1] = r, this.visible_area[2] = o - s, this.visible_area[3] = a - r;
  }
  onMouse(t) {
    if (!this.enabled)
      return;
    var i = this.element, n = i.getBoundingClientRect();
    let s = t;
    var r = s.clientX - n.left, o = s.clientY - n.top;
    s.canvasX = r, s.canvasX = o, s.dragging = this.dragging;
    var a = !this.viewport || this.viewport && r >= this.viewport[0] && r < this.viewport[0] + this.viewport[2] && o >= this.viewport[1] && o < this.viewport[1] + this.viewport[3];
    if (s.type == u.pointerevents_method + "down" && a)
      this.dragging = !0, u.pointerListenerRemove(i, "move", this._binded_mouse_callback), u.pointerListenerAdd(document, "move", this._binded_mouse_callback), u.pointerListenerAdd(document, "up", this._binded_mouse_callback);
    else if (s.type == u.pointerevents_method + "move") {
      var l = r - this.last_mouse[0], h = o - this.last_mouse[1];
      this.dragging && this.mouseDrag(l, h);
    } else
      s.type == u.pointerevents_method + "up" ? (this.dragging = !1, u.pointerListenerRemove(document, "move", this._binded_mouse_callback), u.pointerListenerRemove(document, "up", this._binded_mouse_callback), u.pointerListenerAdd(i, "move", this._binded_mouse_callback)) : a && (s.type == "mousewheel" || s.type == "wheel" || s.type == "DOMMouseScroll") && (s.eventType = "mousewheel", s.type == "wheel" ? s.wheel = -s.deltaY : s.wheel = s.wheelDeltaY != null ? s.wheelDeltaY : s.detail * -60, s.delta = s.wheelDelta ? s.wheelDelta / 40 : s.deltaY ? -s.deltaY / 3 : 0, this.changeDeltaScale(1 + s.delta * 0.05, [s.clientX, s.clientY]));
    if (this.last_mouse[0] = r, this.last_mouse[1] = o, a)
      return s.preventDefault(), s.stopPropagation(), !1;
  }
  toCanvasContext(t) {
    t.scale(this.scale, this.scale), t.translate(this.offset[0], this.offset[1]);
  }
  convertOffsetToCanvas(t) {
    return [
      (t[0] + this.offset[0]) * this.scale,
      (t[1] + this.offset[1]) * this.scale
    ];
  }
  convertCanvasToOffset(t, i = [0, 0]) {
    return i[0] = t[0] / this.scale - this.offset[0], i[1] = t[1] / this.scale - this.offset[1], i;
  }
  mouseDrag(t, i) {
    this.offset[0] += t / this.scale, this.offset[1] += i / this.scale, this.onredraw && this.onredraw(this);
  }
  changeScale(t, i) {
    if (t < this.min_scale ? t = this.min_scale : t > this.max_scale && (t = this.max_scale), t != this.scale && this.element) {
      var n = this.element.getBoundingClientRect();
      if (n) {
        i = i || [
          n.width * 0.5,
          n.height * 0.5
        ], i[0] -= n.left, i[1] -= n.top;
        var s = this.convertCanvasToOffset(i);
        this.scale = t, Math.abs(this.scale - 1) < 0.01 && (this.scale = 1);
        var r = this.convertCanvasToOffset(i), o = [
          r[0] - s[0],
          r[1] - s[1]
        ];
        this.offset[0] += o[0], this.offset[1] += o[1], this.onredraw && this.onredraw(this);
      }
    }
  }
  changeDeltaScale(t, i) {
    this.changeScale(this.scale * t, i);
  }
  reset() {
    this.scale = 1, this.offset[0] = 0, this.offset[1] = 0;
  }
}
class ge {
  processMouseDown(t) {
    if (this.set_canvas_dirty_on_mouse_event && (this.dirty_canvas = !0), !this.graph)
      return;
    let i = t;
    this.adjustMouseEvent(i);
    var n = this.getCanvasWindow();
    n.document, N.active_canvas = this;
    var s = i.clientX, r = i.clientY;
    this.ds.viewport = this.viewport;
    var o = !this.viewport || this.viewport && s >= this.viewport[0] && s < this.viewport[0] + this.viewport[2] && r >= this.viewport[1] && r < this.viewport[1] + this.viewport[3];
    if (this.skip_events || (u.pointerListenerRemove(this.canvas, "move", this._mousemove_callback), u.pointerListenerAdd(n.document, "move", this._mousemove_callback, !0), u.pointerListenerAdd(n.document, "up", this._mouseup_callback, !0)), !!o) {
      var a = this.graph.getNodeOnPos(i.canvasX, i.canvasY, this.visible_nodes, 5), l = !1, h = u.getTime(), p = !(i instanceof PointerEvent) || !i.isPrimary, f = h - this.last_mouseclick < 300 && p;
      if (this.mouse[0] = i.clientX, this.mouse[1] = i.clientY, this.offset_mouse[0] = i.offsetX, this.offset_mouse[1] = i.offsetY, this.graph_mouse[0] = i.canvasX, this.graph_mouse[1] = i.canvasY, this.last_click_position = [this.mouse[0], this.mouse[1]], this.last_click_position_offset = [this.offset_mouse[0], this.offset_mouse[1]], this.pointer_is_down && p ? this.pointer_is_double = !0 : this.pointer_is_double = !1, this.pointer_is_down = !0, this.canvas.focus(), X.closeAllContextMenus(n), this.search_box && this.search_box.close(), !(this.onMouse && this.onMouse(i) === !0)) {
        if (i.which == 1 && !this.pointer_is_double) {
          if (i.ctrlKey && this.allow_interaction && !this.read_only && (this.dragging_rectangle = new Float32Array(4), this.dragging_rectangle[0] = i.canvasX, this.dragging_rectangle[1] = i.canvasY, this.dragging_rectangle[2] = 1, this.dragging_rectangle[3] = 1, l = !0), u.alt_drag_do_clone_nodes && i.altKey && a && this.allow_interaction && !l && !this.read_only) {
            let P = a.clone();
            P && (P.pos[0] += 5, P.pos[1] += 5, this.graph.add(P, { doCalcSize: !1 }), a = P, l = !0, m || (this.allow_dragnodes && (this.graph.beforeChange(), this.node_dragged = a), this.selected_nodes[a.id] || this.processNodeSelected(a, i)));
          }
          var c = !1;
          if (a && this.allow_interaction && !l && !this.read_only) {
            if (!this.live_mode && !a.flags.pinned && this.bringToFront(a), !this.connecting_node && !a.flags.collapsed && !this.live_mode)
              if (!l && a.resizable !== !1 && u.isInsideRectangle(
                i.canvasX,
                i.canvasY,
                a.pos[0] + a.size[0] - 5,
                a.pos[1] + a.size[1] - 5,
                10,
                10
              ))
                this.graph.beforeChange(), this.resizing_node = a, this.canvas.style.cursor = "se-resize", l = !0;
              else {
                if (a.outputs)
                  for (var v = 0, g = a.outputs.length; v < g; ++v) {
                    var d = a.outputs[v], _ = a.getConnectionPos(!1, v);
                    if (u.isInsideRectangle(
                      i.canvasX,
                      i.canvasY,
                      _[0] - 15,
                      _[1] - 10,
                      30,
                      20
                    )) {
                      this.connecting_node = a, this.connecting_output = d, this.connecting_output.slot_index = v, this.connecting_pos = a.getConnectionPos(!1, v), this.connecting_slot = v, u.shift_click_do_break_link_from && i.shiftKey && a.disconnectOutput(v), f ? a.onOutputDblClick && a.onOutputDblClick(v, i) : a.onOutputClick && a.onOutputClick(v, i), l = !0;
                      break;
                    }
                  }
                if (a.inputs)
                  for (var v = 0, g = a.inputs.length; v < g; ++v) {
                    var y = a.inputs[v], _ = a.getConnectionPos(!0, v);
                    if (u.isInsideRectangle(
                      i.canvasX,
                      i.canvasY,
                      _[0] - 15,
                      _[1] - 10,
                      30,
                      20
                    )) {
                      if (f ? a.onInputDblClick && a.onInputDblClick(v, i) : a.onInputClick && a.onInputClick(v, i), y.link !== null) {
                        var b = this.graph.links[y.link];
                        u.click_do_break_link_to && (a.disconnectInput(v), this.dirty_bgcanvas = !0, l = !0), (this.allow_reconnect_links || //this.move_destination_link_without_shift ||
                        i.shiftKey) && (u.click_do_break_link_to || a.disconnectInput(v), this.connecting_node = this.graph._nodes_by_id[b.origin_id], this.connecting_slot = b.origin_slot, this.connecting_output = this.connecting_node.outputs[this.connecting_slot], this.connecting_pos = this.connecting_node.getConnectionPos(!1, this.connecting_slot), this.dirty_bgcanvas = !0, l = !0);
                      }
                      l || (this.connecting_node = a, this.connecting_input = y, this.connecting_input.slot_index = v, this.connecting_pos = a.getConnectionPos(!0, v), this.connecting_slot = v, this.dirty_bgcanvas = !0, l = !0);
                    }
                  }
              }
            if (!l) {
              var m = !1, E = [i.canvasX - a.pos[0], i.canvasY - a.pos[1]], T = this.processNodeWidgets(a, this.graph_mouse, i);
              T && (m = !0, this.node_widget = [a, T]), f && this.selected_nodes[a.id] && (a.onDblClick && a.onDblClick(i, E, this), this.processNodeDblClicked(a), m = !0), a.onMouseDown && a.onMouseDown(i, E, this) ? m = !0 : (a.subgraph && !a.skip_subgraph_button && !a.flags.collapsed && E[0] > a.size[0] - u.NODE_TITLE_HEIGHT && E[1] < 0 && setTimeout(() => {
                this.openSubgraph(a.subgraph);
              }, 10), this.live_mode && (c = !0, m = !0)), m || (this.allow_dragnodes && (this.graph.beforeChange(), this.node_dragged = a), this.selected_nodes[a.id] || this.processNodeSelected(a, i)), this.dirty_canvas = !0;
            }
          } else if (!l) {
            let P = !1;
            if (a && a.subgraph && !a.skip_subgraph_button) {
              var E = [i.canvasX - a.pos[0], i.canvasY - a.pos[1]];
              !a.flags.collapsed && E[0] > a.size[0] - u.NODE_TITLE_HEIGHT && E[1] < 0 && (P = !0, setTimeout(() => {
                this.openSubgraph(a.subgraph);
              }, 10));
            }
            if (!P) {
              if (this.allow_interaction && !this.read_only) {
                const F = this.findLinkCenterAtPos(i.canvasX, i.canvasY);
                F != null && (this.showLinkMenu(F, i), this.over_link_center = null);
              }
              if (this.selected_group = this.graph.getGroupOnPos(i.canvasX, i.canvasY), this.selected_group_resizing = !1, this.selected_group && !this.read_only && this.allow_interaction) {
                i.ctrlKey && (this.dragging_rectangle = null);
                var O = u.distance([i.canvasX, i.canvasY], [this.selected_group.pos[0] + this.selected_group.size[0], this.selected_group.pos[1] + this.selected_group.size[1]]);
                O * this.ds.scale < 10 ? this.selected_group_resizing = !0 : this.selected_group.recomputeInsideNodes();
              }
              f && !this.read_only && this.allow_searchbox && this.allow_interaction && (this.showSearchBox(i), i.preventDefault(), i.stopPropagation()), c = !0;
            }
          }
          !l && c && this.allow_dragcanvas && (this.dragging_canvas = !0);
        } else if (i.which == 2) {
          if (u.middle_click_slot_add_default_node && a && this.allow_interaction && !l && !this.read_only && !this.connecting_node && !a.flags.collapsed && !this.live_mode) {
            var A = null, M = null, L = null;
            if (a.outputs)
              for (var v = 0, g = a.outputs.length; v < g; ++v) {
                var d = a.outputs[v], _ = a.getConnectionPos(!1, v);
                if (u.isInsideRectangle(i.canvasX, i.canvasY, _[0] - 15, _[1] - 10, 30, 20)) {
                  A = d, M = v, L = !0;
                  break;
                }
              }
            if (a.inputs)
              for (var v = 0, g = a.inputs.length; v < g; ++v) {
                var y = a.inputs[v], _ = a.getConnectionPos(!0, v);
                if (u.isInsideRectangle(i.canvasX, i.canvasY, _[0] - 15, _[1] - 10, 30, 20)) {
                  A = y, M = v, L = !1;
                  break;
                }
              }
            if (A && M !== !1) {
              var B = 0.5 - (M + 1) / (L ? a.outputs.length : a.inputs.length), G = a.getBounding(), z = [
                L ? G[0] + G[2] : G[0],
                i.canvasY - 80
                // + node_bounding[0]/this.canvas.width*66 // vertical "derive"
              ];
              this.createDefaultNodeForSlot("AUTO", {
                nodeFrom: L ? a : null,
                slotFrom: L ? M : null,
                nodeTo: L ? null : a,
                slotTo: L ? null : M,
                position: z,
                posAdd: [L ? 30 : -30, -B * 130],
                posSizeFix: [L ? 0 : -1, 0]
                //-alphaPosY*2*/
              });
            }
          }
        } else if ((i.which == 3 || this.pointer_is_double) && this.allow_interaction && !l && !this.read_only) {
          let P = null;
          if (a)
            P = { type: "node", item: a }, Object.keys(this.selected_nodes).length && (this.selected_nodes[a.id] || i.shiftKey || i.ctrlKey || i.metaKey) ? this.selected_nodes[a.id] || this.selectNodes([a], !0) : this.selectNodes([a]);
          else {
            const F = this.findLinkCenterAtPos(i.canvasX, i.canvasY);
            F != null && (this.over_link_center = null, this.dirty_canvas = !0, P = { type: "link", item: F });
          }
          this.processContextMenu(P, i);
        }
        if (this.selected_group_moving = !1, this.selected_group && !this.selected_group_resizing) {
          var pe = this.selected_group.fontSize || u.DEFAULT_GROUP_FONT_SIZE, D = pe * 1.4;
          u.isInsideRectangle(i.canvasX, i.canvasY, this.selected_group.pos[0], this.selected_group.pos[1], this.selected_group.size[0], D) && (this.selected_group_moving = !0);
        }
        return this.last_mouse[0] = i.clientX, this.last_mouse[1] = i.clientY, this.last_mouseclick = u.getTime(), this.last_mouse_dragging = !0, this.graph.change(), (!n.document.activeElement || n.document.activeElement.nodeName.toLowerCase() != "input" && n.document.activeElement.nodeName.toLowerCase() != "textarea") && i.preventDefault(), i.stopPropagation(), this.onMouseDown && this.onMouseDown(i), !1;
      }
    }
  }
  processMouseMove(t) {
    let i = t;
    if (this.autoresize && this.resize(), this.set_canvas_dirty_on_mouse_event && (this.dirty_canvas = !0), !this.graph)
      return;
    N.active_canvas = this, this.adjustMouseEvent(i);
    let n = [i.clientX, i.clientY];
    this.mouse[0] = n[0], this.mouse[1] = n[1];
    let s = [
      n[0] - this.last_mouse[0],
      n[1] - this.last_mouse[1]
    ];
    if (this.last_mouse = n, this.offset_mouse[0] = i.offsetX, this.offset_mouse[1] = i.offsetY, this.graph_mouse[0] = i.canvasX, this.graph_mouse[1] = i.canvasY, this.block_click)
      return i.preventDefault(), !1;
    i.dragging = this.last_mouse_dragging, this.node_widget && (this.processNodeWidgets(
      this.node_widget[0],
      this.graph_mouse,
      i,
      this.node_widget[1]
    ), this.dirty_canvas = !0);
    const r = this.selected_group;
    if (this.selected_group && !this.selected_group_resizing && !this.selected_group_moving && (this.selected_group = null), this.dragging_rectangle)
      this.dragging_rectangle[2] = i.canvasX - this.dragging_rectangle[0], this.dragging_rectangle[3] = i.canvasY - this.dragging_rectangle[1], this.dirty_canvas = !0;
    else if (this.selected_group && !this.read_only && this.allow_interaction) {
      if (this.selected_group_resizing)
        this.selected_group.size = [
          i.canvasX - this.selected_group.pos[0],
          i.canvasY - this.selected_group.pos[1]
        ];
      else {
        var o = s[0] / this.ds.scale, a = s[1] / this.ds.scale;
        this.selected_group.move(o, a, i.ctrlKey), this.selected_group._nodes.length && (this.dirty_canvas = !0);
      }
      this.dirty_bgcanvas = !0;
    } else if (this.dragging_canvas)
      this.ds.offset[0] += s[0] / this.ds.scale, this.ds.offset[1] += s[1] / this.ds.scale, this.dirty_canvas = !0, this.dirty_bgcanvas = !0;
    else {
      const b = this.allow_interaction && !this.read_only;
      this.connecting_node && (this.dirty_canvas = !0);
      var l = this.graph.getNodeOnPos(i.canvasX, i.canvasY, this.visible_nodes);
      if (b)
        for (var h = 0, p = this.graph._nodes.length; h < p; ++h) {
          let m = this.graph._nodes[h];
          if (m.mouseOver && l != m) {
            m.mouseOver = !1, this.node_over && this.node_over.onMouseLeave && this.node_over.onMouseLeave(i, [i.canvasX - this.node_over.pos[0], i.canvasY - this.node_over.pos[1]], this);
            const E = this.node_over;
            this.node_over = null, this.dirty_canvas = !0, this.onHoverChange && E != this.node_over && this.onHoverChange(this.node_over, E);
          }
        }
      if (l) {
        if (l.redraw_on_mouse && (this.dirty_canvas = !0), b) {
          if (!l.mouseOver) {
            l.mouseOver = !0;
            const m = this.node_over;
            this.node_over = l, this.dirty_canvas = !0, this.onHoverChange && m != this.node_over && this.onHoverChange(this.node_over, m), l.onMouseEnter && l.onMouseEnter(i, [i.canvasX - l.pos[0], i.canvasY - l.pos[1]], this);
          }
          if (l.onMouseMove && l.onMouseMove(i, [i.canvasX - l.pos[0], i.canvasY - l.pos[1]], this), this.connecting_node) {
            if (this.connecting_output) {
              var f = this._highlight_input || [0, 0];
              if (!this.isOverNodeBox(l, i.canvasX, i.canvasY)) {
                var c = this.isOverNodeInput(l, i.canvasX, i.canvasY, f);
                if (c != -1 && l.inputs[c]) {
                  var v = l.inputs[c].type;
                  u.isValidConnection(this.connecting_output.type, v) && (this._highlight_input = f, this._highlight_input_slot = l.inputs[c]);
                } else
                  this._highlight_input = null, this._highlight_input_slot = null;
              }
            } else if (this.connecting_input) {
              var f = this._highlight_output || [0, 0];
              if (!this.isOverNodeBox(l, i.canvasX, i.canvasY)) {
                var c = this.isOverNodeOutput(l, i.canvasX, i.canvasY, f);
                if (c != -1 && l.outputs[c]) {
                  var v = l.outputs[c].type;
                  u.isValidConnection(this.connecting_input.type, v) && (this._highlight_output = f);
                } else
                  this._highlight_output = null;
              }
            }
          }
          this.canvas && (u.isInsideRectangle(
            i.canvasX,
            i.canvasY,
            l.pos[0] + l.size[0] - 5,
            l.pos[1] + l.size[1] - 5,
            5,
            5
          ) ? this.canvas.style.cursor = "se-resize" : this.canvas.style.cursor = "crosshair");
        }
      } else {
        var g = this.findLinkCenterAtPos(i.canvasX, i.canvasY);
        g != this.over_link_center && (this.over_link_center = g, this.dirty_canvas = !0), this.canvas && (this.canvas.style.cursor = "");
      }
      if (b) {
        if (this.node_capturing_input && this.node_capturing_input != l && this.node_capturing_input.onMouseMove && this.node_capturing_input.onMouseMove(i, [i.canvasX - this.node_capturing_input.pos[0], i.canvasY - this.node_capturing_input.pos[1]], this), this.node_dragged && !this.live_mode) {
          for (const m in this.selected_nodes) {
            var d = this.selected_nodes[m];
            d.pos[0] += s[0] / this.ds.scale, d.pos[1] += s[1] / this.ds.scale;
          }
          this.dirty_canvas = !0, this.dirty_bgcanvas = !0;
        }
        if (this.resizing_node && !this.live_mode) {
          var _ = [i.canvasX - this.resizing_node.pos[0], i.canvasY - this.resizing_node.pos[1]], y = this.resizing_node.computeSize();
          _[0] = Math.max(y[0], _[0]), _[1] = Math.max(y[1], _[1]), this.resizing_node.setSize(_), this.canvas.style.cursor = "se-resize", this.dirty_canvas = !0, this.dirty_bgcanvas = !0;
        }
      }
    }
    return r && !this.selected_group_resizing && !this.selected_group_moving && (this.selected_group = r), i.preventDefault(), !1;
  }
  processMouseUp(t) {
    let i = t;
    var n = !(i instanceof PointerEvent) || !i.isPrimary;
    if (!n)
      return !1;
    if (this.set_canvas_dirty_on_mouse_event && (this.dirty_canvas = !0), !!this.graph) {
      var s = this.getCanvasWindow(), r = s.document;
      N.active_canvas = this, this.skip_events || (u.pointerListenerRemove(r, "move", this._mousemove_callback, !0), u.pointerListenerAdd(this.canvas, "move", this._mousemove_callback, !0), u.pointerListenerRemove(r, "up", this._mouseup_callback, !0)), this.adjustMouseEvent(i);
      var o = u.getTime();
      if (i.click_time = o - this.last_mouseclick, this.last_mouse_dragging = !1, this.last_click_position = null, this.block_click && (this.block_click = !1), i.which == 1) {
        if (this.node_widget && this.processNodeWidgets(this.node_widget[0], this.graph_mouse, i), this.node_widget = null, this.selected_group) {
          var a = this.selected_group.pos[0] - Math.round(this.selected_group.pos[0]), l = this.selected_group.pos[1] - Math.round(this.selected_group.pos[1]);
          this.selected_group.move(a, l, i.ctrlKey), this.selected_group.pos[0] = Math.round(
            this.selected_group.pos[0]
          ), this.selected_group.pos[1] = Math.round(
            this.selected_group.pos[1]
          ), this.selected_group._nodes.length && (this.dirty_canvas = !0), this.selected_group = null;
        }
        this.selected_group_resizing = !1;
        var h = this.graph.getNodeOnPos(
          i.canvasX,
          i.canvasY,
          this.visible_nodes
        );
        if (this.dragging_rectangle) {
          if (this.graph) {
            var p = this.graph._nodes, f = new Float32Array(4), c = Math.abs(this.dragging_rectangle[2]), v = Math.abs(this.dragging_rectangle[3]), g = this.dragging_rectangle[2] < 0 ? this.dragging_rectangle[0] - c : this.dragging_rectangle[0], d = this.dragging_rectangle[3] < 0 ? this.dragging_rectangle[1] - v : this.dragging_rectangle[1];
            if (this.dragging_rectangle[0] = g, this.dragging_rectangle[1] = d, this.dragging_rectangle[2] = c, this.dragging_rectangle[3] = v, !h || c > 10 && v > 10) {
              for (var _ = [], y = 0; y < p.length; ++y) {
                var b = p[y];
                b.getBounding(f), u.overlapBounding(
                  this.dragging_rectangle,
                  f
                ) && _.push(b);
              }
              _.length && this.selectNodes(_, i.shiftKey);
            } else
              this.selectNodes([h], i.shiftKey || i.ctrlKey);
          }
          this.dragging_rectangle = null;
        } else if (this.connecting_node) {
          this.dirty_canvas = !0, this.dirty_bgcanvas = !0;
          var m = this.connecting_output || this.connecting_input, E = m.type;
          if (h) {
            if (this.connecting_output) {
              var T = this.isOverNodeInput(
                h,
                i.canvasX,
                i.canvasY
              );
              T != -1 ? this.connecting_node.connect(this.connecting_slot, h, T) : this.connecting_node.connectByTypeInput(this.connecting_slot, h, E);
            } else if (this.connecting_input) {
              var T = this.isOverNodeOutput(
                h,
                i.canvasX,
                i.canvasY
              );
              T != -1 ? h.connect(T, this.connecting_node, this.connecting_slot) : this.connecting_node.connectByTypeOutput(this.connecting_slot, h, E);
            }
          } else
            u.release_link_on_empty_shows_menu && (i.shiftKey && this.allow_searchbox ? this.connecting_output ? this.showSearchBox(i, { node_from: this.connecting_node, slotFrom: this.connecting_output, type_filter_in: this.connecting_output.type }) : this.connecting_input && this.showSearchBox(i, { node_to: this.connecting_node, slotFrom: this.connecting_input, type_filter_out: this.connecting_input.type }) : this.connecting_output ? this.showConnectionMenu({ nodeFrom: this.connecting_node, slotFrom: this.connecting_output, e: i }) : this.connecting_input && this.showConnectionMenu({ nodeTo: this.connecting_node, slotTo: this.connecting_input, e: i }));
          this.connecting_output = null, this.connecting_input = null, this.connecting_pos = null, this.connecting_node = null, this.connecting_slot = -1;
        } else if (this.resizing_node)
          this.dirty_canvas = !0, this.dirty_bgcanvas = !0, this.graph.afterChange(this.resizing_node), this.resizing_node = null;
        else if (this.node_dragged) {
          var h = this.node_dragged;
          h && i.click_time < 300 && h.isShowingTitle(!0) && u.isInsideRectangle(
            i.canvasX,
            i.canvasY,
            h.pos[0],
            h.pos[1] - u.NODE_TITLE_HEIGHT,
            u.NODE_TITLE_HEIGHT,
            u.NODE_TITLE_HEIGHT
          ) && h.collapse(), this.dirty_canvas = !0, this.dirty_bgcanvas = !0, this.node_dragged.pos[0] = Math.round(this.node_dragged.pos[0]), this.node_dragged.pos[1] = Math.round(this.node_dragged.pos[1]), (this.graph.config.align_to_grid || this.align_to_grid) && this.node_dragged.alignToGrid(), this.onNodeMoved && this.onNodeMoved(this.node_dragged), this.graph.afterChange(this.node_dragged), this.node_dragged = null;
        } else {
          var h = this.graph.getNodeOnPos(
            i.canvasX,
            i.canvasY,
            this.visible_nodes
          );
          !h && i.click_time < 300 && this.deselectAllNodes(), this.dirty_canvas = !0, this.dragging_canvas = !1, this.node_over && this.node_over.onMouseUp && this.node_over.onMouseUp(i, [i.canvasX - this.node_over.pos[0], i.canvasY - this.node_over.pos[1]], this), this.node_capturing_input && this.node_capturing_input.onMouseUp && this.node_capturing_input.onMouseUp(i, [
            i.canvasX - this.node_capturing_input.pos[0],
            i.canvasY - this.node_capturing_input.pos[1]
          ], this);
        }
      } else
        i.which == 2 ? (this.dirty_canvas = !0, this.dragging_canvas = !1) : i.which == 3 && (this.dirty_canvas = !0, this.dragging_canvas = !1);
      return n && (this.pointer_is_down = !1, this.pointer_is_double = !1), this.graph.change(), i.stopPropagation(), i.preventDefault(), !1;
    }
  }
  processMouseWheel(t) {
    let i = t;
    if (!(!this.graph || !this.allow_dragcanvas)) {
      var n = i.wheelDeltaY != null ? i.wheelDeltaY : i.detail * -60;
      this.adjustMouseEvent(i);
      var s = i.clientX, r = i.clientY, o = !this.viewport || this.viewport && s >= this.viewport[0] && s < this.viewport[0] + this.viewport[2] && r >= this.viewport[1] && r < this.viewport[1] + this.viewport[3];
      if (o) {
        var a = this.ds.scale;
        return n > 0 ? a *= 1.1 : n < 0 && (a *= 1 / 1.1), this.ds.changeScale(a, [i.clientX, i.clientY]), this.graph.change(), i.preventDefault(), !1;
      }
    }
  }
}
const ie = class {
  /** changes the zoom level of the graph (default is 1), you can pass also a place used to pivot the zoom */
  setZoom(e, t) {
    this.ds.changeScale(e, t), this.maxZoom && this.ds.scale > this.maxZoom ? this.scale = this.maxZoom : this.minZoom && this.ds.scale < this.minZoom && (this.scale = this.minZoom);
  }
  /** brings a node to front (above all other nodes) */
  bringToFront(e) {
    var t = this.graph._nodes.indexOf(e);
    t != -1 && (this.graph._nodes.splice(t, 1), this.graph._nodes.push(e));
  }
  /** sends a node to the back (below all other nodes) */
  sendToBack(e) {
    var t = this.graph._nodes.indexOf(e);
    t != -1 && (this.graph._nodes.splice(t, 1), this.graph._nodes.unshift(e));
  }
  /** checks which nodes are visible (inside the camera area) */
  computeVisibleNodes(e, t = []) {
    var i = t;
    i.length = 0, e = e || this.graph._nodes;
    for (var n = 0, s = e.length; n < s; ++n) {
      var r = e[n];
      this.live_mode && !r.onDrawBackground && !r.onDrawForeground || u.overlapBounding(this.visible_area, r.getBounding(ie.temp)) && i.push(r);
    }
    return i;
  }
  /** renders the whole canvas content, by rendering in two separated canvas, one containing the background grid and the connections, and one containing the nodes) */
  draw(e = !1, t = !1) {
    if (!(!this.canvas || this.canvas.width == 0 || this.canvas.height == 0)) {
      var i = u.getTime();
      this.render_time = (i - this.last_draw_time) * 1e-3, this.last_draw_time = i, this.graph && this.ds.computeVisibleArea(this.viewport), (this.dirty_bgcanvas || t || this.always_render_background || this.graph && this.graph._last_trigger_time && i - this.graph._last_trigger_time < 1e3) && this.drawBackCanvas(), (this.dirty_canvas || e) && this.drawFrontCanvas(), this.fps = this.render_time ? 1 / this.render_time : 0, this.frame += 1;
    }
  }
  /** draws the front canvas (the one containing all the nodes) */
  drawFrontCanvas() {
    this.dirty_canvas = !1, this.ctx || (this.ctx = this.canvas.getContext("2d"));
    var e = this.ctx;
    if (e) {
      var t = this.canvas, i = this.viewport || this.dirty_area;
      if (i && (e.save(), e.beginPath(), e.rect(i[0], i[1], i[2], i[3]), e.clip()), this.clear_background && (i ? e.clearRect(i[0], i[1], i[2], i[3]) : e.clearRect(0, 0, t.width, t.height)), this.bgcanvas == this.canvas ? this.drawBackCanvas() : e.drawImage(this.bgcanvas, 0, 0), this.onRender && this.onRender(t, e), this.show_info && this.renderInfo(e, i ? i[0] : 0, i ? i[1] : 0), this.graph) {
        e.save(), this.ds.toCanvasContext(e);
        for (var n = this.computeVisibleNodes(
          null,
          this.visible_nodes
        ), s = 0; s < n.length; ++s) {
          var r = n[s];
          e.save(), e.translate(r.pos[0], r.pos[1]), this.drawNode(r, e), e.restore();
        }
        if (this.render_execution_order && this.drawExecutionOrder(e), this.graph.config.links_ontop && (this.live_mode || this.drawConnections(e)), this.connecting_pos != null) {
          e.lineWidth = this.connections_width;
          var o = null, a = this.connecting_output || this.connecting_input, l = a.type, h = a.dir;
          h == null && (this.connecting_output ? h = this.connecting_node.horizontal ? w.DOWN : w.RIGHT : h = this.connecting_node.horizontal ? w.UP : w.LEFT);
          var p = a.shape;
          switch (l) {
            case I.EVENT:
              o = u.EVENT_LINK_COLOR;
              break;
            default:
              o = u.CONNECTING_LINK_COLOR;
          }
          if (this.renderLink(
            e,
            this.connecting_pos,
            [this.graph_mouse[0], this.graph_mouse[1]],
            null,
            !1,
            null,
            o,
            h,
            w.CENTER
          ), e.beginPath(), p === k.BOX_SHAPE ? (e.rect(
            this.connecting_pos[0] - 6 + 0.5,
            this.connecting_pos[1] - 5 + 0.5,
            14,
            10
          ), e.fill(), e.beginPath(), e.rect(
            this.graph_mouse[0] - 6 + 0.5,
            this.graph_mouse[1] - 5 + 0.5,
            14,
            10
          )) : p === k.ARROW_SHAPE ? (e.moveTo(this.connecting_pos[0] + 8, this.connecting_pos[1] + 0.5), e.lineTo(this.connecting_pos[0] - 4, this.connecting_pos[1] + 6 + 0.5), e.lineTo(this.connecting_pos[0] - 4, this.connecting_pos[1] - 6 + 0.5), e.closePath()) : (e.arc(
            this.connecting_pos[0],
            this.connecting_pos[1],
            4,
            0,
            Math.PI * 2
          ), e.fill(), e.beginPath(), e.arc(
            this.graph_mouse[0],
            this.graph_mouse[1],
            4,
            0,
            Math.PI * 2
          )), e.fill(), e.fillStyle = "#ffcc00", this._highlight_input) {
            e.beginPath();
            var f = this._highlight_input_slot.shape;
            f === k.ARROW_SHAPE ? (e.moveTo(this._highlight_input[0] + 8, this._highlight_input[1] + 0.5), e.lineTo(this._highlight_input[0] - 4, this._highlight_input[1] + 6 + 0.5), e.lineTo(this._highlight_input[0] - 4, this._highlight_input[1] - 6 + 0.5), e.closePath()) : e.arc(
              this._highlight_input[0],
              this._highlight_input[1],
              6,
              0,
              Math.PI * 2
            ), e.fill();
          }
          this._highlight_output && (e.beginPath(), f === k.ARROW_SHAPE ? (e.moveTo(this._highlight_output[0] + 8, this._highlight_output[1] + 0.5), e.lineTo(this._highlight_output[0] - 4, this._highlight_output[1] + 6 + 0.5), e.lineTo(this._highlight_output[0] - 4, this._highlight_output[1] - 6 + 0.5), e.closePath()) : e.arc(
            this._highlight_output[0],
            this._highlight_output[1],
            6,
            0,
            Math.PI * 2
          ), e.fill());
        }
        this.dragging_rectangle && (e.strokeStyle = "#FFF", e.strokeRect(
          this.dragging_rectangle[0],
          this.dragging_rectangle[1],
          this.dragging_rectangle[2],
          this.dragging_rectangle[3]
        )), this.over_link_center && this.render_link_tooltip ? this.drawLinkTooltip(e, this.over_link_center) : this.onDrawLinkTooltip && this.onDrawLinkTooltip(e, null, this), this.onDrawForeground && this.onDrawForeground(e, this.visible_area), e.restore();
      }
      this._graph_stack && this._graph_stack.length && this.render_subgraph_panels && this.drawSubgraphPanel(e), this.onDrawOverlay && this.onDrawOverlay(e), i && e.restore();
    }
  }
  /**
   * draws the panel in the corner that shows subgraph properties
   * @method drawSubgraphPanel
   **/
  drawSubgraphPanel(e) {
    var t = this.graph, i = t._subgraph_node;
    if (!i) {
      console.warn("subgraph without subnode");
      return;
    }
    this.drawSubgraphPanelLeft(t, i, e), this.drawSubgraphPanelRight(t, i, e);
  }
  drawSubgraphPanelLeft(e, t, i) {
    var n = t.inputs ? t.inputs.length : 0, s = 200, r = Math.floor(u.NODE_SLOT_HEIGHT * 1.6);
    if (i.fillStyle = "#111", i.globalAlpha = 0.8, i.beginPath(), i.roundRect(10, 10, s, (n + 1) * r + 50, [8]), i.fill(), i.globalAlpha = 1, i.fillStyle = "#888", i.font = "14px Arial", i.textAlign = "left", i.fillText("Graph Inputs", 20, 34), this.drawButton(s - 20, 20, 20, 20, "X", "#151515", void 0, void 0, !0)) {
      this.closeSubgraph();
      return;
    }
    var o = 50;
    if (i.font = "14px Arial", t.inputs)
      for (var a = 0; a < t.inputs.length; ++a) {
        var l = t.inputs[a];
        l.not_subgraph_input || (i.fillStyle = "#9C9", i.beginPath(), i.arc(s - 16, o, 5, 0, 2 * Math.PI), i.fill(), i.fillStyle = "#AAA", i.fillText(l.name, 30, o + r * 0.75), i.fillStyle = "#777", i.fillText(J(l.type), 130, o + r * 0.75), o += r);
      }
    this.drawButton(20, o + 2, s - 20, r - 2, "+", "#151515", "#222") && this.showSubgraphPropertiesDialog(t);
  }
  drawSubgraphPanelRight(e, t, i) {
    var n = t.outputs ? t.outputs.length : 0, s = this.bgcanvas.width, r = 200, o = Math.floor(u.NODE_SLOT_HEIGHT * 1.6);
    i.fillStyle = "#111", i.globalAlpha = 0.8, i.beginPath(), i.roundRect(s - r - 10, 10, r, (n + 1) * o + 50, [8]), i.fill(), i.globalAlpha = 1, i.fillStyle = "#888", i.font = "14px Arial", i.textAlign = "left";
    var a = "Graph Outputs", l = i.measureText(a).width;
    if (i.fillText(a, s - l - 20, 34), this.drawButton(s - r, 20, 20, 20, "X", "#151515", void 0, void 0, !0)) {
      this.closeSubgraph();
      return;
    }
    var h = 50;
    if (i.font = "14px Arial", t.outputs)
      for (var p = 0; p < t.outputs.length; ++p) {
        var f = t.outputs[p];
        f.not_subgraph_output || (i.fillStyle = "#9C9", i.beginPath(), i.arc(s - r + 16, h, 5, 0, 2 * Math.PI), i.fill(), i.fillStyle = "#AAA", i.fillText(f.name, s - r + 30, h + o * 0.75), i.fillStyle = "#777", i.fillText(J(f.type), s - r + 130, h + o * 0.75), h += o);
      }
    this.drawButton(s - r, h + 2, r - 20, o - 2, "+", "#151515", "#222") && this.showSubgraphPropertiesDialogRight(t);
  }
  //Draws a button into the canvas overlay and computes if it was clicked using the immediate gui paradigm
  drawButton(e, t, i, n, s, r = u.NODE_DEFAULT_COLOR, o = "#555", a = u.NODE_TEXT_COLOR, l = !1) {
    const h = !this.block_click && (l || this.allow_interaction && !this.read_only);
    var p = this.ctx, f = this.offset_mouse, c = h && u.isInsideRectangle(f[0], f[1], e, t, i, n);
    f = this.last_click_position_offset;
    var v = h && f && this.pointer_is_down && u.isInsideRectangle(f[0], f[1], e, t, i, n);
    p.fillStyle = c ? o : r, v && (p.fillStyle = "#AAA"), p.beginPath(), p.roundRect(e, t, i, n, [4]), p.fill(), s != null && s.constructor == String && (p.fillStyle = a, p.textAlign = "center", p.font = (n * 0.65 | 0) + "px Arial", p.fillText(s, e + i * 0.5, t + n * 0.75), p.textAlign = "left");
    var g = v && h;
    return v && this.blockClick(), g;
  }
  /** draws every group area in the background */
  drawGroups(e, t) {
    if (this.graph) {
      var i = this.graph._groups;
      t.save(), t.globalAlpha = 0.5 * this.editor_alpha;
      for (var n = 0; n < i.length; ++n) {
        var s = i[n];
        if (u.overlapBounding(this.visible_area, s._bounding)) {
          t.fillStyle = s.color || "#335", t.strokeStyle = s.color || "#335";
          var r = s._pos, o = s._size;
          t.globalAlpha = 0.25 * this.editor_alpha, t.beginPath(), t.rect(r[0] + 0.5, r[1] + 0.5, o[0], o[1]), t.fill(), t.globalAlpha = this.editor_alpha, t.stroke(), t.beginPath(), t.moveTo(r[0] + o[0], r[1] + o[1]), t.lineTo(r[0] + o[0] - 10, r[1] + o[1]), t.lineTo(r[0] + o[0], r[1] + o[1] - 10), t.fill();
          var a = s.font_size || u.DEFAULT_GROUP_FONT_SIZE;
          t.font = a + "px Arial", t.textAlign = "left", t.fillText(s.title, r[0] + 4, r[1] + a);
        }
      }
      t.restore();
    }
  }
  /** draws some useful stats in the corner of the canvas */
  renderInfo(e, t = 10, i) {
    i = i || this.canvas.height - 80, e.save(), e.translate(t, i), e.font = "10px Arial", e.fillStyle = "#888", e.textAlign = "left", this.graph ? (e.fillText("T: " + this.graph.globaltime.toFixed(2) + "s", 5, 13 * 1), e.fillText("I: " + this.graph.iteration, 5, 13 * 2), e.fillText("N: " + this.graph._nodes.length + " [" + this.visible_nodes.length + "]", 5, 13 * 3), e.fillText("V: " + this.graph._version, 5, 13 * 4), e.fillText("FPS:" + this.fps.toFixed(2), 5, 13 * 5)) : e.fillText("No graph selected", 5, 13 * 1), e.restore();
  }
  /** draws the back canvas (the one containing the background and the connections) */
  drawBackCanvas() {
    var e = this.bgcanvas;
    (e.width != this.canvas.width || e.height != this.canvas.height) && (e.width = this.canvas.width, e.height = this.canvas.height), this.bgctx || (this.bgctx = this.bgcanvas.getContext("2d"));
    var t = this.bgctx;
    let i = this.viewport || [0, 0, t.canvas.width, t.canvas.height];
    if (this.clear_background && t.clearRect(i[0], i[1], i[2], i[3]), this._graph_stack && this._graph_stack.length && this.render_subgraph_stack_header) {
      t.save();
      const o = this._graph_stack[this._graph_stack.length - 1].graph, a = this.graph._subgraph_node;
      t.strokeStyle = a.bgcolor, t.lineWidth = 10, t.strokeRect(1, 1, e.width - 2, e.height - 2), t.lineWidth = 1, t.font = "40px Arial", t.textAlign = "center", t.fillStyle = a.bgcolor || "#AAA";
      let l = "";
      for (let h = 1; h < this._graph_stack.length; ++h)
        l += o._subgraph_node.getTitle() + " >> ";
      t.fillText(
        l + a.getTitle(),
        e.width * 0.5,
        40
      ), t.restore();
    }
    let n = !1;
    if (this.onRenderBackground && this.onRenderBackground(e, t) && (n = !0), this.viewport || (t.restore(), t.setTransform(1, 0, 0, 1, 0, 0)), this.visible_links.length = 0, this.graph) {
      if (t.save(), this.ds.toCanvasContext(t), this.ds.scale < 1.5 && !n && this.clear_background_color && (t.fillStyle = this.clear_background_color, t.fillRect(
        this.visible_area[0],
        this.visible_area[1],
        this.visible_area[2],
        this.visible_area[3]
      )), this.background_image && this.ds.scale > 0.5 && !n) {
        this.zoom_modify_alpha ? t.globalAlpha = (1 - 0.5 / this.ds.scale) * this.editor_alpha : t.globalAlpha = this.editor_alpha, t.imageSmoothingEnabled = t.imageSmoothingEnabled = !1, (!this._bg_img || this._bg_img.name != this.background_image) && (this._bg_img = new Image(), this._bg_img.name = this.background_image, this._bg_img.src = this.background_image, this._bg_img.onload = () => {
          this.draw(!0, !0);
        });
        var s = null;
        this._pattern == null && this._bg_img.width > 0 ? (s = t.createPattern(this._bg_img, "repeat"), this._pattern_img = this._bg_img, this._pattern = s) : s = this._pattern, s && (t.fillStyle = s, t.fillRect(
          this.visible_area[0],
          this.visible_area[1],
          this.visible_area[2],
          this.visible_area[3]
        ), t.fillStyle = "transparent"), t.globalAlpha = 1, t.imageSmoothingEnabled = t.imageSmoothingEnabled = !0;
      }
      this.graph._groups.length && !this.live_mode && this.drawGroups(e, t), this.onDrawBackground && this.onDrawBackground(t, this.visible_area), u.debug && (t.fillStyle = "red", t.fillRect(this.visible_area[0] + 10, this.visible_area[1] + 10, this.visible_area[2] - 20, this.visible_area[3] - 20)), this.render_canvas_border && (t.strokeStyle = "#235", t.strokeRect(0, 0, e.width, e.height)), this.render_connections_shadows ? (t.shadowColor = "#000", t.shadowOffsetX = 0, t.shadowOffsetY = 0, t.shadowBlur = 6) : t.shadowColor = "rgba(0,0,0,0)", !this.live_mode && this.render_connections && this.drawConnections(t), t.shadowColor = "rgba(0,0,0,0)", t.restore();
    }
    this.dirty_bgcanvas = !1, this.dirty_canvas = !0;
  }
  /** draws the given node inside the canvas */
  drawNode(e, t) {
    this.current_node = e;
    var i = e.color || e.constructor.color || u.NODE_DEFAULT_COLOR, n = e.bgcolor || e.constructor.bgcolor || u.NODE_DEFAULT_BGCOLOR;
    e.mouseOver;
    var s = this.ds.scale < 0.6;
    if (this.live_mode) {
      e.flags.collapsed || (t.shadowColor = "transparent", e.onDrawForeground && e.onDrawForeground(t, this, this.canvas));
      return;
    }
    var r = this.editor_alpha;
    if (t.globalAlpha = r, this.render_shadows && !s ? (t.shadowColor = u.DEFAULT_SHADOW_COLOR, t.shadowOffsetX = 2 * this.ds.scale, t.shadowOffsetY = 2 * this.ds.scale, t.shadowBlur = 3 * this.ds.scale) : t.shadowColor = "transparent", !(e.flags.collapsed && e.onDrawCollapsed && e.onDrawCollapsed(t, this) == !0)) {
      var o = e.shape || k.BOX_SHAPE, a = ie.temp_vec2;
      ie.temp_vec2.set(e.size);
      var l = e.horizontal;
      if (e.flags.collapsed) {
        t.font = this.inner_text_font;
        var h = e.getTitle ? e.getTitle() : e.title;
        h != null && (e._collapsed_width = Math.min(
          e.size[0],
          t.measureText(h).width + u.NODE_TITLE_HEIGHT * 2
        ), a[0] = e._collapsed_width, a[1] = 0);
      }
      e.clip_area && (t.save(), t.beginPath(), o == k.BOX_SHAPE ? t.rect(0, 0, a[0], a[1]) : o == k.ROUND_SHAPE ? t.roundRect(0, 0, a[0], a[1], [10]) : o == k.CIRCLE_SHAPE && t.arc(
        a[0] * 0.5,
        a[1] * 0.5,
        a[0] * 0.5,
        0,
        Math.PI * 2
      ), t.clip()), e.has_errors && (n = "red"), this.drawNodeShape(
        e,
        t,
        [a[0], a[1]],
        i,
        n,
        e.is_selected,
        e.mouseOver
      ), t.shadowColor = "transparent", e.onDrawForeground && e.onDrawForeground(t, this, this.canvas), t.textAlign = l ? "center" : "left", t.font = this.inner_text_font;
      var p = !s, f = this.connecting_output, c = this.connecting_input;
      t.lineWidth = 1;
      var v = 0, g = [0, 0];
      if (e.flags.collapsed) {
        if (this.render_collapsed_slots) {
          var A = null, M = null;
          if (e.inputs)
            for (let G = 0; G < e.inputs.length; G++) {
              let z = e.inputs[G];
              if (z.link != null) {
                A = z;
                break;
              }
            }
          if (e.outputs)
            for (let G = 0; G < e.outputs.length; G++) {
              let z = e.outputs[G];
              !z.links || !z.links.length || (M = z);
            }
          if (A) {
            var L = 0, B = u.NODE_TITLE_HEIGHT * -0.5;
            l && (L = e._collapsed_width * 0.5, B = -u.NODE_TITLE_HEIGHT), t.fillStyle = "#686", t.beginPath(), A.shape === k.BOX_SHAPE ? t.rect(L - 7 + 0.5, B - 4, 14, 8) : A.shape === k.ARROW_SHAPE ? (t.moveTo(L + 8, B), t.lineTo(L + -4, B - 4), t.lineTo(L + -4, B + 4), t.closePath()) : t.arc(L, B, 4, 0, Math.PI * 2), t.fill();
          }
          if (M) {
            var L = e._collapsed_width, B = u.NODE_TITLE_HEIGHT * -0.5;
            l && (L = e._collapsed_width * 0.5, B = 0), t.fillStyle = "#686", t.strokeStyle = "black", t.beginPath(), M.shape === k.BOX_SHAPE ? t.rect(L - 7 + 0.5, B - 4, 14, 8) : M.shape === k.ARROW_SHAPE ? (t.moveTo(L + 6, B), t.lineTo(L - 6, B - 4), t.lineTo(L - 6, B + 4), t.closePath()) : t.arc(L, B, 4, 0, Math.PI * 2), t.fill();
          }
        }
      } else {
        if (e.inputs)
          for (var d = 0; d < e.inputs.length; d++) {
            var _ = e.inputs[d], y = _.type, b = _.shape;
            t.globalAlpha = r, this.connecting_output && !u.isValidConnection(_.type, f.type) ? t.globalAlpha = 0.4 * r : t.globalAlpha = r, t.fillStyle = _.link != null ? _.color_on || N.DEFAULT_CONNECTION_COLORS_BY_TYPE[y] || N.DEFAULT_CONNECTION_COLORS.input_on : _.color_off || N.DEFAULT_CONNECTION_COLORS_BY_TYPE_OFF[y] || N.DEFAULT_CONNECTION_COLORS_BY_TYPE[y] || N.DEFAULT_CONNECTION_COLORS.input_off;
            var m = e.getConnectionPos(!0, d, [g[0], g[1]]);
            m[0] -= e.pos[0], m[1] -= e.pos[1], v < m[1] + u.NODE_SLOT_HEIGHT * 0.5 && (v = m[1] + u.NODE_SLOT_HEIGHT * 0.5), t.beginPath();
            var E = !0;
            if (_.shape === k.BOX_SHAPE ? l ? t.rect(
              m[0] - 5 + 0.5,
              m[1] - 8 + 0.5,
              10,
              14
            ) : t.rect(
              m[0] - 6 + 0.5,
              m[1] - 5 + 0.5,
              14,
              10
            ) : b === k.ARROW_SHAPE ? (t.moveTo(m[0] + 8, m[1] + 0.5), t.lineTo(m[0] - 4, m[1] + 6 + 0.5), t.lineTo(m[0] - 4, m[1] - 6 + 0.5), t.closePath()) : b === k.GRID_SHAPE ? (t.rect(m[0] - 4, m[1] - 4, 2, 2), t.rect(m[0] - 1, m[1] - 4, 2, 2), t.rect(m[0] + 2, m[1] - 4, 2, 2), t.rect(m[0] - 4, m[1] - 1, 2, 2), t.rect(m[0] - 1, m[1] - 1, 2, 2), t.rect(m[0] + 2, m[1] - 1, 2, 2), t.rect(m[0] - 4, m[1] + 2, 2, 2), t.rect(m[0] - 1, m[1] + 2, 2, 2), t.rect(m[0] + 2, m[1] + 2, 2, 2), E = !1) : s ? t.rect(m[0] - 4, m[1] - 4, 8, 8) : t.arc(m[0], m[1], 4, 0, Math.PI * 2), t.fill(), p) {
              var T = _.label != null ? _.label : _.name;
              T && (t.fillStyle = u.NODE_TEXT_COLOR, l || _.dir == w.UP ? t.fillText(T, m[0], m[1] - 10) : t.fillText(T, m[0] + 10, m[1] + 5));
            }
          }
        if (t.textAlign = l ? "center" : "right", t.strokeStyle = "black", e.outputs)
          for (let G = 0; G < e.outputs.length; G++) {
            let z = e.outputs[G];
            var y = z.type, b = z.shape;
            this.connecting_input && !u.isValidConnection(c.type, y) ? t.globalAlpha = 0.4 * r : t.globalAlpha = r;
            var m = e.getConnectionPos(!1, G, g);
            m[0] -= e.pos[0], m[1] -= e.pos[1], v < m[1] + u.NODE_SLOT_HEIGHT * 0.5 && (v = m[1] + u.NODE_SLOT_HEIGHT * 0.5), t.fillStyle = z.links && z.links.length ? z.color_on || N.DEFAULT_CONNECTION_COLORS_BY_TYPE[y] || N.DEFAULT_CONNECTION_COLORS.output_on : z.color_off || N.DEFAULT_CONNECTION_COLORS_BY_TYPE_OFF[y] || N.DEFAULT_CONNECTION_COLORS_BY_TYPE[y] || N.DEFAULT_CONNECTION_COLORS.output_off, t.beginPath();
            var E = !0;
            if (b === k.BOX_SHAPE ? l ? t.rect(
              m[0] - 5 + 0.5,
              m[1] - 8 + 0.5,
              10,
              14
            ) : t.rect(
              m[0] - 6 + 0.5,
              m[1] - 5 + 0.5,
              14,
              10
            ) : b === k.ARROW_SHAPE ? (t.moveTo(m[0] + 8, m[1] + 0.5), t.lineTo(m[0] - 4, m[1] + 6 + 0.5), t.lineTo(m[0] - 4, m[1] - 6 + 0.5), t.closePath()) : b === k.GRID_SHAPE ? (t.rect(m[0] - 4, m[1] - 4, 2, 2), t.rect(m[0] - 1, m[1] - 4, 2, 2), t.rect(m[0] + 2, m[1] - 4, 2, 2), t.rect(m[0] - 4, m[1] - 1, 2, 2), t.rect(m[0] - 1, m[1] - 1, 2, 2), t.rect(m[0] + 2, m[1] - 1, 2, 2), t.rect(m[0] - 4, m[1] + 2, 2, 2), t.rect(m[0] - 1, m[1] + 2, 2, 2), t.rect(m[0] + 2, m[1] + 2, 2, 2), E = !1) : s ? t.rect(m[0] - 4, m[1] - 4, 8, 8) : t.arc(m[0], m[1], 4, 0, Math.PI * 2), t.fill(), !s && E && t.stroke(), p) {
              var T = z.label != null ? z.label : z.name;
              T && (t.fillStyle = u.NODE_TEXT_COLOR, l || z.dir == w.DOWN ? t.fillText(T, m[0], m[1] - 8) : t.fillText(T, m[0] - 10, m[1] + 5));
            }
          }
        if (t.textAlign = "left", t.globalAlpha = 1, e.widgets) {
          var O = v;
          (l || e.widgets_up) && (O = 2), e.widgets_start_y != null && (O = e.widgets_start_y), this.drawNodeWidgets(
            e,
            O,
            t,
            this.node_widget && this.node_widget[0] == e ? this.node_widget[1] : null
          );
        }
      }
      e.clip_area && t.restore(), t.globalAlpha = 1;
    }
  }
  /** used by this.over_link_center */
  drawLinkTooltip(e, t) {
    var i = t._pos;
    if (this.allow_interaction && !this.read_only && (e.fillStyle = "black", e.beginPath(), e.arc(i[0], i[1], 3, 0, Math.PI * 2), e.fill()), t.data != null && !(this.onDrawLinkTooltip && this.onDrawLinkTooltip(e, t, this) == !0)) {
      var n = t.data, s = null;
      if (n.constructor === Number ? s = n.toFixed(2) : n.constructor === String ? s = '"' + n + '"' : n.constructor === Boolean ? s = String(n) : n.toToolTip ? s = n.toToolTip() : s = "[" + n.constructor.name + "]", s != null) {
        s = s.substr(0, 30), e.font = "14px Courier New";
        var r = e.measureText(s), o = r.width + 20, a = 24;
        e.shadowColor = "black", e.shadowOffsetX = 2, e.shadowOffsetY = 2, e.shadowBlur = 3, e.fillStyle = "#454", e.beginPath(), e.roundRect(i[0] - o * 0.5, i[1] - 15 - a, o, a, [3]), e.moveTo(i[0] - 10, i[1] - 15), e.lineTo(i[0] + 10, i[1] - 15), e.lineTo(i[0], i[1] - 5), e.fill(), e.shadowColor = "transparent", e.textAlign = "center", e.fillStyle = "#CEC", e.fillText(s, i[0], i[1] - 15 - a * 0.3);
      }
    }
  }
  /** draws the shape of the given node in the canvas */
  drawNodeShape(e, t, i, n, s, r, o) {
    t.strokeStyle = n, t.fillStyle = s;
    var a = u.NODE_TITLE_HEIGHT, l = this.ds.scale < 0.5, h = e.shape || e.constructor.shape || k.ROUND_SHAPE, p = e.titleMode, f = e.isShowingTitle(o), c = ie.tmp_area;
    c[0] = 0, c[1] = f ? -a : 0, c[2] = i[0] + 1, c[3] = f ? i[1] + a : i[1];
    var v = t.globalAlpha;
    if (t.beginPath(), h == k.BOX_SHAPE || l ? t.fillRect(c[0], c[1], c[2], c[3]) : h == k.ROUND_SHAPE || h == k.CARD_SHAPE ? t.roundRect(
      c[0],
      c[1],
      c[2],
      c[3],
      h == k.CARD_SHAPE ? [this.round_radius, this.round_radius, 0, 0] : [this.round_radius]
    ) : h == k.CIRCLE_SHAPE && t.arc(
      i[0] * 0.5,
      i[1] * 0.5,
      i[0] * 0.5,
      0,
      Math.PI * 2
    ), t.fill(), !e.flags.collapsed && f && (t.shadowColor = "transparent", t.fillStyle = "rgba(0,0,0,0.2)", t.fillRect(0, -1, c[2], 2)), t.shadowColor = "transparent", e.onDrawBackground && e.onDrawBackground(t, this, this.canvas, this.graph_mouse), f || p == se.TRANSPARENT_TITLE) {
      if (e.onDrawTitleBar)
        e.onDrawTitleBar(t, this, a, i, this.ds.scale, n);
      else if (p != se.TRANSPARENT_TITLE && (e.constructor.title_color || this.render_title_colored)) {
        var g = e.constructor.title_color || n;
        if (e.flags.collapsed && (t.shadowColor = u.DEFAULT_SHADOW_COLOR), this.use_gradients) {
          var d = N.gradients[g];
          d || (d = N.gradients[g] = t.createLinearGradient(0, 0, 400, 0), d.addColorStop(0, g), d.addColorStop(1, "#000")), t.fillStyle = d;
        } else
          t.fillStyle = g;
        t.beginPath(), h == k.BOX_SHAPE || l ? t.rect(0, -a, i[0] + 1, a) : (h == k.ROUND_SHAPE || h == k.CARD_SHAPE) && t.roundRect(
          0,
          -a,
          i[0] + 1,
          a,
          e.flags.collapsed ? [this.round_radius] : [this.round_radius, this.round_radius, 0, 0]
        ), t.fill(), t.shadowColor = "transparent";
      }
      var _ = null;
      u.node_box_coloured_by_mode && Oe[e.mode] && (_ = Oe[e.mode]), u.node_box_coloured_when_on && (_ = e.action_triggered ? "#FFF" : e.execute_triggered ? "#AAA" : _);
      var y = 10;
      if (e.onDrawTitleBox ? e.onDrawTitleBox(t, this, a, i, this.ds.scale) : h == k.ROUND_SHAPE || h == k.CIRCLE_SHAPE || h == k.CARD_SHAPE ? (l && (t.fillStyle = "black", t.beginPath(), t.arc(
        a * 0.5,
        a * -0.5,
        y * 0.5 + 1,
        0,
        Math.PI * 2
      ), t.fill()), t.fillStyle = e.boxcolor || _ || u.NODE_DEFAULT_BOXCOLOR, l ? t.fillRect(a * 0.5 - y * 0.5, a * -0.5 - y * 0.5, y, y) : (t.beginPath(), t.arc(
        a * 0.5,
        a * -0.5,
        y * 0.5,
        0,
        Math.PI * 2
      ), t.fill())) : (l && (t.fillStyle = "black", t.fillRect(
        (a - y) * 0.5 - 1,
        (a + y) * -0.5 - 1,
        y + 2,
        y + 2
      )), t.fillStyle = e.boxcolor || _ || u.NODE_DEFAULT_BOXCOLOR, t.fillRect(
        (a - y) * 0.5,
        (a + y) * -0.5,
        y,
        y
      )), t.globalAlpha = v, e.onDrawTitleText && e.onDrawTitleText(
        t,
        this,
        a,
        i,
        this.ds.scale,
        this.title_text_font,
        r
      ), !l) {
        t.font = this.title_text_font;
        var b = String(e.getTitle());
        b && (r ? t.fillStyle = u.NODE_SELECTED_TITLE_COLOR : t.fillStyle = e.constructor.title_text_color || this.node_title_color, e.flags.collapsed ? (t.textAlign = "left", t.fillText(
          b.substr(0, 20),
          //avoid urls too long
          a,
          // + measure.width * 0.5,
          u.NODE_TITLE_TEXT_Y - a
        ), t.textAlign = "left") : (t.textAlign = "left", t.fillText(
          b,
          a,
          u.NODE_TITLE_TEXT_Y - a
        )));
      }
      if (!e.flags.collapsed && e.subgraph && !e.skip_subgraph_button) {
        var m = u.NODE_TITLE_HEIGHT, E = e.size[0] - m, T = u.isInsideRectangle(this.graph_mouse[0] - e.pos[0], this.graph_mouse[1] - e.pos[1], E + 2, -m + 2, m - 4, m - 4);
        t.fillStyle = T ? "#888" : "#555", h == k.BOX_SHAPE || l ? t.fillRect(E + 2, -m + 2, m - 4, m - 4) : (t.beginPath(), t.roundRect(E + 2, -m + 2, m - 4, m - 4, [4]), t.fill()), t.fillStyle = "#333", t.beginPath(), t.moveTo(E + m * 0.2, -m * 0.6), t.lineTo(E + m * 0.8, -m * 0.6), t.lineTo(E + m * 0.5, -m * 0.3), t.fill();
      }
      e.onDrawTitle && e.onDrawTitle(t, this);
    }
    r && (e.onBounding && e.onBounding(c), p == se.TRANSPARENT_TITLE && (c[1] -= a, c[3] += a), t.lineWidth = 1, t.globalAlpha = 0.8, t.beginPath(), h == k.BOX_SHAPE ? t.rect(
      -6 + c[0],
      -6 + c[1],
      12 + c[2],
      12 + c[3]
    ) : h == k.ROUND_SHAPE || h == k.CARD_SHAPE && e.flags.collapsed ? t.roundRect(
      -6 + c[0],
      -6 + c[1],
      12 + c[2],
      12 + c[3],
      [this.round_radius * 2]
    ) : h == k.CARD_SHAPE ? t.roundRect(
      -6 + c[0],
      -6 + c[1],
      12 + c[2],
      12 + c[3],
      [this.round_radius * 2, 2, this.round_radius * 2, 2]
    ) : h == k.CIRCLE_SHAPE && t.arc(
      i[0] * 0.5,
      i[1] * 0.5,
      i[0] * 0.5 + 6,
      0,
      Math.PI * 2
    ), t.strokeStyle = u.NODE_BOX_OUTLINE_COLOR, t.stroke(), t.strokeStyle = n, t.globalAlpha = 1), e.execute_triggered > 0 && e.execute_triggered--, e.action_triggered > 0 && e.action_triggered--;
  }
  /** draws every connection visible in the canvas */
  drawConnections(e) {
    var t = u.getTime(), i = this.visible_area;
    let n = ie.margin_area;
    n[0] = i[0] - 20, n[1] = i[1] - 20, n[2] = i[2] + 40, n[3] = i[3] + 40, e.lineWidth = this.connections_width, e.fillStyle = "#AAA", e.strokeStyle = "#AAA", e.globalAlpha = this.editor_alpha;
    for (var s = this.graph._nodes, r = 0, o = s.length; r < o; ++r) {
      var a = s[r];
      if (!(!a.inputs || !a.inputs.length))
        for (var l = 0; l < a.inputs.length; ++l) {
          var h = a.inputs[l];
          if (!h || h.link == null)
            continue;
          var p = h.link, f = this.graph.links[p];
          if (!f)
            continue;
          var c = this.graph.getNodeById(f.origin_id);
          if (c == null)
            continue;
          var v = f.origin_slot, g = null;
          v == -1 ? g = [
            c.pos[0] + 10,
            c.pos[1] + 10
          ] : g = c.getConnectionPos(
            !1,
            v,
            ie.tempA
          );
          var d = a.getConnectionPos(!0, l, ie.tempB);
          let O = ie.link_bounding;
          if (O[0] = g[0], O[1] = g[1], O[2] = d[0] - g[0], O[3] = d[1] - g[1], O[2] < 0 && (O[0] += O[2], O[2] = Math.abs(O[2])), O[3] < 0 && (O[1] += O[3], O[3] = Math.abs(O[3])), !!u.overlapBounding(O, n)) {
            var _ = c.outputs[v], y = a.inputs[l];
            if (!(!_ || !y)) {
              var b = _.dir || (c.horizontal ? w.DOWN : w.RIGHT), m = y.dir || (a.horizontal ? w.UP : w.LEFT);
              if (this.renderLink(
                e,
                g,
                d,
                f,
                !1,
                !1,
                null,
                b,
                m
              ), f && f._last_time && t - f._last_time < 1e3) {
                var E = 2 - (t - f._last_time) * 2e-3, T = e.globalAlpha;
                e.globalAlpha = T * E, this.renderLink(
                  e,
                  g,
                  d,
                  f,
                  !0,
                  !0,
                  "white",
                  b,
                  m
                ), e.globalAlpha = T;
              }
            }
          }
        }
    }
    e.globalAlpha = 1;
  }
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
  renderLink(e, t, i, n, s, r, o, a, l, h) {
    n && this.visible_links.push(n), !o && n && (o = n.color || this.link_type_colors[n.type]), o || (o = this.default_link_color), n != null && this.highlighted_links[n.id] && (o = "#FFF"), a = a || w.RIGHT, l = l || w.LEFT;
    var p = u.distance(t, i);
    this.render_connections_border && this.ds.scale > 0.6 && (e.lineWidth = this.connections_width + 4), e.lineJoin = "round", h = h || 1, h > 1 && (e.lineWidth = 0.5), e.beginPath();
    for (var f = 0; f < h; f += 1) {
      var c = (f - (h - 1) * 0.5) * 5;
      if (this.links_render_mode == de.SPLINE_LINK) {
        e.moveTo(t[0], t[1] + c);
        var v = 0, g = 0, d = 0, _ = 0;
        switch (a) {
          case w.LEFT:
            v = p * -0.25;
            break;
          case w.RIGHT:
            v = p * 0.25;
            break;
          case w.UP:
            g = p * -0.25;
            break;
          case w.DOWN:
            g = p * 0.25;
            break;
        }
        switch (l) {
          case w.LEFT:
            d = p * -0.25;
            break;
          case w.RIGHT:
            d = p * 0.25;
            break;
          case w.UP:
            _ = p * -0.25;
            break;
          case w.DOWN:
            _ = p * 0.25;
            break;
        }
        e.bezierCurveTo(
          t[0] + v,
          t[1] + g + c,
          i[0] + d,
          i[1] + _ + c,
          i[0],
          i[1] + c
        );
      } else if (this.links_render_mode == de.LINEAR_LINK) {
        e.moveTo(t[0], t[1] + c);
        var v = 0, g = 0, d = 0, _ = 0;
        switch (a) {
          case w.LEFT:
            v = -1;
            break;
          case w.RIGHT:
            v = 1;
            break;
          case w.UP:
            g = -1;
            break;
          case w.DOWN:
            g = 1;
            break;
        }
        switch (l) {
          case w.LEFT:
            d = -1;
            break;
          case w.RIGHT:
            d = 1;
            break;
          case w.UP:
            _ = -1;
            break;
          case w.DOWN:
            _ = 1;
            break;
        }
        var y = 15;
        e.lineTo(
          t[0] + v * y,
          t[1] + g * y + c
        ), e.lineTo(
          i[0] + d * y,
          i[1] + _ * y + c
        ), e.lineTo(i[0], i[1] + c);
      } else if (this.links_render_mode == de.STRAIGHT_LINK) {
        e.moveTo(t[0], t[1]);
        var b = t[0], m = t[1], E = i[0], T = i[1];
        a == w.RIGHT ? b += 10 : m += 10, l == w.LEFT ? E -= 10 : T -= 10, e.lineTo(b, m), e.lineTo((b + E) * 0.5, m), e.lineTo((b + E) * 0.5, T), e.lineTo(E, T), e.lineTo(i[0], i[1]);
      } else
        return;
    }
    this.render_connections_border && this.ds.scale > 0.6 && !s && (e.strokeStyle = "rgba(0,0,0,0.5)", e.stroke()), e.lineWidth = this.connections_width, e.fillStyle = e.strokeStyle = o, e.stroke();
    var O = this.computeConnectionPoint(t, i, 0.5, a, l);
    if (n && n._pos && (n._pos[0] = O[0], n._pos[1] = O[1]), this.ds.scale >= 0.6 && this.highquality_render && l != w.CENTER) {
      if (this.render_connection_arrows) {
        var A = this.computeConnectionPoint(
          t,
          i,
          0.25,
          a,
          l
        ), M = this.computeConnectionPoint(
          t,
          i,
          0.26,
          a,
          l
        ), L = this.computeConnectionPoint(
          t,
          i,
          0.75,
          a,
          l
        ), B = this.computeConnectionPoint(
          t,
          i,
          0.76,
          a,
          l
        ), G = 0, z = 0;
        this.render_curved_connections ? (G = -Math.atan2(M[0] - A[0], M[1] - A[1]), z = -Math.atan2(B[0] - L[0], B[1] - L[1])) : z = G = i[1] > t[1] ? 0 : Math.PI, e.save(), e.translate(A[0], A[1]), e.rotate(G), e.beginPath(), e.moveTo(-5, -3), e.lineTo(0, 7), e.lineTo(5, -3), e.fill(), e.restore(), e.save(), e.translate(L[0], L[1]), e.rotate(z), e.beginPath(), e.moveTo(-5, -3), e.lineTo(0, 7), e.lineTo(5, -3), e.fill(), e.restore();
      }
      e.beginPath(), e.arc(O[0], O[1], 5, 0, Math.PI * 2), e.fill();
    }
    if (r) {
      e.fillStyle = o;
      for (var f = 0; f < 5; ++f) {
        var pe = (u.getTime() * 1e-3 + f * 0.2) % 1, O = this.computeConnectionPoint(
          t,
          i,
          pe,
          a,
          l
        );
        e.beginPath(), e.arc(O[0], O[1], 5, 0, 2 * Math.PI), e.fill();
      }
    }
  }
  computeConnectionPoint(e, t, i, n = w.RIGHT, s = w.LEFT) {
    var r = u.distance(e, t), o = e, a = [e[0], e[1]], l = [t[0], t[1]], h = t;
    switch (n) {
      case w.LEFT:
        a[0] += r * -0.25;
        break;
      case w.RIGHT:
        a[0] += r * 0.25;
        break;
      case w.UP:
        a[1] += r * -0.25;
        break;
      case w.DOWN:
        a[1] += r * 0.25;
        break;
    }
    switch (s) {
      case w.LEFT:
        l[0] += r * -0.25;
        break;
      case w.RIGHT:
        l[0] += r * 0.25;
        break;
      case w.UP:
        l[1] += r * -0.25;
        break;
      case w.DOWN:
        l[1] += r * 0.25;
        break;
    }
    var p = (1 - i) * (1 - i) * (1 - i), f = 3 * ((1 - i) * (1 - i)) * i, c = 3 * (1 - i) * (i * i), v = i * i * i, g = p * o[0] + f * a[0] + c * l[0] + v * h[0], d = p * o[1] + f * a[1] + c * l[1] + v * h[1];
    return [g, d];
  }
  drawExecutionOrder(e) {
    e.shadowColor = "transparent", e.globalAlpha = 0.25, e.textAlign = "center", e.strokeStyle = "white", e.globalAlpha = 0.75;
    for (var t = this.visible_nodes, i = 0; i < t.length; ++i) {
      var n = t[i];
      e.fillStyle = "black", e.fillRect(
        n.pos[0] - u.NODE_TITLE_HEIGHT,
        n.pos[1] - u.NODE_TITLE_HEIGHT,
        u.NODE_TITLE_HEIGHT,
        u.NODE_TITLE_HEIGHT
      ), n.order == 0 && e.strokeRect(
        n.pos[0] - u.NODE_TITLE_HEIGHT + 0.5,
        n.pos[1] - u.NODE_TITLE_HEIGHT + 0.5,
        u.NODE_TITLE_HEIGHT,
        u.NODE_TITLE_HEIGHT
      ), e.fillStyle = "#FFF", e.fillText(
        "" + n.order,
        n.pos[0] + u.NODE_TITLE_HEIGHT * -0.5,
        n.pos[1] - 6
      );
    }
    e.globalAlpha = 1;
  }
  /** draws the widgets stored inside a node */
  drawNodeWidgets(e, t, i, n) {
    if (!(!e.widgets || !e.widgets.length)) {
      var s = e.size[0], r = e.widgets;
      t += 2;
      var o = u.NODE_WIDGET_HEIGHT, a = this.ds.scale > 0.5;
      i.save(), i.globalAlpha = this.editor_alpha;
      for (var l = u.WIDGET_OUTLINE_COLOR, h = u.WIDGET_BGCOLOR, p = u.WIDGET_TEXT_COLOR, f = u.WIDGET_SECONDARY_TEXT_COLOR, c = 15, v = 0; v < r.length; ++v) {
        var g = r[v];
        if (!g.hidden) {
          var d = t;
          g.y && (d = g.y), g.last_y = d, i.strokeStyle = l, i.fillStyle = "#222", i.textAlign = "left", g.disabled && (i.globalAlpha *= 0.5);
          var _ = g.width || s;
          switch (g.type) {
            case "button":
              g.clicked && (i.fillStyle = "#AAA", g.clicked = !1, this.dirty_canvas = !0), i.fillRect(c, d, _ - c * 2, o), a && !g.disabled && !u.ignore_all_widget_events && i.strokeRect(c, d, _ - c * 2, o), a && (i.textAlign = "center", i.fillStyle = p, i.fillText(g.name, _ * 0.5, d + o * 0.7));
              break;
            case "toggle":
              i.textAlign = "left", i.strokeStyle = l, i.fillStyle = h, i.beginPath(), a ? i.roundRect(c, d, _ - c * 2, o, [o * 0.5]) : i.rect(c, d, _ - c * 2, o), i.fill(), a && !g.disabled && !u.ignore_all_widget_events && i.stroke(), i.fillStyle = g.value ? "#89A" : "#333", i.beginPath(), i.arc(_ - c * 2, d + o * 0.5, o * 0.36, 0, Math.PI * 2), i.fill(), a && (i.fillStyle = f, g.name != null && i.fillText(g.name, c * 2, d + o * 0.7), i.fillStyle = g.value ? p : f, i.textAlign = "right", i.fillText(
                g.value ? g.options.on || "true" : g.options.off || "false",
                _ - 40,
                d + o * 0.7
              ));
              break;
            case "slider":
              i.fillStyle = h, i.fillRect(c, d, _ - c * 2, o);
              var y = g.options.max - g.options.min, b = (g.value - g.options.min) / y;
              if (i.fillStyle = n == g ? "#89A" : "#678", i.fillRect(c, d, b * (_ - c * 2), o), a && !g.disabled && i.strokeRect(c, d, _ - c * 2, o), g.marker) {
                var m = (+g.marker - g.options.min) / y;
                i.fillStyle = "#AA9", i.fillRect(c + m * (_ - c * 2), d, 2, o);
              }
              a && (i.textAlign = "center", i.fillStyle = p, i.fillText(
                g.name + "  " + Number(g.value).toFixed(3),
                _ * 0.5,
                d + o * 0.7
              ));
              break;
            case "number":
            case "combo":
              if (i.textAlign = "left", i.strokeStyle = l, i.fillStyle = h, i.beginPath(), a ? i.roundRect(c, d, _ - c * 2, o, [o * 0.5]) : i.rect(c, d, _ - c * 2, o), i.fill(), a)
                if (!g.disabled && !u.ignore_all_widget_events && i.stroke(), i.fillStyle = p, !g.disabled && !u.ignore_all_widget_events && (i.beginPath(), i.moveTo(c + 16, d + 5), i.lineTo(c + 6, d + o * 0.5), i.lineTo(c + 16, d + o - 5), i.fill(), i.beginPath(), i.moveTo(_ - c - 16, d + 5), i.lineTo(_ - c - 6, d + o * 0.5), i.lineTo(_ - c - 16, d + o - 5), i.fill()), i.fillStyle = f, i.fillText(g.name, c * 2 + 5, d + o * 0.7), i.fillStyle = p, i.textAlign = "right", g.type == "number")
                  i.fillText(
                    Number(g.value).toFixed(
                      g.options.precision !== void 0 ? g.options.precision : 3
                    ),
                    _ - c * 2 - 20,
                    d + o * 0.7
                  );
                else {
                  var E = g.value;
                  if (g.options.values) {
                    var T = g.options.values;
                    T.constructor === Function && (T = T()), T && T.constructor !== Array && (E = T[g.value]);
                  }
                  i.fillText(
                    E,
                    _ - c * 2 - 20,
                    d + o * 0.7
                  );
                }
              break;
            case "string":
            case "text":
              i.textAlign = "left", i.strokeStyle = l, i.fillStyle = h, i.beginPath(), a ? i.roundRect(c, d, _ - c * 2, o, [o * 0.5]) : i.rect(c, d, _ - c * 2, o), i.fill(), a && (g.disabled || i.stroke(), i.save(), i.beginPath(), i.rect(c, d, _ - c * 2, o), i.clip(), i.fillStyle = f, g.name != null && i.fillText(g.name, c * 2, d + o * 0.7), i.fillStyle = p, i.textAlign = "right", i.fillText(String(g.value).substr(0, g.options.max_length || 30), _ - c * 2, d + o * 0.7), i.restore());
              break;
            default:
              g.draw && g.draw(i, e, _, d, o);
              break;
          }
          t += (g.computeSize ? g.computeSize(_)[1] : o) + 4, i.globalAlpha = this.editor_alpha;
        }
      }
      i.restore(), i.textAlign = "left";
    }
  }
};
let U = ie;
U.temp = new Float32Array(4);
U.temp_vec2 = new Float32Array(2);
U.tmp_area = new Float32Array(4);
U.margin_area = new Float32Array(4);
U.link_bounding = new Float32Array(4);
U.tempA = [0, 0];
U.tempB = [0, 0];
class me {
  constructor(t = "Group") {
    this.fontSize = u.DEFAULT_GROUP_FONT_SIZE, this._nodes = [], this.graph = null, this._bounding = new Float32Array([10, 10, 140, 80]), this.title = t, this.color = N.node_colors.pale_blue ? N.node_colors.pale_blue.groupcolor : "#AAA", this._pos = this._bounding.subarray(0, 2), this._size = this._bounding.subarray(2, 4);
  }
  get bounding() {
    return this._bounding;
  }
  get pos() {
    return [this._pos[0], this._pos[1]];
  }
  set pos(t) {
    !t || t.length < 2 || (this._pos[0] = t[0], this._pos[1] = t[1]);
  }
  get size() {
    return [this._size[0], this._size[1]];
  }
  set size(t) {
    !t || t.length < 2 || (this._size[0] = Math.max(140, t[0]), this._size[1] = Math.max(80, t[1]));
  }
  configure(t) {
    t.bounding, this.title = t.title, this._bounding.set(t.bounding), this.color = t.color, this.font = t.font;
  }
  serialize() {
    const t = this._bounding;
    return {
      title: this.title,
      bounding: [
        Math.round(t[0]),
        Math.round(t[1]),
        Math.round(t[2]),
        Math.round(t[3])
      ],
      color: this.color,
      font: this.font
    };
  }
  move(t, i, n) {
    if (this._pos[0] += t, this._pos[1] += i, !n)
      for (var s = 0; s < this._nodes.length; ++s) {
        var r = this._nodes[s];
        r.pos[0] += t, r.pos[1] += i;
      }
  }
  recomputeInsideNodes() {
    this._nodes.length = 0;
    for (var t = this.graph._nodes, i = new Float32Array(4), n = 0; n < t.length; ++n) {
      var s = t[n];
      s.getBounding(i), u.overlapBounding(this._bounding, i) && this._nodes.push(s);
    }
  }
  /** checks if a point is inside the shape of a node */
  isPointInside(t, i, n = 0, s = !1) {
    var r = this.graph && this.graph.isLive() ? 0 : u.NODE_TITLE_HEIGHT;
    return s && (r = 0), this.pos[0] - 4 - n < t && this.pos[0] + this.size[0] + 4 + n > t && this.pos[1] - r - n < i && this.pos[1] + this.size[1] + n > i;
  }
  /** Forces to redraw or the main canvas (LGraphNode) or the bg canvas (links) */
  setDirtyCanvas(t, i = !1) {
    this.graph && this.graph.sendActionToCanvas("setDirty", [t, i]);
  }
}
class he {
  constructor(t, i, n, s, r, o) {
    this.data = null, this._pos = [0, 0], this._last_time = 0, this.id = t, this.type = i, this.origin_id = n, this.origin_slot = s, this.target_id = r, this.target_slot = o;
  }
  static configure(t) {
    return t instanceof Array ? new he(t[0], t[5], t[1], t[2], t[3], t[4]) : new he(t.id, t.type, t.origin_id, t.origin_slot, t.target_id, t.target_slot);
  }
  serialize() {
    return [
      this.id,
      this.origin_id,
      this.origin_slot,
      this.target_id,
      this.target_slot,
      this.type
    ];
  }
}
let _e;
const He = new Uint8Array(16);
function Fe() {
  if (!_e && (_e = typeof crypto < "u" && crypto.getRandomValues && crypto.getRandomValues.bind(crypto), !_e))
    throw new Error("crypto.getRandomValues() not supported. See https://github.com/uuidjs/uuid#getrandomvalues-not-supported");
  return _e(He);
}
const K = [];
for (let e = 0; e < 256; ++e)
  K.push((e + 256).toString(16).slice(1));
function Ue(e, t = 0) {
  return (K[e[t + 0]] + K[e[t + 1]] + K[e[t + 2]] + K[e[t + 3]] + "-" + K[e[t + 4]] + K[e[t + 5]] + "-" + K[e[t + 6]] + K[e[t + 7]] + "-" + K[e[t + 8]] + K[e[t + 9]] + "-" + K[e[t + 10]] + K[e[t + 11]] + K[e[t + 12]] + K[e[t + 13]] + K[e[t + 14]] + K[e[t + 15]]).toLowerCase();
}
const ze = typeof crypto < "u" && crypto.randomUUID && crypto.randomUUID.bind(crypto), ke = {
  randomUUID: ze
};
function oe(e, t, i) {
  if (ke.randomUUID && !t && !e)
    return ke.randomUUID();
  e = e || {};
  const n = e.random || (e.rng || Fe)();
  if (n[6] = n[6] & 15 | 64, n[8] = n[8] & 63 | 128, t) {
    i = i || 0;
    for (let s = 0; s < 16; ++s)
      t[i + s] = n[s];
    return t;
  }
  return Ue(n);
}
const Ne = class {
  constructor(e) {
    this.desc = "", this.pos = [0, 0], this.subgraph = null, this.skip_subgraph_button = !1, this.priority = 0, this.removable = !0, this.clonable = !0, this.collapsable = !0, this.titleMode = se.NORMAL_TITLE, this.serialize_widgets = !1, this.hide_in_node_lists = !1, this.block_delete = !1, this.ignore_remove = !1, this.last_serialization = null, this._relative_id = null, this.exec_version = 0, this.action_call = null, this.execute_triggered = 0, this.action_triggered = 0, this.console = [], this.title = e || "Unnamed", this.size = [u.NODE_WIDTH, 60], this.graph = null, this.pos = [10, 10], u.use_uuids ? this.id = oe() : this.id = -1, this.type = null, this.inputs = [], this.outputs = [], this.connections = [], this.properties = {}, this.properties_info = [], this.flags = {};
  }
  get slotLayout() {
    return "slotLayout" in this.constructor ? this.constructor.slotLayout : null;
  }
  /** configure a node from an object containing the serialized info */
  configure(e) {
    this.graph && this.graph._version++;
    for (var t in e) {
      if (t == "properties") {
        for (var i in e.properties)
          this.properties[i] = e.properties[i], this.onPropertyChanged && this.onPropertyChanged(i, e.properties[i]);
        continue;
      }
      e[t] != null && (typeof e[t] == "object" ? this[t] && this[t].configure ? this[t].configure(e[t]) : this[t] = u.cloneObject(e[t], this[t]) : this[t] = e[t]);
    }
    e.title || (this.title = ye(this, "title") || this.title);
    const n = e.bgColor;
    if (n != null && (this.bgcolor || (this.bgcolor = n)), this.inputs)
      for (let o = 0; o < this.inputs.length; ++o) {
        let a = this.inputs[o], l = this.graph ? this.graph.links[a.link] : null;
        a.properties || (a.properties = {}), this.onConnectionsChange && this.onConnectionsChange(W.INPUT, o, !0, l, a), this.onInputAdded && this.onInputAdded(a);
      }
    if (this.outputs)
      for (var s = 0; s < this.outputs.length; ++s) {
        let o = this.outputs[s];
        if (o.properties || (o.properties = {}), !!o.links) {
          for (let a = 0; a < o.links.length; ++a) {
            let l = this.graph ? this.graph.links[o.links[a]] : null;
            this.onConnectionsChange && this.onConnectionsChange(W.OUTPUT, s, !0, l, o);
          }
          this.onOutputAdded && this.onOutputAdded(o);
        }
      }
    if (this.widgets) {
      for (var s = 0; s < this.widgets.length; ++s) {
        var r = this.widgets[s];
        r && r.options && r.options.property && this.properties[r.options.property] && (r.value = JSON.parse(JSON.stringify(this.properties[r.options.property])));
      }
      if (e.widgets_values)
        for (var s = 0; s < e.widgets_values.length; ++s)
          this.widgets[s] && (this.widgets[s].value = e.widgets_values[s]);
    }
    this.onConfigure && this.onConfigure(e);
  }
  /** serialize the content */
  serialize() {
    let e = {
      id: this.id,
      type: this.type,
      pos: this.pos,
      size: this.size,
      flags: u.cloneObject(this.flags),
      order: this.order,
      mode: this.mode
    };
    if (this.constructor === Ne && this.last_serialization)
      return this.last_serialization;
    if (this.inputs && (e.inputs = this.inputs), this.outputs) {
      for (var t = 0; t < this.outputs.length; t++)
        delete this.outputs[t]._data;
      e.outputs = this.outputs;
    }
    if (this.title && this.title != this.constructor.title && (e.title = this.title), this.properties && (e.properties = u.cloneObject(this.properties)), this.widgets && this.serialize_widgets) {
      e.widgets_values = [];
      for (var t = 0; t < this.widgets.length; ++t)
        this.widgets[t] ? e.widgets_values[t] = this.widgets[t].value : e.widgets_values[t] = null;
    }
    return e.type || (e.type = this.constructor.type), this.color && (e.color = this.color), this.bgcolor && (e.bgcolor = this.bgcolor), this.boxcolor && (e.boxcolor = this.boxcolor), this.shape && (e.shape = this.shape), this.onSerialize && this.onSerialize(e), e;
  }
  /** Creates a clone of this node  */
  clone(e = { forNode: {} }) {
    var t = u.createNode(this.type);
    if (!t)
      return null;
    var i = u.cloneObject(this.serialize());
    if (i.inputs)
      for (var n = 0; n < i.inputs.length; ++n)
        i.inputs[n].link = null;
    if (i.outputs)
      for (var n = 0; n < i.outputs.length; ++n)
        i.outputs[n].links && (i.outputs[n].links.length = 0);
    return delete i.id, u.use_uuids && (i.id = oe()), t.configure(i), t;
  }
  /** serialize and stringify */
  toString() {
    return JSON.stringify(this.serialize());
  }
  /** get the title string */
  getTitle() {
    return this.title || this.constructor.title;
  }
  getRootGraph() {
    var t;
    let e = this.graph;
    for (; e && e._is_subgraph; )
      e = (t = e._subgraph_node) == null ? void 0 : t.graph;
    return e == null || e._is_subgraph ? null : e;
  }
  *iterateParentSubgraphNodes() {
    var t;
    let e = this.graph._subgraph_node;
    for (; e; )
      yield e, e = (t = e.graph) == null ? void 0 : t._subgraph_node;
  }
  /** sets the value of a property */
  setProperty(e, t) {
    var r;
    if (this.properties || (this.properties = {}), t !== this.properties[e]) {
      var i = this.properties[e];
      if (this.properties[e] = t, this.graph && this.graph._version++, this.onPropertyChanged && this.onPropertyChanged(e, t, i) === !1 && (this.properties[e] = i), this.widgets)
        for (var n = 0; n < this.widgets.length; ++n) {
          var s = this.widgets[n];
          if (s && ((r = s.options) == null ? void 0 : r.property) == e) {
            s.value = t;
            break;
          }
        }
    }
  }
  getInputSlotProperty(e, t) {
    if (!(!this.inputs || !this.graph) && !(e == -1 || e >= this.inputs.length)) {
      var i = this.inputs[e];
      if (i)
        return i.properties || (i.properties = {}), i.properties[t];
    }
  }
  getOutputSlotProperty(e, t) {
    if (!(!this.outputs || !this.graph) && !(e == -1 || e >= this.outputs.length)) {
      var i = this.outputs[e];
      if (i)
        return i.properties || (i.properties = {}), i.properties[t];
    }
  }
  setInputSlotProperty(e, t, i) {
    if (!(!this.inputs || !this.graph) && !(e == -1 || e >= this.inputs.length)) {
      var n = this.inputs[e];
      if (n && (n.properties || (n.properties = {}), i !== n.properties[t])) {
        var s = n.properties[t];
        n.properties[t] = i, this.graph && this.graph._version++, this.onSlotPropertyChanged && this.onSlotPropertyChanged(W.INPUT, e, n, t, i, s) === !1 && (n.properties[t] = s);
      }
    }
  }
  setOutputSlotProperty(e, t, i) {
    if (!(!this.outputs || !this.graph) && !(e == -1 || e >= this.outputs.length)) {
      var n = this.outputs[e];
      if (n && (n.properties || (n.properties = {}), i !== n.properties[t])) {
        var s = n.properties[t];
        n.properties[t] = i, this.graph && this.graph._version++, this.onSlotPropertyChanged && this.onSlotPropertyChanged(W.OUTPUT, e, n, t, i, s) === !1 && (n.properties[t] = s);
      }
    }
  }
  /** sets the output data */
  setOutputData(e, t) {
    if (!(!this.outputs || !this.graph) && !(e == -1 || e >= this.outputs.length)) {
      var i = this.outputs[e];
      if (i && (u.serialize_slot_data ? i._data = t : i._data = void 0, this.outputs[e].links))
        for (var n = 0; n < this.outputs[e].links.length; n++) {
          var s = this.outputs[e].links[n], r = this.graph.links[s];
          r && (r.data = t);
        }
    }
  }
  /** sets the output data */
  setOutputDataType(e, t) {
    if (this.outputs && !(e == -1 || e >= this.outputs.length)) {
      var i = this.outputs[e];
      if (i && (i.type = t, this.outputs[e].links))
        for (let n = this.outputs[e].links.length - 1; n >= 0; n--) {
          const s = this.outputs[e].links[n], r = this.graph.links[s];
          if (r) {
            r.type = t;
            const o = this.graph.getNodeById(r.target_id);
            if (o) {
              const a = o.getInputInfo(r.target_slot);
              a && !u.isValidConnection(t, a.type) && o.disconnectInput(r.target_slot);
            }
          }
        }
    }
  }
  *iterateInputInfo() {
    for (let e = 0; e < this.inputs.length; e++)
      yield this.inputs[e];
  }
  /**
   * Retrieves the input data (data traveling through the connection) from one slot
   * @param slot
   * @param force_update if set to true it will force the connected node of this slot to output data into this link
   * @return data or if it is not connected returns undefined
   */
  getInputData(e, t) {
    if (!(!this.inputs || !this.graph) && !(e >= this.inputs.length || this.inputs[e].link == null)) {
      var i = this.inputs[e].link, n = this.graph.links[i];
      if (!n)
        return u.debug && console.error(`Link not found in slot ${e}!`, this, this.inputs[e], i), null;
      if (!t)
        return n.data;
      var s = this.graph.getNodeById(n.origin_id);
      return s && (s.updateOutputData ? s.updateOutputData(n.origin_slot) : s.onExecute && s.onExecute(null, {})), n.data;
    }
  }
  /**
   * Retrieves the input data type (in case this supports multiple input types)
   * @param slot
   * @return datatype in string format
   */
  getInputDataType(e) {
    if (!this.inputs || e >= this.inputs.length || this.inputs[e].link == null)
      return null;
    var t = this.inputs[e].link, i = this.graph.links[t];
    if (!i)
      return u.debug && console.error(`Link not found in slot ${e}!`, this, this.inputs[e], t), null;
    var n = this.graph.getNodeById(i.origin_id);
    if (!n)
      return i.type;
    var s = n.outputs[i.origin_slot];
    return s && s.type != -1 ? s.type : null;
  }
  /**
   * Retrieves the input data from one slot using its name instead of slot number
   * @param slot_name
   * @param force_update if set to true it will force the connected node of this slot to output data into this link
   * @return data or if it is not connected returns null
   */
  getInputDataByName(e, t) {
    var i = this.findInputSlotIndexByName(e);
    return i == -1 ? null : this.getInputData(i, t);
  }
  /** tells you if there is a connection in one input slot */
  isInputConnected(e) {
    return this.inputs ? e < this.inputs.length && this.inputs[e].link != null : !1;
  }
  /** tells you info about an input connection (which node, type, etc) */
  getInputInfo(e) {
    return this.inputs && e < this.inputs.length ? this.inputs[e] : null;
  }
  /**
   * Returns the link info in the connection of an input slot
   * @param {number} slot
   * @return {LLink} object or null
   */
  getInputLink(e) {
    if (!this.inputs || !this.graph)
      return null;
    if (e < this.inputs.length) {
      var t = this.inputs[e];
      return this.graph.links[t.link];
    }
    return null;
  }
  /** returns the node connected in the input slot */
  getInputNode(e) {
    if (!this.inputs || !this.graph)
      return null;
    if (e < this.inputs.length) {
      const i = this.inputs[e].link, n = this.graph.links[i];
      if (!n)
        return u.debug && console.error(`Link not found in slot ${e}!`, this, this.inputs[e], i), null;
      var t = this.graph.getNodeById(n.origin_id);
      if (t)
        return t;
    }
    return null;
  }
  /** returns the value of an input with this name, otherwise checks if there is a property with that name */
  getInputOrProperty(e) {
    if (!this.inputs || !this.inputs.length || !this.graph)
      return this.properties ? this.properties[e] : null;
    for (var t = 0, i = this.inputs.length; t < i; ++t) {
      var n = this.inputs[t];
      if (e == n.name && n.link != null) {
        var s = this.graph.links[n.link];
        if (s)
          return s.data;
      }
    }
    return this.properties[e];
  }
  /** sets the input data type */
  setInputDataType(e, t) {
    if (!(!this.inputs || !this.graph) && !(e == -1 || e >= this.inputs.length)) {
      var i = this.inputs[e];
      if (i && (i.type = t, i.link)) {
        const n = i.link, s = this.graph.links[n];
        s.type = t;
        const r = this.graph.getNodeById(s.origin_id);
        if (r) {
          const o = r.getOutputInfo(s.origin_slot);
          o && !u.isValidConnection(o.type, t) && r.disconnectOutput(s.origin_slot);
        }
      }
    }
  }
  /**
   * Returns the output slot in another node that an input in this node is connected to.
   * @param {number} slot
   * @return {LLink} object or null
   */
  getOutputSlotConnectedTo(e) {
    if (!this.outputs || !this.graph)
      return null;
    if (e >= 0 && e < this.outputs.length) {
      var t = this.inputs[e];
      if (t.link) {
        const i = this.graph.links[t.link];
        return this.graph.getNodeById(i.origin_id).outputs[i.origin_slot];
      }
    }
    return null;
  }
  *iterateOutputInfo() {
    for (let e = 0; e < this.outputs.length; e++)
      yield this.outputs[e];
  }
  /** tells you the last output data that went in that slot */
  getOutputData(e) {
    if (!this.outputs || !this.graph || e >= this.outputs.length)
      return null;
    var t = this.outputs[e];
    return t._data;
  }
  /**
   * Returns the link info in the connection of an output slot
   * @param {number} slot
   * @return {LLink} object or null
   */
  getOutputLinks(e) {
    if (!this.outputs || !this.graph)
      return [];
    if (e >= 0 && e < this.outputs.length) {
      var t = this.outputs[e];
      if (t.links) {
        var i = [];
        for (const n of t.links)
          i.push(this.graph.links[n]);
        return i;
      }
    }
    return [];
  }
  /**
   * Returns the input slots in other nodes that an output in this node is connected to.
   * @param {number} slot
   * @return {LLink} object or null
   */
  getInputSlotsConnectedTo(e) {
    if (!this.outputs || !this.graph)
      return [];
    if (e >= 0 && e < this.outputs.length) {
      var t = this.outputs[e];
      if (t.links) {
        var i = [];
        for (const n of t.links) {
          const s = this.graph.links[n], r = this.graph.getNodeById(s.target_id);
          i.push(r.inputs[s.target_slot]);
        }
        return i;
      }
    }
    return [];
  }
  /** tells you info about an output connection (which node, type, etc) */
  getOutputInfo(e) {
    return this.outputs && e < this.outputs.length ? this.outputs[e] : null;
  }
  /** tells you if there is a connection in one output slot */
  isOutputConnected(e) {
    return !this.outputs || !this.graph ? !1 : e < this.outputs.length && this.outputs[e].links && this.outputs[e].links.length > 0;
  }
  /** tells you if there is any connection in the output slots */
  isAnyOutputConnected() {
    if (!this.outputs || !this.graph)
      return !1;
    for (var e = 0; e < this.outputs.length; ++e)
      if (this.outputs[e].links && this.outputs[e].links.length)
        return !0;
    return !1;
  }
  /** retrieves all the nodes connected to this output slot */
  getOutputNodes(e) {
    if (!this.outputs || this.outputs.length == 0 || !this.graph || e >= this.outputs.length)
      return null;
    var t = this.outputs[e];
    if (!t.links || t.links.length == 0)
      return null;
    for (var i = [], n = 0; n < t.links.length; n++) {
      var s = t.links[n], r = this.graph.links[s];
      if (r) {
        var o = this.graph.getNodeById(r.target_id);
        o && i.push(o);
      }
    }
    return i;
  }
  *iterateAllLinks() {
    if (this.graph) {
      for (const e of this.iterateInputInfo())
        if (e.link) {
          const t = this.graph.links[e.link];
          t && (yield t);
        }
      for (const e of this.iterateOutputInfo())
        if (e.links != null)
          for (const t of e.links) {
            const i = this.graph.links[t];
            i && (yield i);
          }
    }
  }
  addOnTriggerInput() {
    var e = this.findInputSlotIndexByName("onTrigger");
    if (e == -1) {
      //!trigS ||
      return this.addInput("onTrigger", I.EVENT, { optional: !0, nameLocked: !0 }), this.findInputSlotIndexByName("onTrigger");
    }
    return e;
  }
  addOnExecutedOutput() {
    var e = this.findOutputSlotIndexByName("onExecuted");
    if (e == -1) {
      //!trigS ||
      return this.addOutput("onExecuted", I.ACTION, { optional: !0, nameLocked: !0 }), this.findOutputSlotIndexByName("onExecuted");
    }
    return e;
  }
  onAfterExecuteNode(e, t) {
    var i = this.findOutputSlotIndexByName("onExecuted");
    i != -1 && this.triggerSlot(i, e, null, t);
  }
  changeMode(e) {
    switch (e) {
      case Z.ON_EVENT:
        break;
      case Z.ON_TRIGGER:
        this.addOnTriggerInput(), this.addOnExecutedOutput();
        break;
      case Z.NEVER:
        break;
      case Z.ALWAYS:
        break;
      case Z.ON_REQUEST:
        break;
      default:
        return !1;
    }
    return this.mode = e, !0;
  }
  doExecute(e, t = {}) {
    this.onExecute && (t.action_call || (t.action_call = this.id + "_exec_" + Math.floor(Math.random() * 9999)), this.graph.nodes_executing[this.id] = !0, this.onExecute(e, t), this.graph.nodes_executing[this.id] = !1, this.exec_version = this.graph.iteration, t && t.action_call && (this.action_call = t.action_call, this.graph.nodes_executedAction[this.id] = t.action_call)), this.execute_triggered = 2, this.onAfterExecuteNode && this.onAfterExecuteNode(e, t);
  }
  /**
   * Triggers an action, wrapped by logics to control execution flow
   * @method actionDo
   * @param {String} action name
   * @param {*} param
   */
  actionDo(e, t, i = {}) {
    this.onAction && (i.action_call || (i.action_call = this.id + "_" + (e || "action") + "_" + Math.floor(Math.random() * 9999)), this.graph.nodes_actioning[this.id] = e || "actioning", this.onAction(e, t, i), this.graph.nodes_actioning[this.id] = !1, i && i.action_call && (this.action_call = i.action_call, this.graph.nodes_executedAction[this.id] = i.action_call)), this.action_triggered = 2, this.onAfterExecuteNode && this.onAfterExecuteNode(t, i);
  }
  /**  Triggers an event in this node, this will trigger any output with the same name */
  trigger(e, t, i) {
    if (!(!this.outputs || !this.outputs.length)) {
      this.graph && (this.graph._last_trigger_time = u.getTime());
      for (var n = 0; n < this.outputs.length; ++n) {
        var s = this.outputs[n];
        !s || s.type !== I.EVENT || e && s.name != e || this.triggerSlot(n, t, null, i);
      }
    }
  }
  /**
   * Triggers an slot event in this node
   * @param slot the index of the output slot
   * @param param
   * @param link_id in case you want to trigger and specific output link in a slot
   */
  triggerSlot(e, t, i, n = {}) {
    if (this.outputs) {
      if (e == null) {
        console.error("slot must be a number");
        return;
      }
      typeof e != "number" && console.warn("slot must be a number, use node.trigger('name') if you want to use a string");
      var s = this.outputs[e];
      if (s) {
        var r = s.links;
        if (!(!r || !r.length)) {
          this.graph && (this.graph._last_trigger_time = u.getTime());
          for (var o = 0; o < r.length; ++o) {
            var a = r[o];
            if (!(i != null && i != a)) {
              var l = this.graph.links[r[o]];
              if (l) {
                l._last_time = u.getTime();
                var h = this.graph.getNodeById(l.target_id);
                if (h) {
                  if (h.inputs[l.target_slot], n.link = l, n.originNode = this, h.mode === Z.ON_TRIGGER)
                    n.action_call || (n.action_call = this.id + "_trigg_" + Math.floor(Math.random() * 9999)), h.onExecute && h.doExecute(t, n);
                  else if (h.onAction) {
                    n.action_call || (n.action_call = this.id + "_act_" + Math.floor(Math.random() * 9999));
                    const p = h.inputs[l.target_slot];
                    h.actionDo(p.name, t, n);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  /**
   * clears the trigger slot animation
   * @param slot the index of the output slot
   * @param link_id in case you want to trigger and specific output link in a slot
   */
  clearTriggeredSlot(e, t) {
    if (this.outputs) {
      var i = this.outputs[e];
      if (i) {
        var n = i.links;
        if (!(!n || !n.length))
          for (var s = 0; s < n.length; ++s) {
            var r = n[s];
            if (!(t != null && t != r)) {
              var o = this.graph.links[n[s]];
              o && (o._last_time = 0);
            }
          }
      }
    }
  }
  /**
   * changes node size and triggers callback
   * @method setSize
   * @param {vec2} size
   */
  setSize(e) {
    this.size = e, this.onResize && this.onResize(this.size);
  }
  /**
   * add a new property to this node
   * @param name
   * @param default_value
   * @param type string defining the output type ("vec3","number",...)
   * @param extra_info this can be used to have special properties of the property (like values, etc)
   */
  addProperty(e, t, i, n) {
    var s = { name: e, type: i, default_value: t };
    if (n)
      for (var r in n)
        s[r] = n[r];
    return this.properties_info || (this.properties_info = []), this.properties_info.push(s), this.properties || (this.properties = {}), this.properties[e] = t, s;
  }
  hasProperty(e) {
    return this.properties != null && e in this.properties;
  }
  /**
   * add a new output slot to use in this node
   * @param name
   * @param type string defining the output type ("vec3","number",...)
   * @param extra_info this can be used to have special properties of an output (label, special color, position, etc)
   */
  addOutput(e, t = I.DEFAULT, i) {
    var n = { name: e, type: t, links: [], properties: {} };
    if (i)
      for (var s in i)
        n[s] = i[s];
    return (n.shape == null || n.shape == k.DEFAULT) && (t == "array" ? n.shape = k.GRID_SHAPE : (t === I.EVENT || t === I.ACTION) && (n.shape = k.BOX_SHAPE)), (t === I.EVENT || t === I.ACTION) && (n.shape = k.BOX_SHAPE), this.outputs || (this.outputs = []), this.outputs.push(n), this.onOutputAdded && this.onOutputAdded(n), u.auto_load_slot_types && u.registerNodeAndSlotType(this, t, !0), this.setSize(this.computeSize()), this.setDirtyCanvas(!0, !0), n;
  }
  /** remove an existing output slot */
  removeOutput(e) {
    const t = this.outputs[e];
    this.disconnectOutput(e), this.outputs.splice(e, 1);
    for (var i = e; i < this.outputs.length; ++i)
      if (!(!this.outputs[i] || !this.outputs[i].links))
        for (var n = this.outputs[i].links, s = 0; s < n.length; ++s) {
          var r = this.graph.links[n[s]];
          r && (r.origin_slot -= 1);
        }
    this.setSize(this.computeSize()), this.onOutputRemoved && this.onOutputRemoved(e, t), this.setDirtyCanvas(!0, !0);
  }
  moveOutput(e, t) {
    const i = this.outputs[e];
    if (i == null || t < 0 || t > this.outputs.length - 1)
      return;
    const n = this.outputs[t];
    if (i.links)
      for (const s of i.links) {
        const r = this.graph.links[s];
        r.origin_slot = t;
      }
    if (n.links)
      for (const s of n.links) {
        const r = this.graph.links[s];
        r.origin_slot = e;
      }
    this.outputs[t] = i, this.outputs[e] = n;
  }
  /**
   * add a new input slot to use in this node
   * @param name
   * @param type string defining the input type ("vec3","number",...), it its a generic one use 0
   * @param extra_info this can be used to have special properties of an input (label, color, position, etc)
   */
  addInput(e, t = I.DEFAULT, i) {
    var n = { name: e, type: t, link: null, properties: {} };
    if (i)
      for (var s in i)
        n[s] = i[s];
    return (n.shape == null || n.shape == k.DEFAULT) && (t == "array" ? n.shape = k.GRID_SHAPE : (t === I.EVENT || t === I.ACTION) && (n.shape = k.BOX_SHAPE)), this.inputs || (this.inputs = []), this.inputs.push(n), this.setSize(this.computeSize()), this.onInputAdded && this.onInputAdded(n), u.registerNodeAndSlotType(this, t), this.setDirtyCanvas(!0, !0), n;
  }
  /** remove an existing input slot */
  removeInput(e) {
    this.disconnectInput(e);
    for (var t = this.inputs.splice(e, 1), i = e; i < this.inputs.length; ++i)
      if (this.inputs[i]) {
        var n = this.graph.links[this.inputs[i].link];
        n && (n.target_slot -= 1);
      }
    this.setSize(this.computeSize()), this.onInputRemoved && this.onInputRemoved(e, t[0]), this.setDirtyCanvas(!0, !0);
  }
  moveInput(e, t) {
    const i = this.inputs[e];
    if (i == null || t < 0 || t > this.inputs.length - 1)
      return;
    const n = this.inputs[t];
    if (i.link != null) {
      const s = this.graph.links[i.link];
      s.target_slot = t;
    }
    if (n.link != null) {
      const s = this.graph.links[n.link];
      s.target_slot = e;
    }
    this.inputs[t] = i, this.inputs[e] = n;
  }
  /**
   * add an special connection to this node (used for special kinds of graphs)
   * @param name
   * @param type string defining the input type ("vec3","number",...)
   * @param pos position of the connection inside the node
   * @param direction if is input or output
   */
  addConnection(e, t, i, n) {
    let s = {
      name: e,
      type: t,
      pos: i,
      direction: n,
      links: null
    };
    return this.connections.push(s), s;
  }
  /** computes the size of a node according to its inputs and output slots */
  computeSize(e = [0, 0]) {
    const t = ye(this, "overrideSize");
    if (t)
      return t.concat();
    var i = Math.max(
      this.inputs ? this.inputs.length : 1,
      this.outputs ? this.outputs.length : 1
    ), n = e;
    i = Math.max(i, 1);
    var s = u.NODE_TEXT_SIZE, r = d(this.title), o = 0, a = 0;
    if (this.inputs)
      for (var l = 0, h = this.inputs.length; l < h; ++l) {
        var p = this.inputs[l], f = p.label || p.name || "", c = d(f);
        o < c && (o = c);
      }
    if (this.outputs)
      for (var l = 0, h = this.outputs.length; l < h; ++l) {
        var v = this.outputs[l], f = v.label || v.name || "", c = d(f);
        a < c && (a = c);
      }
    if (n[0] = Math.max(o + a + 10, r), n[0] = Math.max(n[0], u.NODE_WIDTH), this.widgets && this.widgets.length)
      for (const _ of this.widgets)
        n[0] = Math.max(n[0], _.width || u.NODE_WIDTH * 1.5);
    n[1] = (this.constructor.slot_start_y || 0) + i * u.NODE_SLOT_HEIGHT;
    var g = 0;
    if (this.widgets && this.widgets.length) {
      for (var l = 0, h = this.widgets.length; l < h; ++l) {
        const b = this.widgets[l];
        b.hidden || (b.computeSize ? g += b.computeSize(n[0])[1] + 4 : g += u.NODE_WIDGET_HEIGHT + 4);
      }
      g += 8;
    }
    this.widgets_up ? n[1] = Math.max(n[1], g) : this.widgets_start_y != null ? n[1] = Math.max(n[1], g + this.widgets_start_y) : n[1] += g;
    function d(_) {
      return _ ? s * _.length * 0.6 : 0;
    }
    return this.constructor.min_height && n[1] < this.constructor.min_height && (n[1] = this.constructor.min_height), n[1] += 6, n;
  }
  /**
   * returns all the info available about a property of this node.
   *
   * @method getPropertyInfo
   * @param {String} property name of the property
   * @return {Object} the object with all the available info
  */
  getPropertyInfo(e) {
    var t = null;
    if (this.properties_info) {
      for (var i = 0; i < this.properties_info.length; ++i)
        if (this.properties_info[i].name == e) {
          t = this.properties_info[i];
          break;
        }
    }
    return this.constructor["@" + e] && (t = this.constructor["@" + e]), this.constructor.widgets_info && this.constructor.widgets_info[e] && (t = this.constructor.widgets_info[e]), !t && this.onGetPropertyInfo && (t = this.onGetPropertyInfo(e)), t || (t = {}), t.type || (t.type = typeof this.properties[e]), t.widget == "combo" && (t.type = "enum"), t;
  }
  /**
   * https://github.com/jagenjo/litegraph.js/blob/master/guides/README.md#node-widgets
   * @return created widget
   */
  addWidget(e, t, i, n, s) {
    this.widgets || (this.widgets = []), !s && n && n.constructor === Object && (s = n, n = null), s && s.constructor === String && (s = { property: s }), n && n.constructor === String && (s || (s = {}), s.property = n, n = null), n && n.constructor !== Function && (console.warn("addWidget: callback must be a function"), n = null);
    var r = {
      type: e.toLowerCase(),
      name: t,
      value: i,
      callback: n,
      options: s || {}
    };
    if (r.options.y !== void 0 && (r.y = r.options.y), !n && !r.options.callback && !r.options.property && console.warn("LiteGraph addWidget(...) without a callback or property assigned"), e == "combo" && !r.options.values)
      throw "LiteGraph addWidget('combo',...) requires to pass values in options: { values:['red','blue'] }";
    return this.widgets.push(r), this.setSize(this.computeSize()), r;
  }
  addCustomWidget(e) {
    return this.widgets || (this.widgets = []), this.widgets.push(e), this.setSize(this.computeSize()), e;
  }
  setWidgetHidden(e, t) {
    e.hidden = t, this.setSize(this.computeSize());
  }
  /**
   * returns the bounding of the object, used for rendering purposes
   * @return [x, y, width, height]
   */
  getBounding(e) {
    return e = e || new Float32Array(4), e[0] = this.pos[0] - 4, e[1] = this.pos[1] - u.NODE_TITLE_HEIGHT, e[2] = this.size[0] + 4, e[3] = this.flags.collapsed ? u.NODE_TITLE_HEIGHT : this.size[1] + u.NODE_TITLE_HEIGHT, this.onBounding && this.onBounding(e), e;
  }
  /** checks if a point is inside the shape of a node */
  isPointInside(e, t, i = 0, n = !1) {
    var s = this.graph && this.graph.isLive() ? 0 : u.NODE_TITLE_HEIGHT;
    if (n && (s = 0), this.flags && this.flags.collapsed) {
      if (u.isInsideRectangle(
        e,
        t,
        this.pos[0] - i,
        this.pos[1] - u.NODE_TITLE_HEIGHT - i,
        (this._collapsed_width || u.NODE_COLLAPSED_WIDTH) + 2 * i,
        u.NODE_TITLE_HEIGHT + 2 * i
      ))
        return !0;
    } else if (this.pos[0] - 4 - i < e && this.pos[0] + this.size[0] + 4 + i > e && this.pos[1] - s - i < t && this.pos[1] + this.size[1] + i > t)
      return !0;
    return !1;
  }
  /** checks if a point is inside a node slot, and returns info about which slot */
  getSlotInPosition(e, t) {
    var i = [0, 0];
    if (this.inputs)
      for (var n = 0, s = this.inputs.length; n < s; ++n) {
        var r = this.inputs[n];
        if (this.getConnectionPos(!0, n, i), u.isInsideRectangle(
          e,
          t,
          i[0] - 10,
          i[1] - 5,
          20,
          10
        ))
          return { input: r, slot: n, link_pos: i };
      }
    if (this.outputs)
      for (var n = 0, s = this.outputs.length; n < s; ++n) {
        var o = this.outputs[n];
        if (this.getConnectionPos(!1, n, i), u.isInsideRectangle(
          e,
          t,
          i[0] - 10,
          i[1] - 5,
          20,
          10
        ))
          return { output: o, slot: n, link_pos: i };
      }
    return null;
  }
  is(e) {
    const t = e.__LITEGRAPH_TYPE__;
    return t != null && this.type === t;
  }
  /**
   * returns the input slot with a given name (used for dynamic slots), -1 if not found
   * for compatibility purposes only, please prefer `findInputSlotIndexByName`
   * @param name the name of the slot
   * @return the slot (-1 if not found)
   */
  findInputSlot(e) {
    return this.findInputSlotIndexByName(e);
  }
  /**
   * returns the output slot with a given name (used for dynamic slots), -1 if not found
   * for compatibility purposes only, please prefer `findOutputSlotIndexByName`
   * @param name the name of the slot
   * @return  the slot (-1 if not found)
   */
  findOutputSlot(e) {
    return this.findOutputSlotIndexByName(e);
  }
  /**
   * returns the input slot with a given name (used for dynamic slots), -1 if not found
   * @param name the name of the slot
   * @return the slot (-1 if not found)
   */
  findInputSlotIndexByName(e, t = !1, i) {
    if (!this.inputs)
      return -1;
    for (var n = 0, s = this.inputs.length; n < s; ++n)
      if (!(t && this.inputs[n].link && this.inputs[n].link != null) && !(i && i.includes(this.inputs[n].type)) && (!e || e == this.inputs[n].name))
        return n;
    return -1;
  }
  findInputSlotByName(e, t = !1, i) {
    if (!this.inputs)
      return null;
    for (var n = 0, s = this.inputs.length; n < s; ++n)
      if (!(t && this.inputs[n].link && this.inputs[n].link != null) && !(i && i.includes(this.inputs[n].type)) && (!e || e == this.inputs[n].name))
        return this.inputs[n];
    return null;
  }
  /**
   * returns the output slot with a given name (used for dynamic slots), -1 if not found
   * @param name the name of the slot
   * @return  the slot (-1 if not found)
   */
  findOutputSlotIndexByName(e, t = !1, i) {
    if (!this.outputs)
      return -1;
    for (var n = 0, s = this.outputs.length; n < s; ++n)
      if (!(t && this.outputs[n].links && this.outputs[n].links != null) && !(i && i.includes(this.outputs[n].type)) && (!e || e == this.outputs[n].name))
        return n;
    return -1;
  }
  findOutputSlotByName(e, t = !1, i) {
    if (!this.outputs)
      return null;
    for (var n = 0, s = this.outputs.length; n < s; ++n)
      if (!(t && this.outputs[n].links && this.outputs[n].links != null) && !(i && i.includes(this.outputs[n].type)) && (!e || e == this.outputs[n].name))
        return this.outputs[n];
    return null;
  }
  /**
   * findSlotByType for INPUTS
   */
  findInputSlotIndexByType(e, t = !1, i = !1) {
    return this.findSlotByType(!0, e, !1, t, i);
  }
  /**
   * findSlotByType for OUTPUTS
   */
  findOutputSlotIndexByType(e, t = !1, i = !1) {
    return this.findSlotByType(!1, e, !1, t, i);
  }
  /**
   * findSlotByType for INPUTS
   */
  findInputSlotByType(e, t = !1, i = !1) {
    return this.findSlotByType(!0, e, !1, t, i);
  }
  /**
   * findSlotByType for OUTPUTS
   */
  findOutputSlotByType(e, t = !1, i = !1) {
    return this.findSlotByType(!1, e, !1, t, i);
  }
  /**
   * returns the output (or input) slot with a given type, -1 if not found
   * @method findSlotByType
   * @param {boolean} input uise inputs instead of outputs
   * @param {string} type the type of the slot
   * @param {boolean} preferFreeSlot if we want a free slot (if not found, will return the first of the type anyway)
   * @return {number_or_object} the slot (-1 if not found)
   */
  findSlotByType(e, t, i, n = !1, s = !1) {
    n = n || !1, s = s || !1;
    var r = e ? this.inputs : this.outputs;
    if (!r)
      return i ? null : -1;
    (t == "" || t == "*") && (t = 0);
    for (var o = 0, a = r.length; o < a; ++o) {
      var l = (t + "").toLowerCase().split(","), h = r[o].type == "0" || r[o].type == "*" ? "0" : r[o].type;
      let p = (h + "").toLowerCase().split(",");
      for (let f = 0; f < l.length; f++)
        for (let c = 0; c < p.length; c++)
          if (l[f] == "_event_" && (l[f] = I.EVENT), p[f] == "_event_" && (p[f] = I.EVENT), l[f] == "*" && (l[f] = I.DEFAULT), p[f] == "*" && (p[f] = I.DEFAULT), l[f] == p[c]) {
            let v = r[o];
            if (n && v.links && v.links !== null || v.link && v.link !== null)
              continue;
            return i ? v : o;
          }
    }
    if (n && !s)
      for (var o = 0, a = r.length; o < a; ++o) {
        var l = (t + "").toLowerCase().split(","), h = r[o].type == "0" || r[o].type == "*" ? "0" : r[o].type;
        let g = (h + "").toLowerCase().split(",");
        for (let d = 0; d < l.length; d++)
          for (let _ = 0; _ < g.length; _++)
            if (l[d] == "*" && (l[d] = I.DEFAULT), g[d] == "*" && (g[d] = I.DEFAULT), l[d] == g[_])
              return i ? r[o] : o;
      }
    return i ? null : -1;
  }
  /**
   * connect this node output to the input of another node BY TYPE
   * @method connectByType
   * @param {number_or_string} slot (could be the number of the slot or the string with the name of the slot)
   * @param {LGraphNode} node the target node
   * @param {string} target_type the input slot type of the target node
   * @return {Object} the link_info is created, otherwise null
   */
  connectByTypeInput(e, t, i, n = {}) {
    var s = {
      createEventInCase: !0,
      firstFreeIfOutputGeneralInCase: !0,
      generalTypeInCase: !0
    }, r = Object.assign(s, n);
    t && t.constructor === Number && (t = this.graph.getNodeById(t));
    let o = i;
    i === I.EVENT ? o = I.ACTION : i === I.ACTION && (o = I.EVENT);
    let a = t.findInputSlotIndexByType(o, !0);
    if (a >= 0 && a !== null)
      return u.debug && console.debug("CONNbyTYPE type " + i + " for " + a), this.connect(e, t, a);
    if (u.debug && console.log("type " + i + " not found or not free?"), r.createEventInCase && i == I.EVENT)
      return u.debug && console.debug("connect WILL CREATE THE onTrigger " + i + " to " + t), this.connect(e, t, -1);
    if (r.generalTypeInCase) {
      let l = t.findInputSlotIndexByType(I.DEFAULT, !0, !0);
      if (u.debug && console.debug("connect TO a general type (*, 0), if not found the specific type ", i, " to ", t, "RES_SLOT:", l), l >= 0)
        return this.connect(e, t, l);
    }
    if (r.firstFreeIfOutputGeneralInCase && (i == 0 || i == "*" || i == "")) {
      let l = t.findInputSlotIndexByName(null, !0, [I.EVENT]);
      if (u.debug && console.debug("connect TO TheFirstFREE ", i, " to ", t, "RES_SLOT:", l), l >= 0)
        return this.connect(e, t, l);
    }
    return u.debug && console.error("no way to connect type: ", i, " to targetNODE ", t), null;
  }
  /**
   * connect this node input to the output of another node BY TYPE
   * @method connectByType
   * @param {number_or_string} slot (could be the number of the slot or the string with the name of the slot)
   * @param {LGraphNode} node the target node
   * @param {string} target_type the output slot type of the target node
   * @return {Object} the link_info is created, otherwise null
   */
  connectByTypeOutput(e, t, i, n = {}) {
    var s = {
      createEventInCase: !0,
      firstFreeIfInputGeneralInCase: !0,
      generalTypeInCase: !0
    }, r = Object.assign(s, n);
    t && t.constructor === Number && (t = this.graph.getNodeById(t));
    let o = i;
    if (i === I.EVENT ? o = I.ACTION : i === I.ACTION && (o = I.EVENT), a = t.findOutputSlotIndexByType(o, !0), a >= 0 && a !== null)
      return console.debug("CONNbyTYPE OUT! type " + i + " for " + a + " to " + o), t.connect(a, this, e);
    if (r.generalTypeInCase) {
      var a = t.findOutputSlotIndexByType(0, !0, !0);
      if (a >= 0)
        return t.connect(a, this, e);
    }
    if ((r.createEventInCase && i == I.EVENT || i == I.ACTION) && u.do_add_triggers_slots) {
      var a = t.addOnExecutedOutput();
      return t.connect(a, this, e);
    }
    if (r.firstFreeIfInputGeneralInCase && (i == 0 || i == "*" || i == "")) {
      let l = t.findOutputSlotIndexByName(null, !0, [I.EVENT, I.ACTION]);
      if (l >= 0)
        return t.connect(l, this, e);
    }
    return console.error("no way to connect byOUT type: ", i, " to sourceNODE ", t), console.error("type OUT! " + i + " not found or not free?"), null;
  }
  /**
   * connect this node output to the input of another node
   * @param slot (could be the number of the slot or the string with the name of the slot)
   * @param  targetNode the target node
   * @param  targetSlot the input slot of the target node (could be the number of the slot or the string with the name of the slot, or -1 to connect a trigger)
   * @return {Object} the linkInfo is created, otherwise null
   */
  connect(e, t, i) {
    if (i = i || 0, !this.graph)
      throw new Error("Connect: Error, node doesn't belong to any graph. Nodes must be added first to a graph before connecting them.");
    if (typeof e == "string") {
      if (e = this.findOutputSlotIndexByName(e), e == -1)
        return u.debug && console.error("Connect: Error, no slot of name " + e), null;
    } else if (!this.outputs || e >= this.outputs.length)
      return u.debug && console.error("Connect: Error, slot number not found"), null;
    if (t && t.constructor === Number && (t = this.graph.getNodeById(t)), !t)
      throw "target node is null";
    if (t == this)
      return u.debug && console.error("Connect: Error, can't connect node to itself!"), null;
    if (!t.graph)
      throw new Error("Connect: Error, target node doesn't belong to any graph. Nodes must be added first to a graph before connecting them.");
    if (typeof i == "string") {
      if (i = t.findInputSlotIndexByName(i), i == -1)
        return u.debug && console.error(
          "Connect: Error, no slot of name " + i
        ), null;
    } else if (i === I.EVENT)
      if (u.do_add_triggers_slots)
        t.changeMode(Z.ON_TRIGGER), i = t.findInputSlotIndexByName("onTrigger");
      else
        return u.debug && console.error("Connect: Error, can't connect event target slot"), null;
    else if (!t.inputs || i >= t.inputs.length)
      return u.debug && console.error("Connect: Error, slot number not found"), null;
    var n = !1, s = t.inputs[i], r = null, o = this.outputs[e];
    if (!this.outputs[e])
      return u.debug && (console.warn("Connect: Invalid slot passed: " + e), console.warn(this.outputs)), null;
    if (t.onBeforeConnectInput && (i = t.onBeforeConnectInput(i)), i === -1 || i === null || !u.isValidConnection(o.type, s.type))
      return this.setDirtyCanvas(!1, !0), n && this.graph.connectionChange(this, r), console.warn("Connect: Invalid connection: ", i, o.type, s.type), null;
    if (u.debug && console.debug("valid connection", o.type, s.type), t.onConnectInput && t.onConnectInput(i, o.type, o, this, e) === !1)
      return u.debug && console.debug("onConnectInput blocked", o.type, s.type), null;
    if (this.onConnectOutput && this.onConnectOutput(e, s.type, s, t, i) === !1)
      return u.debug && console.debug("onConnectOutput blocked", o.type, s.type), null;
    if (t.inputs[i] && t.inputs[i].link != null && (this.graph.beforeChange(), t.disconnectInput(i, { doProcessChange: !1 }), n = !0), o.links !== null && o.links.length)
      switch (o.type) {
        case I.EVENT:
          u.allow_multi_output_for_events || (this.graph.beforeChange(), this.disconnectOutput(e, null, { doProcessChange: !1 }), n = !0);
          break;
      }
    let a;
    return u.use_uuids ? a = oe() : a = ++this.graph.last_link_id, r = new he(
      a,
      s.type || o.type,
      this.id,
      e,
      t.id,
      i
    ), this.graph.links[r.id] && console.error("Link already exists in graph!", r.id, r, this.graph.links[r.id]), this.graph.links[r.id] = r, o.links == null && (o.links = []), o.links.push(r.id), t.inputs[i].link = r.id, this.graph && this.graph._version++, this.onConnectionsChange && this.onConnectionsChange(
      W.OUTPUT,
      e,
      !0,
      r,
      o
    ), t.onConnectionsChange && t.onConnectionsChange(
      W.INPUT,
      i,
      !0,
      r,
      s
    ), this.graph && this.graph.onNodeConnectionChange && (this.graph.onNodeConnectionChange(
      W.INPUT,
      t,
      i,
      this,
      e
    ), this.graph.onNodeConnectionChange(
      W.OUTPUT,
      this,
      e,
      t,
      i
    )), this.setDirtyCanvas(!1, !0), this.graph.afterChange(), this.graph.connectionChange(this, r), r;
  }
  /**
   * disconnect one output to an specific node
   * @param slot (could be the number of the slot or the string with the name of the slot)
   * @param targetNode the target node to which this slot is connected [Optional, if not targetNode is specified all nodes will be disconnected]
   * @return if it was disconnected successfully
   */
  disconnectOutput(e, t, i) {
    if (typeof e == "string") {
      if (e = this.findOutputSlotIndexByName(e), e == -1)
        return u.debug && console.error("Connect: Error, no slot of name " + e), !1;
    } else if (!this.outputs || e >= this.outputs.length)
      return u.debug && console.error("Connect: Error, slot number not found"), !1;
    var n = this.outputs[e];
    if (!n || !n.links || n.links.length == 0)
      return !1;
    if (t) {
      if (t.constructor === Number && (t = this.graph.getNodeById(t)), !t)
        throw "Target Node not found";
      for (var s = 0, r = n.links.length; s < r; s++) {
        var o = n.links[s], a = this.graph.links[o];
        if (a.target_id == t.id) {
          n.links.splice(s, 1);
          var l = t.inputs[a.target_slot];
          l.link = null, delete this.graph.links[o], this.graph && this.graph._version++, t.onConnectionsChange && t.onConnectionsChange(
            W.INPUT,
            a.target_slot,
            !1,
            a,
            l
          ), this.onConnectionsChange && this.onConnectionsChange(
            W.OUTPUT,
            e,
            !1,
            a,
            n
          ), this.graph && this.graph.onNodeConnectionChange && this.graph.onNodeConnectionChange(
            W.OUTPUT,
            this,
            e
          ), this.graph && this.graph.onNodeConnectionChange && (this.graph.onNodeConnectionChange(
            W.OUTPUT,
            this,
            e
          ), this.graph.onNodeConnectionChange(
            W.INPUT,
            t,
            a.target_slot
          ));
          break;
        }
      }
    } else {
      for (var s = 0, r = n.links.length; s < r; s++) {
        var o = n.links[s], a = this.graph.links[o];
        if (a) {
          var t = this.graph.getNodeById(a.target_id), l = null;
          this.graph && this.graph._version++, t && (l = t.inputs[a.target_slot], l.link = null, t.onConnectionsChange && t.onConnectionsChange(
            W.INPUT,
            a.target_slot,
            !1,
            a,
            l
          ), this.graph && this.graph.onNodeConnectionChange && this.graph.onNodeConnectionChange(
            W.INPUT,
            t,
            a.target_slot
          )), delete this.graph.links[o], this.onConnectionsChange && this.onConnectionsChange(
            W.OUTPUT,
            e,
            !1,
            a,
            n
          ), this.graph && this.graph.onNodeConnectionChange && (this.graph.onNodeConnectionChange(
            W.OUTPUT,
            this,
            e
          ), this.graph.onNodeConnectionChange(
            W.INPUT,
            t,
            a.target_slot
          ));
        }
      }
      n.links = null;
    }
    return this.setDirtyCanvas(!1, !0), this.graph.connectionChange(this), !0;
  }
  /**
   * disconnect one input
   * @param slot (could be the number of the slot or the string with the name of the slot)
   * @return if it was disconnected successfully
   */
  disconnectInput(e, t = {}) {
    if (typeof e == "string") {
      if (e = this.findInputSlotIndexByName(e), e == -1)
        return u.debug && console.error("Connect: Error, no slot of name " + e), !1;
    } else if (!this.inputs || e >= this.inputs.length)
      return u.debug && console.error("Connect: Error, slot number not found"), !1;
    var i = this.inputs[e];
    if (!i)
      return !1;
    var n = this.inputs[e].link;
    if (n != null) {
      this.inputs[e].link = null;
      var s = this.graph.links[n];
      if (s) {
        var r = this.graph.getNodeById(s.origin_id);
        if (!r)
          return !1;
        var o = r.outputs[s.origin_slot];
        if (!o || !o.links || o.links.length == 0)
          return !1;
        for (var a = 0, l = o.links.length; a < l; a++)
          if (o.links[a] == n) {
            o.links.splice(a, 1);
            break;
          }
        delete this.graph.links[n], this.graph && this.graph._version++, this.onConnectionsChange && this.onConnectionsChange(
          W.INPUT,
          e,
          !1,
          s,
          i
        ), r.onConnectionsChange && r.onConnectionsChange(
          W.OUTPUT,
          a,
          !1,
          s,
          o
        ), this.graph && this.graph.onNodeConnectionChange && (this.graph.onNodeConnectionChange(
          W.OUTPUT,
          r,
          a
        ), this.graph.onNodeConnectionChange(W.INPUT, this, e));
      }
    }
    return this.setDirtyCanvas(!1, !0), this.graph && this.graph.connectionChange(this), !0;
  }
  /**
   * returns the center of a connection point in canvas coords
   * @param is_input true if if a input slot, false if it is an output
   * @param slot (could be the number of the slot or the string with the name of the slot)
   * @param out a place to store the output, to free garbage
   * @return the position
   **/
  getConnectionPos(e, t, i = [0, 0], n = !1) {
    var s = 0;
    e && this.inputs && (s = this.inputs.length), !e && this.outputs && (s = this.outputs.length);
    var r = u.NODE_SLOT_HEIGHT * 0.5;
    if (this.flags.collapsed && !n) {
      var o = this._collapsed_width || u.NODE_COLLAPSED_WIDTH;
      return this.horizontal ? (i[0] = this.pos[0] + o * 0.5, e ? i[1] = this.pos[1] - u.NODE_TITLE_HEIGHT : i[1] = this.pos[1]) : (e ? i[0] = this.pos[0] : i[0] = this.pos[0] + o, i[1] = this.pos[1] - u.NODE_TITLE_HEIGHT * 0.5), i;
    }
    return e && t == -1 ? (i[0] = this.pos[0] + u.NODE_TITLE_HEIGHT * 0.5, i[1] = this.pos[1] + u.NODE_TITLE_HEIGHT * 0.5, i) : e && s > t && this.inputs[t].pos ? (i[0] = this.pos[0] + this.inputs[t].pos[0], i[1] = this.pos[1] + this.inputs[t].pos[1], i) : !e && s > t && this.outputs[t].pos ? (i[0] = this.pos[0] + this.outputs[t].pos[0], i[1] = this.pos[1] + this.outputs[t].pos[1], i) : this.horizontal ? (i[0] = this.pos[0] + (t + 0.5) * (this.size[0] / s), e ? i[1] = this.pos[1] - u.NODE_TITLE_HEIGHT : i[1] = this.pos[1] + this.size[1], i) : (e ? i[0] = this.pos[0] + r : i[0] = this.pos[0] + this.size[0] + 1 - r, i[1] = this.pos[1] + (t + 0.7) * u.NODE_SLOT_HEIGHT + (this.constructor.slot_start_y || 0), i);
  }
  /** Force align to grid */
  alignToGrid() {
    this.pos[0] = u.CANVAS_GRID_SIZE * Math.round(this.pos[0] / u.CANVAS_GRID_SIZE), this.pos[1] = u.CANVAS_GRID_SIZE * Math.round(this.pos[1] / u.CANVAS_GRID_SIZE);
  }
  /** Console output */
  trace(e) {
    this.console || (this.console = []), this.console.push(e), this.console.length > Ne.MAX_CONSOLE && this.console.shift(), this.graph.onNodeTrace && this.graph.onNodeTrace(this, e);
  }
  /** Forces to redraw or the main canvas (LGraphNode) or the bg canvas (links) */
  setDirtyCanvas(e, t = !1) {
    this.graph && this.graph.sendActionToCanvas("setDirty", [e, t]);
  }
  loadImage(e) {
    var t = new Image();
    t.src = u.node_images_path + e;
    var i = this;
    return t.onload = function() {
      i.setDirtyCanvas(!0);
    }, t;
  }
  /** Allows to get onMouseMove and onMouseUp events even if the mouse is out of focus */
  captureInput(e) {
    if (!(!this.graph || !this.graph.list_of_graphcanvas))
      for (var t = this.graph.list_of_graphcanvas, i = 0; i < t.length; ++i) {
        var n = t[i];
        !e && n.node_capturing_input != this || (n.node_capturing_input = e ? this : null);
      }
  }
  isShowingTitle(e) {
    return this.titleMode == se.TRANSPARENT_TITLE || this.titleMode == se.NO_TITLE ? !1 : (this.titleMode == se.AUTOHIDE_TITLE && e, !0);
  }
  /** Collapse the node to make it smaller on the canvas */
  collapse(e = !1) {
    this.graph._version++, !(this.collapsable === !1 && !e) && (this.flags.collapsed ? this.flags.collapsed = !1 : this.flags.collapsed = !0, this.setDirtyCanvas(!0, !0));
  }
  /** Forces the node to do not move or realign on Z */
  pin(e) {
    this.graph._version++, e === void 0 ? this.flags.pinned = !this.flags.pinned : this.flags.pinned = e;
  }
  localToScreen(e, t, i) {
    return [
      (e + this.pos[0]) * i.ds.scale + i.ds.offset[0],
      (t + this.pos[1]) * i.ds.scale + i.ds.offset[1]
    ];
  }
  getOptionalSlots() {
    return ye(this, "optionalSlots");
  }
};
let ae = Ne;
ae.MAX_CONSOLE = 100;
function we() {
  let e = [];
  return e = e.concat(Ae), e = e.concat([I.ACTION]), e = e.concat(u.slot_types_in.map((t) => t.toUpperCase())), e;
}
function We() {
  return we().map(J);
}
class $ extends ae {
  constructor(t) {
    super(t), this.properties = {
      name: "",
      type: "number",
      value: 0,
      subgraphID: null
    }, this.nameInGraph = "", this.clonable = !1, this.size = [180, 90];
    let i = this;
    this.nameWidget = this.addWidget(
      "text",
      "Name",
      this.properties.name,
      this.setName.bind(this)
    ), u.graph_inputs_outputs_use_combo_widget ? this.typeWidget = this.addWidget(
      "combo",
      "Type",
      J(this.properties.type),
      this.setType.bind(this),
      { values: We }
    ) : this.typeWidget = this.addWidget(
      "text",
      "Type",
      J(this.properties.type),
      this.setType.bind(this)
    ), this.valueWidget = this.addWidget(
      "number",
      "Value",
      this.properties.value,
      function(n) {
        i.setProperty("value", n);
      }
    ), this.widgets_up = !0;
  }
  setName(t) {
    if (t == null || t === this.properties.name)
      return;
    const i = this.getParentSubgraph();
    i && (t = i.getValidGraphInputName(t), this.setProperty("name", t));
  }
  setType(t) {
    t || (t = "*");
    let i = t;
    t === "-1" || t === "Action" ? i = I.ACTION : t === "-2" || t === "Event" ? i = I.EVENT : t === "0" && (i = "*"), this.setProperty("type", i);
  }
  onConfigure() {
    this.updateType();
  }
  getParentSubgraph() {
    var t, i;
    return (i = (t = this.graph._subgraph_node) == null ? void 0 : t.graph) == null ? void 0 : i.getNodeById(this.properties.subgraphID);
  }
  /** ensures the type in the node output and the type in the associated graph input are the same */
  updateType() {
    var t = this.properties.type;
    this.typeWidget.value = J(t);
    const i = this.outputs[0];
    i.type != t && (u.isValidConnection(i.type, t) || this.disconnectOutput(0), i.type = t), t == "array" ? i.shape = k.GRID_SHAPE : t === I.EVENT || t === I.ACTION ? i.shape = k.BOX_SHAPE : i.shape = k.DEFAULT, t == "number" ? (this.valueWidget.type = "number", this.valueWidget.value = 0) : t == "boolean" ? (this.valueWidget.type = "toggle", this.valueWidget.value = !0) : t == "string" ? (this.valueWidget.type = "text", this.valueWidget.value = "") : (this.valueWidget.type = null, this.valueWidget.value = null), this.properties.value = this.valueWidget.value, this.graph && this.nameInGraph && Se(t) ? (this.graph.changeInputType(this.nameInGraph, t), i.type !== t && this.setOutputDataType(0, t)) : console.error("[GraphInput] Can't change output to type", t, this.graph, this.nameInGraph);
  }
  /** this is executed AFTER the property has changed */
  onPropertyChanged(t, i) {
    if (t == "name") {
      if (i == "" || i == this.nameInGraph || i == "enabled")
        return !1;
      this.graph && (this.nameInGraph ? this.graph.renameInput(this.nameInGraph, i) : this.graph.addInput(i, "" + this.properties.type, null)), this.nameWidget.value = i, this.nameInGraph = i;
    } else
      t == "type" && this.updateType();
  }
  getTitle() {
    return this.flags.collapsed ? this.properties.name : this.title;
  }
  onAction(t, i) {
    this.properties.type == I.EVENT && this.triggerSlot(0, i);
  }
  onExecute() {
    var t = this.properties.name, i = this.graph.inputs[t];
    if (!i) {
      this.setOutputData(0, this.properties.value);
      return;
    }
    this.setOutputData(0, i.value !== void 0 ? i.value : this.properties.value);
  }
  onRemoved() {
    this.nameInGraph && this.graph.removeInput(this.nameInGraph);
  }
}
$.slotLayout = {
  inputs: [],
  outputs: [
    { name: "", type: "number" }
  ]
};
u.registerNodeType({
  class: $,
  title: "Input",
  desc: "Input of the graph",
  type: "graph/input",
  hide_in_node_lists: !0
});
function Le() {
  let e = [];
  return e = e.concat(Ae), e = e.concat([I.EVENT]), e = e.concat(u.slot_types_out), e;
}
function Ye() {
  return Le().map(J);
}
class Q extends ae {
  constructor(t) {
    super(t), this.properties = {
      name: "",
      type: "number",
      subgraphID: null
    }, this.nameInGraph = "", this.clonable = !1, this.size = [180, 60], this.nameWidget = this.addWidget(
      "text",
      "Name",
      this.properties.name,
      this.setName.bind(this)
    ), u.graph_inputs_outputs_use_combo_widget ? this.typeWidget = this.addWidget(
      "combo",
      "Type",
      J(this.properties.type),
      this.setType.bind(this),
      { values: Ye }
    ) : this.typeWidget = this.addWidget(
      "text",
      "Type",
      J(this.properties.type),
      this.setType.bind(this)
    ), this.widgets_up = !0;
  }
  setName(t) {
    if (t == null || t === this.properties.name)
      return;
    const i = this.getParentSubgraph();
    i && (t = i.getValidGraphOutputName(t), this.setProperty("name", t));
  }
  setType(t) {
    t || (t = "*");
    let i = t;
    t === "-1" || t === "Action" ? i = I.ACTION : t === "-2" || t === "Event" ? i = I.EVENT : t === "0" && (i = "*"), this.setProperty("type", i);
  }
  onConfigure() {
    this.updateType();
  }
  getParentSubgraph() {
    var t, i;
    return (i = (t = this.graph._subgraph_node) == null ? void 0 : t.graph) == null ? void 0 : i.getNodeById(this.properties.subgraphID);
  }
  updateType() {
    var t = this.properties.type;
    const i = this.inputs[0];
    this.typeWidget && (this.typeWidget.value = J(t)), t == "array" ? i.shape = k.GRID_SHAPE : t === I.EVENT || t === I.ACTION ? i.shape = k.BOX_SHAPE : i.shape = k.DEFAULT, i.type != t && ((t == "action" || t == "event") && (t = I.EVENT), u.isValidConnection(i.type, t) || this.disconnectInput(0), i.type = t), this.graph && this.nameInGraph && Se(t) ? (this.graph.changeOutputType(this.nameInGraph, t), i.type !== t && this.setInputDataType(0, t)) : console.error("Can't change GraphOutput to type", t, this.graph, this.nameInGraph);
  }
  /** this is executed AFTER the property has changed */
  onPropertyChanged(t, i) {
    if (t == "name") {
      if (i == "" || i == this.nameInGraph || i == "enabled")
        return !1;
      this.graph ? this.nameInGraph ? this.graph.renameOutput(this.nameInGraph, i) : this.graph.addOutput(i, "" + this.properties.type, null) : console.error("[GraphOutput] missing graph!", t, i), this.nameWidget.value = i, this.nameInGraph = i;
    } else
      t == "type" && this.updateType();
  }
  getTitle() {
    return this.flags.collapsed ? this.properties.name : this.title;
  }
  onAction(t, i, n) {
    const s = this.getParentSubgraph();
    if (!s)
      return;
    const r = s.findOutputSlotIndexByName(this.properties.name);
    r == null || s.outputs[r] == null || s.triggerSlot(r, i);
  }
  onExecute() {
    const t = this.getInputData(0);
    this.graph.setOutputData(this.properties.name, t);
  }
  onRemoved() {
    this.nameInGraph && this.graph.removeOutput(this.nameInGraph);
  }
}
Q.slotLayout = {
  inputs: [
    { name: "", type: "" }
  ],
  outputs: []
};
u.registerNodeType({
  class: Q,
  title: "Output",
  desc: "Output of the graph",
  type: "graph/output",
  hide_in_node_lists: !0
});
var xe = /* @__PURE__ */ ((e) => (e[e.STATUS_STOPPED = 1] = "STATUS_STOPPED", e[e.STATUS_RUNNING = 2] = "STATUS_RUNNING", e))(xe || {});
const De = class {
  constructor(e) {
    this.supported_types = null, this.vars = {}, this.extra = {}, this.inputs = {}, this.outputs = {}, this.links = {}, this.list_of_graphcanvas = [], this._nodes = [], this._groups = [], this._nodes_by_id = {}, this._nodes_executable = null, this._nodes_in_order = [], this._version = -1, this._last_trigger_time = 0, this._is_subgraph = !1, this._subgraph_node = null, this.nodes_executing = [], this.nodes_actioning = [], this.nodes_executedAction = [], this.execution_timer_id = -1, this.execution_time = 0, this.errors_in_execution = !1, u.debug && console.log("Graph created"), this.list_of_graphcanvas = null, this.clear(), e && this.configure(e);
  }
  getSupportedTypes() {
    return this.supported_types || De.DEFAULT_SUPPORTED_TYPES;
  }
  /*
   * Gets the root graph above any subgraphs.
   */
  getRootGraph() {
    const e = Array.from(this.iterateParentGraphs()), t = e[e.length - 1];
    return t._is_subgraph ? null : t;
  }
  *iterateParentGraphs() {
    var t;
    let e = this;
    for (; e; )
      yield e, e = (t = e._subgraph_node) == null ? void 0 : t.graph;
  }
  /** Removes all nodes from this graph */
  clear() {
    if (this.stop(), this.status = 1, this.last_node_id = 0, this.last_link_id = 0, this._version = -1, this._nodes)
      for (var e = 0; e < this._nodes.length; ++e) {
        var t = this._nodes[e];
        t.onRemoved && t.onRemoved();
      }
    this._nodes = [], this._nodes_by_id = {}, this._nodes_in_order = [], this._nodes_executable = null, this._groups = [], this.links = {}, this.iteration = 0, this.config = {}, this.vars = {}, this.extra = {}, this.globaltime = 0, this.runningtime = 0, this.fixedtime = 0, this.fixedtime_lapse = 0.01, this.elapsed_time = 0.01, this.last_update_time = 0, this.starttime = 0, this.catch_errors = !0, this.nodes_executing = [], this.nodes_actioning = [], this.nodes_executedAction = [], this.inputs = {}, this.outputs = {}, this.change(), this.sendActionToCanvas("clear");
  }
  /** Attach Canvas to this graph */
  attachCanvas(e) {
    if (!(e instanceof N))
      throw "attachCanvas expects a LGraphCanvas instance";
    e.graph && e.graph != this && e.graph.detachCanvas(e), e.graph = this, this.list_of_graphcanvas || (this.list_of_graphcanvas = []), this.list_of_graphcanvas.push(e);
  }
  /** Detach Canvas to this graph */
  detachCanvas(e) {
    if (this.list_of_graphcanvas) {
      var t = this.list_of_graphcanvas.indexOf(e);
      t != -1 && (e.graph = null, this.list_of_graphcanvas.splice(t, 1));
    }
  }
  /**
   * Starts running this graph every interval milliseconds.
   * @param interval amount of milliseconds between executions, if 0 then it renders to the monitor refresh rate
   */
  start(e) {
    if (this.status != 2) {
      this.status = 2, this.onPlayEvent && this.onPlayEvent(), this.sendEventToAllNodes("onStart"), this.starttime = u.getTime(), this.last_update_time = this.starttime, e = e || 0;
      var t = this;
      if (e == 0 && typeof window < "u" && window.requestAnimationFrame) {
        let i = function() {
          t.execution_timer_id == -1 && (window.requestAnimationFrame(i), t.onBeforeStep && t.onBeforeStep(), t.runStep(1, !t.catch_errors), t.onAfterStep && t.onAfterStep());
        };
        this.execution_timer_id = -1, i();
      } else
        this.execution_timer_id = setInterval(function() {
          t.onBeforeStep && t.onBeforeStep(), t.runStep(1, !t.catch_errors), t.onAfterStep && t.onAfterStep();
        }, e);
    }
  }
  /** Stops the execution loop of the graph */
  stop() {
    this.status != 1 && (this.status = 1, this.onStopEvent && this.onStopEvent(), this.execution_timer_id != null && (this.execution_timer_id != -1 && clearInterval(this.execution_timer_id), this.execution_timer_id = null), this.sendEventToAllNodes("onStop"));
  }
  /**
   * Run N steps (cycles) of the graph
   * @param num number of steps to run, default is 1
   * @param do_not_catch_errors if you want to try/catch errors
   */
  runStep(e = 1, t = !1, i) {
    var n = u.getTime();
    this.globaltime = 1e-3 * (n - this.starttime);
    let s = this._nodes_executable ? this._nodes_executable : this._nodes;
    if (s) {
      if (i = i || s.length, t) {
        for (var r = 0; r < e; r++) {
          for (var o = 0; o < i; ++o) {
            var a = s[o];
            a.mode == Z.ALWAYS && a.onExecute && a.doExecute();
          }
          this.fixedtime += this.fixedtime_lapse, this.onExecuteStep && this.onExecuteStep();
        }
        this.onAfterExecute && this.onAfterExecute();
      } else
        try {
          for (var r = 0; r < e; r++) {
            for (var o = 0; o < i; ++o) {
              var a = s[o];
              a.mode == Z.ALWAYS && a.onExecute && a.onExecute(null, {});
            }
            this.fixedtime += this.fixedtime_lapse, this.onExecuteStep && this.onExecuteStep();
          }
          this.onAfterExecute && this.onAfterExecute(), this.errors_in_execution = !1;
        } catch (p) {
          if (this.errors_in_execution = !0, u.throw_errors)
            throw p;
          u.debug && console.log("Error during execution: " + p), this.stop();
        }
      var l = u.getTime(), h = l - n;
      h == 0 && (h = 1), this.execution_time = 1e-3 * h, this.globaltime += 1e-3 * h, this.iteration += 1, this.elapsed_time = (l - this.last_update_time) * 1e-3, this.last_update_time = l, this.nodes_executing = [], this.nodes_actioning = [], this.nodes_executedAction = [];
    }
  }
  /**
   * Updates the graph execution order according to relevance of the nodes (nodes with only outputs have more relevance than
   * nodes with only inputs.
   */
  updateExecutionOrder() {
    this._nodes_in_order = this.computeExecutionOrder(!1), this._nodes_executable = [];
    for (var e = 0; e < this._nodes_in_order.length; ++e)
      if (this._nodes_in_order[e].onExecute) {
        let t = this._nodes_in_order[e];
        this._nodes_executable.push(t);
      }
  }
  *computeExecutionOrderRecursive(e = !1, t) {
    for (const i of this.computeExecutionOrder(e, t))
      if (yield i, i.is(ne))
        for (const n of i.subgraph.computeExecutionOrderRecursive(e, t))
          yield n;
  }
  /** This is more internal, it computes the executable nodes in order and returns it */
  computeExecutionOrder(e = !1, t) {
    for (var i = [], n = [], s = {}, r = {}, o = {}, a = 0, _ = this._nodes.length; a < _; ++a) {
      var l = this._nodes[a];
      if (!(e && !l.onExecute)) {
        s[l.id] = l;
        var h = 0;
        if (l.inputs)
          for (var p = 0, f = l.inputs.length; p < f; p++)
            l.inputs[p] && l.inputs[p].link != null && (h += 1);
        h == 0 ? (n.push(l), t && (l._level = 1)) : (t && (l._level = 0), o[l.id] = h);
      }
    }
    for (; n.length != 0; ) {
      let y = n.shift();
      if (i.push(y), delete s[y.id], !!y.outputs)
        for (var a = 0; a < y.outputs.length; a++) {
          var c = y.outputs[a];
          if (!(c == null || c.links == null || c.links.length == 0))
            for (var p = 0; p < c.links.length; p++) {
              var v = c.links[p], g = this.links[v];
              if (g && !r[g.id]) {
                var d = this.getNodeById(g.target_id);
                if (d == null) {
                  r[g.id] = !0;
                  continue;
                }
                t && (!d._level || d._level <= y._level) && (d._level = y._level + 1), r[g.id] = !0, o[d.id] -= 1, o[d.id] == 0 && n.push(d);
              }
            }
        }
    }
    for (let y of Object.keys(s).sort())
      i.push(s[y]);
    i.length != this._nodes.length && u.debug && console.warn("something went wrong, nodes missing");
    for (var _ = i.length, a = 0; a < _; ++a)
      i[a].order = a;
    i = i.sort(function(y, b) {
      var m = y.constructor.priority || y.priority || 0, E = b.constructor.priority || b.priority || 0;
      return m == E ? y.order - b.order : m - E;
    });
    for (var a = 0; a < _; ++a)
      i[a].order = a;
    return i;
  }
  /**
   * Returns all the nodes that could affect this one (ancestors) by crawling all the inputs recursively.
   * It doesn't include the node itself
   * @return an array with all the LGraphNodes that affect this node, in order of execution
   */
  getAncestors(e) {
    for (var t = [], i = [e], n = {}; i.length; ) {
      var s = i.shift();
      if (s.inputs) {
        !n[s.id] && s != e && (n[s.id] = !0, t.push(s));
        for (var r = 0; r < s.inputs.length; ++r) {
          var o = s.getInputNode(r);
          o && t.indexOf(o) == -1 && i.push(o);
        }
      }
    }
    return t.sort(function(a, l) {
      return a.order - l.order;
    }), t;
  }
  /**
   * Positions every node in a more readable manner
   */
  arrange(e = 100, t = ue.HORIZONTAL_LAYOUT) {
    const i = this.computeExecutionOrder(!1, !0), n = [];
    for (let r = 0; r < i.length; ++r) {
      const o = i[r], a = o._level || 1;
      n[a] || (n[a] = []), n[a].push(o);
    }
    let s = e;
    for (let r = 0; r < n.length; ++r) {
      const o = n[r];
      if (!o)
        continue;
      let a = 100, l = e + u.NODE_TITLE_HEIGHT;
      for (let h = 0; h < o.length; ++h) {
        const p = o[h];
        p.pos[0] = t == ue.VERTICAL_LAYOUT ? l : s, p.pos[1] = t == ue.VERTICAL_LAYOUT ? s : l;
        const f = t == ue.VERTICAL_LAYOUT ? 1 : 0;
        p.size[f] > a && (a = p.size[f]);
        const c = t == ue.VERTICAL_LAYOUT ? 0 : 1;
        l += p.size[c] + e + u.NODE_TITLE_HEIGHT;
      }
      s += a + e;
    }
    this.setDirtyCanvas(!0, !0);
  }
  /**
   * Returns the amount of time the graph has been running in milliseconds
   * @return number of milliseconds the graph has been running
   */
  getTime() {
    return this.globaltime;
  }
  /**
   * Returns the amount of time accumulated using the fixedtime_lapse var. This is used in context where the time increments should be constant
   * @return number of milliseconds the graph has been running
   */
  getFixedTime() {
    return this.fixedtime;
  }
  /**
   * Returns the amount of time it took to compute the latest iteration. Take into account that this number could be not correct
   * if the nodes are using graphical actions
   * @return number of milliseconds it took the last cycle
   */
  getElapsedTime() {
    return this.elapsed_time;
  }
  /**
   * Iterates all nodes in this graph *excluding* subgraphs.
   */
  *iterateNodesInOrder() {
    const e = this._nodes_in_order ? this._nodes_in_order : this._nodes || [];
    for (const t of e)
      yield t;
  }
  /**
   * Iterates all nodes in this graph and subgraphs.
   */
  *iterateNodesInOrderRecursive() {
    const e = this._nodes_in_order ? this._nodes_in_order : this._nodes || [];
    for (const t of e)
      if (yield t, t.subgraph != null)
        for (const i of t.subgraph.iterateNodesInOrderRecursive())
          yield i;
  }
  /**
   * Iterates all nodes in this graph *excluding* subgraphs.
   */
  *iterateNodesOfClass(e) {
    const t = e.__LITEGRAPH_TYPE__;
    if (t != null)
      for (const i of this.iterateNodesInOrder())
        i.type === t && (yield i);
  }
  /**
   * Iterates all nodes in this graph *excluding* subgraphs.
   */
  *iterateNodesOfClassRecursive(e) {
    const t = e.__LITEGRAPH_TYPE__;
    if (t != null)
      for (const i of this.iterateNodesInOrderRecursive())
        i.type === t && (yield i);
  }
  /**
   * Iterates all nodes in this graph *excluding* subgraphs.
   */
  *iterateNodesOfTypeRecursive(e) {
    for (const t of this.iterateNodesInOrderRecursive())
      t.type === e && (yield t);
  }
  /**
   * Sends an event to all the nodes, useful to trigger stuff
   * @param eventName the name of the event (function to be called)
   * @param params parameters in array format
   */
  sendEventToAllNodes(e, t = [], i = Z.ALWAYS) {
    var n = this._nodes_in_order ? this._nodes_in_order : this._nodes;
    if (n)
      for (const s of this.iterateNodesInOrder()) {
        if (s.type === "basic/subgraph" && e != "onExecute") {
          s.mode == i && s.sendEventToAllNodes(e, t, i);
          continue;
        }
        !s[e] || s.mode != i || (t === void 0 ? s[e]() : t && t.constructor === Array ? s[e].apply(s, t) : s[e](t));
      }
  }
  sendActionToCanvas(e, t = []) {
    if (this.list_of_graphcanvas)
      for (var i = 0; i < this.list_of_graphcanvas.length; ++i) {
        var n = this.list_of_graphcanvas[i];
        n[e] && n[e].apply(n, t);
      }
  }
  addGroup(e) {
    return this._groups.push(e), this.setDirtyCanvas(!0), this.change(), e.graph = this, this._version++, e;
  }
  /**
   * Adds a new node instance to this graph
   * @param node the instance of the node
   */
  add(e, t = {}) {
    if (e.id != -1 && this._nodes_by_id[e.id] != null && (console.warn(
      "LiteGraph: there is already a node with this ID, changing it",
      e.id
    ), u.use_uuids ? e.id = oe() : e.id = ++this.last_node_id), t.pos && (isNaN(t.pos[0]) || isNaN(t.pos[1])))
      throw "LiteGraph: Node position contained NaN(s)!";
    if (this._nodes.length >= u.MAX_NUMBER_OF_NODES)
      throw "LiteGraph: max number of nodes in a graph reached";
    return u.use_uuids ? e.id || (e.id = oe()) : e.id == null || e.id == -1 ? e.id = ++this.last_node_id : this.last_node_id < e.id && (this.last_node_id = e.id), e.graph = this, this._version++, this._nodes.push(e), this._nodes_by_id[e.id] = e, t.pos && (e.pos = t.pos), e.onAdded && e.onAdded(this), this.config.align_to_grid && e.alignToGrid(), t.skipComputeOrder || this.updateExecutionOrder(), this.onNodeAdded && this.onNodeAdded(e, t), this.setDirtyCanvas(!0), this.change(), e;
  }
  /** Removes a node from the graph */
  remove(e, t = {}) {
    if (e instanceof me) {
      var i = this._groups.indexOf(e);
      i != -1 && this._groups.splice(i, 1), e.graph = null, this._version++, this.setDirtyCanvas(!0, !0), this.change();
      return;
    }
    if (this._nodes_by_id[e.id] != null && !e.ignore_remove) {
      if (this.beforeChange(), e.inputs)
        for (var n = 0; n < e.inputs.length; n++) {
          var s = e.inputs[n];
          s.link != null && e.disconnectInput(n);
        }
      if (e.outputs)
        for (var n = 0; n < e.outputs.length; n++) {
          let l = e.outputs[n];
          l.links != null && l.links.length && e.disconnectOutput(n);
        }
      if (e.onRemoved && e.onRemoved(t), e.graph = null, this._version++, this.list_of_graphcanvas)
        for (var n = 0; n < this.list_of_graphcanvas.length; ++n) {
          var r = this.list_of_graphcanvas[n];
          r.selected_nodes[e.id] && delete r.selected_nodes[e.id], r.node_dragged == e && (r.node_dragged = null);
        }
      var o = this._nodes.indexOf(e);
      o != -1 && this._nodes.splice(o, 1), delete this._nodes_by_id[e.id], this.onNodeRemoved && this.onNodeRemoved(e, t), this.sendActionToCanvas("checkPanels"), this.setDirtyCanvas(!0, !0), this.afterChange(), this.change(), this.updateExecutionOrder();
    }
  }
  /** Returns a node by its id. */
  getNodeById(e) {
    return e == null ? null : this._nodes_by_id[e];
  }
  /** Returns a node by its id. */
  getNodeByIdRecursive(e) {
    const t = this.getNodeById(e);
    if (t != null)
      return t;
    for (const i of this.iterateNodesOfClass(ne)) {
      const n = i.subgraph.getNodeByIdRecursive(e);
      if (n)
        return n;
    }
    return null;
  }
  /**
   * Returns a list of nodes that matches a class
   * @param classObject the class itself (not an string)
   * @return a list with all the nodes of this type
   */
  findNodesByClass(e, t = []) {
    t.length = 0;
    for (const i of this.iterateNodesOfClass(e))
      t.push(i);
    return t;
  }
  /**
   * Returns a list of nodes that matches a type
   * @param type the name of the node type
   * @return a list with all the nodes of this type
   */
  findNodesByType(i, t = []) {
    var i = i.toLowerCase();
    t.length = 0;
    for (var n = 0, s = this._nodes.length; n < s; ++n)
      this._nodes[n].type.toLowerCase() == i && t.push(this._nodes[n]);
    return t;
  }
  /**
   * Returns a list of nodes that matches a class
   * @param classObject the class itself (not an string)
   * @return a list with all the nodes of this type
   */
  findNodesByClassRecursive(e, t = []) {
    t.length = 0;
    for (const i of this.iterateNodesOfClassRecursive(e))
      t.push(i);
    return t;
  }
  /**
   * Returns a list of nodes that matches a type
   * @param type the name of the node type
   * @return a list with all the nodes of this type
   */
  findNodesByTypeRecursive(i, t = []) {
    var i = i.toLowerCase();
    t.length = 0;
    for (const n of this.iterateNodesOfTypeRecursive(i))
      t.push(n);
    return t;
  }
  /**
   * Returns the first node that matches a name in its title
   * @param title the name of the node to search
   * @return the node or null
   */
  findNodeByTitle(e) {
    for (var t = 0, i = this._nodes.length; t < i; ++t)
      if (this._nodes[t].title == e)
        return this._nodes[t];
    return null;
  }
  /**
   * Returns a list of nodes that matches a name
   * @param title the name of the node to search
   * @return a list with all the nodes with this name
   */
  findNodesByTitle(e) {
    for (var t = [], i = 0, n = this._nodes.length; i < n; ++i)
      this._nodes[i].title == e && t.push(this._nodes[i]);
    return t;
  }
  /**
   * Returns the top-most node in this position of the canvas
   * @param x the x coordinate in canvas space
   * @param y the y coordinate in canvas space
   * @param nodesList a list with all the nodes to search from, by default is all the nodes in the graph
   * @return the node at this position or null
   */
  getNodeOnPos(e, t, i, n) {
    i = i || this._nodes;
    for (var s = null, r = i.length - 1; r >= 0; r--) {
      var o = i[r], a = o.titleMode == se.NO_TITLE;
      if (o.isPointInside(e, t, n, a))
        return o;
    }
    return s;
  }
  /**
   * Returns the top-most group in that position
   * @param x the x coordinate in canvas space
   * @param y the y coordinate in canvas space
   * @return the group or null
   */
  getGroupOnPos(e, t) {
    for (var i = this._groups.length - 1; i >= 0; i--) {
      var n = this._groups[i];
      if (n.isPointInside(e, t, 2, !0))
        return n;
    }
    return null;
  }
  /**
   * Checks that the node type matches the node type registered, used when replacing a nodetype by a newer version during execution
   * this replaces the ones using the old version with the new version
   * @method checkNodeTypes
   */
  checkNodeTypes() {
    for (var e = !1, t = 0; t < this._nodes.length; t++) {
      var i = this._nodes[t], n = u.registered_node_types[i.type];
      if (i.constructor != n.class) {
        console.log("node being replaced by newer version: " + i.type);
        var s = u.createNode(i.type);
        e = !0, this._nodes[t] = s, s.configure(i.serialize()), s.graph = this, this._nodes_by_id[s.id] = s, i.inputs && (s.inputs = i.inputs.concat()), i.outputs && (s.outputs = i.outputs.concat());
      }
    }
    return this.updateExecutionOrder(), e;
  }
  // ********** GLOBALS *****************
  onAction(e, t, i = {}) {
    for (const n of this.iterateNodesOfClass($))
      if (n.properties.name == e) {
        n.actionDo(e, t, i);
        break;
      }
  }
  trigger(e, t) {
    this.onTrigger && this.onTrigger(e, t);
  }
  triggerSlot(e, t) {
    this.onTrigger && this.onTrigger(e, t);
  }
  /** Tell this graph it has a global graph input of this type */
  addInput(e, t, i) {
    var n = this.inputs[e];
    n || (this.beforeChange(), this.inputs[e] = { name: e, type: t, value: i }, this._version++, this.afterChange(), this.onInputAdded && this.onInputAdded(e, t, i), this.onInputsOutputsChange && this.onInputsOutputsChange());
  }
  /** Assign a data to the global graph input */
  setInputData(e, t) {
    var i = this.inputs[e];
    i && (i.value = t);
  }
  /** Returns the current value of a global graph input */
  getInputData(e) {
    var t = this.inputs[e];
    return t ? t.value : null;
  }
  /** Changes the name of a global graph input */
  renameInput(e, t) {
    if (t != e)
      return this.inputs[e] ? this.inputs[t] ? (console.error("there is already one input with that name"), !1) : (this.inputs[t] = this.inputs[e], delete this.inputs[e], this._version++, this.onInputRenamed && this.onInputRenamed(e, t), this.onInputsOutputsChange && this.onInputsOutputsChange(), !0) : !1;
  }
  /** Changes the type of a global graph input */
  changeInputType(e, t) {
    if (!this.inputs[e])
      return !1;
    if (this.inputs[e].type && String(this.inputs[e].type).toLowerCase() == String(t).toLowerCase())
      return;
    const i = this.inputs[e].type;
    return this.inputs[e].type = t, this._version++, this.onInputTypeChanged && this.onInputTypeChanged(e, i, t), !0;
  }
  /** Removes a global graph input */
  removeInput(e) {
    return this.inputs[e] ? (delete this.inputs[e], this._version++, this.onInputRemoved && this.onInputRemoved(e), this.onInputsOutputsChange && this.onInputsOutputsChange(), !0) : !1;
  }
  /** Creates a global graph output */
  addOutput(e, t, i) {
    this.outputs[e] = { name: e, type: t, value: i }, this._version++, this.onOutputAdded && this.onOutputAdded(e, t, i), this.onInputsOutputsChange && this.onInputsOutputsChange();
  }
  /** Assign a data to the global output */
  setOutputData(e, t) {
    var i = this.outputs[e];
    i && (i.value = t);
  }
  /** Returns the current value of a global graph output */
  getOutputData(e) {
    var t = this.outputs[e];
    return t ? t.value : null;
  }
  /** Renames a global graph output */
  renameOutput(e, t) {
    return this.outputs[e] ? this.outputs[t] ? (console.error("there is already one output with that name"), !1) : (this.outputs[t] = this.outputs[e], delete this.outputs[e], this._version++, this.onOutputRenamed && this.onOutputRenamed(e, t), this.onInputsOutputsChange && this.onInputsOutputsChange(), !0) : !1;
  }
  /** Changes the type of a global graph output */
  changeOutputType(e, t) {
    if (!this.outputs[e])
      return !1;
    if (this.outputs[e].type && String(this.outputs[e].type).toLowerCase() == String(t).toLowerCase())
      return;
    const i = this.outputs[e].type;
    return this.outputs[e].type = t, this._version++, this.onOutputTypeChanged && this.onOutputTypeChanged(e, i, t), !0;
  }
  /** Removes a global graph output */
  removeOutput(e) {
    return this.outputs[e] ? (delete this.outputs[e], this._version++, this.onOutputRemoved && this.onOutputRemoved(e), this.onInputsOutputsChange && this.onInputsOutputsChange(), !0) : !1;
  }
  /* TODO implement
      triggerInput(name: string, value: any): void {
          var nodes = this.findNodesByTitle(name);
          for (var i = 0; i < nodes.length; ++i) {
              nodes[i].onTrigger(value);
          }
      }
  
      setCallback(name: string, func: (...args: any[]) => any): void {
          var nodes = this.findNodesByTitle(name);
          for (var i = 0; i < nodes.length; ++i) {
              nodes[i].setTrigger(func);
          }
      }
      */
  /** used for undo, called before any change is made to the graph */
  beforeChange(e) {
    this.onBeforeChange && this.onBeforeChange(this, e), this.sendActionToCanvas("onBeforeChange", [this]);
  }
  /** used to resend actions, called after any change is made to the graph */
  afterChange(e) {
    this.onAfterChange && this.onAfterChange(this, e), this.sendActionToCanvas("onAfterChange", [this]);
  }
  connectionChange(e, t) {
    this.updateExecutionOrder(), this.onConnectionChange && this.onConnectionChange(e), this._version++, this.sendActionToCanvas("onConnectionChange");
  }
  /** returns if the graph is in live mode */
  isLive() {
    if (!this.list_of_graphcanvas)
      return !1;
    for (var e = 0; e < this.list_of_graphcanvas.length; ++e) {
      var t = this.list_of_graphcanvas[e];
      if (t.live_mode)
        return !0;
    }
    return !1;
  }
  /** clears the triggered slot animation in all links (stop visual animation) */
  clearTriggeredSlots() {
    for (var e in this.links) {
      var t = this.links[e];
      t && t._last_time && (t._last_time = 0);
    }
  }
  /* Called when something visually changed (not the graph!) */
  change() {
    u.debug && console.log("Graph changed"), this.sendActionToCanvas("setDirty", [!0, !0]), this.onChange && this.onChange(this);
  }
  setDirtyCanvas(e = !1, t = !1) {
    this.sendActionToCanvas("setDirty", [e, t]);
  }
  /** Destroys a link */
  removeLink(e) {
    var t = this.links[e];
    if (t) {
      var i = this.getNodeById(t.target_id);
      i && i.disconnectInput(t.target_slot);
    }
  }
  /** Creates a Object containing all the info about this graph, it can be serialized */
  serialize() {
    for (var e = [], t = 0, i = this._nodes.length; t < i; ++t)
      e.push(this._nodes[t].serialize());
    var n = [];
    for (const h in this.links) {
      var s = this.links[h];
      if (!s.serialize) {
        console.error(
          "weird LLink bug, link info is not a LLink but a regular object",
          s
        );
        var r = he.configure(s);
        for (var o in s)
          r[o] = s[o];
        this.links[h] = r, s = r;
      }
      n.push(s.serialize());
    }
    for (var a = [], t = 0; t < this._groups.length; ++t)
      a.push(this._groups[t].serialize());
    var l = {
      last_node_id: this.last_node_id,
      last_link_id: this.last_link_id,
      nodes: e,
      links: n,
      groups: a,
      config: this.config,
      extra: this.extra,
      version: u.VERSION
    };
    return this.onSerialize && this.onSerialize(l), l;
  }
  /**
   * Configure a graph from a JSON string
   * @param data configure a graph from a JSON string
   * @returns if there was any error parsing
   */
  configure(e, t) {
    if (e) {
      t || this.clear();
      var i = e.nodes;
      if (e.links && e.links.constructor === Array) {
        for (var n = [], s = 0; s < e.links.length; ++s) {
          var r = e.links[s];
          if (!r) {
            console.warn("serialized graph link data contains errors, skipping.");
            continue;
          }
          var o = he.configure(r);
          n[o.id] = o;
        }
        e.links = n;
      }
      for (const c in e)
        c == "nodes" || c == "groups" || (this[c] = e[c]);
      var a = !1;
      if (this._nodes = [], i) {
        for (var s = 0, l = i.length; s < l; ++s) {
          var h = i[s], p = u.createNode(h.type, h.title);
          p || (console.error(
            "Node not found or has errors: " + h.type
          ), p = new ae(), p.last_serialization = h, p.has_errors = !0, a = !0), p.id = h.id, this.add(p, { addedBy: "configure", skipComputeOrder: !0 });
        }
        for (var s = 0, l = i.length; s < l; ++s) {
          var h = i[s], p = this.getNodeById(h.id);
          p && p.configure(h);
        }
      }
      if (this._groups.length = 0, e.groups)
        for (var s = 0; s < e.groups.length; ++s) {
          var f = new me();
          f.configure(e.groups[s]), this.addGroup(f);
        }
      return this.updateExecutionOrder(), this.extra = e.extra || {}, this.onConfigure && this.onConfigure(e), this._version++, this.setDirtyCanvas(!0, !0), a;
    }
  }
  load(e, t) {
    var i = this;
    if (e.constructor === File || e.constructor === Blob) {
      var n = new FileReader();
      n.addEventListener("load", function(r) {
        var o = JSON.parse(n.result);
        i.configure(o), t && t(o);
      }), n.readAsText(e);
      return;
    }
    var s = new XMLHttpRequest();
    s.open("GET", e, !0), s.send(null), s.onload = function(r) {
      if (s.status !== 200) {
        console.error("Error loading graph:", s.status, s.response);
        return;
      }
      var o = JSON.parse(s.response);
      i.configure(o), t && t(o);
    }, s.onerror = function(r) {
      console.error("Error loading graph:", r);
    };
  }
};
let Pe = De;
Pe.DEFAULT_SUPPORTED_TYPES = ["number", "string", "boolean"];
function Re(e) {
  const t = { nodeIDs: {}, linkIDs: {} };
  for (const i of e.nodes) {
    const n = i.id, s = oe();
    if (i.id = s, t.nodeIDs[n] || t.nodeIDs[s])
      throw new Error(
        `New/old node UUID wasn't unique in changed map! ${n} ${s}`
      );
    t.nodeIDs[n] = s, t.nodeIDs[s] = n;
  }
  for (const i of e.links) {
    const n = i[0], s = oe();
    if (i[0] = s, t.linkIDs[n] || t.linkIDs[s])
      throw new Error(
        `New/old link UUID wasn't unique in changed map! ${n} ${s}`
      );
    t.linkIDs[n] = s, t.linkIDs[s] = n;
    const r = i[1], o = i[3];
    if (!t.nodeIDs[r])
      throw new Error(`Old node UUID not found in mapping! ${r}`);
    if (i[1] = t.nodeIDs[r], !t.nodeIDs[o])
      throw new Error(`Old node UUID not found in mapping! ${o}`);
    i[3] = t.nodeIDs[o];
  }
  for (const i of e.nodes) {
    for (const n of i.inputs)
      n.link && (n.link = t.linkIDs[n.link]);
    for (const n of i.outputs)
      n.links && (n.links = n.links.map((s) => t.linkIDs[s]));
  }
  for (const i of e.nodes)
    if (i.type === "graph/subgraph") {
      const n = Re(
        i.subgraph
      );
      t.nodeIDs = { ...t.nodeIDs, ...n.nodeIDs }, t.linkIDs = { ...t.linkIDs, ...n.linkIDs };
    }
  return t;
}
function Ve(e, t) {
  for (const i of e.iterateNodesInOrderRecursive())
    i.onReassignID && i.onReassignID(t);
}
const Me = class extends ae {
  constructor(e, t) {
    super(e), this.properties = {
      enabled: !0
    }, this.size = [140, 80], this.enabled = !0, this.subgraph = (t || Me.default_lgraph_factory)(), this.subgraph._subgraph_node = this, this.subgraph._is_subgraph = !0;
    const i = (n, s) => {
      const r = s.bind(this);
      return function(...o) {
        n == null || n.apply(this, o), r(...o);
      };
    };
    this.subgraph.onTrigger = i(
      this.subgraph.onTrigger,
      this.onSubgraphTrigger
    ), this.subgraph.onNodeAdded = i(
      this.subgraph.onNodeAdded,
      this.onSubgraphNodeAdded
    ), this.subgraph.onNodeRemoved = i(
      this.subgraph.onNodeRemoved,
      this.onSubgraphNodeRemoved
    ), this.subgraph.onInputAdded = i(
      this.subgraph.onInputAdded,
      this.onSubgraphNewInput
    ), this.subgraph.onInputRenamed = i(
      this.subgraph.onInputRenamed,
      this.onSubgraphRenamedInput
    ), this.subgraph.onInputTypeChanged = i(
      this.subgraph.onInputTypeChanged,
      this.onSubgraphTypeChangeInput
    ), this.subgraph.onInputRemoved = i(
      this.subgraph.onInputRemoved,
      this.onSubgraphRemovedInput
    ), this.subgraph.onOutputAdded = i(
      this.subgraph.onOutputAdded,
      this.onSubgraphNewOutput
    ), this.subgraph.onOutputRenamed = i(
      this.subgraph.onOutputRenamed,
      this.onSubgraphRenamedOutput
    ), this.subgraph.onOutputTypeChanged = i(
      this.subgraph.onOutputTypeChanged,
      this.onSubgraphTypeChangeOutput
    ), this.subgraph.onOutputRemoved = i(
      this.subgraph.onOutputRemoved,
      this.onSubgraphRemovedOutput
    );
  }
  // getRootGraph(): LGraph | null {
  //     const graphs = Array.from(this.iterateParentGraphs());
  //     const graph = graphs[graphs.length - 1]
  //     // console.warn(graph._is_subgraph)
  //     if (graph._is_subgraph)
  //         return null;
  //     return graph;
  // }
  *iterateParentGraphs() {
    var t;
    let e = this.graph;
    for (; e; )
      yield e, e = (t = e._subgraph_node) == null ? void 0 : t.graph;
  }
  onDblClick(e, t, i) {
    var n = this;
    setTimeout(function() {
      i.openSubgraph(n.subgraph);
    }, 10);
  }
  onAction(e, t, i) {
    const { originNode: n, link: s } = i;
    if (!n || !s)
      return;
    const r = s.target_slot;
    this.getInnerGraphInputByIndex(r).triggerSlot(0, t);
  }
  onExecute() {
    if (this.enabled = this.getInputOrProperty("enabled"), !!this.enabled) {
      if (this.inputs)
        for (var e = 0; e < this.inputs.length; e++) {
          var t = this.inputs[e], i = this.getInputData(e);
          this.subgraph.setInputData(t.name, i);
        }
      if (this.subgraph.runStep(), this.outputs)
        for (var e = 0; e < this.outputs.length; e++) {
          var n = this.outputs[e], i = this.subgraph.getOutputData(n.name);
          this.setOutputData(e, i);
        }
    }
  }
  sendEventToAllNodes(e, t, i) {
    this.enabled && this.subgraph.sendEventToAllNodes(e, t, i);
  }
  onDrawBackground(e, t, i, n) {
  }
  // override onMouseDown(e, localpos, graphcanvas)
  // {
  // 	var y = this.size[1] - LiteGraph.NODE_TITLE_HEIGHT + 0.5;
  // 	if(localpos[1] > y)
  // 	{
  // 		graphcanvas.showSubgraphPropertiesDialog(this);
  // 	}
  // }
  // override onMouseDown(e: MouseEventExt, localpos: Vector2, graphcanvas: LGraphCanvas): boolean | undefined {
  //     var y = this.size[1] - LiteGraph.NODE_TITLE_HEIGHT + 0.5;
  //     console.log(0)
  //     if (localpos[1] > y) {
  //         if (localpos[0] < this.size[0] / 2) {
  //             console.log(1)
  //             graphcanvas.showSubgraphPropertiesDialog(this);
  //         } else {
  //             console.log(2)
  //             graphcanvas.showSubgraphPropertiesDialogRight(this);
  //         }
  //     }
  //     return false;
  // }
  computeSize() {
    var e = this.inputs ? this.inputs.length : 0, t = this.outputs ? this.outputs.length : 0;
    return [
      200,
      Math.max(e, t) * u.NODE_SLOT_HEIGHT + u.NODE_SLOT_HEIGHT * 0.5
    ];
  }
  //**** INPUTS ***********************************
  onSubgraphTrigger(e, t) {
  }
  onSubgraphNodeAdded(e, t) {
    var i, n;
    (i = this.graph) != null && i.onNodeAdded && (t.subgraphs || (t.subgraphs = []), t.subgraphs.push(this), (n = this.graph) == null || n.onNodeAdded(e, t));
  }
  onSubgraphNodeRemoved(e, t) {
    var i, n;
    (i = this.graph) != null && i.onNodeRemoved && (t.subgraphs || (t.subgraphs = []), t.subgraphs.push(this), (n = this.graph) == null || n.onNodeRemoved(e, t));
  }
  onSubgraphNewInput(e, t) {
    var i = this.findInputSlotIndexByName(e);
    i == -1 && this.addInput(e, t);
  }
  onSubgraphRenamedInput(e, t) {
    var i = this.findInputSlotIndexByName(e);
    if (i != -1) {
      var n = this.getInputInfo(i);
      n.name = t;
    }
  }
  onSubgraphTypeChangeInput(e, t, i) {
    var n = this.findInputSlotIndexByName(e);
    if (n != -1) {
      var s = this.getInputInfo(n);
      s.type = i;
    }
  }
  onSubgraphRemovedInput(e) {
    var t = this.findInputSlotIndexByName(e);
    t != -1 && this.removeInput(t);
  }
  //**** OUTPUTS ***********************************
  onSubgraphNewOutput(e, t) {
    var i = this.findOutputSlotIndexByName(e);
    i == -1 && this.addOutput(e, t);
  }
  onSubgraphRenamedOutput(e, t) {
    var i = this.findOutputSlotIndexByName(e);
    if (i != -1) {
      var n = this.getOutputInfo(i);
      n.name = t;
    }
  }
  onSubgraphTypeChangeOutput(e, t, i) {
    var n = this.findOutputSlotIndexByName(e);
    if (n != -1) {
      var s = this.getOutputInfo(n);
      s.type = i;
    }
  }
  onSubgraphRemovedOutput(e) {
    var t = this.findOutputSlotIndexByName(e);
    t != -1 && this.removeOutput(t);
  }
  // *****************************************************
  getExtraMenuOptions(e, t) {
    var i = this;
    return [
      {
        content: "Open",
        callback: function() {
          e.openSubgraph(i.subgraph);
        }
      }
    ];
  }
  onResize(e) {
    console.error("TEST subgraph resize");
  }
  serialize() {
    var e = ae.prototype.serialize.call(this);
    return e.subgraph = this.subgraph.serialize(), e;
  }
  //no need to define node.configure, the default method detects node.subgraph and passes the object to node.subgraph.configure()
  onConfigure(e) {
    super.onConfigure && super.onConfigure(e), this.subgraph._is_subgraph = !0, this.subgraph._subgraph_node = this;
    for (const t of this.subgraph.iterateNodesInOrder())
      (t.is($) || t.is(Q)) && (t.properties.subgraphID = this.id);
  }
  onReassignID() {
    for (const e of this.subgraph.iterateNodesInOrder())
      (e.is($) || e.is(Q)) && (e.properties.subgraphID = this.id);
  }
  clone(e = { forNode: {} }) {
    var s, r, o, a;
    var t = u.createNode(this.type), i = this.serialize();
    let n = null;
    if (u.use_uuids) {
      const l = u.cloneObject(i.subgraph);
      n = Re(l), i.subgraph = l;
    }
    return delete i.id, delete i.inputs, delete i.outputs, t.configure(i), u.use_uuids && Ve(t.subgraph, n), (s = e.forNode)[r = this.id] || (s[r] = {}), e.forNode[this.id].subgraphNewIDMapping = n, (o = e.forNode)[a = t.id] || (o[a] = {}), e.forNode[t.id].subgraphNewIDMapping = n, t;
  }
  buildFromNodes(e) {
    var _, y;
    if (e = e.filter((b) => !b.is($) && !b.is(Q)), e.length === 0)
      return;
    const t = {}, i = {}, n = {}, s = e.reduce((b, m) => (b[m.id] = m, b), {});
    let r = Number.MAX_SAFE_INTEGER, o = 0, a = Number.MAX_SAFE_INTEGER, l = 0;
    for (const b of Object.values(e))
      r = Math.min(b.pos[0], r), o = Math.max(b.pos[0] + b.size[0], o), a = Math.min(b.pos[1], a), l = Math.max(b.pos[1] + b.size[1], l);
    const h = {};
    for (const b of e) {
      h[b.id] = b;
      for (let m = 0; m < b.inputs.length; m++) {
        const E = b.getInputLink(m);
        if (E) {
          const T = b.getConnectionPos(!0, m), O = b.getInputInfo(m), A = b.getInputNode(m);
          A && (h[A.id] = A), s[E.origin_id] != null ? n[E.id] = [E, T] : t[E.id] = [E, T, O.name];
        }
      }
      for (let m = 0; m < b.outputs.length; m++) {
        const E = b.getOutputLinks(m);
        for (const T of E) {
          const O = b.getConnectionPos(!1, m), A = b.getOutputInfo(m), M = b.graph.getNodeById(T.target_id);
          M && (h[M.id] = M), s[T.target_id] != null ? n[T.id] = [T, O] : i[T.id] = [T, O, A.name];
        }
      }
    }
    const p = Object.values(t), f = Object.values(i);
    p.sort((b, m) => b[1][1] - m[1][1]), f.sort((b, m) => b[1][1] - m[1][1]), u.debug && (console.debug("NODES", Object.keys(e)), console.debug("IN", Object.keys(t)), console.debug("OUT", Object.keys(i)), console.debug("INNER", Object.keys(n)));
    const c = {}, v = {};
    for (const b of e) {
      const m = [b.pos[0] - r, b.pos[1] - a], E = b.id;
      b.graph.remove(b, { removedBy: "moveIntoSubgraph" }), this.subgraph.add(b, {
        addedBy: "moveIntoSubgraph",
        prevNodeID: E
      }), b.pos = m, h[E] = b, h[b.id] = b;
    }
    let g = 0, d = 0;
    for (const [b, m, E] of p) {
      let T = null;
      if (c[b.origin_id] && (T = c[b.origin_id][b.origin_slot]), !T && (T = this.addGraphInput(E, b.type, [
        -200,
        g
      ]), g += T.innerNode.size[1] + u.NODE_SLOT_HEIGHT, !T)) {
        console.error(
          "Failed creating subgraph output pair!",
          b
        );
        continue;
      }
      const O = h[b.origin_id], A = h[b.target_id];
      O.connect(b.origin_slot, this, T.outerInputIndex), T.innerNode.connect(0, A, b.target_slot), c[_ = b.origin_id] || (c[_] = {}), c[b.origin_id][b.origin_slot] = T;
    }
    for (const [b, m, E] of f) {
      let T = null;
      if (v[b.target_id] && (T = v[b.target_id][b.target_slot]), !T && (T = this.addGraphOutput(E, b.type, [
        o - r + 200,
        d
      ]), d += T.innerNode.size[1] + u.NODE_SLOT_HEIGHT, !T)) {
        console.error(
          "Failed creating subgraph output pair!",
          b
        );
        continue;
      }
      const O = h[b.origin_id], A = h[b.target_id];
      O.connect(b.origin_slot, T.innerNode, 0), this.connect(T.outerOutputIndex, A, b.target_slot), v[y = b.target_id] || (v[y] = {}), v[b.target_id][b.origin_slot] = T;
    }
    for (const [b, m] of Object.values(n)) {
      const E = h[b.origin_id], T = h[b.target_id];
      E.connect(
        b.origin_slot,
        T,
        b.target_slot
      );
    }
  }
  addGraphInput(e, t, i) {
    e = this.getValidGraphInputName(e);
    const n = u.createNode($);
    if (n == null)
      return null;
    let s = t;
    t === I.EVENT ? s = I.ACTION : t === I.ACTION && (t = I.EVENT), console.warn("[Subgraph] addGraphInput", e, t, s, i), n.setProperty("name", e), n.setProperty("type", t), n.properties.subgraphID = this.id, this.subgraph.add(n);
    const r = n.computeSize();
    i && (n.pos = [
      i[0] - r[0] * 0.5,
      i[1] - r[1] * 0.5
    ]), this.subgraph.addInput(e, s, null);
    const o = this.inputs.length - 1, a = this.inputs[o];
    return { innerNode: n, outerInput: a, outerInputIndex: o };
  }
  addGraphOutput(e, t, i) {
    e = this.getValidGraphOutputName(e);
    const n = u.createNode(Q);
    if (n == null)
      return null;
    let s = t;
    t === I.EVENT ? t = I.ACTION : t === I.ACTION && (s = I.EVENT), console.warn("[Subgraph] addGraphOutput", e, t, s, i), n.setProperty("name", e), n.setProperty("type", t), n.properties.subgraphID = this.id, this.subgraph.add(n);
    const r = n.computeSize();
    i && (n.pos = [i[0], i[1] - r[1] * 0.5]), this.subgraph.addOutput(e, s, null);
    const o = this.outputs.length - 1, a = this.outputs[o];
    return { innerNode: n, outerOutput: a, outerOutputIndex: o };
  }
  removeGraphInput(e) {
    if (this.findInputSlotIndexByName(e) == null) {
      console.error("[Subgraph] No input in slot!", e);
      return;
    }
    const i = this.subgraph.findNodesByClass($).filter((n) => n.properties.name === e);
    if (i.length > 0)
      for (const n of i)
        this.subgraph.remove(n);
    else {
      console.warn(
        "[Subgraph] No GraphInputs found on input removal",
        e
      );
      const n = this.findInputSlotIndexByName(e);
      n !== -1 && this.removeInput(n);
    }
  }
  removeGraphOutput(e) {
    if (this.findOutputSlotIndexByName(e) == null) {
      console.error("[Subgraph] No output in slot!", e);
      return;
    }
    const i = this.subgraph.findNodesByClass(Q).filter((n) => n.properties.name === e);
    if (i.length > 0)
      for (const n of i)
        this.subgraph.remove(n);
    else {
      console.warn(
        "[Subgraph] No GraphOutputs found on output removal",
        e
      );
      const n = this.findOutputSlotIndexByName(e);
      n !== -1 && this.removeOutput(n);
    }
  }
  getValidGraphInputName(e) {
    e || (e = "newInput");
    let t = e, i = this.getInnerGraphInput(t), n = 1;
    for (; i != null; )
      t = `${e}_${n++}`, i = this.getInnerGraphInput(t);
    return t;
  }
  getValidGraphOutputName(e) {
    e || (e = "newOutput");
    let t = e, i = this.getInnerGraphOutput(t), n = 1;
    for (; i != null; )
      t = `${e}_${n++}`, i = this.getInnerGraphOutput(t);
    return t;
  }
  getInnerGraphOutput(e) {
    return this.subgraph._nodes.find((i) => i.is(Q) && i.properties.name === e) || null;
  }
  getInnerGraphInput(e) {
    return this.subgraph._nodes.find((i) => i.is($) && i.properties.name === e) || null;
  }
  getInnerGraphOutputByIndex(e) {
    const t = this.getOutputInfo(e);
    return t ? this.getInnerGraphOutput(t.name) : null;
  }
  getInnerGraphInputByIndex(e) {
    const t = this.getInputInfo(e);
    return t ? this.getInnerGraphInput(t.name) : null;
  }
  moveNodesToParentGraph(e) {
    if (e = e.filter((g) => !g.is($) && !g.is(Q)), e.length === 0)
      return;
    const t = this, i = t.graph;
    let n = Number.MAX_SAFE_INTEGER, s = 0, r = Number.MAX_SAFE_INTEGER, o = 0;
    for (const g of Object.values(e))
      n = Math.min(g.pos[0], n), s = Math.max(g.pos[0] + g.size[0], s), r = Math.min(g.pos[1], r), o = Math.max(g.pos[1] + g.size[1], o);
    const a = s - n, l = o - r, h = t.pos[0] + t.size[0] / 2 - a / 2, p = t.pos[1] + t.size[1] / 2 - l / 2, f = {}, c = {};
    for (const [g, d] of e.entries())
      c[d.id] = d;
    for (const g of e)
      for (const d of g.iterateAllLinks()) {
        const _ = d.target_id === g.id, y = g.getConnectionPos(
          _,
          _ ? d.target_slot : d.origin_slot
        );
        c[d.origin_id] != null && c[d.target_id] != null && (f[d.id] = [d, y]);
      }
    const v = {};
    for (const [g, d] of e.entries()) {
      const _ = [
        d.pos[0] - n + h,
        d.pos[1] - r + p
      ], y = d.id;
      d.graph.remove(d, { removedBy: "moveOutOfSubgraph" }), i.add(d, { addedBy: "moveOutOfSubgraph", prevNodeID: y }), d.pos = _, v[y] = d;
    }
    for (const [g, d] of Object.values(f)) {
      const _ = c[g.origin_id], y = c[g.target_id];
      _.connect(g.origin_slot, y, g.target_slot);
    }
    return v;
  }
  convertNodesToSubgraphInputs(e) {
    var a;
    if (e = e.filter((l) => !l.is($) && !l.is(Q)), e.length === 0)
      return;
    const t = ve(e, (l) => l.id), i = [], n = {}, s = this.subgraph;
    for (const l of e)
      for (const h of l.iterateAllLinks()) {
        if (t[h.origin_id] == null)
          throw new Error(
            "Can't convert to input with an origin link outward"
          );
        if (t[h.target_id] == null) {
          i.push(h);
          const p = [0, 0];
          l.getConnectionPos(!1, h.target_slot, p), n[l.id] = [
            [l.pos[0], l.pos[1]],
            p
          ];
        }
      }
    const r = this.moveNodesToParentGraph(e), o = {};
    for (const l of i) {
      const h = s.getNodeById(l.target_id), p = h.getInputInfo(l.target_slot);
      o[a = l.origin_id] || (o[a] = {});
      let f = o[l.origin_id][l.origin_slot];
      if (f == null) {
        const v = this.getValidGraphInputName(p.name);
        f = this.addGraphInput(v, p.type), o[l.origin_id][l.origin_slot] = f;
        const [g, d] = n[l.origin_id], _ = f.innerNode.pos, y = f.innerNode.computeSize(), b = f.innerNode.getConnectionPos(!0, 0), m = [
          f.innerNode.pos[0] - b[0],
          f.innerNode.pos[1] - b[1]
        ], E = [
          d[0] + m[0] - y[0],
          d[1] + m[1]
        ];
        console.warn(
          "newPos",
          _,
          "size",
          f.innerNode.size,
          "connPos",
          d,
          "newConPos",
          b,
          "offset",
          m
        ), f.innerNode.pos = E;
      }
      r[l.origin_id].connect(l.origin_slot, this, f.outerInputIndex), f.innerNode.connect(0, h, l.target_slot);
    }
  }
  convertNodesToSubgraphOutputs(e) {
    var a;
    if (e = e.filter((l) => !l.is($) && !l.is(Q)), e.length === 0)
      return;
    const t = ve(e, (l) => l.id), i = [], n = {}, s = this.subgraph;
    for (const l of e)
      for (const h of l.iterateAllLinks())
        if (t[h.origin_id] == null) {
          i.push(h);
          const p = [0, 0];
          l.getConnectionPos(!0, h.origin_slot, p), n[l.id] = [
            [l.pos[0], l.pos[1]],
            p
          ];
        } else if (t[h.target_id] == null)
          throw new Error(
            "Can't convert to input with an origin link outward"
          );
    const r = this.moveNodesToParentGraph(e), o = {};
    for (const l of i) {
      const h = s.getNodeById(l.origin_id), p = h.getOutputInfo(l.origin_slot);
      o[a = l.target_id] || (o[a] = {});
      let f = o[l.target_id][l.target_slot];
      if (f == null) {
        const v = this.getValidGraphOutputName(p.name);
        f = this.addGraphOutput(v, p.type), o[l.target_id][l.target_slot] = f;
        const [g, d] = n[l.target_id], _ = f.innerNode.getConnectionPos(!0, 0), y = [
          f.innerNode.pos[0] - _[0],
          f.innerNode.pos[1] - _[1]
        ], b = [
          d[0] + y[0],
          d[1] + y[1]
        ];
        f.innerNode.pos = b;
      }
      const c = r[l.target_id];
      h.connect(l.origin_slot, f.innerNode, 0), this.connect(f.outerOutputIndex, c, l.target_slot);
    }
  }
};
let ne = Me;
ne.default_lgraph_factory = () => new Pe();
ne.slotLayout = {
  inputs: [],
  outputs: []
};
ne.propertyLayout = [
  { name: "enabled", defaultValue: !0 }
];
ne.optionalSlots = {
  outputs: [{ name: "enabled", type: "boolean" }]
};
u.registerNodeType({
  class: ne,
  title: "Subgraph",
  desc: "Graph inside a node",
  title_color: "#334",
  type: "graph/subgraph"
});
class C {
  static onMenuCollapseAll() {
  }
  static onMenuNodeEdit() {
  }
  // refactor: there are different dialogs, some uses createDialog some dont
  prompt(t = "", i, n, s, r = !1, o = null) {
    var a = this, l = document.createElement("div");
    if (l.is_modified = !1, l.className = "graphdialog rounded", r) {
      let T = 5;
      typeof i != "string" && (i = JSON.stringify(i, null, 2));
      const O = (i.match(/\n/g) || "").length + 1;
      T = Te(O, 5, 10), l.innerHTML = `
<span class='name'></span>
<textarea autofocus rows='${T}' cols='30' class='value'></textarea>
<button class='rounded'>OK</button>
`;
    } else
      l.innerHTML = `
<span class='name'></span>
<input autofocus type='text' class='value'/>
<button class='rounded'>OK</button>`;
    l.close = function() {
      a.prompt_box = null, l.parentNode && l.parentNode.removeChild(l);
    };
    var h = N.active_canvas, p = h.canvas;
    p.parentNode.appendChild(l), this.ds.scale > 1 && (l.style.transform = "scale(" + this.ds.scale + ")");
    var f = null, c = 0;
    u.pointerListenerAdd(l, "leave", function(T) {
      c || u.dialog_close_on_mouse_leave && !l.is_modified && u.dialog_close_on_mouse_leave && T.buttons === 0 && (f = setTimeout(l.close, u.dialog_close_on_mouse_leave_delay));
    }), u.pointerListenerAdd(l, "enter", function(T) {
      u.dialog_close_on_mouse_leave && f && clearTimeout(f);
    });
    var v = l.querySelectorAll("select");
    v && v.forEach(function(T) {
      T.addEventListener("click", function(O) {
        c++;
      }), T.addEventListener("blur", function(O) {
        c = 0;
      }), T.addEventListener("change", function(O) {
        c = -1;
      });
    }), a.prompt_box && a.prompt_box.close(), a.prompt_box = l;
    var g = l.querySelector(".name");
    g.innerText = t;
    let d = l.querySelector(".value");
    d.value = i;
    var _ = d;
    if (_.addEventListener("keydown", function(T) {
      if (l.is_modified = !0, T.keyCode == 27)
        l.close();
      else if (T.keyCode == 13 && T.target instanceof Element && T.target.localName != "textarea")
        n && n(this.value), l.close();
      else
        return;
      T.preventDefault(), T.stopPropagation();
    }), o)
      for (const [T, O] of Object.entries(o))
        _.style[T] = O;
    var y = l.querySelector("button");
    y.addEventListener("click", function(T) {
      n && n(_.value), a.setDirty(!0), l.close();
    });
    var b = p.getBoundingClientRect(), m = -20, E = -20;
    return b && (m -= b.left, E -= b.top), s ? (l.style.left = s.clientX + "px", l.style.top = s.clientY + "px") : (l.style.left = p.width * 0.5 + m + "px", l.style.top = p.height * 0.5 + E + "px"), console.warn(l.style.left, l.style.top), console.warn(s), setTimeout(function() {
      _.focus();
    }, 10), Ee(l), l;
  }
  showSearchBox(t, i = {}) {
    var n = {
      slotFrom: null,
      node_from: null,
      node_to: null,
      do_type_filter: u.search_filter_enabled,
      type_filter_in: null,
      type_filter_out: null,
      show_general_if_none_on_typefilter: !0,
      show_general_after_typefiltered: !0,
      hide_on_mouse_leave: u.search_hide_on_mouse_leave,
      show_all_if_empty: !0,
      show_all_on_open: u.search_show_all_on_open
    };
    i = Object.assign(n, i);
    var s = this, r = N.active_canvas, o = r.canvas, a = o.ownerDocument || document;
    let l = t;
    var h = document.createElement("div");
    h.className = "litegraph litesearchbox graphdialog rounded", h.innerHTML = "<span class='name'>Search</span> <input autofocus type='text' class='value rounded'/>", i.do_type_filter && (h.innerHTML += "<select class='slot_in_type_filter'><option value=''></option></select>", h.innerHTML += "<select class='slot_out_type_filter'><option value=''></option></select>"), h.innerHTML += "<div class='helper'></div>", a.fullscreenElement ? a.fullscreenElement.appendChild(h) : (a.body.appendChild(h), a.body.style.overflow = "hidden");
    let p = null, f = null;
    if (i.do_type_filter && (p = h.querySelector(".slot_in_type_filter"), f = h.querySelector(".slot_out_type_filter")), h.close = function() {
      s.search_box = null, this.blur(), o.focus(), a.body.style.overflow = "", setTimeout(function() {
        s.canvas.focus();
      }, 20), h.parentNode && h.parentNode.removeChild(h);
    }, this.ds.scale > 1 && (h.style.transform = "scale(" + this.ds.scale + ")"), i.hide_on_mouse_leave) {
      var c = 0, v = null;
      u.pointerListenerAdd(h, "enter", function(D) {
        v && (clearTimeout(v), v = null);
      }), u.pointerListenerAdd(h, "leave", function(D) {
        c || (v = setTimeout(function() {
          h.close();
        }, 500));
      }), i.do_type_filter && (p.addEventListener("click", function(D) {
        c++;
      }), p.addEventListener("blur", function(D) {
        c = 0;
      }), p.addEventListener("change", function(D) {
        c = -1;
      }), f.addEventListener("click", function(D) {
        c++;
      }), f.addEventListener("blur", function(D) {
        c = 0;
      }), f.addEventListener("change", function(D) {
        c = -1;
      }));
    }
    s.search_box && s.search_box.close(), s.search_box = h;
    var g = h.querySelector(".helper"), d = null, _ = null, y = null;
    const b = (D) => {
      if (D)
        if (s.onSearchBoxSelection)
          s.onSearchBoxSelection(D, l, r);
        else {
          var P = u.searchbox_extras[D.toLowerCase()];
          P && (D = P.type), r.graph.beforeChange();
          var F = u.createNode(D);
          if (F && (F.pos = r.convertEventToCanvasOffset(
            l
          ), r.graph.add(F)), P && P.data) {
            if (P.data.properties)
              for (var V in P.data.properties)
                F.addProperty("" + V, P.data.properties[V]);
            if (P.data.inputs) {
              F.inputs = [];
              for (var V in P.data.inputs)
                F.addInput(
                  P.data.inputs[V][0],
                  P.data.inputs[V][1]
                );
            }
            if (P.data.outputs) {
              F.outputs = [];
              for (var V in P.data.outputs)
                F.addOutput(
                  P.data.outputs[V][0],
                  P.data.outputs[V][1]
                );
            }
            P.data.title && (F.title = P.data.title), P.data.json && F.configure(P.data.json);
          }
          if (i.node_from) {
            var R = null;
            switch (typeof i.slotFrom) {
              case "string":
                R = i.node_from.findOutputSlotIndexByName(i.slotFrom);
                break;
              case "object":
                i.slotFrom.name ? R = i.node_from.findOutputSlotIndexByName(i.slotFrom.name) : R = -1, R == -1 && typeof i.slotFrom.slot_index < "u" && (R = i.slotFrom.slot_index);
                break;
              case "number":
                R = i.slotFrom;
                break;
              default:
                R = 0;
            }
            R = R, typeof i.node_from.outputs[R] !== void 0 && R !== null && R > -1 && i.node_from.connectByTypeInput(R, F, i.node_from.outputs[R].type);
          }
          if (i.node_to) {
            var R = null;
            switch (typeof i.slotFrom) {
              case "string":
                R = i.node_to.findInputSlotIndexByName(i.slotFrom);
                break;
              case "number":
                R = i.slotFrom;
                break;
              default:
                R = 0;
            }
            typeof i.node_to.inputs[R] !== void 0 && R !== null && R > -1 && i.node_to.connectByTypeOutput(R, F, i.node_to.inputs[R].type);
          }
          r.graph.afterChange();
        }
      h.close();
    }, m = (D) => {
      var P = y;
      y && y.classList.remove("selected"), y ? (y = D ? y.nextSibling : y.previousSibling, y || (y = P)) : y = D ? g.childNodes[0] : g.childNodes[g.childNodes.length], y && (y.classList.add("selected"), y.scrollIntoView({ block: "end", behavior: "smooth" }));
    }, E = (D, P, F, V, R, le = {}) => {
      const ee = Object.assign({
        skipFilter: !1,
        inTypeOverride: null,
        outTypeOverride: null
      }, le), fe = u.registered_node_types[D];
      if (fe.hide_in_node_lists || P && fe.filter != P || (!i.show_all_if_empty || F) && D.toLowerCase().indexOf(F) === -1)
        return !1;
      if (i.do_type_filter && !ee.skipFilter) {
        const Y = D;
        let H = V == null ? void 0 : V.value;
        if (ee.inTypeOverride != null && (H = ee.inTypeOverride), V && H && u.registered_slot_in_types[H] && u.registered_slot_in_types[H].nodes) {
          var te = u.registered_slot_in_types[H].nodes.includes(Y);
          if (te === !1)
            return !1;
        }
        if (H = R == null ? void 0 : R.value, ee.outTypeOverride != null && (H = ee.outTypeOverride), R && H && u.registered_slot_out_types[H] && u.registered_slot_out_types[H].nodes) {
          var te = u.registered_slot_out_types[H].nodes.includes(Y);
          if (te === !1)
            return !1;
        }
      }
      return !0;
    }, T = () => {
      _ = null;
      var D = O.value;
      if (d = null, g.innerHTML = "", !D && !i.show_all_if_empty)
        return;
      if (s.onSearchBox) {
        var P = s.onSearchBox(g, D, r);
        if (P)
          for (var F = 0; F < P.length; ++F)
            te(P[F]);
      } else {
        var V = 0;
        D = D.toLowerCase();
        var R = r.filter || r.graph.filter;
        let Y, H;
        i.do_type_filter && s.search_box ? (Y = s.search_box.querySelector(".slot_in_type_filter"), H = s.search_box.querySelector(".slot_out_type_filter")) : (Y = null, H = null);
        for (const x in u.searchbox_extras) {
          var le = u.searchbox_extras[x];
          if (!((!i.show_all_if_empty || D) && le.desc.toLowerCase().indexOf(D) === -1)) {
            var be = u.registered_node_types[le.type];
            if (!(be && be.filter != R) && E(le.type, R, D, Y, H) && (te(le.desc, "searchbox_extra"), N.search_limit !== -1 && V++ > N.search_limit))
              break;
          }
        }
        var ee = null;
        if (Array.prototype.filter)
          var fe = Object.keys(u.registered_node_types), ee = fe.filter((q) => E(q, R, D, Y, H));
        else {
          ee = [];
          for (const x in u.registered_node_types)
            E(x, R, D, Y, H) && ee.push(x);
        }
        for (var F = 0; F < ee.length && (te(ee[F]), !(N.search_limit !== -1 && V++ > N.search_limit)); F++)
          ;
        if (i.show_general_after_typefiltered && (Y != null && Y.value || H != null && H.value)) {
          let x = [];
          for (const q in u.registered_node_types)
            E(q, R, D, Y, H, { inTypeOverride: Y && Y.value ? "*" : null, outTypeOverride: H && H.value ? "*" : null }) && x.push(q);
          for (let q = 0; q < x.length && (te(x[q], "generic_type"), !(N.search_limit !== -1 && V++ > N.search_limit)); q++)
            ;
        }
        if ((Y != null && Y.value || H != null && H.value) && (g == null ? void 0 : g.childNodes.length) == 0 && i.show_general_if_none_on_typefilter) {
          let x = [];
          for (const q in u.registered_node_types)
            E(q, R, D, Y, H, { skipFilter: !0 }) && x.push(q);
          for (let q = 0; q < x.length && (te(x[q], "not_in_filter"), !(N.search_limit !== -1 && V++ > N.search_limit)); q++)
            ;
        }
      }
      function te(Y, H) {
        var x = document.createElement("div");
        d || (d = Y), x.innerText = Y, x.dataset.type = escape(Y), x.className = "litegraph lite-search-item", H && (x.className += " " + H), x.addEventListener("click", function(q) {
          b(unescape(this.dataset.type));
        }), g.appendChild(x);
      }
    };
    var O = h.querySelector("input");
    if (O && (O.addEventListener("blur", function(D) {
      this.focus();
    }), O.addEventListener("keydown", function(D) {
      if (D.keyCode == 38)
        m(!1);
      else if (D.keyCode == 40)
        m(!0);
      else if (D.keyCode == 27)
        h.close();
      else if (D.keyCode == 13)
        y ? b(y.innerHTML) : d ? b(d) : h.close();
      else {
        _ && clearInterval(_), _ = setTimeout(T, u.search_box_refresh_interval_ms);
        return;
      }
      return D.preventDefault(), D.stopPropagation(), D.stopImmediatePropagation(), !0;
    })), i.do_type_filter) {
      if (p) {
        var A = u.slot_types_in, M = A.length;
        (i.type_filter_in == I.EVENT || i.type_filter_in == I.ACTION) && (i.type_filter_in = "_event_");
        for (var L = 0; L < M; L++) {
          var B = document.createElement("option");
          B.value = A[L], B.innerHTML = A[L], p.appendChild(B), i.type_filter_in !== null && (i.type_filter_in + "").toLowerCase() == (A[L] + "").toLowerCase() && (B.selected = !0);
        }
        p.addEventListener("change", T);
      }
      if (f) {
        var A = u.slot_types_out, M = A.length;
        (i.type_filter_out == I.EVENT || i.type_filter_out == I.ACTION) && (i.type_filter_out = "_event_");
        for (var L = 0; L < M; L++) {
          var B = document.createElement("option");
          B.value = A[L], B.innerHTML = A[L], f.appendChild(B), i.type_filter_out !== null && (i.type_filter_out + "").toLowerCase() == (A[L] + "").toLowerCase() && (B.selected = !0);
        }
        f.addEventListener("change", T);
      }
    }
    var G = o.getBoundingClientRect(), z = (l ? l.clientX : G.left + G.width * 0.5) - 80, pe = (l ? l.clientY : G.top + G.height * 0.5) - 20;
    return h.style.left = z + "px", h.style.top = pe + "px", l.layerY > G.height - 200 && (g.style.maxHeight = G.height - l.layerY - 20 + "px"), O.focus(), i.show_all_on_open && T(), h;
  }
  showShowNodePanel(t) {
    this.closePanels();
    var i = this.getCanvasWindow(), n = this, s = this.createPanel(t.title || "", {
      closable: !0,
      window: i,
      onOpen: function() {
      },
      onClose: function() {
        n.node_panel = null;
      }
    });
    n.node_panel = s, s.id = "node-panel", s.node = t, s.classList.add("settings");
    function r() {
      s.content.innerHTML = "", s.addHTML("<span class='node_type'>" + t.type + "</span><span class='node_desc'>" + (t.constructor.desc || "") + "</span><span class='separator'></span>"), s.addHTML("<h3>Properties</h3>");
      var o = function(f, c) {
        switch (n.graph.beforeChange(t), f) {
          case "Title":
            t.title = c;
            break;
          case "Mode":
            var v = Object.values(re).indexOf(c);
            v >= Z.ALWAYS && re[v] ? t.changeMode(v) : console.warn("unexpected mode: " + c);
            break;
          case "Color":
            N.node_colors[c] ? (t.color = N.node_colors[c].color, t.bgcolor = N.node_colors[c].bgcolor) : console.warn("unexpected color: " + c);
            break;
          default:
            t.setProperty(f, c);
            break;
        }
        n.graph.afterChange(), n.dirty_canvas = !0;
      };
      s.addWidget("string", "Title", t.title, {}, o), s.addWidget("combo", "Mode", re[t.mode], { values: re }, o);
      var a = "";
      t.color !== void 0 && (a = Object.keys(N.node_colors).filter(function(f) {
        return N.node_colors[f].color == t.color;
      })[0]), s.addWidget("combo", "Color", a, { values: Object.keys(N.node_colors) }, o);
      for (var l in t.properties) {
        var h = t.properties[l], p = t.getPropertyInfo(l);
        p.type, !(t.onAddPropertyToPanel && t.onAddPropertyToPanel(l, s)) && s.addWidget(p.widget || p.type, l, h, p, o);
      }
      s.addSeparator(), t.onShowCustomPanelInfo && t.onShowCustomPanelInfo(s), s.footer.innerHTML = "", s.addButton("Delete", function() {
        t.block_delete || (t.graph.remove(t), s.close());
      }).classList.add("delete");
    }
    s.inner_showCodePad = function(o) {
      s.classList.remove("settings"), s.classList.add("centered"), s.alt_content.innerHTML = "<textarea class='code'></textarea>";
      var a = s.alt_content.querySelector("textarea"), l = function() {
        s.toggleAltContent(!1), s.toggleFooterVisibility(!0), a.parentNode.removeChild(a), s.classList.add("settings"), s.classList.remove("centered"), r();
      };
      a.value = t.properties[o], a.addEventListener("keydown", function(f) {
        f.code == "Enter" && f.ctrlKey && (t.setProperty(o, a.value), l());
      }), s.toggleAltContent(!0), s.toggleFooterVisibility(!1), a.style.height = "calc(100% - 40px)";
      var h = s.addButton("Assign", function() {
        t.setProperty(o, a.value), l();
      });
      s.alt_content.appendChild(h);
      var p = s.addButton("Close", l);
      p.style.float = "right", s.alt_content.appendChild(p);
    }, r(), this.canvas.parentNode.appendChild(s);
  }
  showSubgraphPropertiesDialog(t) {
    console.log("showing subgraph properties dialog");
    var i = this.canvas.parentNode.querySelector(".subgraph_dialog");
    i && i.close();
    var n = this.createPanel("Subgraph Inputs", { closable: !0, width: 500 });
    n.node = t, n.classList.add("subgraph_dialog");
    const s = t;
    var r = s.subgraph;
    if (!r) {
      console.warn("subnode without subgraph!");
      return;
    }
    function o() {
      if (n.clear(), t.inputs)
        for (var d = 0; d < t.inputs.length; ++d) {
          var _ = t.inputs[d];
          if (_.not_subgraph_input)
            continue;
          var y = `
<button class="delete">&#10005;</button>
<button class="move_up">↑</button>
<button class="move_down">↓</button>
<span class='bullet_icon'></span>
<span class='name'></span>
<span class='type'></span>`, b = n.addHTML(y, "subgraph_property");
          b.dataset.name = _.name, b.dataset.slot = "" + d, b.querySelector(".name").innerText = _.name, b.querySelector(".type").innerText = J(_.type), b.querySelector(".delete").addEventListener("click", function(T) {
            const O = this.parentNode.dataset.name;
            s.removeGraphInput(O), o();
          });
          const m = b.querySelector(".move_up");
          m.disabled = d <= 0, m.addEventListener("click", function(T) {
            const O = +this.parentNode.dataset.slot;
            O < 0 || (s.moveInput(O, O - 1), o());
          });
          const E = b.querySelector(".move_down");
          E.disabled = d >= t.inputs.length - 1, E.addEventListener("click", function(T) {
            const O = +this.parentNode.dataset.slot;
            O > t.inputs.length - 1 || (s.moveInput(O, O + 1), o());
          });
        }
    }
    var a = `
+
<span class='label'>Name</span>
<input class='name'/>
<span class='label'>Type</span>
<select class='type'></select>
<button>+</button>`, l = n.addHTML(a, "subgraph_property extra", !0);
    const h = l.querySelector(".name"), p = l.querySelector(".type"), f = l.querySelector("button");
    for (const d of we()) {
      var c = document.createElement("option");
      c.value = d, c.innerHTML = J(d), p.appendChild(c), d === "*" && (c.selected = !0);
    }
    const v = () => {
      const d = h.value;
      let _ = p.value;
      _ === "-1" ? _ = I.ACTION : _ === "-2" && (_ = I.EVENT), !(!d || t.findInputSlotIndexByName(d) != -1) && (this.addGraphInputNode(t, d, _), h.value = "", p.value = "", o(), h.focus());
    }, g = (d) => {
      d.keyCode == 13 ? (v(), d.preventDefault()) : d.keyCode == 27 && (n.close(), d.preventDefault());
    };
    return f.addEventListener("click", v), h.addEventListener("keydown", g), p.addEventListener("keydown", g), o(), this.canvas.parentNode.appendChild(n), h.focus(), n;
  }
  showSubgraphPropertiesDialogRight(t) {
    var i = this.canvas.parentNode.querySelector(".subgraph_dialog");
    i && i.close();
    var n = this.createPanel("Subgraph Outputs", { closable: !0, width: 500 });
    n.node = t, n.classList.add("subgraph_dialog");
    const s = t;
    if (!s.subgraph) {
      console.warn("subnode without subgraph!");
      return;
    }
    function o() {
      if (n.clear(), t.outputs)
        for (var d = 0; d < t.outputs.length; ++d) {
          var _ = t.outputs[d];
          if (_.not_subgraph_output)
            continue;
          var y = `
<button>&#10005;</button>
<button class="move_up">↑</button>
<button class="move_down">↓</button>
<span class='bullet_icon'></span>
<span class='name'></span>
<span class='type'></span>`, b = n.addHTML(y, "subgraph_property");
          b.dataset.name = _.name, b.dataset.slot = "" + d, b.querySelector(".name").innerText = _.name, b.querySelector(".type").innerText = J(_.type), b.querySelector("button").addEventListener("click", function(T) {
            const O = this.parentNode.dataset.name;
            s.removeGraphOutput(O), o();
          });
          const m = b.querySelector(".move_up");
          m.disabled = d <= 0, m.addEventListener("click", function(T) {
            const O = +this.parentNode.dataset.slot;
            O < 0 || (s.moveOutput(O, O - 1), o());
          });
          const E = b.querySelector(".move_down");
          E.disabled = d >= t.outputs.length - 1, E.addEventListener("click", function(T) {
            const O = +this.parentNode.dataset.slot;
            O > t.outputs.length - 1 || (s.moveOutput(O, O + 1), o());
          });
        }
    }
    var a = `
+
<span class='label'>Name</span>
<input class='name'/>
<span class='label'>Type</span>
<select class='type'></select>
<button>+</button>`, l = n.addHTML(a, "subgraph_property extra", !0);
    const h = l.querySelector(".name"), p = l.querySelector(".type"), f = l.querySelector("button");
    for (const d of Le()) {
      var c = document.createElement("option");
      c.value = d, c.innerHTML = J(d), p.appendChild(c), d === "*" && (c.selected = !0);
    }
    const v = () => {
      const d = h.value;
      let _ = p.value;
      _ === "-1" ? _ = I.ACTION : _ === "-2" && (_ = I.EVENT), !(!d || t.findOutputSlotIndexByName(d) != -1) && (this.addGraphOutputNode(t, d, _), h.value = "", p.value = "", o(), h.focus());
    }, g = (d) => {
      d.keyCode == 13 ? (v(), d.preventDefault()) : d.keyCode == 27 && (n.close(), d.preventDefault());
    };
    return f.addEventListener("click", v), h.addEventListener("keydown", g), p.addEventListener("keydown", g), o(), this.canvas.parentNode.appendChild(n), h.focus(), n;
  }
  showConnectionMenu(t = {}) {
    var i = t.nodeFrom && t.slotFrom, n = !i && t.nodeTo && t.slotTo;
    if (!i && !n)
      return console.warn("No data passed to showConnectionMenu"), !1;
    var s = i ? t.nodeFrom : t.nodeTo;
    const r = i ? t.slotFrom : t.slotTo;
    let o;
    var a = null;
    switch (typeof r) {
      case "string":
        a = i ? s.findOutputSlotIndexByName(r) : s.findInputSlotIndexByName(r), o = i ? s.outputs[r] : s.inputs[r];
        break;
      case "object":
        o = r, a = i ? s.findOutputSlotIndexByName(o.name) : s.findInputSlotIndexByName(o.name);
        break;
      case "number":
        a = r, o = i ? s.outputs[a] : s.inputs[a];
        break;
      default:
        return console.error("Can't get slot information", r), !1;
    }
    var l = [{ content: "Add Node" }, j.SEPARATOR];
    s.graph._is_subgraph && (i ? l.push({ content: "Add Subgraph Output" }) : l.push({ content: "Add Subgraph Input" }), l.push(j.SEPARATOR)), this.allow_searchbox && (l.push({ content: "Search" }), l.push(j.SEPARATOR));
    var h = o.type == I.EVENT ? "_event_" : o.type, p = i ? u.slot_types_default_out : u.slot_types_default_in;
    const f = p[h];
    if (console.warn("FROMSL", p, f), p && p[h])
      if (Array.isArray(f))
        for (var c of f) {
          const b = typeof c == "string" ? c : (c == null ? void 0 : c.title) || (c == null ? void 0 : c.node);
          l.push({ content: b, value: c });
        }
      else
        throw new Error(`Invalid default slot specifier, must be an array: ${f}`);
    const v = (b) => {
      const m = s.graph._subgraph_node, E = [b.canvasX, b.canvasY];
      m.addGraphInput(o.name, o.type, E).innerNode.connect(0, s, a);
    }, g = (b) => {
      const m = s.graph._subgraph_node, E = [b.canvasX, b.canvasY], T = m.addGraphOutput(o.name, o.type, E);
      s.connect(a, T.innerNode, 0);
    }, d = (b) => {
      const m = Object.assign(t, {
        position: [t.e.canvasX, t.e.canvasY]
      });
      var E = this.createDefaultNodeForSlot(b, m);
      E ? console.log("node created", b) : console.error("node not in defaults", b);
    }, _ = (b, m, E) => {
      switch (b.content) {
        case "Add Node":
          N.onMenuAdd(b, m, E, y, function(T) {
            i ? t.nodeFrom.connectByTypeInput(a, T, h) : t.nodeTo.connectByTypeOutput(a, T, h);
          });
          break;
        case "Add Subgraph Input":
          v(this.adjustMouseEvent(E));
          break;
        case "Add Subgraph Output":
          g(this.adjustMouseEvent(E));
          break;
        case "Search":
          i ? this.showSearchBox(E, { node_from: t.nodeFrom, slotFrom: o, type_filter_in: h }) : this.showSearchBox(E, { node_to: t.nodeTo, slotFrom: o, type_filter_out: h });
          break;
        default:
          d(b.value);
          break;
      }
    };
    var y = new X(l, {
      event: t.e,
      title: (o && o.name != "" ? o.name + (h ? " | " : "") : "") + (o && h ? h : ""),
      callback: _
    });
    return !1;
  }
  getLinkMenuOptions(t) {
    const i = this.graph.getNodeById(t.origin_id), n = this.graph.getNodeById(t.target_id);
    let s = null;
    i && i.outputs && i.outputs[t.origin_slot] && (s = i.outputs[t.origin_slot].type);
    let r = null;
    n && n.outputs && n.outputs[t.target_slot] && (r = n.inputs[t.target_slot].type);
    const o = (p) => {
      console.debug("node autoconnect"), !(!p.inputs || !p.inputs.length || !p.outputs || !p.outputs.length) && i.connectByTypeInput(t.origin_slot, p, s) && (p.connectByTypeInput(t.target_slot, n, r), p.pos[0] -= p.size[0] * 0.5);
    }, a = (p, f, c, v, g) => {
      N.onMenuAdd(p, f, c, v, o);
    }, l = () => {
      this.graph.removeLink(t.id);
    };
    let h = [
      {
        content: "Add Node",
        has_submenu: !0,
        callback: a
      },
      j.SEPARATOR,
      {
        content: "Delete",
        has_submenu: !0,
        callback: l
      },
      j.SEPARATOR
    ];
    return this.graph.onGetLinkMenuOptions && (h = this.graph.onGetLinkMenuOptions(h, t)), i.getExtraLinkOptions && (h = i.getExtraLinkOptions(this, t, W.OUTPUT, h)), n.getExtraLinkOptions && (h = n.getExtraLinkOptions(this, t, W.INPUT, h)), h;
  }
  showLinkMenu(t, i) {
    const n = this.getLinkMenuOptions(t);
    return new X(n, {
      event: i,
      title: t.data != null ? t.data.constructor.name : null,
      extra: t
    }), !1;
  }
  /*
   * Shows a popup for editing one of the LGraphNode.properties.
   */
  showEditPropertyValue(t, i, n = {}) {
    if (!t || t.properties[i] === void 0 || u.ignore_all_widget_events)
      return;
    var s = t.getPropertyInfo(i), r = s.type, o = "";
    if (r == "string" || r == "number" || r == "array" || r == "object")
      if (s.multiline) {
        let d = t.properties[i], _ = 5;
        if (r !== "string") {
          d = JSON.stringify(d, null, 2);
          const y = (d.match(/\n/g) || "").length + 1;
          _ = Te(y, 5, 10);
        }
        o = "<textarea autofocus type='text' rows='" + _ + "' cols='30' class='value'>" + (d || "") + "</textarea>";
      } else
        o = "<input autofocus type='text' class='value'/>";
    else if ((r == "enum" || r == "combo") && s.values) {
      o = "<select autofocus type='text' class='value'>";
      for (var a in s.values) {
        var l = a;
        s.values instanceof Array && (l = s.values[a]), o += "<option value='" + l + "' " + (l == t.properties[i] ? "selected" : "") + ">" + s.values[a] + "</option>";
      }
      o += "</select>";
    } else if (r == "boolean" || r == "toggle")
      o = "<input autofocus type='checkbox' class='value' " + (t.properties[i] ? "checked" : "") + "/>";
    else {
      console.warn("unknown type: " + r);
      return;
    }
    var h = this.createDialog(
      "<span class='name'>" + (s.label ? s.label : i) + "</span>" + o + "<button>OK</button>",
      n
    ), p = null;
    if ((r == "enum" || r == "combo") && s.values)
      p = h.querySelector("select"), p.addEventListener("change", function(d) {
        h.modified(), v(d.target.value);
      });
    else if (r == "boolean" || r == "toggle")
      p = h.querySelector("input"), p && p.addEventListener("click", function(d) {
        h.modified(), v(!!p.checked);
      });
    else if (s.multiline ? p = h.querySelector("textarea") : p = h.querySelector("input"), p) {
      p.addEventListener("blur", function(_) {
        this.focus();
      });
      let d = t.properties[i] !== void 0 ? t.properties[i] : "";
      if (r !== "string") {
        let _ = null;
        s.multiline && (_ = 2), d = JSON.stringify(d, null, _);
      }
      if (p.value = d, p.addEventListener("keydown", function(_) {
        let y = !1;
        _.keyCode == 27 ? (h.close(), y = !0) : _.keyCode == 13 && !s.multiline ? (c(), y = !0) : _.keyCode != 13 && h.modified(), y && (_.preventDefault(), _.stopPropagation());
      }), s.inputStyle)
        for (const [_, y] of Object.entries(s.inputStyle))
          p.style[_] = y;
    }
    p && p.focus();
    const f = () => {
      n.onclose && n.onclose(), h.close(), t.setDirtyCanvas(!0, !0);
    }, c = () => {
      r != "boolean" && r != "toggle" ? v(p.value) : f();
    }, v = (d) => {
      s && s.values && s.values.constructor === Object && s.values[d] != null && (d = s.values[d]), typeof t.properties[i] == "number" && (d = Number(d)), (r == "array" || r == "object") && (d = JSON.parse(d)), t.setProperty(i, d), f();
    };
    var g = h.querySelector("button");
    return g.addEventListener("click", c), Ee(h), h;
  }
  // TODO refactor, theer are different dialog, some uses createDialog, some dont
  createDialog(t, i = { checkForInput: !1, closeOnLeave: !0, closeOnLeave_checkModified: !0 }) {
    var n = document.createElement("div");
    n.className = "graphdialog", n.innerHTML = t, n.is_modified = !1;
    var s = this.canvas.getBoundingClientRect(), r = -20, o = -20;
    if (s && (r -= s.left, o -= s.top), i.position ? (r = i.position[0], o = i.position[1]) : i.event ? (r = i.event.clientX, o = i.event.clientY) : (r += this.canvas.width * 0.5, o += this.canvas.height * 0.5), n.style.left = r + "px", n.style.top = o + "px", this.canvas.parentNode.appendChild(n), i.checkForInput) {
      var a = n.querySelectorAll("input"), l = !1;
      a && a.forEach(function(c) {
        c.addEventListener("keydown", function(v) {
          if (n.modified(), v.keyCode == 27)
            n.close();
          else if (v.keyCode != 13)
            return;
          v.preventDefault(), v.stopPropagation();
        }), l || c.focus();
      });
    }
    n.modified = function() {
      n.is_modified = !0;
    }, n.close = function() {
      n.parentNode && n.parentNode.removeChild(n);
    };
    var h = null, p = 0;
    n.addEventListener("mouseleave", function(c) {
      p || (i.closeOnLeave || u.dialog_close_on_mouse_leave) && !n.is_modified && u.dialog_close_on_mouse_leave && c.buttons === 0 && (h = setTimeout(n.close, u.dialog_close_on_mouse_leave_delay));
    }), n.addEventListener("mouseenter", function(c) {
      (i.closeOnLeave || u.dialog_close_on_mouse_leave) && h && clearTimeout(h);
    });
    var f = n.querySelectorAll("select");
    return f && f.forEach(function(c) {
      c.addEventListener("click", function(v) {
        p++;
      }), c.addEventListener("blur", function(v) {
        p = 0;
      }), c.addEventListener("change", function(v) {
        p = -1;
      });
    }), n;
  }
  getCanvasMenuOptions() {
    var t = null;
    if (this.getMenuOptions ? t = this.getMenuOptions(this) : (t = [
      {
        content: "Add Node",
        has_submenu: !0,
        callback: N.onMenuAdd
      },
      { content: "Add Group", callback: N.onGroupAdd }
      //{ content: "Arrange", callback: that.graph.arrange },
      //{content:"Collapse All", callback: LGraphCanvas.onMenuCollapseAll }
    ], this._graph_stack && this._graph_stack.length > 0 && t.push(j.SEPARATOR, {
      content: "Close subgraph",
      callback: this.closeSubgraph.bind(this)
    })), this.getExtraMenuOptions) {
      var i = this.getExtraMenuOptions(this, t);
      i && (t = t.concat(i));
    }
    return t;
  }
  getNodeMenuOptions(t) {
    let i = [];
    t.getMenuOptions ? i = t.getMenuOptions(this) : (i = [
      {
        content: "Inputs",
        has_submenu: !0,
        disabled: !0,
        callback: N.showMenuNodeOptionalInputs
      },
      {
        content: "Outputs",
        has_submenu: !0,
        disabled: !0,
        callback: N.showMenuNodeOptionalOutputs
      },
      j.SEPARATOR,
      {
        content: "Properties",
        has_submenu: !0,
        disabled: u.ignore_all_widget_events,
        callback: N.onShowMenuNodeProperties
      },
      j.SEPARATOR,
      {
        content: "Title",
        value: { name: "title", type: "string" },
        callback: N.onShowPropertyEditor
      },
      {
        content: "Mode",
        has_submenu: !0,
        callback: N.onMenuNodeMode
      }
    ], t.resizable !== !1 && i.push({
      content: "Resize",
      callback: N.onMenuResizeNode
    }), i.push(
      {
        content: "Collapse",
        callback: N.onMenuNodeCollapse
      },
      { content: "Pin", callback: N.onMenuNodePin },
      {
        content: "Colors",
        has_submenu: !0,
        callback: N.onMenuNodeColors
      },
      {
        content: "Shapes",
        has_submenu: !0,
        callback: N.onMenuNodeShapes
      },
      j.SEPARATOR
    ));
    const n = t.getOptionalSlots();
    if (n && (n.inputs && n.inputs.length > 0 && typeof i[0] == "object" && (i[0].disabled = !1), n.outputs && n.outputs.length && typeof i[1] == "object" && (i[1].disabled = !1)), t.getExtraMenuOptions) {
      var s = t.getExtraMenuOptions(this, i);
      s && (s.push(j.SEPARATOR), i = s.concat(i));
    }
    t.clonable !== !1 && i.push({
      content: "Clone",
      callback: N.onMenuNodeClone
    }), i.push({
      content: "To Subgraph",
      callback: N.onMenuNodeToSubgraph
    });
    let r = Object.values(this.selected_nodes || {});
    if (r.length || (r = [t]), r = r.filter((o) => !o.is($) && !o.is(Q)), i.push({
      content: "To Parent Graph",
      disabled: !t.graph._is_subgraph || r.length === 0,
      callback: N.onMenuNodeToParentGraph
    }), t.graph._is_subgraph) {
      const o = (p) => {
        let f = 0;
        const c = ve(p, (v) => v.id);
        for (const v of p)
          for (const g of v.iterateAllLinks()) {
            if (c[g.origin_id] == null)
              return 0;
            c[g.target_id] == null && (f += 1);
          }
        return f;
      }, a = (p) => {
        let f = 0;
        const c = ve(p, (v) => v.id);
        for (const v of p)
          for (const g of v.iterateAllLinks())
            if (c[g.origin_id] == null)
              f += 1;
            else if (c[g.target_id] == null)
              return 0;
        return f;
      }, l = o(r);
      i.push({
        content: "To Subgraph Input" + (l > 1 ? "s" : ""),
        disabled: l === 0,
        callback: N.onMenuNodeToSubgraphInputs
      });
      const h = a(r);
      i.push({
        content: "To Subgraph Output" + (h > 1 ? "s" : ""),
        disabled: h === 0,
        callback: N.onMenuNodeToSubgraphOutputs
      });
    }
    return i.push(j.SEPARATOR, {
      content: "Remove",
      disabled: !(t.removable !== !1 && !t.block_delete),
      callback: N.onMenuNodeRemove
    }), t.graph && t.graph.onGetNodeMenuOptions && (i = t.graph.onGetNodeMenuOptions(i, t)), i;
  }
  getGroupMenuOptions(t) {
    var i = [
      {
        content: "Title",
        value: { name: "title", type: "string" },
        callback: N.onShowPropertyEditor
      },
      {
        content: "Color",
        has_submenu: !0,
        callback: N.onMenuNodeColors
      },
      {
        content: "Font size",
        value: { name: "fontSize", type: "number" },
        callback: N.onShowPropertyEditor
      },
      j.SEPARATOR,
      { content: "Remove", callback: N.onMenuNodeRemove }
    ];
    return i;
  }
  /** Called when mouse right click */
  processContextMenu(t, i) {
    var n = N.active_canvas, s = n.getCanvasWindow();
    let r = i, o = null, a = null, l = null;
    t != null && (l = t.item, t.type === "node" && (o = t.item), t.type === "link" && (a = t.item));
    let h = null;
    var p = {
      event: r,
      extra: l
    };
    o != null && (p.title = o.type);
    let f = null;
    o != null && (f = o.getSlotInPosition(r.canvasX, r.canvasY), N.active_node = o);
    const c = (y) => {
      const b = y.slot;
      o.graph.beforeChange(), b.input ? o.removeInput(b.slot) : b.output && o.removeOutput(b.slot), o.graph.afterChange();
    }, v = (y) => {
      var b = y.slot;
      o.graph.beforeChange(), b.output ? o.disconnectOutput(b.slot) : b.input && o.disconnectInput(b.slot), o.graph.afterChange();
    }, g = (y) => {
      var b = y.slot, m = b.input ? o.getInputInfo(b.slot) : o.getOutputInfo(b.slot), E = this.createDialog(
        "<span class='name'>Name</span><input autofocus type='text'/><button>OK</button>",
        p
      ), T = E.querySelector("input");
      T && m && (T.value = m.label || "");
      var O = () => {
        o.graph.beforeChange(), T.value && (m && (m.label = T.value), this.setDirty(!0)), E.close(), o.graph.afterChange();
      };
      E.querySelector("button").addEventListener("click", O), T.addEventListener("keydown", function(A) {
        if (E.is_modified = !0, A.keyCode == 27)
          E.close();
        else if (A.keyCode == 13)
          O();
        else if (A.keyCode != 13 && A.target instanceof Element && A.target.localName != "textarea")
          return;
        A.preventDefault(), A.stopPropagation();
      }), T.focus();
    };
    if (f) {
      if (h = [], o.getSlotMenuOptions)
        h = o.getSlotMenuOptions(f);
      else {
        f && f.output && f.output.links && f.output.links.length && h.push({ content: "Disconnect Links", slot: f, callback: v });
        var d = f.input || f.output;
        d.removable && h.push(
          d.locked ? "Cannot remove" : { content: "Remove Slot", slot: f, callback: c }
        ), d.nameLocked || h.push({ content: "Rename Slot", slot: f, callback: g });
      }
      const y = (f.input ? f.input.type : f.output.type) || "*";
      p.title = J(y);
    } else if (o)
      h = this.getNodeMenuOptions(o);
    else if (a)
      h = this.getLinkMenuOptions(a);
    else {
      h = this.getCanvasMenuOptions();
      var _ = this.graph.getGroupOnPos(
        r.canvasX,
        r.canvasY
      );
      _ && h.push(j.SEPARATOR, {
        content: "Edit Group",
        has_submenu: !0,
        submenu: {
          title: "Group",
          extra: _,
          options: this.getGroupMenuOptions(_)
        }
      });
    }
    h && new X(h, p, s);
  }
  createPanel(t, i = {}) {
    var n = i.window || window, s = document.createElement("div");
    if (s.className = "litegraph dialog", s.innerHTML = `
<div class='dialog-header'><span class='dialog-title'></span></div>
<div class='dialog-content'></div>
<div style='display:none;' class='dialog-alt-content'></div>
<div class='dialog-footer'></div>`, s.header = s.querySelector(".dialog-header"), i.width && (s.style.width = i.width + (i.width.constructor === Number ? "px" : "")), i.height && (s.style.height = i.height + (i.height.constructor === Number ? "px" : "")), i.closable) {
      var r = document.createElement("span");
      r.innerHTML = "&#10005;", r.classList.add("close"), r.addEventListener("click", function() {
        s.close();
      }), s.header.appendChild(r);
    }
    return i.onOpen && (s.onOpen = i.onOpen), i.onClose && (s.onClose = i.onClose), s.title_element = s.querySelector(".dialog-title"), s.title_element.innerText = t, s.content = s.querySelector(".dialog-content"), s.alt_content = s.querySelector(".dialog-alt-content"), s.footer = s.querySelector(".dialog-footer"), s.close = function() {
      s.onClose && typeof s.onClose == "function" && s.onClose(), s.parentNode && s.parentNode.removeChild(s), this.parentNode && this.parentNode.removeChild(this);
    }, s.toggleAltContent = function(o = !1) {
      if (typeof o < "u")
        var a = o ? "block" : "none", l = o ? "none" : "block";
      else
        var a = s.alt_content.style.display != "block" ? "block" : "none", l = s.alt_content.style.display != "block" ? "none" : "block";
      s.alt_content.style.display = a, s.content.style.display = l;
    }, s.toggleFooterVisibility = function(o = !1) {
      if (typeof o < "u")
        var a = o ? "block" : "none";
      else
        var a = s.footer.style.display != "block" ? "block" : "none";
      s.footer.style.display = a;
    }, s.clear = function() {
      this.content.innerHTML = "";
    }, s.addHTML = function(o, a, l) {
      var h = document.createElement("div");
      return a && (h.className = a), h.innerHTML = o, l ? s.footer.appendChild(h) : s.content.appendChild(h), h;
    }, s.addButton = function(o, a, l) {
      var h = document.createElement("button");
      return h.innerText = o, h.options = l, h.classList.add("btn"), h.addEventListener("click", a), s.footer.appendChild(h), h;
    }, s.addSeparator = function() {
      var o = document.createElement("div");
      return o.className = "separator", s.content.appendChild(o), o;
    }, s.addWidget = function(o, a, l, h = {}, p) {
      var f = String(l);
      o = o.toLowerCase(), o == "number" && (f = l.toFixed(3));
      var c = document.createElement("div");
      c.className = "property", c.innerHTML = "<span class='property_name'></span><span class='property_value'></span>";
      let v = c.querySelector(".property_name");
      v.innerText = h.label || a;
      var g = c.querySelector(".property_value");
      if (g.innerText = f, c.dataset.property = a, c.dataset.type = h.type || o, c.options = h, c.value = l, o == "code")
        c.addEventListener("click", function(_) {
          s.inner_showCodePad(this.dataset.property);
        });
      else if (o == "boolean")
        c.classList.add("boolean"), l && c.classList.add("bool-on"), c.addEventListener("click", function() {
          var _ = this.dataset.property;
          this.value = !this.value, this.classList.toggle("bool-on");
          const y = this.querySelector(".property_value");
          y.innerText = this.value ? "true" : "false", d(_, this.value);
        });
      else if (o == "string" || o == "number")
        g.setAttribute("contenteditable", "true"), g.addEventListener("keydown", function(_) {
          _.code == "Enter" && (o != "string" || !_.shiftKey) && (_.preventDefault(), this.blur());
        }), g.addEventListener("blur", function() {
          let _ = this.innerText;
          const y = this.parentNode;
          var b = y.dataset.property, m = y.dataset.type;
          m == "number" && (_ = Number(_)), d(b, _);
        });
      else if ((o == "enum" || o == "combo") && "values" in h) {
        var f = N.getPropertyPrintableValue(l, h.values);
        g.innerText = f, g.addEventListener("click", function(y) {
          let b = h.values || [];
          typeof b == "function" && (console.error("Values by callback not supported in panel.addWidget!", b), b = []);
          var E = this.parentNode.dataset.property, T = this;
          let O = Array.from(b).map((M) => ({ content: M }));
          new X(O, {
            event: y,
            className: "dark",
            callback: A
          }, n);
          function A(M, L, B) {
            return T.innerText = M.content, d(E, M.content), !1;
          }
        });
      }
      s.content.appendChild(c);
      function d(_, y) {
        h.callback && h.callback(_, y, h), p && p(_, y, h);
      }
      return c;
    }, s.onOpen && typeof s.onOpen == "function" && s.onOpen(), s;
  }
  checkPanels() {
    if (this.canvas)
      for (var t = this.canvas.parentNode.querySelectorAll(".litegraph.dialog"), i = 0; i < t.length; ++i) {
        var n = t[i];
        if (n.node && (n.node.graph || n.close(), n.node.graph != this.graph)) {
          if (n.node.is(ne) && this.graph._is_subgraph && this.graph === n.node.subgraph)
            continue;
          n.close();
        }
      }
  }
  closePanels() {
    var t = document.querySelector("#node-panel");
    t && t.close();
    var t = document.querySelector("#option-panel");
    t && t.close();
  }
}
C.onShowPropertyEditor = function(e, t, i, n, s) {
  var r = e.value, o = r.name, a = s[o], l = document.createElement("div");
  l.is_modified = !1, l.className = "graphdialog", l.innerHTML = "<span class='name'></span><input autofocus type='text' class='value'/><button>OK</button>", l.close = function() {
    l.parentNode && l.parentNode.removeChild(l);
  };
  var h = l.querySelector(".name");
  h.innerText = o;
  var p = l.querySelector(".value");
  if (p && (p.value = a, p.addEventListener("blur", function(E) {
    this.focus();
  }), p.addEventListener("keydown", function(E) {
    if (l.is_modified = !0, E.keyCode == 27)
      l.close();
    else if (E.keyCode == 13)
      _();
    else if (E.keyCode != 13 && E.target instanceof Element && E.target.localName != "textarea")
      return;
    E.preventDefault(), E.stopPropagation();
  }), r.inputStyle))
    for (const [E, T] of Object.entries(r.inputStyle))
      p.style[E] = T;
  var f = N.active_canvas, c = f.canvas, v = c.getBoundingClientRect(), g = -20, d = -20;
  v && (g -= v.left, d -= v.top), i ? (l.style.left = i.clientX + g + "px", l.style.top = i.clientY + d + "px") : (l.style.left = c.width * 0.5 + g + "px", l.style.top = c.height * 0.5 + d + "px");
  const _ = () => {
    p && y(p.value);
  }, y = (E) => {
    r.type == "number" ? E = Number(E) : r.type == "boolean" && (E = !!E);
    const T = s[o];
    s[o] = E, s.onJSPropertyChanged && s.onJSPropertyChanged(o, E, T) === !1 && (s[o] = T), l.parentNode && l.parentNode.removeChild(l), s.setDirtyCanvas(!0, !0);
  };
  var b = l.querySelector("button");
  b.addEventListener("click", _), c.parentNode.appendChild(l), p && p.focus();
  var m = null;
  l.addEventListener("mouseleave", function(E) {
    u.dialog_close_on_mouse_leave && !l.is_modified && u.dialog_close_on_mouse_leave && E.buttons === 0 && (m = setTimeout(l.close, u.dialog_close_on_mouse_leave_delay));
  }), l.addEventListener("mouseenter", function(E) {
    u.dialog_close_on_mouse_leave && m && clearTimeout(m);
  }), Ee(l);
};
C.onGroupAdd = function(e, t, i, n) {
  var s = N.active_canvas;
  s.getCanvasWindow();
  var r = new me();
  r.pos = s.convertEventToCanvasOffset(i), s.graph.addGroup(r);
};
C.onMenuAdd = function(e, t, i, n, s) {
  var r = N.active_canvas, o = r.getCanvasWindow(), a = r.graph;
  if (!a)
    return;
  function l(h, p) {
    var f = u.getNodeTypesCategories(r.filter || a.filter).filter(function(g) {
      return g.startsWith(h);
    }), c = [];
    f.map(function(g) {
      if (g) {
        var d = new RegExp("^(" + h + ")"), _ = g.replace(d, "").split("/")[0], y = h === "" ? _ + "/" : h + _ + "/", b = _;
        b.indexOf("::") != -1 && (b = b.split("::")[1]);
        var m = c.findIndex(function(E) {
          return E.value === y;
        });
        m === -1 && c.push(
          {
            value: y,
            content: b,
            has_submenu: !0,
            callback: function(E, T, O, A) {
              l(E.value, A);
            }
          }
        );
      }
    });
    var v = u.getNodeTypesInCategory(h.slice(0, -1), r.filter || a.filter);
    v.map(function(g) {
      if (!g.hide_in_node_lists) {
        var d = {
          value: g.class,
          content: g.title,
          has_submenu: !1,
          callback: function(_, y, b, m) {
            var E = m.getFirstEvent();
            r.graph.beforeChange();
            var T = u.createNode(_.value);
            T && (T.pos = r.convertEventToCanvasOffset(E), r.graph.add(T)), s && s(T), r.graph.afterChange();
          }
        };
        c.push(d);
      }
    }), new X(c, { event: i, parentMenu: p }, o);
  }
  return l("", n), !1;
};
C.showMenuNodeOptionalInputs = function(e, t, i, n, s) {
  if (!s)
    return;
  var r = this, o = N.active_canvas, a = o.getCanvasWindow();
  let l = s.getOptionalSlots().inputs, h = [];
  if (l)
    for (let v = 0; v < l.length; v++) {
      let g = l[v];
      if (!g) {
        h.push(j.SEPARATOR);
        continue;
      }
      let { name: d, type: _, options: y } = g;
      y || (y = {}), y.label && (d = y.label), y.removable = !0;
      var p = { content: d, value: g };
      _ == I.ACTION && (p.className = "event"), h.push(p);
    }
  if (s.onMenuNodeInputs) {
    var f = s.onMenuNodeInputs(h);
    f && (h = f);
  }
  if (!h.length) {
    console.log("no input entries");
    return;
  }
  new X(
    h,
    {
      event: i,
      callback: c,
      parentMenu: n,
      node: s
    },
    a
  );
  function c(v, g, d, _) {
    if (s && (v.callback && v.callback.call(r, s, v, d, _), v.value)) {
      let y = v.value;
      s.graph.beforeChange(), s.addInput(y.name, y.type, y.options), s.onNodeOptionalInputAdd && s.onNodeOptionalInputAdd(v.value), s.setDirtyCanvas(!0, !0), s.graph.afterChange();
    }
  }
  return !1;
};
C.showMenuNodeOptionalOutputs = function(e, t, i, n, s) {
  if (!s)
    return;
  var r = this, o = N.active_canvas, a = o.getCanvasWindow(), l = s.getOptionalSlots().outputs, h = [];
  if (l)
    for (var p = 0; p < l.length; p++) {
      var f = l[p];
      if (!f) {
        h.push(j.SEPARATOR);
        continue;
      }
      let { name: d, type: _, options: y } = f;
      if (!(s.flags && s.flags.skip_repeated_outputs && s.findOutputSlotIndexByName(f[0]) != -1)) {
        y || (y = {}), y.label && (d = y.label), y.removable = !0;
        var c = { content: d, value: [d, _, y] };
        _ == I.EVENT && (c.className = "event"), h.push(c);
      }
    }
  if (this.onMenuNodeOutputs && (h = this.onMenuNodeOutputs(h)), u.do_add_triggers_slots && s.findOutputSlotIndexByName("onExecuted") == -1 && h.push({ content: "On Executed", value: ["onExecuted", I.EVENT, { nameLocked: !0 }], className: "event" }), s.onMenuNodeOutputs) {
    var v = s.onMenuNodeOutputs(h);
    v && (h = v);
  }
  if (!h.length)
    return;
  let g = function(d, _, y, b) {
    if (s && (d.callback && d.callback.call(r, s, d, y, b), !!d.value)) {
      var m = d.value[1];
      if (m && (m.constructor === Object || m.constructor === Array)) {
        var E = [];
        for (var T in m)
          E.push({ content: T, value: m[T] });
        return new X(E, {
          event: y,
          callback: g,
          parentMenu: n,
          node: s
        }), !1;
      } else {
        const O = d.value;
        s.graph.beforeChange(), s.addOutput(O.name, O.type, O.options), s.onNodeOptionalOutputAdd && s.onNodeOptionalOutputAdd(d.value), s.setDirtyCanvas(!0, !0), s.graph.afterChange();
      }
    }
  };
  return new X(
    h,
    {
      event: i,
      callback: g,
      parentMenu: n,
      node: s
    },
    a
  ), !1;
};
C.onMenuResizeNode = function(e, t, i, n, s) {
  if (s) {
    var r = function(l) {
      l.size = l.computeSize(), l.onResize && l.onResize(l.size);
    }, o = N.active_canvas;
    if (!o.selected_nodes || Object.keys(o.selected_nodes).length <= 1)
      r(s);
    else
      for (var a in o.selected_nodes)
        r(o.selected_nodes[a]);
    s.setDirtyCanvas(!0, !0);
  }
};
C.onShowMenuNodeProperties = function(e, t, i, n, s) {
  if (!s || !s.properties)
    return;
  var r = N.active_canvas, o = r.getCanvasWindow(), a = [];
  for (var l in s.properties) {
    var h = s.properties[l] !== void 0 ? s.properties[l] : " ";
    typeof h == "object" && (h = JSON.stringify(h));
    var p = s.getPropertyInfo(l);
    (p.type == "enum" || p.type == "combo") && (h = N.getPropertyPrintableValue(h, p.values)), h = N.decodeHTML(h), a.push({
      content: "<span class='property_name'>" + (p.label ? p.label : l) + "</span><span class='property_value'>" + h + "</span>",
      value: l
    });
  }
  if (!a.length)
    return;
  new X(
    a,
    {
      event: i,
      callback: f,
      parentMenu: n,
      allow_html: !0,
      node: s
    },
    o
  );
  function f(c, v, g, d) {
    if (s) {
      var _ = this.getBoundingClientRect();
      r.showEditPropertyValue(s, c.value, {
        position: [_.left, _.top]
      });
    }
  }
  return !1;
};
C.onResizeNode = function(e, t, i, n, s) {
  s && (s.size = s.computeSize(), s.setDirtyCanvas(!0, !0));
};
C.onMenuNodeCollapse = function(e, t, i, n, s) {
  s.graph.beforeChange(
    /*?*/
  );
  var r = function(l) {
    l.collapse();
  }, o = N.active_canvas;
  if (!o.selected_nodes || Object.keys(o.selected_nodes).length <= 1)
    r(s);
  else
    for (var a in o.selected_nodes)
      r(o.selected_nodes[a]);
  s.graph.afterChange(
    /*?*/
  );
};
C.onMenuNodePin = function(e, t, i, n, s) {
  s.pin();
};
C.onMenuNodeMode = function(e, t, i, n, s) {
  let r = Array.from(re).map((a) => ({ content: a }));
  new X(
    r,
    { event: i, callback: o, parentMenu: n, node: s }
  );
  function o(a) {
    if (s) {
      var l = Object.values(re).indexOf(a.content), h = function(c) {
        l >= Z.ALWAYS && re[l] ? c.changeMode(l) : (console.warn("unexpected mode: " + a), c.changeMode(Z.ALWAYS));
      }, p = N.active_canvas;
      if (!p.selected_nodes || Object.keys(p.selected_nodes).length <= 1)
        h(s);
      else
        for (var f in p.selected_nodes)
          h(p.selected_nodes[f]);
    }
  }
  return !1;
};
C.onMenuNodeColors = function(e, t, i, n, s) {
  if (!s)
    throw "no node for color";
  var r = [];
  r.push({
    value: null,
    content: "<span style='display: block; padding-left: 4px;'>No color</span>"
  });
  for (let l in N.node_colors) {
    var o = N.node_colors[l];
    let h = {
      value: l,
      content: "<span style='display: block; color: #999; padding-left: 4px; border-left: 8px solid " + o.color + "; background-color:" + o.bgcolor + "'>" + l + "</span>"
    };
    r.push(h);
  }
  new X(r, {
    event: i,
    callback: a,
    parentMenu: n,
    node: s,
    allow_html: !0
  });
  function a(l) {
    if (s) {
      var h = l.value ? N.node_colors[l.value] : null, p = function(v) {
        h ? v instanceof me ? v.color = h.groupcolor : (v.color = h.color, v.bgcolor = h.bgcolor) : (delete v.color, v instanceof ae && delete v.bgcolor);
      }, f = N.active_canvas;
      if (!f.selected_nodes || Object.keys(f.selected_nodes).length <= 1)
        p(s);
      else
        for (var c in f.selected_nodes)
          p(f.selected_nodes[c]);
      s.setDirtyCanvas(!0, !0);
    }
  }
  return !1;
};
C.onMenuNodeShapes = function(e, t, i, n, s) {
  if (!s)
    throw "no node passed";
  const r = Array.from(Ie).map((a) => ({ content: a }));
  new X(r, {
    event: i,
    callback: o,
    parentMenu: n,
    node: s
  });
  function o(a) {
    if (s) {
      s.graph.beforeChange(
        /*?*/
      );
      var l = function(f) {
        f.shape = Ie.indexOf(a.content);
      }, h = N.active_canvas;
      if (!h.selected_nodes || Object.keys(h.selected_nodes).length <= 1)
        l(s);
      else
        for (var p in h.selected_nodes)
          l(h.selected_nodes[p]);
      s.graph.afterChange(
        /*?*/
      ), s.setDirtyCanvas(!0);
    }
  }
  return !1;
};
C.onMenuNodeRemove = function(e, t, i, n, s) {
  if (!s)
    throw "no node passed";
  var r = s.graph;
  r.beforeChange();
  var o = function(h) {
    h.removable !== !1 && r.remove(h);
  }, a = N.active_canvas;
  if (!a.selected_nodes || Object.keys(a.selected_nodes).length <= 1)
    o(s);
  else
    for (var l in a.selected_nodes)
      o(a.selected_nodes[l]);
  r.afterChange(), s.setDirtyCanvas(!0, !0);
};
C.onMenuNodeToSubgraph = function(e, t, i, n, s) {
  var r = s.graph, o = N.active_canvas;
  if (o) {
    var a = Object.values(o.selected_nodes || {});
    a.length || (a = [s]);
    var l = u.createNode("graph/subgraph", null, { constructorArgs: [null] });
    l.pos = s.pos.concat(), r.add(l), l.buildFromNodes(a), o.deselectAllNodes(), s.setDirtyCanvas(!0, !0);
  }
};
C.onMenuNodeToSubgraphInputs = function(e, t, i, n, s) {
  var r = N.active_canvas;
  if (!r)
    return;
  const o = s.graph._subgraph_node;
  if (!s.graph._is_subgraph || !o) {
    console.error("[To Subgraph Inputs] Current graph is not a subgraph!", s.graph);
    return;
  }
  let a = Object.values(r.selected_nodes || {});
  a.length || (a = [s]), o.convertNodesToSubgraphInputs(a), r.deselectAllNodes(), s.setDirtyCanvas(!0, !0);
};
C.onMenuNodeToSubgraphOutputs = function(e, t, i, n, s) {
  var r = N.active_canvas;
  if (!r)
    return;
  const o = s.graph._subgraph_node;
  if (!s.graph._is_subgraph || !o) {
    console.error("[To Subgraph Outputs] Current graph is not a subgraph!", s.graph);
    return;
  }
  let a = Object.values(r.selected_nodes || {});
  a.length || (a = [s]), o.convertNodesToSubgraphOutputs(a), r.deselectAllNodes(), s.setDirtyCanvas(!0, !0);
};
C.onMenuNodeToParentGraph = function(e, t, i, n, s) {
  var r = N.active_canvas;
  if (!r)
    return;
  const o = s.graph._subgraph_node;
  if (!s.graph._is_subgraph || !o) {
    console.error("[To Parent Graph] Current graph is not a subgraph!", s.graph);
    return;
  }
  let a = Object.values(r.selected_nodes || {});
  a.length || (a = [s]), o.moveNodesToParentGraph(a), r.deselectAllNodes(), s.setDirtyCanvas(!0, !0);
};
C.onMenuNodeClone = function(e, t, i, n, s) {
  var r = N.active_canvas;
  (!r.selected_nodes || Object.keys(r.selected_nodes).length <= 1) && r.selectNode(s), r.cloneSelection();
};
const ce = class {
  constructor(e, t, i = {}) {
    this.link_type_colors = {}, this.node_panel = null, this.options_panel = null, this.render_time = 0, this.allow_dragcanvas = !0, this.allow_dragnodes = !0, this.allow_interaction = !0, this.allow_reconnect_links = !0, this.allow_searchbox = !0, this.always_render_background = !1, this.background_image = ce.DEFAULT_BACKGROUND_IMAGE, this.block_click = !1, this.clear_background = !0, this.clear_background_color = "#222", this.connecting_pos = null, this.connecting_slot = null, this.connecting_input = null, this.connecting_output = null, this.connections_width = 3, this.current_node = null, this.drag_mode = !1, this.dragging_rectangle = null, this.ds = new Be(), this.editor_alpha = 1, this.filter = null, this.highquality_render = !0, this.skip_events = !1, this.last_mouse_position = [0, 0], this.last_click_position = [0, 0], this.last_click_position_offset = [0, 0], this.last_mouse_dragging = !1, this.links_render_mode = de.SPLINE_LINK, this.live_mode = !1, this.mouse = [0, 0], this.offset_mouse = [0, 0], this.graph_mouse = [0, 0], this.node_widget = null, this.maxZoom = null, this.minZoom = null, this.multi_select = !1, this.over_link_center = null, this.pause_rendering = !1, this.read_only = !1, this.render_canvas_border = !0, this.render_collapsed_slots = !0, this.render_connection_arrows = !1, this.render_connections_border = !0, this.render_connections_shadows = !1, this.render_connections = !0, this.render_curved_connections = !1, this.render_execution_order = !1, this.render_link_tooltip = !0, this.render_only_selected = !0, this.render_shadows = !0, this.render_title_colored = !0, this.render_subgraph_panels = !0, this.render_subgraph_stack_header = !0, this.round_radius = 8, this.set_canvas_dirty_on_mouse_event = !0, this.show_info = !0, this.use_gradients = !1, this.visible_links = [], this.zoom_modify_alpha = !0, this.pointer_is_down = !1, this.pointer_is_double = !1, this._highlight_input = null, this._highlight_input_slot = null, this._highlight_output = null, this._graph_stack = [], this._bg_img = null, this._pattern = null, this._pattern_img = null, this.search_box = null, this.prompt_box = null, this._events_binded = !1, this.resizing_node = null, typeof e == "string" && (e = document.querySelector(e)), this.skip_events = i.skip_events || !1, this.title_text_font = "" + u.NODE_TEXT_SIZE + "px Arial", this.inner_text_font = "normal " + u.NODE_SUBTEXT_SIZE + "px Arial", this.node_title_color = u.NODE_TITLE_COLOR, this.default_link_color = u.LINK_COLOR, this.link_type_colors = u.cloneObject(ce.DEFAULT_LINK_TYPE_COLORS), this.canvas_mouse = this.graph_mouse, this.visible_area = this.ds.visible_area, this.viewport = i.viewport || null, t && t.attachCanvas(this), this.setCanvas(e, i.skip_events), this.clear(), i.skip_render || this.startRendering(), this.autoresize = i.autoresize;
  }
  static getFileExtension(e) {
    var t = e.indexOf("?");
    t != -1 && (e = e.substr(0, t));
    var i = e.lastIndexOf(".");
    return i == -1 ? "" : e.substr(i + 1).toLowerCase();
  }
  static decodeHTML(e) {
    var t = document.createElement("div");
    return t.innerText = e, t.innerHTML;
  }
  static getPropertyPrintableValue(e, t) {
    if (!t || t.constructor === Array)
      return String(e);
    if (t.constructor === Object) {
      var i = "";
      for (var n in t)
        if (t[n] == e) {
          i = n;
          break;
        }
      return String(e) + " (" + i + ")";
    }
  }
  get scale() {
    return this.ds.scale;
  }
  set scale(e) {
    this.ds.scale = e;
  }
  /** clears all the data inside */
  clear() {
    this.frame = 0, this.last_draw_time = 0, this.render_time = 0, this.fps = 0, this.dragging_rectangle = null, this.selected_nodes = {}, this.selected_group = null, this.visible_nodes = [], this.node_dragged = null, this.node_over = null, this.node_capturing_input = null, this.connecting_node = null, this.highlighted_links = {}, this.dragging_canvas = !1, this.dirty_canvas = !0, this.dirty_bgcanvas = !0, this.dirty_area = null, this.node_in_panel = null, this.node_widget = null, this.last_mouse = [0, 0], this.last_mouseclick = 0, this.pointer_is_down = !1, this.pointer_is_double = !1, this.onClear && this.onClear();
  }
  /** assigns a graph, you can reassign graphs to the same canvas */
  setGraph(e, t = !1) {
    if (this.graph != e) {
      if (t || this.clear(), !e && this.graph) {
        this.graph.detachCanvas(this);
        return;
      }
      e.attachCanvas(this), this._graph_stack && (this._graph_stack = null), this.setDirty(!0, !0);
    }
  }
  /** opens a graph contained inside a node in the current graph */
  openSubgraph(e) {
    if (!e)
      throw "graph cannot be null";
    if (this.graph == e)
      throw "graph cannot be the same";
    if (this.clear(), this.graph) {
      this._graph_stack || (this._graph_stack = []);
      const i = [this.ds.offset[0], this.ds.offset[1]];
      this._graph_stack.push({ graph: this.graph, offset: i, scale: this.ds.scale });
    }
    u.debug && (console.warn("SubGraph opened", e), console.warn("Graph inputs", e.inputs), console.warn("Graph outputs", e.outputs)), e.attachCanvas(this);
    const t = [0, 0];
    if (e._nodes.length > 0) {
      let i = Number.MAX_SAFE_INTEGER, n = 0, s = Number.MAX_SAFE_INTEGER, r = 0;
      for (const o of e.iterateNodesInOrder())
        i = Math.min(o.pos[0], i), n = Math.max(o.pos[0] + o.size[0], n), s = Math.min(o.pos[1], s), r = Math.max(o.pos[1] + o.size[1], r);
      t[0] = -(i + (n - i) / 2) + this.canvas.width / 2, t[1] = -(s + (r - s) / 2) + this.canvas.height / 2;
    }
    this.ds.offset = t, this.ds.scale = 1, this.checkPanels(), this.setDirty(!0, !0);
  }
  closeAllSubgraphs() {
    for (; this._graph_stack && this._graph_stack.length > 0; )
      this.closeSubgraph();
  }
  /** closes a subgraph contained inside a node */
  closeSubgraph() {
    if (!(!this._graph_stack || this._graph_stack.length == 0)) {
      var e = this.graph._subgraph_node, { graph: t, offset: i, scale: n } = this._graph_stack.pop();
      this.selected_nodes = {}, this.highlighted_links = {}, t.attachCanvas(this), this.setDirty(!0, !0), e && (this.centerOnNode(e), this.selectNodes([e])), this.ds.offset = i, this.ds.scale = n;
    }
  }
  /** assigns a canvas */
  setCanvas(e, t = !1) {
    if (e && typeof e == "string" && (e = document.getElementById(e), !e))
      throw "Error creating LiteGraph canvas: Canvas not found";
    if (e = e, e !== this.canvas && (!e && this.canvas && (t || this.unbindEvents()), this.canvas = e, this.ds.element = e, !!e)) {
      if (e.className += " lgraphcanvas", e.data = this, e.tabIndex = 1, this.bgcanvas = null, this.bgcanvas || (this.bgcanvas = document.createElement("canvas"), this.bgcanvas.width = this.canvas.width, this.bgcanvas.height = this.canvas.height), e.getContext == null)
        throw e.localName != "canvas" ? "Element supplied for LGraphCanvas must be a <canvas> element, you passed a " + e.localName : "This browser doesn't support Canvas";
      t || this.bindEvents(), this.adjustCanvasForHiDPI();
    }
  }
  //used in some events to capture them
  _doNothing(e) {
    return e.preventDefault(), !1;
  }
  _doReturnTrue(e) {
    return e.preventDefault(), !0;
  }
  /** binds mouse, keyboard, touch and drag events to the canvas */
  bindEvents() {
    if (this._events_binded) {
      console.warn("LGraphCanvas: events already binded");
      return;
    }
    var e = this.canvas, t = this.getCanvasWindow(), i = t.document;
    this._mousedown_callback = this.processMouseDown.bind(this), this._mousewheel_callback = this.processMouseWheel.bind(this), this._mousemove_callback = this.processMouseMove.bind(this), this._mouseup_callback = this.processMouseUp.bind(this), u.pointerListenerAdd(e, "down", this._mousedown_callback, !0), e.addEventListener("mousewheel", this._mousewheel_callback, !1), u.pointerListenerAdd(e, "up", this._mouseup_callback, !0), u.pointerListenerAdd(e, "move", this._mousemove_callback), e.addEventListener("contextmenu", this._doNothing), e.addEventListener(
      "DOMMouseScroll",
      this._mousewheel_callback,
      !1
    ), this._key_callback = this.processKey.bind(this), e.addEventListener("keydown", this._key_callback, !0), i.addEventListener("keyup", this._key_callback, !0), this._ondrop_callback = this.processDrop.bind(this), e.addEventListener("dragover", this._doNothing, !1), e.addEventListener("dragend", this._doNothing, !1), e.addEventListener("drop", this._ondrop_callback, !1), e.addEventListener("dragenter", this._doReturnTrue, !1), this._events_binded = !0;
  }
  /** unbinds mouse events from the canvas */
  unbindEvents() {
    if (!this._events_binded) {
      console.warn("LGraphCanvas: no events binded");
      return;
    }
    u.debug && console.log("pointerevents: unbindEvents");
    var e = this.getCanvasWindow(), t = e.document;
    u.pointerListenerRemove(this.canvas, "move", this._mousedown_callback), u.pointerListenerRemove(this.canvas, "up", this._mousedown_callback), u.pointerListenerRemove(this.canvas, "down", this._mousedown_callback), this.canvas.removeEventListener(
      "mousewheel",
      this._mousewheel_callback
    ), this.canvas.removeEventListener(
      "DOMMouseScroll",
      this._mousewheel_callback
    ), this.canvas.removeEventListener("keydown", this._key_callback), t.removeEventListener("keyup", this._key_callback), this.canvas.removeEventListener("contextmenu", this._doNothing), this.canvas.removeEventListener("drop", this._ondrop_callback), this.canvas.removeEventListener("dragenter", this._doReturnTrue), this._mousedown_callback = null, this._mousewheel_callback = null, this._key_callback = null, this._ondrop_callback = null, this._events_binded = !1;
  }
  /**
   * this function allows to render the canvas using WebGL instead of Canvas2D
   * this is useful if you plant to render 3D objects inside your nodes, it uses litegl.js for webgl and canvas2DtoWebGL to emulate the Canvas2D calls in webGL
   **/
  enableWebGL() {
  }
  /**
   * marks as dirty the canvas, this way it will be rendered again
   * @param fg if the foreground canvas is dirty (the one containing the nodes)
   * @param bg if the background canvas is dirty (the one containing the wires)
   */
  setDirty(e = !1, t = !1) {
    e && (this.dirty_canvas = !0), t && (this.dirty_bgcanvas = !0);
  }
  /**
   * Used to attach the canvas in a popup
   * @return the window where the canvas is attached (the DOM root node)
   */
  getCanvasWindow() {
    if (!this.canvas)
      return window;
    var e = this.canvas.ownerDocument;
    return e.defaultView;
  }
  adjustCanvasForHiDPI(e) {
    if (e || (e = window.devicePixelRatio), e == 1 || !this.canvas.parentNode)
      return;
    const t = this.canvas.parentNode.getBoundingClientRect(), { width: i, height: n } = t;
    this.canvas.width = i * e, this.canvas.height = n * e, this.canvas.style.width = i + "px", this.canvas.style.height = n + "px", this.canvas.getContext("2d").scale(e, e);
  }
  /** starts rendering the content of the canvas when needed */
  startRendering() {
    if (this.is_rendering)
      return;
    this.is_rendering = !0, e.call(this);
    function e() {
      this.pause_rendering || this.draw();
      var t = this.getCanvasWindow();
      this.is_rendering && t.requestAnimationFrame(e.bind(this));
    }
  }
  /** stops rendering the content of the canvas (to save resources) */
  stopRendering() {
    this.is_rendering = !1;
  }
  //used to block future mouse events (because of im gui)
  blockClick() {
    this.block_click = !0, this.last_mouseclick = 0;
  }
  createDefaultNodeForSlot(e, t = {}) {
    var i = this, n = t.nodeFrom && t.slotFrom !== null, s = !n && t.nodeTo && t.slotTo !== null;
    if (t = { ...{
      position: [0, 0],
      posAdd: [0, 0],
      posSizeFix: [0, 0]
    }, ...t }, !n && !s)
      return console.warn("No data passed to createDefaultNodeForSlot " + t.nodeFrom + " " + t.slotFrom + " " + t.nodeTo + " " + t.slotTo), !1;
    if (!e)
      return console.warn("No type to createDefaultNodeForSlot"), !1;
    var o = n ? t.nodeFrom : t.nodeTo, a = n ? t.slotFrom : t.slotTo, l = null;
    switch (typeof a) {
      case "string":
        l = n ? o.findOutputSlotIndexByName(a) : o.findInputSlotIndexByName(a), a = n ? o.outputs[a] : o.inputs[a];
        break;
      case "object":
        l = n ? o.findOutputSlotIndexByName(a.name) : o.findInputSlotIndexByName(a.name);
        break;
      case "number":
        l = a, a = n ? o.outputs[a] : o.inputs[a];
        break;
      case "undefined":
      default:
        return console.warn("Cant get slot information " + a), !1;
    }
    a = a, (!a || !l) && console.warn("createDefaultNodeForSlot bad slotX " + a + " " + l);
    var h = a.type == I.EVENT ? "_event_" : a.type, p = n ? u.slot_types_default_out : u.slot_types_default_in;
    const f = p[h];
    if (p && f) {
      a.link !== null || a.links && a.links.length > 0;
      let _ = null;
      if (Array.isArray(f)) {
        for (var c in f)
          if (e == p[h][c] || e == "AUTO") {
            _ = p[h][c], u.debug && console.log("opts.nodeType == slotTypesDefault[fromSlotType][typeX] :: " + e);
            break;
          }
      } else
        throw new Error(`Invalid default slot specifier, must be an array: ${f}`);
      if (_) {
        var v = null;
        typeof _ == "object" && _.node && (v = _, _ = _.node);
        var g = u.createNode(_);
        if (g) {
          if (v) {
            if (v.properties)
              for (var d in v.properties)
                g.addProperty(d, v.properties[d]);
            if (v.inputs) {
              g.inputs = [];
              for (var d in v.inputs)
                g.addOutput(
                  v.inputs[d][0],
                  v.inputs[d][1]
                );
            }
            if (v.outputs) {
              g.outputs = [];
              for (var d in v.outputs)
                g.addOutput(
                  v.outputs[d][0],
                  v.outputs[d][1]
                );
            }
            v.title && (g.title = v.title), v.json && g.configure(v.json);
          }
          console.warn("PLACING", g.type, t);
          const y = t.position[0] + t.posAdd[0] + (t.posSizeFix[0] ? t.posSizeFix[0] * g.size[0] : 0), b = t.position[1] + t.posAdd[1] + (t.posSizeFix[1] ? t.posSizeFix[1] * g.size[1] : 0), m = [y, b];
          return i.graph.add(g, { pos: m }), n ? t.nodeFrom.connectByTypeInput(l, g, h) : t.nodeTo.connectByTypeOutput(l, g, h), n && s && console.debug("connecting in between"), !0;
        } else
          console.log("failed creating " + _);
      }
    }
    return !1;
  }
  /** returns true if a position (in graph space) is on top of a node little corner box */
  isOverNodeBox(e, t, i) {
    var n = u.NODE_TITLE_HEIGHT;
    return !!u.isInsideRectangle(
      t,
      i,
      e.pos[0] + 2,
      e.pos[1] + 2 - n,
      n - 4,
      n - 4
    );
  }
  /** returns slot index if a position (in graph space) is on top of a node input slot */
  isOverNodeInput(e, t, i, n) {
    if (e.inputs)
      for (var s = 0, r = e.inputs.length; s < r; ++s) {
        var o = e.getConnectionPos(!0, s), a = !1;
        if (e.horizontal ? a = u.isInsideRectangle(
          t,
          i,
          o[0] - 5,
          o[1] - 10,
          10,
          20
        ) : a = u.isInsideRectangle(
          t,
          i,
          o[0] - 10,
          o[1] - 5,
          40,
          10
        ), a)
          return n && (n[0] = o[0], n[1] = o[1]), s;
      }
    return -1;
  }
  /**
   * returns the INDEX if a position (in graph space) is on top of a node output slot
   * @method isOverNodeOuput
   **/
  isOverNodeOutput(e, t, i, n) {
    if (e.outputs)
      for (var s = 0, r = e.outputs.length; s < r; ++s) {
        e.outputs[s];
        var o = e.getConnectionPos(!1, s), a = !1;
        if (e.horizontal ? a = u.isInsideRectangle(
          t,
          i,
          o[0] - 5,
          o[1] - 10,
          10,
          20
        ) : a = u.isInsideRectangle(
          t,
          i,
          o[0] - 10,
          o[1] - 5,
          40,
          10
        ), a)
          return n && (n[0] = o[0], n[1] = o[1]), s;
      }
    return -1;
  }
  findLinkCenterAtPos(e, t) {
    for (let i = 0; i < this.visible_links.length; ++i) {
      const n = this.visible_links[i];
      if (this.graph && this.graph.links[n.id] == null)
        continue;
      const s = n._pos;
      if (!(!s || e < s[0] - 4 || e > s[0] + 4 || t < s[1] - 4 || t > s[1] + 4))
        return n;
    }
    return null;
  }
  /** process a key event */
  processKey(e) {
    if (!this.graph)
      return;
    var t = !1;
    if (u.debug && console.log("processKey", e), e.target instanceof Element && e.target.localName == "input")
      return;
    const i = this.allow_interaction && !this.read_only;
    if (e.type == "keydown") {
      if (e.keyCode == 32 && !(e.metaKey || e.ctrlKey || e.shiftKey) && (this.dragging_canvas = !0, t = !0), e.keyCode == 27 && !(e.metaKey || e.ctrlKey || e.shiftKey) && (this.node_panel && this.node_panel.close(), this.options_panel && this.options_panel.close(), t = !0), i && (e.keyCode == 65 && e.ctrlKey && (this.selectNodes(), t = !0), e.code == "KeyX" && (e.metaKey || e.ctrlKey) && !e.shiftKey && this.selected_nodes && (this.cutToClipboard(), t = !0), e.code == "KeyC" && (e.metaKey || e.ctrlKey) && !e.shiftKey && this.selected_nodes && (this.copyToClipboard(), t = !0), e.code == "KeyV" && (e.metaKey || e.ctrlKey) && !e.shiftKey && this.pasteFromClipboard(), e.code == "KeyD" && (e.metaKey || e.ctrlKey) && !e.shiftKey && (this.cloneSelection(), t = !0), (e.keyCode == 46 || e.keyCode == 8) && e.target instanceof Element && e.target.localName != "input" && e.target.localName != "textarea" && (this.deleteSelectedNodes(), t = !0), this.selected_nodes))
        for (var n in this.selected_nodes)
          this.selected_nodes[n].onKeyDown && this.selected_nodes[n].onKeyDown(e);
    } else if (e.type == "keyup" && (e.keyCode == 32 && (this.dragging_canvas = !1), i && this.selected_nodes))
      for (var n in this.selected_nodes)
        this.selected_nodes[n].onKeyUp && this.selected_nodes[n].onKeyUp(e);
    if (this.graph.change(), t)
      return e.preventDefault(), e.stopImmediatePropagation(), !1;
  }
  cutToClipboard() {
    this.copyToClipboard(), this.deleteSelectedNodes();
  }
  copyToClipboard() {
    var e = {
      nodes: [],
      nodeCloneData: {},
      links: []
    }, t = 0, i = [];
    for (var n in this.selected_nodes) {
      var s = this.selected_nodes[n];
      s._relative_id = t, i.push(s), t += 1;
    }
    for (let h = 0; h < i.length; ++h) {
      let p = i[h];
      if (!p.clonable)
        continue;
      const f = { forNode: {} };
      let c = p.clone(f);
      if (!c) {
        console.warn("node type not found: " + p.type);
        continue;
      }
      if (e.nodes.push(c.serialize()), e.nodeCloneData[c.id] = {
        prevNodeID: p.id,
        cloneData: f
      }, p.inputs && p.inputs.length)
        for (var r = 0; r < p.inputs.length; ++r) {
          var o = p.inputs[r];
          if (!(!o || o.link == null)) {
            var a = this.graph.links[o.link];
            if (a) {
              var l = this.graph.getNodeById(
                a.origin_id
              );
              !l || !this.selected_nodes[l.id] || !this.selected_nodes[l.id].clonable || e.links.push([
                l._relative_id,
                a.origin_slot,
                //j,
                p._relative_id,
                a.target_slot
              ]);
            }
          }
        }
    }
    localStorage.setItem(
      "litegrapheditor_clipboard",
      JSON.stringify(e)
    );
  }
  pasteFromClipboard() {
    var e = localStorage.getItem("litegrapheditor_clipboard");
    if (e) {
      this.graph.beforeChange();
      for (var t = JSON.parse(e), i = null, n = null, s = 0; s < t.nodes.length; ++s)
        i ? (i[0] > t.nodes[s].pos[0] && (i[0] = t.nodes[s].pos[0], n[0] = s), i[1] > t.nodes[s].pos[1] && (i[1] = t.nodes[s].pos[1], n[1] = s)) : (i = [t.nodes[s].pos[0], t.nodes[s].pos[1]], n = [s, s]);
      for (var r = [], s = 0; s < t.nodes.length; ++s) {
        var o = t.nodes[s], a = u.createNode(o.type);
        if (a) {
          a.configure(o), a.pos[0] += this.graph_mouse[0] - i[0], a.pos[1] += this.graph_mouse[1] - i[1];
          const { cloneData: c, prevNodeID: v } = t.nodeCloneData[a.id];
          this.graph.add(a, { doProcessChange: !1, addedBy: "paste", prevNodeID: v, cloneData: c }), r.push(a);
        }
      }
      for (var s = 0; s < t.links.length; ++s) {
        var l = t.links[s], h = r[l[0]], p = r[l[2]];
        h && p ? h.connect(l[1], p, l[3]) : console.warn("Warning, nodes missing on pasting");
      }
      this.selectNodes(r), this.graph.afterChange();
    }
  }
  cloneSelection() {
    if (!this.selected_nodes || Object.keys(this.selected_nodes).length === 0)
      return;
    this.graph.beforeChange();
    const e = {}, t = [], i = {};
    for (const r of Object.values(this.selected_nodes))
      for (const o of r.iterateAllLinks())
        this.selected_nodes[o.origin_id] && this.selected_nodes[o.target_id] && t.push(o);
    const n = function(r) {
      if (r.clonable == !1)
        return;
      const o = r.id, a = { forNode: {} }, l = r.clone(a);
      l && (i[o] = l, l.pos = [r.pos[0] + 5, r.pos[1] + 5], r.graph.add(l, { addedBy: "cloneSelection", prevNodeID: o, prevNode: r, cloneData: a }), e[l.id] = l);
    };
    for (var s in this.selected_nodes)
      n(this.selected_nodes[s]);
    for (const r of t) {
      const o = i[r.origin_id], a = i[r.target_id];
      o && a && o.connect(r.origin_slot, a, r.target_slot);
    }
    Object.keys(e).length && this.selectNodes(Object.values(e)), this.graph.afterChange(), this.setDirty(!0, !0);
  }
  processDrop(e) {
    let t = e;
    t.preventDefault(), this.adjustMouseEvent(t);
    var i = t.clientX, n = t.clientY, s = !this.viewport || this.viewport && i >= this.viewport[0] && i < this.viewport[0] + this.viewport[2] && n >= this.viewport[1] && n < this.viewport[1] + this.viewport[3];
    if (s) {
      var r = [t.canvasX, t.canvasY], o = this.graph ? this.graph.getNodeOnPos(r[0], r[1]) : null;
      if (!o) {
        var a = null;
        this.onDropItem && (a = this.onDropItem(t)), a || this.checkDropItem(t);
        return;
      }
      if (o.onDropFile || o.onDropData) {
        var l = t.dataTransfer.files;
        if (l && l.length)
          for (var h = 0; h < l.length; h++) {
            var p = t.dataTransfer.files[0], f = p.name;
            if (ce.getFileExtension(f), o.onDropFile && o.onDropFile(p), o.onDropData) {
              var c = new FileReader();
              c.onload = function(g) {
                var d = g.target.result;
                o.onDropData(d, f, p);
              };
              var v = p.type.split("/")[0];
              v == "text" || v == "" ? c.readAsText(p) : v == "image" ? c.readAsDataURL(p) : c.readAsArrayBuffer(p);
            }
          }
      }
      return !!(o.onDropItem && o.onDropItem(t) || this.onDropItem && this.onDropItem(t));
    }
  }
  checkDropItem(e) {
    let t = e;
    if (t.dataTransfer.files.length) {
      var i = t.dataTransfer.files[0], n = ce.getFileExtension(i.name).toLowerCase(), s = u.node_types_by_file_extension[n];
      if (s) {
        this.graph.beforeChange();
        var r = u.createNode(s.type);
        r.pos = [t.canvasX, t.canvasY], this.graph.add(r), r.onDropFile && r.onDropFile(i), this.graph.afterChange();
      }
    }
  }
  processNodeDblClicked(e) {
    this.onShowNodePanel ? this.onShowNodePanel(e) : this.showShowNodePanel(e), this.onNodeDblClicked && this.onNodeDblClicked(e), this.setDirty(!0);
  }
  processNodeSelected(e, t) {
    this.selectNode(e, t && (t.shiftKey || t.ctrlKey || this.multi_select)), this.onNodeSelected && this.onNodeSelected(e);
  }
  /** selects a given node (or adds it to the current selection) */
  selectNode(e, t = !1) {
    e == null ? this.deselectAllNodes() : this.selectNodes([e], t);
  }
  /** selects several nodes (or adds them to the current selection) */
  selectNodes(e, t = !1) {
    t || this.deselectAllNodes(), e = e || this.graph._nodes, typeof e == "string" && (e = [e]);
    for (var i in e) {
      var n = e[i];
      if (n.is_selected) {
        this.deselectNode(n);
        continue;
      }
      if (!n.is_selected && n.onSelected && n.onSelected(), n.is_selected = !0, this.selected_nodes[n.id] = n, n.inputs)
        for (var s = 0; s < n.inputs.length; ++s)
          this.highlighted_links[n.inputs[s].link] = !0;
      if (n.outputs)
        for (var s = 0; s < n.outputs.length; ++s) {
          var r = n.outputs[s];
          if (r.links)
            for (var o = 0; o < r.links.length; ++o)
              this.highlighted_links[r.links[o]] = !0;
        }
    }
    this.onSelectionChange && this.onSelectionChange(this.selected_nodes), this.setDirty(!0);
  }
  /** removes a node from the current selection */
  deselectNode(e) {
    if (e.is_selected) {
      if (e.onDeselected && e.onDeselected(), e.is_selected = !1, this.onNodeDeselected && this.onNodeDeselected(e), e.inputs)
        for (var t = 0; t < e.inputs.length; ++t)
          delete this.highlighted_links[e.inputs[t].link];
      if (e.outputs)
        for (var t = 0; t < e.outputs.length; ++t) {
          var i = e.outputs[t];
          if (i.links)
            for (var n = 0; n < i.links.length; ++n)
              delete this.highlighted_links[i.links[n]];
        }
    }
  }
  /** removes all nodes from the current selection */
  deselectAllNodes() {
    if (this.graph) {
      for (var e = this.graph._nodes, t = 0, i = e.length; t < i; ++t) {
        var n = e[t];
        n.is_selected && (n.onDeselected && n.onDeselected(), n.is_selected = !1, this.onNodeDeselected && this.onNodeDeselected(n));
      }
      this.selected_nodes = {}, this.current_node = null, this.highlighted_links = {}, this.onSelectionChange && this.onSelectionChange(this.selected_nodes), this.setDirty(!0);
    }
  }
  /** deletes all nodes in the current selection from the graph */
  deleteSelectedNodes() {
    this.graph.beforeChange();
    for (var e in this.selected_nodes) {
      var t = this.selected_nodes[e];
      if (!t.block_delete) {
        if (t.inputs && t.inputs.length && t.outputs && t.outputs.length && u.isValidConnection(t.inputs[0].type, t.outputs[0].type) && t.inputs[0].link && t.outputs[0].links && t.outputs[0].links.length) {
          var i = t.graph.links[t.inputs[0].link], n = t.graph.links[t.outputs[0].links[0]], s = t.getInputNode(0), r = t.getOutputNodes(0)[0];
          s && r && s.connect(i.origin_slot, r, n.target_slot);
        }
        this.graph.remove(t), this.onNodeDeselected && this.onNodeDeselected(t);
      }
    }
    this.selected_nodes = {}, this.current_node = null, this.highlighted_links = {}, this.setDirty(!0), this.graph.afterChange();
  }
  /** centers the camera on a given node */
  centerOnNode(e) {
    this.ds.offset[0] = -e.pos[0] - e.size[0] * 0.5 + this.canvas.width * 0.5 / this.ds.scale, this.ds.offset[1] = -e.pos[1] - e.size[1] * 0.5 + this.canvas.height * 0.5 / this.ds.scale, this.setDirty(!0, !0);
  }
  /**
   * adds some useful properties to a mouse event, like the position in graph coordinates
   * @method adjustMouseEvent
   **/
  adjustMouseEvent(e) {
    let t = e;
    var i = 0, n = 0;
    if (this.canvas) {
      var s = this.canvas.getBoundingClientRect();
      i = t.clientX - s.left, n = t.clientY - s.top;
    } else
      i = t.clientX, n = t.clientY;
    return this.last_mouse_position[0] = i, this.last_mouse_position[1] = n, t.canvasX = i / this.ds.scale - this.ds.offset[0], t.canvasY = n / this.ds.scale - this.ds.offset[1], t;
  }
  /** process an event on widgets */
  processNodeWidgets(e, t, i, n) {
    if (!e.widgets || !e.widgets.length || u.ignore_all_widget_events)
      return null;
    for (var s = t[0] - e.pos[0], r = t[1] - e.pos[1], o = e.size[0], a = this, l = this.getCanvasWindow(), h = 0; h < e.widgets.length; ++h) {
      var p = e.widgets[h];
      if (!(!p || p.disabled)) {
        var f = p.computeSize ? p.computeSize(o)[1] : u.NODE_WIDGET_HEIGHT, c = p.width || o;
        if (!(p != n && (s < 6 || s > c - 12 || r < p.last_y || r > p.last_y + f || p.last_y === void 0))) {
          var v = p.value;
          switch (p.type) {
            case "button":
              i.type === u.pointerevents_method + "down" && (p.callback && setTimeout(function() {
                p.callback(p, a, e, t, i);
              }, 20), p.clicked = !0, this.dirty_canvas = !0);
              break;
            case "slider":
              p.options.max - p.options.min;
              var g = Te((s - 15) / (c - 30), 0, 1);
              p.value = p.options.min + (p.options.max - p.options.min) * g, p.callback && setTimeout(function() {
                E(p, p.value);
              }, 20), this.dirty_canvas = !0;
              break;
            case "number":
            case "combo":
              var v = p.value;
              if (i.type == u.pointerevents_method + "move" && p.type == "number")
                i.deltaX && (p.value += i.deltaX * (p.options.step || 0.1)), p.options.min != null && p.value < p.options.min && (p.value = p.options.min), p.options.max != null && p.value > p.options.max && (p.value = p.options.max);
              else if (i.type == u.pointerevents_method + "down") {
                var d = p.options.values;
                if (d && typeof d == "function") {
                  let O = p.options.values;
                  d = O(p, e);
                }
                var _ = null;
                p.type != "number" && (_ = Array.isArray(d) ? d : Object.keys(d));
                var y = s < 40 ? -1 : s > c - 40 ? 1 : 0;
                if (p.type == "number")
                  p.value += y * (p.options.step || 0.1), p.options.min != null && p.value < p.options.min && (p.value = p.options.min), p.options.max != null && p.value > p.options.max && (p.value = p.options.max);
                else if (y) {
                  var b = -1;
                  this.last_mouseclick = 0, d.constructor === Object ? b = _.indexOf(String(p.value)) + y : b = _.indexOf(p.value) + y, b >= _.length && (b = _.length - 1), b < 0 && (b = 0), Array.isArray(d) ? p.value = d[b] : p.value = b;
                } else {
                  let O = function(M, L, B) {
                    let G = M.content;
                    return d != _ && (G = m.indexOf(G)), this.value = G, E(this, G), a.dirty_canvas = !0, !1;
                  };
                  var m = d != _ ? Object.values(d) : d;
                  let A = Array.from(m).map((M) => ({ content: M }));
                  new X(
                    A,
                    {
                      scale: Math.max(1, this.ds.scale),
                      event: i,
                      className: "dark",
                      callback: O.bind(p)
                    },
                    l
                  );
                }
              } else if (i.type == u.pointerevents_method + "up" && p.type == "number") {
                var y = s < 40 ? -1 : s > c - 40 ? 1 : 0;
                i.click_time < 200 && y == 0 && this.prompt(
                  "Value",
                  p.value,
                  function(A) {
                    this.value = Number(A), E(this, this.value);
                  }.bind(p),
                  i
                );
              }
              v != p.value && setTimeout(
                function() {
                  E(this, this.value);
                }.bind(p),
                20
              ), this.dirty_canvas = !0;
              break;
            case "toggle":
              i.type == u.pointerevents_method + "down" && (p.value = !p.value, setTimeout(function() {
                E(p, p.value);
              }, 20));
              break;
            case "string":
            case "text":
              i.type == u.pointerevents_method + "down" && this.prompt(
                "Value",
                p.value,
                function(O) {
                  this.value = O, E(this, O);
                }.bind(p),
                i,
                p.options ? p.options.multiline : !1,
                p.options.inputStyle
              );
              break;
            default:
              p.mouse && (this.dirty_canvas = p.mouse(i, [s, r], e));
              break;
          }
          return v != p.value && (e.onWidgetChanged && e.onWidgetChanged(p, v), e.graph._version++), p;
        }
      }
    }
    function E(T, O) {
      T.value = O, T.options && T.options.property && e.properties[T.options.property] !== void 0 && e.setProperty(T.options.property, O), T.callback && T.callback(T.value, a, e, t, i);
    }
    return null;
  }
  adjustNodesSize() {
    for (var e = this.graph._nodes, t = 0; t < e.length; ++t)
      e[t].size = e[t].computeSize();
    this.setDirty(!0, !0);
  }
  /** resizes the canvas to a given size, if no size is passed, then it tries to fill the parentNode */
  resize(e, t) {
    if (!e && !t) {
      var i = this.canvas.parentNode;
      e = i.offsetWidth, t = i.offsetHeight;
    }
    this.canvas.width == e && this.canvas.height == t || (this.canvas.width = e, this.canvas.height = t, this.bgcanvas.width = this.canvas.width, this.bgcanvas.height = this.canvas.height, this.adjustCanvasForHiDPI(), this.setDirty(!0, !0));
  }
  isAreaClicked(e, t, i, n, s) {
    var r = this.offset_mouse;
    u.isInsideRectangle(r[0], r[1], e, t, i, n), r = this.last_click_position;
    var o = r && u.isInsideRectangle(r[0], r[1], e, t, i, n), a = o && !this.block_click;
    return o && s && this.blockClick(), a;
  }
  /**
   * switches to live mode (node shapes are not rendered, only the content)
   * this feature was designed when graphs where meant to create user interfaces
   **/
  switchLiveMode(e) {
    if (!e) {
      this.live_mode = !this.live_mode, this.dirty_canvas = !0, this.dirty_bgcanvas = !0;
      return;
    }
    var t = this, i = this.live_mode ? 1.1 : 0.9;
    this.live_mode && (this.live_mode = !1, this.editor_alpha = 0.1);
    var n = setInterval(function() {
      t.editor_alpha *= i, t.dirty_canvas = !0, t.dirty_bgcanvas = !0, i < 1 && t.editor_alpha < 0.01 && (clearInterval(n), i < 1 && (t.live_mode = !0)), i > 1 && t.editor_alpha > 0.99 && (clearInterval(n), t.editor_alpha = 1);
    }, 1);
  }
  onNodeSelectionChange() {
  }
  touchHandler(e) {
  }
  convertOffsetToCanvas(e) {
    return this.ds.convertOffsetToCanvas(e);
  }
  convertCanvasToOffset(e, t = [0, 0]) {
    return this.ds.convertCanvasToOffset(e, t);
  }
  /** converts event coordinates from canvas2D to graph coordinates */
  convertEventToCanvasOffset(e) {
    var t = this.canvas.getBoundingClientRect();
    return this.convertCanvasToOffset([
      e.clientX - t.left,
      e.clientY - t.top
    ]);
  }
  addGraphInputNode(e, t, i) {
    const n = this.graph.findNodesByClass($).find((o) => o.properties.name === t);
    if (n) {
      this.selectNodes([n]);
      return;
    }
    (!i || i === "") && (i = "*");
    const s = [
      this.canvas.width * 0.25 / this.ds.scale - this.ds.offset[0],
      this.canvas.height * 0.5 / this.ds.scale - this.ds.offset[1]
    ];
    this.graph.beforeChange();
    const r = e.addGraphInput(t, i, s);
    if (r) {
      const o = r.innerNode;
      this.selectNodes([o]), this.graph.afterChange();
    } else
      console.error("graph input node not found:", i);
  }
  addGraphOutputNode(e, t, i) {
    const n = this.graph.findNodesByClass(Q).find((o) => o.properties.name === t);
    if (n) {
      this.selectNodes([n]);
      return;
    }
    (!i || i === "") && (i = "*");
    const s = [
      this.canvas.width * 0.75 / this.ds.scale - this.ds.offset[0],
      this.canvas.height * 0.5 / this.ds.scale - this.ds.offset[1]
    ];
    this.graph.beforeChange();
    const r = e.addGraphOutput(t, i, s);
    if (r) {
      const o = r.innerNode;
      this.selectNodes([o]), this.graph.afterChange();
    } else
      console.error("graph output node not found:", i);
  }
  getCanvasMenuOptions() {
    return C.prototype.getCanvasMenuOptions.apply(this, arguments);
  }
  getNodeMenuOptions(e) {
    return C.prototype.getNodeMenuOptions.apply(this, arguments);
  }
  getLinkMenuOptions(e) {
    return C.prototype.getLinkMenuOptions.apply(this, arguments);
  }
  getGroupMenuOptions(e) {
    return C.prototype.getGroupMenuOptions.apply(this, arguments);
  }
  checkPanels() {
    C.prototype.checkPanels.apply(this, arguments);
  }
  closePanels() {
    C.prototype.closePanels.apply(this, arguments);
  }
  createDialog(e, t) {
    return C.prototype.createDialog.apply(this, arguments);
  }
  createPanel(e, t = {}) {
    return C.prototype.createPanel.apply(this, arguments);
  }
  showSearchBox(e, t = {}) {
    return C.prototype.showSearchBox.apply(this, arguments);
  }
  prompt(e = "", t, i, n, s = !1, r = null) {
    return C.prototype.prompt.apply(this, arguments);
  }
  showConnectionMenu(e = {}) {
    return C.prototype.showConnectionMenu.apply(this, arguments);
  }
  showLinkMenu(e, t) {
    return C.prototype.showLinkMenu.apply(this, arguments);
  }
  showEditPropertyValue(e, t, i) {
    return C.prototype.showEditPropertyValue.apply(this, arguments);
  }
  showShowNodePanel(e) {
    C.prototype.showShowNodePanel.apply(this, arguments);
  }
  showSubgraphPropertiesDialog(e) {
    return C.prototype.showSubgraphPropertiesDialog.apply(this, arguments);
  }
  showSubgraphPropertiesDialogRight(e) {
    return C.prototype.showSubgraphPropertiesDialogRight.apply(this, arguments);
  }
  processContextMenu(e, t) {
    C.prototype.processContextMenu.apply(this, arguments);
  }
  /*
   * Events
   */
  processMouseMove(e) {
    return ge.prototype.processMouseMove.apply(this, arguments);
  }
  processMouseDown(e) {
    return ge.prototype.processMouseDown.apply(this, arguments);
  }
  processMouseUp(e) {
    return ge.prototype.processMouseUp.apply(this, arguments);
  }
  processMouseWheel(e) {
    return ge.prototype.processMouseWheel.apply(this, arguments);
  }
  /*
   * Rendering
   */
  setZoom(e, t) {
    U.prototype.setZoom.apply(this, arguments);
  }
  bringToFront(e) {
    U.prototype.bringToFront.apply(this, arguments);
  }
  sendToBack(e) {
    U.prototype.sendToBack.apply(this, arguments);
  }
  computeVisibleNodes(e, t = []) {
    return U.prototype.computeVisibleNodes.apply(this, arguments);
  }
  draw(e = !1, t = !1) {
    U.prototype.draw.apply(this, arguments);
  }
  drawFrontCanvas() {
    U.prototype.drawFrontCanvas.apply(this, arguments);
  }
  drawSubgraphPanel(e) {
    U.prototype.drawSubgraphPanel.apply(this, arguments);
  }
  drawSubgraphPanelLeft(e, t, i) {
    U.prototype.drawSubgraphPanelLeft.apply(this, arguments);
  }
  drawSubgraphPanelRight(e, t, i) {
    U.prototype.drawSubgraphPanelRight.apply(this, arguments);
  }
  drawButton(e, t, i, n, s, r = u.NODE_DEFAULT_COLOR, o = "#555", a = u.NODE_TEXT_COLOR, l = !0) {
    return U.prototype.drawButton.apply(this, arguments);
  }
  drawBackCanvas() {
    U.prototype.drawBackCanvas.apply(this, arguments);
  }
  renderInfo(e, t = 10, i) {
    U.prototype.renderInfo.apply(this, arguments);
  }
  drawNode(e, t) {
    U.prototype.drawNode.apply(this, arguments);
  }
  drawLinkTooltip(e, t) {
    U.prototype.drawLinkTooltip.apply(this, arguments);
  }
  drawNodeShape(e, t, i, n, s, r, o) {
    U.prototype.drawNodeShape.apply(this, arguments);
  }
  drawConnections(e) {
    U.prototype.drawConnections.apply(this, arguments);
  }
  renderLink(e, t, i, n, s, r, o, a, l, h) {
    U.prototype.renderLink.apply(this, arguments);
  }
  computeConnectionPoint(e, t, i, n = w.RIGHT, s = w.LEFT) {
    return U.prototype.computeConnectionPoint.apply(this, arguments);
  }
  drawExecutionOrder(e) {
    U.prototype.drawExecutionOrder.apply(this, arguments);
  }
  drawNodeWidgets(e, t, i, n) {
    U.prototype.drawNodeWidgets.apply(this, arguments);
  }
  drawGroups(e, t) {
    U.prototype.drawGroups.apply(this, arguments);
  }
  /*
   * ComfyUI Extension
   */
  updateBackground(e, t) {
    this._bg_img = new Image(), this._bg_img.name = e, this._bg_img.src = e, this._bg_img.onload = () => {
      this.draw(!0, !0);
    }, this.background_image = e, this.clear_background = !0, this.clear_background_color = t, this._pattern = null;
  }
};
let N = ce;
N.DEFAULT_BACKGROUND_IMAGE = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAIAAAD/gAIDAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAQBJREFUeNrs1rEKwjAUhlETUkj3vP9rdmr1Ysammk2w5wdxuLgcMHyptfawuZX4pJSWZTnfnu/lnIe/jNNxHHGNn//HNbbv+4dr6V+11uF527arU7+u63qfa/bnmh8sWLBgwYJlqRf8MEptXPBXJXa37BSl3ixYsGDBMliwFLyCV/DeLIMFCxYsWLBMwSt4Be/NggXLYMGCBUvBK3iNruC9WbBgwYJlsGApeAWv4L1ZBgsWLFiwYJmCV/AK3psFC5bBggULloJX8BpdwXuzYMGCBctgwVLwCl7Be7MMFixYsGDBsu8FH1FaSmExVfAxBa/gvVmwYMGCZbBg/W4vAQYA5tRF9QYlv/QAAAAASUVORK5CYII=";
N.node_colors = {
  red: { color: "#322", bgcolor: "#533", groupcolor: "#A88" },
  brown: { color: "#332922", bgcolor: "#593930", groupcolor: "#b06634" },
  green: { color: "#232", bgcolor: "#353", groupcolor: "#8A8" },
  blue: { color: "#223", bgcolor: "#335", groupcolor: "#88A" },
  pale_blue: { color: "#2a363b", bgcolor: "#3f5159", groupcolor: "#3f789e" },
  cyan: { color: "#233", bgcolor: "#355", groupcolor: "#8AA" },
  purple: { color: "#323", bgcolor: "#535", groupcolor: "#a1309b" },
  yellow: { color: "#432", bgcolor: "#653", groupcolor: "#b58b2a" },
  black: { color: "#222", bgcolor: "#000", groupcolor: "#444" }
};
N.DEFAULT_LINK_TYPE_COLORS = {
  [I.ACTION]: u.ACTION_LINK_COLOR,
  [I.EVENT]: u.EVENT_LINK_COLOR,
  number: "#AAA",
  node: "#DCA"
};
N.DEFAULT_CONNECTION_COLORS = {
  input_off: "#778",
  input_on: "#7F7",
  //"#BBD"
  output_off: "#778",
  output_on: "#7F7"
  //"#BBD"
};
N.DEFAULT_CONNECTION_COLORS_BY_TYPE = {
  number: "#7F7",
  string: "#77F",
  boolean: "#F77"
};
N.DEFAULT_CONNECTION_COLORS_BY_TYPE_OFF = {
  number: "#474",
  string: "#447",
  boolean: "#744"
};
N.active_canvas = null;
N.active_node = null;
N.onMenuCollapseAll = C.onMenuCollapseAll;
N.onMenuNodeEdit = C.onMenuNodeEdit;
N.onShowPropertyEditor = C.onShowPropertyEditor;
N.onGroupAdd = C.onGroupAdd;
N.onMenuAdd = C.onMenuAdd;
N.showMenuNodeOptionalInputs = C.showMenuNodeOptionalInputs;
N.showMenuNodeOptionalOutputs = C.showMenuNodeOptionalOutputs;
N.onShowMenuNodeProperties = C.onShowMenuNodeProperties;
N.onResizeNode = C.onResizeNode;
N.onMenuResizeNode = C.onMenuResizeNode;
N.onMenuNodeCollapse = C.onMenuNodeCollapse;
N.onMenuNodePin = C.onMenuNodePin;
N.onMenuNodeMode = C.onMenuNodeMode;
N.onMenuNodeColors = C.onMenuNodeColors;
N.onMenuNodeShapes = C.onMenuNodeShapes;
N.onMenuNodeRemove = C.onMenuNodeRemove;
N.onMenuNodeClone = C.onMenuNodeClone;
N.onMenuNodeToSubgraph = C.onMenuNodeToSubgraph;
N.onMenuNodeToSubgraphInputs = C.onMenuNodeToSubgraphInputs;
N.onMenuNodeToSubgraphOutputs = C.onMenuNodeToSubgraphOutputs;
N.onMenuNodeToParentGraph = C.onMenuNodeToParentGraph;
var j = /* @__PURE__ */ ((e) => (e[e.SEPARATOR = 0] = "SEPARATOR", e))(j || {});
class X {
  static trigger(t, i, n, s) {
    var r = document.createEvent("CustomEvent");
    return r.initCustomEvent(i, !0, !0, n), r.target = s, t.dispatchEvent && t.dispatchEvent(r), r;
  }
  static isCursorOverElement(t, i) {
    var n = t.clientX, s = t.clientY, r = i.getBoundingClientRect();
    return r ? s > r.top && s < r.top + r.height && n > r.left && n < r.left + r.width : !1;
  }
  static closeAllContextMenus(t) {
    t = t || window;
    var i = t.document.querySelectorAll(".litecontextmenu");
    if (i.length) {
      var n = Array.from(i);
      for (const s of n)
        s.close();
    }
  }
  constructor(t, i = {}, n) {
    var g;
    this.options = i;
    var s = this;
    i.parentMenu && (i.parentMenu.constructor !== this.constructor ? (console.error(
      "parentMenu must be of class ContextMenu, ignoring it"
    ), i.parentMenu = null) : (this.parentMenu = i.parentMenu, this.parentMenu.lock = !0, this.parentMenu.current_submenu = this));
    var r = null;
    i.event && (r = i.event.constructor.name), r !== "MouseEvent" && r !== "CustomEvent" && r !== "PointerEvent" && (console.error(
      "Event passed to ContextMenu is not of type MouseEvent or CustomEvent. Ignoring it. (" + r + ")"
    ), i.event = null);
    var o = document.createElement("div");
    if (o.className = "litegraph litecontextmenu litemenubar-panel", i.className && (o.className += " " + i.className), o.style.pointerEvents = "none", setTimeout(function() {
      o.style.pointerEvents = "auto";
    }, 100), u.pointerListenerAdd(
      o,
      "up",
      function(d) {
        return d.preventDefault(), !0;
      },
      !0
    ), o.addEventListener(
      "contextmenu",
      function(d) {
        return d.button != 2 || d.preventDefault(), !1;
      },
      !0
    ), o.close = () => {
      o.parentNode.removeChild(o);
    }, u.pointerListenerAdd(
      o,
      "down",
      function(d) {
        if (d.button == 2)
          return s.close(), d.preventDefault(), !0;
      },
      !0
    ), i.scroll_speed || (i.scroll_speed = 0.1), o.addEventListener("wheel", this.onMouseWheel.bind(this), !0), o.addEventListener("mousewheel", this.onMouseWheel.bind(this), !0), this.root = o, i.title) {
      var a = document.createElement("div");
      a.className = "litemenu-title", a.innerHTML = i.title, o.appendChild(a);
    }
    this.values = [];
    for (let d = 0; d < t.length; d++) {
      let _ = t[d], y = "";
      _ === 0 || _ == null ? y = "" : typeof _ == "string" ? y = _ : y = _.content, this.addItem(y, _, i);
    }
    u.pointerListenerAdd(o, "enter", function(d) {
      o.closing_timer && clearTimeout(o.closing_timer);
    });
    var l = document;
    i.event && i.event.target instanceof Node && (l = i.event.target.ownerDocument), l || (l = document), l.fullscreenElement ? l.fullscreenElement.appendChild(o) : l.body.appendChild(o);
    var h = i.left || 0, p = i.top || 0;
    if (i.event) {
      if (h = i.event.clientX - 10, p = i.event.clientY - 10, i.title && (p -= 20), i.parentMenu) {
        var f = i.parentMenu.root.getBoundingClientRect();
        h = f.left + f.width;
      }
      var c = document.body.getBoundingClientRect(), v = o.getBoundingClientRect();
      c.height == 0 && console.error("document.body height is 0. That is dangerous, set html,body { height: 100%; }"), c.width && h > c.width - v.width - 10 && (h = c.width - v.width - 10), c.height && p > c.height - v.height - 10 && (p = c.height - v.height - 10);
    }
    o.style.left = h + "px", o.style.top = p + "px", i.scale && (o.style.transform = "scale(" + i.scale + ")"), (g = u.onContextMenuCreated) == null || g.call(u, this);
  }
  addItem(t, i, n = {}) {
    var s = this, r = document.createElement("div");
    r.className = "litemenu-entry submenu";
    var o = !1;
    typeof i == "string" && (i = { content: i }), i === 0 || i == null ? r.classList.add("separator") : (r.innerHTML = i.title ? i.title : t, i.disabled && (o = !0, r.classList.add("disabled")), (i.submenu || i.has_submenu) && r.classList.add("has_submenu"), typeof i == "function" ? r.dataset.value = t : r.dataset.value = "" + this.values.length, i.className && (r.className += " " + i.className)), this.values.push(i), this.root.appendChild(r), o || r.addEventListener("click", h), n.autoopen && u.pointerListenerAdd(r, "enter", l);
    let a = this;
    function l(p) {
      var f = this.value;
      !f || !f.has_submenu || h.call(this, p);
    }
    function h(p) {
      let f = parseInt(this.dataset.value);
      var c = a.values[f];
      u.debug && console.debug("ContextMenu inner_onclick", f, c);
      const v = N.active_canvas;
      if (!v)
        return;
      const g = v.adjustMouseEvent(p);
      var d = !0;
      if (s.current_submenu && s.current_submenu.close(g), n.callback) {
        var _ = n.callback.call(
          this,
          c,
          n,
          g,
          s,
          n.node
        );
        _ === !0 && (d = !1);
      }
      if (c && typeof c == "object") {
        if (c.callback && !n.ignore_item_callbacks && c.disabled !== !0) {
          var _ = c.callback.call(
            this,
            c,
            n,
            g,
            s,
            n.extra
          );
          _ === !0 && (d = !1);
        }
        if (c.submenu) {
          if (!c.submenu.options)
            throw "ContextMenu submenu needs options";
          new X(c.submenu.options, {
            callback: c.submenu.callback,
            event: g,
            parentMenu: s,
            ignore_item_callbacks: c.submenu.ignore_item_callbacks,
            title: c.submenu.title,
            extra: c.submenu.extra,
            autoopen: n.autoopen
          }), d = !1;
        }
      }
      d && !s.lock && s.close();
    }
    return r;
  }
  close(t, i) {
    this.root.parentNode && this.root.parentNode.removeChild(this.root), this.parentMenu && !i && (this.parentMenu.lock = !1, this.parentMenu.current_submenu = null, t === void 0 ? this.parentMenu.close() : t && !X.isCursorOverElement(t, this.parentMenu.root) && X.trigger(this.parentMenu.root, u.pointerevents_method + "leave", t)), this.current_submenu && this.current_submenu.close(t, !0), this.root.closing_timer && clearTimeout(this.root.closing_timer);
  }
  getTopMenu() {
    return this.options.parentMenu ? this.options.parentMenu.getTopMenu() : this;
  }
  getFirstEvent() {
    return this.options.parentMenu ? this.options.parentMenu.getFirstEvent() : this.options.event;
  }
  onMouseWheel(t) {
    var i = parseInt(this.root.style.top);
    return this.root.style.top = (i + t.deltaY * this.options.scroll_speed * (this.options.invert_scrolling ? -1 : 1)).toFixed() + "px", t.preventDefault(), !0;
  }
}
export {
  Ae as BASE_SLOT_TYPES,
  k as BuiltInSlotShape,
  I as BuiltInSlotType,
  X as ContextMenu,
  j as ContextMenuSpecialItem,
  w as Dir,
  Be as DragAndScale,
  $ as GraphInput,
  Q as GraphOutput,
  W as LConnectionKind,
  Pe as LGraph,
  N as LGraphCanvas,
  ge as LGraphCanvas_Events,
  U as LGraphCanvas_Rendering,
  C as LGraphCanvas_UI,
  me as LGraphGroup,
  ae as LGraphNode,
  xe as LGraphStatus,
  Xe as LINK_RENDER_MODE_NAMES,
  he as LLink,
  ue as LayoutDirection,
  de as LinkRenderMode,
  u as LiteGraph,
  Oe as NODE_MODE_COLORS,
  re as NODE_MODE_NAMES,
  Z as NodeMode,
  Ie as SLOT_SHAPE_NAMES,
  ne as Subgraph,
  se as TitleMode,
  Te as clamp,
  J as getLitegraphTypeName,
  we as getSlotTypesIn,
  We as getSlotTypesInFormatted,
  Le as getSlotTypesOut,
  Ye as getSlotTypesOutFormatted,
  Ce as getStaticProperty,
  ye as getStaticPropertyOnInstance,
  Se as isValidLitegraphType,
  Ee as makeDraggable,
  Re as reassignGraphIDs,
  ve as toHashMap
};
