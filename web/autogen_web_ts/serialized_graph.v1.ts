/* eslint-disable */
import * as _m0 from "protobufjs/minimal";
import { Any } from "./google/any";

export const protobufPackage = "serialized_graph";

/** Represents a generic vector with two elements */
export interface Vector2 {
  x: number;
  y: number;
}

/** Represents a generic vector with four elements */
export interface Vector4 {
  x: number;
  y: number;
  z: number;
  w: number;
}

/** Represents a serialized graph node */
export interface SerializedLGraphNode {
  id: number;
  type: string;
  pos: Vector2 | undefined;
  size: Vector2 | undefined;
  flags: Any | undefined;
  mode: string;
  inputs: Any[];
  outputs: Any[];
  title: string;
  properties: { [key: string]: Any };
  widgets_values: Any[];
}

export interface SerializedLGraphNode_PropertiesEntry {
  key: string;
  value: Any | undefined;
}

/** Represents a serialized graph group */
export interface SerializedLGraphGroup {
  title: string;
  bounding: Vector4 | undefined;
  color: string;
  font: string;
}

/** Represents a link in the graph */
export interface Link {
  source_node_id: number;
  source_output_slot: number;
  target_node_id: number;
  target_input_slot: number;
  type: string;
}

/** Represents the entire serialized graph */
export interface SerializedGraph {
  last_node_id: number;
  last_link_id: number;
  nodes: SerializedLGraphNode[];
  links: Link[];
  groups: SerializedLGraphGroup[];
  config: Any | undefined;
  version: string;
}

function createBaseVector2(): Vector2 {
  return { x: 0, y: 0 };
}

export const Vector2 = {
  encode(message: Vector2, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.x !== 0) {
      writer.uint32(13).float(message.x);
    }
    if (message.y !== 0) {
      writer.uint32(21).float(message.y);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): Vector2 {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseVector2();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 13) {
            break;
          }

          message.x = reader.float();
          continue;
        case 2:
          if (tag !== 21) {
            break;
          }

          message.y = reader.float();
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  create(base?: DeepPartial<Vector2>): Vector2 {
    return Vector2.fromPartial(base ?? {});
  },
  fromPartial(object: DeepPartial<Vector2>): Vector2 {
    const message = createBaseVector2();
    message.x = object.x ?? 0;
    message.y = object.y ?? 0;
    return message;
  },
};

function createBaseVector4(): Vector4 {
  return { x: 0, y: 0, z: 0, w: 0 };
}

export const Vector4 = {
  encode(message: Vector4, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.x !== 0) {
      writer.uint32(13).float(message.x);
    }
    if (message.y !== 0) {
      writer.uint32(21).float(message.y);
    }
    if (message.z !== 0) {
      writer.uint32(29).float(message.z);
    }
    if (message.w !== 0) {
      writer.uint32(37).float(message.w);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): Vector4 {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseVector4();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 13) {
            break;
          }

          message.x = reader.float();
          continue;
        case 2:
          if (tag !== 21) {
            break;
          }

          message.y = reader.float();
          continue;
        case 3:
          if (tag !== 29) {
            break;
          }

          message.z = reader.float();
          continue;
        case 4:
          if (tag !== 37) {
            break;
          }

          message.w = reader.float();
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  create(base?: DeepPartial<Vector4>): Vector4 {
    return Vector4.fromPartial(base ?? {});
  },
  fromPartial(object: DeepPartial<Vector4>): Vector4 {
    const message = createBaseVector4();
    message.x = object.x ?? 0;
    message.y = object.y ?? 0;
    message.z = object.z ?? 0;
    message.w = object.w ?? 0;
    return message;
  },
};

function createBaseSerializedLGraphNode(): SerializedLGraphNode {
  return {
    id: 0,
    type: "",
    pos: undefined,
    size: undefined,
    flags: undefined,
    mode: "",
    inputs: [],
    outputs: [],
    title: "",
    properties: {},
    widgets_values: [],
  };
}

export const SerializedLGraphNode = {
  encode(message: SerializedLGraphNode, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.id !== 0) {
      writer.uint32(8).int32(message.id);
    }
    if (message.type !== "") {
      writer.uint32(18).string(message.type);
    }
    if (message.pos !== undefined) {
      Vector2.encode(message.pos, writer.uint32(26).fork()).ldelim();
    }
    if (message.size !== undefined) {
      Vector2.encode(message.size, writer.uint32(34).fork()).ldelim();
    }
    if (message.flags !== undefined) {
      Any.encode(message.flags, writer.uint32(42).fork()).ldelim();
    }
    if (message.mode !== "") {
      writer.uint32(50).string(message.mode);
    }
    for (const v of message.inputs) {
      Any.encode(v!, writer.uint32(58).fork()).ldelim();
    }
    for (const v of message.outputs) {
      Any.encode(v!, writer.uint32(66).fork()).ldelim();
    }
    if (message.title !== "") {
      writer.uint32(74).string(message.title);
    }
    Object.entries(message.properties).forEach(([key, value]) => {
      SerializedLGraphNode_PropertiesEntry.encode({ key: key as any, value }, writer.uint32(82).fork()).ldelim();
    });
    for (const v of message.widgets_values) {
      Any.encode(v!, writer.uint32(90).fork()).ldelim();
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): SerializedLGraphNode {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseSerializedLGraphNode();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 8) {
            break;
          }

          message.id = reader.int32();
          continue;
        case 2:
          if (tag !== 18) {
            break;
          }

          message.type = reader.string();
          continue;
        case 3:
          if (tag !== 26) {
            break;
          }

          message.pos = Vector2.decode(reader, reader.uint32());
          continue;
        case 4:
          if (tag !== 34) {
            break;
          }

          message.size = Vector2.decode(reader, reader.uint32());
          continue;
        case 5:
          if (tag !== 42) {
            break;
          }

          message.flags = Any.decode(reader, reader.uint32());
          continue;
        case 6:
          if (tag !== 50) {
            break;
          }

          message.mode = reader.string();
          continue;
        case 7:
          if (tag !== 58) {
            break;
          }

          message.inputs.push(Any.decode(reader, reader.uint32()));
          continue;
        case 8:
          if (tag !== 66) {
            break;
          }

          message.outputs.push(Any.decode(reader, reader.uint32()));
          continue;
        case 9:
          if (tag !== 74) {
            break;
          }

          message.title = reader.string();
          continue;
        case 10:
          if (tag !== 82) {
            break;
          }

          const entry10 = SerializedLGraphNode_PropertiesEntry.decode(reader, reader.uint32());
          if (entry10.value !== undefined) {
            message.properties[entry10.key] = entry10.value;
          }
          continue;
        case 11:
          if (tag !== 90) {
            break;
          }

          message.widgets_values.push(Any.decode(reader, reader.uint32()));
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  create(base?: DeepPartial<SerializedLGraphNode>): SerializedLGraphNode {
    return SerializedLGraphNode.fromPartial(base ?? {});
  },
  fromPartial(object: DeepPartial<SerializedLGraphNode>): SerializedLGraphNode {
    const message = createBaseSerializedLGraphNode();
    message.id = object.id ?? 0;
    message.type = object.type ?? "";
    message.pos = (object.pos !== undefined && object.pos !== null) ? Vector2.fromPartial(object.pos) : undefined;
    message.size = (object.size !== undefined && object.size !== null) ? Vector2.fromPartial(object.size) : undefined;
    message.flags = (object.flags !== undefined && object.flags !== null) ? Any.fromPartial(object.flags) : undefined;
    message.mode = object.mode ?? "";
    message.inputs = object.inputs?.map((e) => Any.fromPartial(e)) || [];
    message.outputs = object.outputs?.map((e) => Any.fromPartial(e)) || [];
    message.title = object.title ?? "";
    message.properties = Object.entries(object.properties ?? {}).reduce<{ [key: string]: Any }>((acc, [key, value]) => {
      if (value !== undefined) {
        acc[key] = Any.fromPartial(value);
      }
      return acc;
    }, {});
    message.widgets_values = object.widgets_values?.map((e) => Any.fromPartial(e)) || [];
    return message;
  },
};

function createBaseSerializedLGraphNode_PropertiesEntry(): SerializedLGraphNode_PropertiesEntry {
  return { key: "", value: undefined };
}

export const SerializedLGraphNode_PropertiesEntry = {
  encode(message: SerializedLGraphNode_PropertiesEntry, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.key !== "") {
      writer.uint32(10).string(message.key);
    }
    if (message.value !== undefined) {
      Any.encode(message.value, writer.uint32(18).fork()).ldelim();
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): SerializedLGraphNode_PropertiesEntry {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseSerializedLGraphNode_PropertiesEntry();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 10) {
            break;
          }

          message.key = reader.string();
          continue;
        case 2:
          if (tag !== 18) {
            break;
          }

          message.value = Any.decode(reader, reader.uint32());
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  create(base?: DeepPartial<SerializedLGraphNode_PropertiesEntry>): SerializedLGraphNode_PropertiesEntry {
    return SerializedLGraphNode_PropertiesEntry.fromPartial(base ?? {});
  },
  fromPartial(object: DeepPartial<SerializedLGraphNode_PropertiesEntry>): SerializedLGraphNode_PropertiesEntry {
    const message = createBaseSerializedLGraphNode_PropertiesEntry();
    message.key = object.key ?? "";
    message.value = (object.value !== undefined && object.value !== null) ? Any.fromPartial(object.value) : undefined;
    return message;
  },
};

function createBaseSerializedLGraphGroup(): SerializedLGraphGroup {
  return { title: "", bounding: undefined, color: "", font: "" };
}

export const SerializedLGraphGroup = {
  encode(message: SerializedLGraphGroup, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.title !== "") {
      writer.uint32(10).string(message.title);
    }
    if (message.bounding !== undefined) {
      Vector4.encode(message.bounding, writer.uint32(18).fork()).ldelim();
    }
    if (message.color !== "") {
      writer.uint32(26).string(message.color);
    }
    if (message.font !== "") {
      writer.uint32(34).string(message.font);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): SerializedLGraphGroup {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseSerializedLGraphGroup();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 10) {
            break;
          }

          message.title = reader.string();
          continue;
        case 2:
          if (tag !== 18) {
            break;
          }

          message.bounding = Vector4.decode(reader, reader.uint32());
          continue;
        case 3:
          if (tag !== 26) {
            break;
          }

          message.color = reader.string();
          continue;
        case 4:
          if (tag !== 34) {
            break;
          }

          message.font = reader.string();
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  create(base?: DeepPartial<SerializedLGraphGroup>): SerializedLGraphGroup {
    return SerializedLGraphGroup.fromPartial(base ?? {});
  },
  fromPartial(object: DeepPartial<SerializedLGraphGroup>): SerializedLGraphGroup {
    const message = createBaseSerializedLGraphGroup();
    message.title = object.title ?? "";
    message.bounding = (object.bounding !== undefined && object.bounding !== null)
      ? Vector4.fromPartial(object.bounding)
      : undefined;
    message.color = object.color ?? "";
    message.font = object.font ?? "";
    return message;
  },
};

function createBaseLink(): Link {
  return { source_node_id: 0, source_output_slot: 0, target_node_id: 0, target_input_slot: 0, type: "" };
}

export const Link = {
  encode(message: Link, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.source_node_id !== 0) {
      writer.uint32(8).int32(message.source_node_id);
    }
    if (message.source_output_slot !== 0) {
      writer.uint32(16).int32(message.source_output_slot);
    }
    if (message.target_node_id !== 0) {
      writer.uint32(24).int32(message.target_node_id);
    }
    if (message.target_input_slot !== 0) {
      writer.uint32(32).int32(message.target_input_slot);
    }
    if (message.type !== "") {
      writer.uint32(42).string(message.type);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): Link {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseLink();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 8) {
            break;
          }

          message.source_node_id = reader.int32();
          continue;
        case 2:
          if (tag !== 16) {
            break;
          }

          message.source_output_slot = reader.int32();
          continue;
        case 3:
          if (tag !== 24) {
            break;
          }

          message.target_node_id = reader.int32();
          continue;
        case 4:
          if (tag !== 32) {
            break;
          }

          message.target_input_slot = reader.int32();
          continue;
        case 5:
          if (tag !== 42) {
            break;
          }

          message.type = reader.string();
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  create(base?: DeepPartial<Link>): Link {
    return Link.fromPartial(base ?? {});
  },
  fromPartial(object: DeepPartial<Link>): Link {
    const message = createBaseLink();
    message.source_node_id = object.source_node_id ?? 0;
    message.source_output_slot = object.source_output_slot ?? 0;
    message.target_node_id = object.target_node_id ?? 0;
    message.target_input_slot = object.target_input_slot ?? 0;
    message.type = object.type ?? "";
    return message;
  },
};

function createBaseSerializedGraph(): SerializedGraph {
  return { last_node_id: 0, last_link_id: 0, nodes: [], links: [], groups: [], config: undefined, version: "" };
}

export const SerializedGraph = {
  encode(message: SerializedGraph, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.last_node_id !== 0) {
      writer.uint32(8).int32(message.last_node_id);
    }
    if (message.last_link_id !== 0) {
      writer.uint32(16).int32(message.last_link_id);
    }
    for (const v of message.nodes) {
      SerializedLGraphNode.encode(v!, writer.uint32(26).fork()).ldelim();
    }
    for (const v of message.links) {
      Link.encode(v!, writer.uint32(34).fork()).ldelim();
    }
    for (const v of message.groups) {
      SerializedLGraphGroup.encode(v!, writer.uint32(42).fork()).ldelim();
    }
    if (message.config !== undefined) {
      Any.encode(message.config, writer.uint32(50).fork()).ldelim();
    }
    if (message.version !== "") {
      writer.uint32(58).string(message.version);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): SerializedGraph {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseSerializedGraph();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 8) {
            break;
          }

          message.last_node_id = reader.int32();
          continue;
        case 2:
          if (tag !== 16) {
            break;
          }

          message.last_link_id = reader.int32();
          continue;
        case 3:
          if (tag !== 26) {
            break;
          }

          message.nodes.push(SerializedLGraphNode.decode(reader, reader.uint32()));
          continue;
        case 4:
          if (tag !== 34) {
            break;
          }

          message.links.push(Link.decode(reader, reader.uint32()));
          continue;
        case 5:
          if (tag !== 42) {
            break;
          }

          message.groups.push(SerializedLGraphGroup.decode(reader, reader.uint32()));
          continue;
        case 6:
          if (tag !== 50) {
            break;
          }

          message.config = Any.decode(reader, reader.uint32());
          continue;
        case 7:
          if (tag !== 58) {
            break;
          }

          message.version = reader.string();
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  create(base?: DeepPartial<SerializedGraph>): SerializedGraph {
    return SerializedGraph.fromPartial(base ?? {});
  },
  fromPartial(object: DeepPartial<SerializedGraph>): SerializedGraph {
    const message = createBaseSerializedGraph();
    message.last_node_id = object.last_node_id ?? 0;
    message.last_link_id = object.last_link_id ?? 0;
    message.nodes = object.nodes?.map((e) => SerializedLGraphNode.fromPartial(e)) || [];
    message.links = object.links?.map((e) => Link.fromPartial(e)) || [];
    message.groups = object.groups?.map((e) => SerializedLGraphGroup.fromPartial(e)) || [];
    message.config = (object.config !== undefined && object.config !== null)
      ? Any.fromPartial(object.config)
      : undefined;
    message.version = object.version ?? "";
    return message;
  },
};

type Builtin = Date | Function | Uint8Array | string | number | boolean | undefined;

export type DeepPartial<T> = T extends Builtin ? T
  : T extends globalThis.Array<infer U> ? globalThis.Array<DeepPartial<U>>
  : T extends ReadonlyArray<infer U> ? ReadonlyArray<DeepPartial<U>>
  : T extends {} ? { [K in keyof T]?: DeepPartial<T[K]> }
  : Partial<T>;
