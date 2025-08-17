/**
 * These contain types from ComfyUI (or LiteGraph) that are either copied or manually determined
 * added here mostly because @comfyorg/comfyui-frontend-types hasn't exported them oir they weren't
 * available.
 */
import type {SerializedGraph} from './index.js';

export type getPngMetadata = (file: File | Blob) => { workflow?: string; prompt?: string };
export type getWebpMetadata = (file: File | Blob) => {
  Workflow?: string;
  workflow?: string;
  Prompt?: string;
  prompt?: string;
};

// Below are types derived from the Serialized version of a workflow.

// export type SerializedLink = [
//   number, // this.id,
//   number, // this.origin_id,
//   number, // this.origin_slot,
//   number, // this.target_id,
//   number, // this.target_slot,
//   string, // this.type
// ];

// interface SerializedNodeInput {
//   name: string;
//   type: string;
//   link: number;
// }
// interface SerializedNodeOutput extends SerializedNodeInput {
//   slot_index: number;
//   links: number[];
// }

// export interface SerializedNode {
//   // id: number;
//   // inputs: SerializedNodeInput[];
//   // outputs: SerializedNodeOutput[];
//   mode: number;
//   order: number;
//   pos: [number, number];
//   properties: any;
//   size: [number, number];
//   type: string;
//   widgets_values: Array<number | string>;
// }

// export interface SerializedGraph {
  // config: any;
  // extra: any;
  // groups: any;
  // last_link_id: number;
  // last_node_id: number;
  // links: SerializedLink[];
  // nodes: SerializedNode[];
// }


/**
 * ComfyUI-Frontend defines a ComfyNodeDef from Zod, but doesn't expose it. This is a shim.
 */
export type ComfyNodeDef = {
	name: string;
	display_name?: string;
	description?: string;
	category: string;
	input?: {
		required?: Record<string, [string | any[]] | [string | any[], any]>;
		optional?: Record<string, [string | any[]] | [string | any[], any]>;
		hidden?: Record<string, [string | any[]] | [string | any[], any]>;
	};
	output?: string[];
	output_name: string[];
	// @rgthree
	output_node?: boolean;
};



// Below are types derived from the formats for the ComfyAPI.

// @rgthree
type ComfyApiInputLink = [
  /** The id string of the connected node. */
  string,
  /** The output index. */
  number,
]


type ComfyApiFormatNode = {
  "inputs": {
    [input_name: string]: string|number|boolean|ComfyApiInputLink,
  },
  "class_type": string,
  "_meta": {
    "title": string,
  }
}

export type ComfyApiFormat = {
  [node_id: string]: ComfyApiFormatNode
}

export type ComfyApiPrompt = {
  workflow: SerializedGraph,
  output: ComfyApiFormat,
}

export type ComfyApiEventDetailStatus = {
  exec_info: {
    queue_remaining: number;
  };
};

export type ComfyApiEventDetailExecutionStart = {
  prompt_id: string;
};

export type ComfyApiEventDetailExecuting = null | string;

export type ComfyApiEventDetailProgress = {
  node: string;
  prompt_id: string;
  max: number;
  value: number;
};

export type ComfyApiEventDetailExecuted = {
  node: string;
  prompt_id: string;
  output: any;
};

export type ComfyApiEventDetailCached = {
  nodes: string[];
  prompt_id: string;
};

export type ComfyApiEventDetailError = {
  prompt_id: string;
  exception_type: string;
  exception_message: string;
  node_id: string;
  node_type: string;
  node_id: string;
  traceback: string;
  executed: any[];
  current_inputs:  {[key: string]: (number[]|string[])};
  current_outputs: {[key: string]: (number[]|string[])};
}
