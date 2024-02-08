/* eslint-disable */
import type { CallContext, CallOptions } from "nice-grpc-common";
import * as _m0 from "protobufjs/minimal";
import { Empty } from "./google/empty";
import { Struct } from "./google/struct";
import { SerializedGraph } from "./serialized_graph.v1";

export const protobufPackage = "comfy_request.v1";

/** These are more direct client-created workflows for client -> server -> worker */

/** Message definition for WorkflowStep */
export interface WorkflowStep {
  class_type: string;
  /** Inputs are too idiosyncratic to be typed specifically */
  inputs: { [key: string]: any } | undefined;
}

/**
 * TO DO: add conditional check for url conformity
 * Two files with the same hash are treated as equivalent; we use file-hashes as filenames.
 * File types returned:
 * image: png, jpg, svg, webp, gif
 * video: mp4
 * data: json (icluding RLE-encoded masks), npy (numpy array for embeddings)
 * TO DO: in the future, we may want more info, such as mask VS image, or latent preview
 */
export interface WorkflowFile {
  /** unique identifier for the file */
  file_hash: string;
  /** ComfyUI terminology: key 'format' */
  mime_type: string;
  reference?: WorkflowFile_FileReference | undefined;
  data?: Uint8Array | undefined;
}

export interface WorkflowFile_FileReference {
  /** string must be a valid url */
  url: string;
  /** Comfy UI terminology: key 'type', values 'temp' | 'output' */
  is_temp: boolean;
}

/**
 * TO DO: add specific buckets for different users perhaps?
 * Private VS public outputs?
 *
 * Right now: output-files are saved to our DO S3 bucket, and all of them are publicly
 * available.
 * Temp-files are saved to our DO S3 temp bucket, which is wiped after 24 hours.
 * Corresponding records exist in Firestore for both.
 * If 'save outputs' is false, then the output files are saved to the temp bucket
 * In the future, API-clients can provide us with S3 secrets which we can use to upload
 * to their specified account.
 * In the future we may allow for S3 presigned urls as well.
 * In the future we may add support for webhook callbacks.
 * In the future we may allow for direct binary outputs (instead of uploading files to S3
 * and then returning urls)
 */
export interface OutputConfig {
  save_outputs: boolean;
  send_latent_previews: boolean;
}

/** client -> server message */
export interface ComfyRequest {
  /** keys are node_ids */
  workflow: { [key: string]: WorkflowStep };
  serialized_graph?: SerializedGraph | undefined;
  input_files: WorkflowFile[];
  output_config: OutputConfig | undefined;
  worker_wait_duration?:
    | number
    | undefined;
  /** redis channel name to publish results to */
  session_id?: string | undefined;
}

export interface ComfyRequest_WorkflowEntry {
  key: string;
  value: WorkflowStep | undefined;
}

/**
 * ComfyUI has 'delete' (remove specific non-running items), and 'interrupt'
 * (stop currently running process) commands. We roll them all into a single endpoint here.
 */
export interface CancelJob {
  job_id: string;
}

/** ComfyUI calls this 'clear' (remove all queued items owned by the user) */
export interface PurgeRoomQueue {
  session_id: string;
}

export interface JobCreated {
  /** created by the server; id of the job in the queue */
  job_id: string;
  /** redis channel to subscribe to for updates */
  session_id: string;
  queue_seconds: number;
  execution_seconds: number;
}

/** Temp-files and latent-previews are not included */
export interface JobOutput {
  job_id: string;
  session_id: string;
  files: WorkflowFile[];
}

/** It's assumed that the consumer knows what session_id it's watching */
export interface ComfyMessage {
  job_id: string;
  user_id: string;
  queue_status?: ComfyMessage_QueueStatus | undefined;
  execution_start?: ComfyMessage_ExecutionStart | undefined;
  executing?: ComfyMessage_Executing | undefined;
  progress?: ComfyMessage_Progress | undefined;
  execution_error?: ComfyMessage_ExecutionError | undefined;
  execution_interrupted?: ComfyMessage_ExecutionInterrupted | undefined;
  execution_cached?: ComfyMessage_ExecutionCached | undefined;
  output?: ComfyMessage_Output | undefined;
  custom_message?: ComfyMessage_CustomMessage | undefined;
}

/**
 * updates queue-display on client. SID's purpose is unknown
 * ComfyUI terminology: 'Status'
 */
export interface ComfyMessage_QueueStatus {
  /** looks like: "99506f0d89b64dbdb09ae567274fb078" */
  sid?: string | undefined;
  queue_remaining: number;
}

/** job-started */
export interface ComfyMessage_ExecutionStart {
}

/**
 * job execution moved to node_id
 * There is a bug in ComfyUI where it'll send an Executing update with node: null at the
 * end of a job; we ignore these
 */
export interface ComfyMessage_Executing {
  node_id: string;
}

/** Updates a node's progress bar; like (value / max) = percent-complete */
export interface ComfyMessage_Progress {
  max: number;
  value: number;
}

/** we remove currentInputs and currentOutputs as they are too large */
export interface ComfyMessage_ExecutionError {
  currentInputs: { [key: string]: any } | undefined;
  currentOutputs: { [key: string]: any } | undefined;
  execution_message: string;
  exception_type: string;
  /** list of nodes executed */
  executed: string[];
  /** node id that threw the error */
  node_id: string;
  node_type: string;
  traceback: string[];
}

export interface ComfyMessage_ExecutionInterrupted {
  /** node-ids that already finished */
  executed: string[];
  node_id: string;
  node_type: string;
}

/** This specifies nodes that were skipped due to their output being cached */
export interface ComfyMessage_ExecutionCached {
  node_ids: string[];
}

/**
 * A node produced an output (temp or saved); display it
 * In the original ComfyUI, it's like output.images[0] = { filename, subfolder, type }, or output.gifs[0]
 * There is also output.animated; an array whose indices are bools corresponding to the images array
 * We simplify all of this
 * ComfyUI terinology; this is called 'Executed', which was confusing
 */
export interface ComfyMessage_Output {
  node_id: string;
  /** Note; in the future, we may need an 'event-type' as well */
  files: WorkflowFile[];
}

/** This is a catch-all; custom-nodes can define their own update-messages */
export interface ComfyMessage_CustomMessage {
  type: string;
  data: { [key: string]: any } | undefined;
}

/**
 * If outputs_only is true, then only Output messages will be returned,
 * otherwise all ComfyMessage types will be returned.
 * By default, output messages consisting of temp-files and latent-previews are
 * not included, but if you want them, set those flags to true.
 */
export interface MessageFilter {
  outputs_only: boolean;
  include_temp_files?: boolean | undefined;
  include_latent_previews?: boolean | undefined;
}

/** By default, all message-types will be returned, unless a filter is applied */
export interface RoomStreamRequest {
  session_id: string;
  filter?: MessageFilter | undefined;
}

export interface JobStreamRequest {
  job_id: string;
  filter?: MessageFilter | undefined;
}

function createBaseWorkflowStep(): WorkflowStep {
  return { class_type: "", inputs: undefined };
}

export const WorkflowStep = {
  encode(message: WorkflowStep, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.class_type !== "") {
      writer.uint32(10).string(message.class_type);
    }
    if (message.inputs !== undefined) {
      Struct.encode(Struct.wrap(message.inputs), writer.uint32(18).fork()).ldelim();
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): WorkflowStep {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseWorkflowStep();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 10) {
            break;
          }

          message.class_type = reader.string();
          continue;
        case 2:
          if (tag !== 18) {
            break;
          }

          message.inputs = Struct.unwrap(Struct.decode(reader, reader.uint32()));
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  create(base?: DeepPartial<WorkflowStep>): WorkflowStep {
    return WorkflowStep.fromPartial(base ?? {});
  },
  fromPartial(object: DeepPartial<WorkflowStep>): WorkflowStep {
    const message = createBaseWorkflowStep();
    message.class_type = object.class_type ?? "";
    message.inputs = object.inputs ?? undefined;
    return message;
  },
};

function createBaseWorkflowFile(): WorkflowFile {
  return { file_hash: "", mime_type: "", reference: undefined, data: undefined };
}

export const WorkflowFile = {
  encode(message: WorkflowFile, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.file_hash !== "") {
      writer.uint32(10).string(message.file_hash);
    }
    if (message.mime_type !== "") {
      writer.uint32(18).string(message.mime_type);
    }
    if (message.reference !== undefined) {
      WorkflowFile_FileReference.encode(message.reference, writer.uint32(26).fork()).ldelim();
    }
    if (message.data !== undefined) {
      writer.uint32(34).bytes(message.data);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): WorkflowFile {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseWorkflowFile();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 10) {
            break;
          }

          message.file_hash = reader.string();
          continue;
        case 2:
          if (tag !== 18) {
            break;
          }

          message.mime_type = reader.string();
          continue;
        case 3:
          if (tag !== 26) {
            break;
          }

          message.reference = WorkflowFile_FileReference.decode(reader, reader.uint32());
          continue;
        case 4:
          if (tag !== 34) {
            break;
          }

          message.data = reader.bytes();
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  create(base?: DeepPartial<WorkflowFile>): WorkflowFile {
    return WorkflowFile.fromPartial(base ?? {});
  },
  fromPartial(object: DeepPartial<WorkflowFile>): WorkflowFile {
    const message = createBaseWorkflowFile();
    message.file_hash = object.file_hash ?? "";
    message.mime_type = object.mime_type ?? "";
    message.reference = (object.reference !== undefined && object.reference !== null)
      ? WorkflowFile_FileReference.fromPartial(object.reference)
      : undefined;
    message.data = object.data ?? undefined;
    return message;
  },
};

function createBaseWorkflowFile_FileReference(): WorkflowFile_FileReference {
  return { url: "", is_temp: false };
}

export const WorkflowFile_FileReference = {
  encode(message: WorkflowFile_FileReference, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.url !== "") {
      writer.uint32(10).string(message.url);
    }
    if (message.is_temp === true) {
      writer.uint32(16).bool(message.is_temp);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): WorkflowFile_FileReference {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseWorkflowFile_FileReference();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 10) {
            break;
          }

          message.url = reader.string();
          continue;
        case 2:
          if (tag !== 16) {
            break;
          }

          message.is_temp = reader.bool();
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  create(base?: DeepPartial<WorkflowFile_FileReference>): WorkflowFile_FileReference {
    return WorkflowFile_FileReference.fromPartial(base ?? {});
  },
  fromPartial(object: DeepPartial<WorkflowFile_FileReference>): WorkflowFile_FileReference {
    const message = createBaseWorkflowFile_FileReference();
    message.url = object.url ?? "";
    message.is_temp = object.is_temp ?? false;
    return message;
  },
};

function createBaseOutputConfig(): OutputConfig {
  return { save_outputs: false, send_latent_previews: false };
}

export const OutputConfig = {
  encode(message: OutputConfig, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.save_outputs === true) {
      writer.uint32(8).bool(message.save_outputs);
    }
    if (message.send_latent_previews === true) {
      writer.uint32(16).bool(message.send_latent_previews);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): OutputConfig {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseOutputConfig();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 8) {
            break;
          }

          message.save_outputs = reader.bool();
          continue;
        case 2:
          if (tag !== 16) {
            break;
          }

          message.send_latent_previews = reader.bool();
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  create(base?: DeepPartial<OutputConfig>): OutputConfig {
    return OutputConfig.fromPartial(base ?? {});
  },
  fromPartial(object: DeepPartial<OutputConfig>): OutputConfig {
    const message = createBaseOutputConfig();
    message.save_outputs = object.save_outputs ?? false;
    message.send_latent_previews = object.send_latent_previews ?? false;
    return message;
  },
};

function createBaseComfyRequest(): ComfyRequest {
  return {
    workflow: {},
    serialized_graph: undefined,
    input_files: [],
    output_config: undefined,
    worker_wait_duration: undefined,
    session_id: undefined,
  };
}

export const ComfyRequest = {
  encode(message: ComfyRequest, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    Object.entries(message.workflow).forEach(([key, value]) => {
      ComfyRequest_WorkflowEntry.encode({ key: key as any, value }, writer.uint32(10).fork()).ldelim();
    });
    if (message.serialized_graph !== undefined) {
      SerializedGraph.encode(message.serialized_graph, writer.uint32(18).fork()).ldelim();
    }
    for (const v of message.input_files) {
      WorkflowFile.encode(v!, writer.uint32(26).fork()).ldelim();
    }
    if (message.output_config !== undefined) {
      OutputConfig.encode(message.output_config, writer.uint32(34).fork()).ldelim();
    }
    if (message.worker_wait_duration !== undefined) {
      writer.uint32(40).uint32(message.worker_wait_duration);
    }
    if (message.session_id !== undefined) {
      writer.uint32(50).string(message.session_id);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): ComfyRequest {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseComfyRequest();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 10) {
            break;
          }

          const entry1 = ComfyRequest_WorkflowEntry.decode(reader, reader.uint32());
          if (entry1.value !== undefined) {
            message.workflow[entry1.key] = entry1.value;
          }
          continue;
        case 2:
          if (tag !== 18) {
            break;
          }

          message.serialized_graph = SerializedGraph.decode(reader, reader.uint32());
          continue;
        case 3:
          if (tag !== 26) {
            break;
          }

          message.input_files.push(WorkflowFile.decode(reader, reader.uint32()));
          continue;
        case 4:
          if (tag !== 34) {
            break;
          }

          message.output_config = OutputConfig.decode(reader, reader.uint32());
          continue;
        case 5:
          if (tag !== 40) {
            break;
          }

          message.worker_wait_duration = reader.uint32();
          continue;
        case 6:
          if (tag !== 50) {
            break;
          }

          message.session_id = reader.string();
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  create(base?: DeepPartial<ComfyRequest>): ComfyRequest {
    return ComfyRequest.fromPartial(base ?? {});
  },
  fromPartial(object: DeepPartial<ComfyRequest>): ComfyRequest {
    const message = createBaseComfyRequest();
    message.workflow = Object.entries(object.workflow ?? {}).reduce<{ [key: string]: WorkflowStep }>(
      (acc, [key, value]) => {
        if (value !== undefined) {
          acc[key] = WorkflowStep.fromPartial(value);
        }
        return acc;
      },
      {},
    );
    message.serialized_graph = (object.serialized_graph !== undefined && object.serialized_graph !== null)
      ? SerializedGraph.fromPartial(object.serialized_graph)
      : undefined;
    message.input_files = object.input_files?.map((e) => WorkflowFile.fromPartial(e)) || [];
    message.output_config = (object.output_config !== undefined && object.output_config !== null)
      ? OutputConfig.fromPartial(object.output_config)
      : undefined;
    message.worker_wait_duration = object.worker_wait_duration ?? undefined;
    message.session_id = object.session_id ?? undefined;
    return message;
  },
};

function createBaseComfyRequest_WorkflowEntry(): ComfyRequest_WorkflowEntry {
  return { key: "", value: undefined };
}

export const ComfyRequest_WorkflowEntry = {
  encode(message: ComfyRequest_WorkflowEntry, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.key !== "") {
      writer.uint32(10).string(message.key);
    }
    if (message.value !== undefined) {
      WorkflowStep.encode(message.value, writer.uint32(18).fork()).ldelim();
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): ComfyRequest_WorkflowEntry {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseComfyRequest_WorkflowEntry();
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

          message.value = WorkflowStep.decode(reader, reader.uint32());
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  create(base?: DeepPartial<ComfyRequest_WorkflowEntry>): ComfyRequest_WorkflowEntry {
    return ComfyRequest_WorkflowEntry.fromPartial(base ?? {});
  },
  fromPartial(object: DeepPartial<ComfyRequest_WorkflowEntry>): ComfyRequest_WorkflowEntry {
    const message = createBaseComfyRequest_WorkflowEntry();
    message.key = object.key ?? "";
    message.value = (object.value !== undefined && object.value !== null)
      ? WorkflowStep.fromPartial(object.value)
      : undefined;
    return message;
  },
};

function createBaseCancelJob(): CancelJob {
  return { job_id: "" };
}

export const CancelJob = {
  encode(message: CancelJob, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.job_id !== "") {
      writer.uint32(10).string(message.job_id);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): CancelJob {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseCancelJob();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 10) {
            break;
          }

          message.job_id = reader.string();
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  create(base?: DeepPartial<CancelJob>): CancelJob {
    return CancelJob.fromPartial(base ?? {});
  },
  fromPartial(object: DeepPartial<CancelJob>): CancelJob {
    const message = createBaseCancelJob();
    message.job_id = object.job_id ?? "";
    return message;
  },
};

function createBasePurgeRoomQueue(): PurgeRoomQueue {
  return { session_id: "" };
}

export const PurgeRoomQueue = {
  encode(message: PurgeRoomQueue, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.session_id !== "") {
      writer.uint32(10).string(message.session_id);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): PurgeRoomQueue {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBasePurgeRoomQueue();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 10) {
            break;
          }

          message.session_id = reader.string();
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  create(base?: DeepPartial<PurgeRoomQueue>): PurgeRoomQueue {
    return PurgeRoomQueue.fromPartial(base ?? {});
  },
  fromPartial(object: DeepPartial<PurgeRoomQueue>): PurgeRoomQueue {
    const message = createBasePurgeRoomQueue();
    message.session_id = object.session_id ?? "";
    return message;
  },
};

function createBaseJobCreated(): JobCreated {
  return { job_id: "", session_id: "", queue_seconds: 0, execution_seconds: 0 };
}

export const JobCreated = {
  encode(message: JobCreated, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.job_id !== "") {
      writer.uint32(10).string(message.job_id);
    }
    if (message.session_id !== "") {
      writer.uint32(18).string(message.session_id);
    }
    if (message.queue_seconds !== 0) {
      writer.uint32(24).uint32(message.queue_seconds);
    }
    if (message.execution_seconds !== 0) {
      writer.uint32(32).uint32(message.execution_seconds);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): JobCreated {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseJobCreated();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 10) {
            break;
          }

          message.job_id = reader.string();
          continue;
        case 2:
          if (tag !== 18) {
            break;
          }

          message.session_id = reader.string();
          continue;
        case 3:
          if (tag !== 24) {
            break;
          }

          message.queue_seconds = reader.uint32();
          continue;
        case 4:
          if (tag !== 32) {
            break;
          }

          message.execution_seconds = reader.uint32();
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  create(base?: DeepPartial<JobCreated>): JobCreated {
    return JobCreated.fromPartial(base ?? {});
  },
  fromPartial(object: DeepPartial<JobCreated>): JobCreated {
    const message = createBaseJobCreated();
    message.job_id = object.job_id ?? "";
    message.session_id = object.session_id ?? "";
    message.queue_seconds = object.queue_seconds ?? 0;
    message.execution_seconds = object.execution_seconds ?? 0;
    return message;
  },
};

function createBaseJobOutput(): JobOutput {
  return { job_id: "", session_id: "", files: [] };
}

export const JobOutput = {
  encode(message: JobOutput, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.job_id !== "") {
      writer.uint32(10).string(message.job_id);
    }
    if (message.session_id !== "") {
      writer.uint32(18).string(message.session_id);
    }
    for (const v of message.files) {
      WorkflowFile.encode(v!, writer.uint32(26).fork()).ldelim();
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): JobOutput {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseJobOutput();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 10) {
            break;
          }

          message.job_id = reader.string();
          continue;
        case 2:
          if (tag !== 18) {
            break;
          }

          message.session_id = reader.string();
          continue;
        case 3:
          if (tag !== 26) {
            break;
          }

          message.files.push(WorkflowFile.decode(reader, reader.uint32()));
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  create(base?: DeepPartial<JobOutput>): JobOutput {
    return JobOutput.fromPartial(base ?? {});
  },
  fromPartial(object: DeepPartial<JobOutput>): JobOutput {
    const message = createBaseJobOutput();
    message.job_id = object.job_id ?? "";
    message.session_id = object.session_id ?? "";
    message.files = object.files?.map((e) => WorkflowFile.fromPartial(e)) || [];
    return message;
  },
};

function createBaseComfyMessage(): ComfyMessage {
  return {
    job_id: "",
    user_id: "",
    queue_status: undefined,
    execution_start: undefined,
    executing: undefined,
    progress: undefined,
    execution_error: undefined,
    execution_interrupted: undefined,
    execution_cached: undefined,
    output: undefined,
    custom_message: undefined,
  };
}

export const ComfyMessage = {
  encode(message: ComfyMessage, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.job_id !== "") {
      writer.uint32(10).string(message.job_id);
    }
    if (message.user_id !== "") {
      writer.uint32(18).string(message.user_id);
    }
    if (message.queue_status !== undefined) {
      ComfyMessage_QueueStatus.encode(message.queue_status, writer.uint32(26).fork()).ldelim();
    }
    if (message.execution_start !== undefined) {
      ComfyMessage_ExecutionStart.encode(message.execution_start, writer.uint32(34).fork()).ldelim();
    }
    if (message.executing !== undefined) {
      ComfyMessage_Executing.encode(message.executing, writer.uint32(42).fork()).ldelim();
    }
    if (message.progress !== undefined) {
      ComfyMessage_Progress.encode(message.progress, writer.uint32(50).fork()).ldelim();
    }
    if (message.execution_error !== undefined) {
      ComfyMessage_ExecutionError.encode(message.execution_error, writer.uint32(58).fork()).ldelim();
    }
    if (message.execution_interrupted !== undefined) {
      ComfyMessage_ExecutionInterrupted.encode(message.execution_interrupted, writer.uint32(66).fork()).ldelim();
    }
    if (message.execution_cached !== undefined) {
      ComfyMessage_ExecutionCached.encode(message.execution_cached, writer.uint32(74).fork()).ldelim();
    }
    if (message.output !== undefined) {
      ComfyMessage_Output.encode(message.output, writer.uint32(82).fork()).ldelim();
    }
    if (message.custom_message !== undefined) {
      ComfyMessage_CustomMessage.encode(message.custom_message, writer.uint32(90).fork()).ldelim();
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): ComfyMessage {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseComfyMessage();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 10) {
            break;
          }

          message.job_id = reader.string();
          continue;
        case 2:
          if (tag !== 18) {
            break;
          }

          message.user_id = reader.string();
          continue;
        case 3:
          if (tag !== 26) {
            break;
          }

          message.queue_status = ComfyMessage_QueueStatus.decode(reader, reader.uint32());
          continue;
        case 4:
          if (tag !== 34) {
            break;
          }

          message.execution_start = ComfyMessage_ExecutionStart.decode(reader, reader.uint32());
          continue;
        case 5:
          if (tag !== 42) {
            break;
          }

          message.executing = ComfyMessage_Executing.decode(reader, reader.uint32());
          continue;
        case 6:
          if (tag !== 50) {
            break;
          }

          message.progress = ComfyMessage_Progress.decode(reader, reader.uint32());
          continue;
        case 7:
          if (tag !== 58) {
            break;
          }

          message.execution_error = ComfyMessage_ExecutionError.decode(reader, reader.uint32());
          continue;
        case 8:
          if (tag !== 66) {
            break;
          }

          message.execution_interrupted = ComfyMessage_ExecutionInterrupted.decode(reader, reader.uint32());
          continue;
        case 9:
          if (tag !== 74) {
            break;
          }

          message.execution_cached = ComfyMessage_ExecutionCached.decode(reader, reader.uint32());
          continue;
        case 10:
          if (tag !== 82) {
            break;
          }

          message.output = ComfyMessage_Output.decode(reader, reader.uint32());
          continue;
        case 11:
          if (tag !== 90) {
            break;
          }

          message.custom_message = ComfyMessage_CustomMessage.decode(reader, reader.uint32());
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  create(base?: DeepPartial<ComfyMessage>): ComfyMessage {
    return ComfyMessage.fromPartial(base ?? {});
  },
  fromPartial(object: DeepPartial<ComfyMessage>): ComfyMessage {
    const message = createBaseComfyMessage();
    message.job_id = object.job_id ?? "";
    message.user_id = object.user_id ?? "";
    message.queue_status = (object.queue_status !== undefined && object.queue_status !== null)
      ? ComfyMessage_QueueStatus.fromPartial(object.queue_status)
      : undefined;
    message.execution_start = (object.execution_start !== undefined && object.execution_start !== null)
      ? ComfyMessage_ExecutionStart.fromPartial(object.execution_start)
      : undefined;
    message.executing = (object.executing !== undefined && object.executing !== null)
      ? ComfyMessage_Executing.fromPartial(object.executing)
      : undefined;
    message.progress = (object.progress !== undefined && object.progress !== null)
      ? ComfyMessage_Progress.fromPartial(object.progress)
      : undefined;
    message.execution_error = (object.execution_error !== undefined && object.execution_error !== null)
      ? ComfyMessage_ExecutionError.fromPartial(object.execution_error)
      : undefined;
    message.execution_interrupted =
      (object.execution_interrupted !== undefined && object.execution_interrupted !== null)
        ? ComfyMessage_ExecutionInterrupted.fromPartial(object.execution_interrupted)
        : undefined;
    message.execution_cached = (object.execution_cached !== undefined && object.execution_cached !== null)
      ? ComfyMessage_ExecutionCached.fromPartial(object.execution_cached)
      : undefined;
    message.output = (object.output !== undefined && object.output !== null)
      ? ComfyMessage_Output.fromPartial(object.output)
      : undefined;
    message.custom_message = (object.custom_message !== undefined && object.custom_message !== null)
      ? ComfyMessage_CustomMessage.fromPartial(object.custom_message)
      : undefined;
    return message;
  },
};

function createBaseComfyMessage_QueueStatus(): ComfyMessage_QueueStatus {
  return { sid: undefined, queue_remaining: 0 };
}

export const ComfyMessage_QueueStatus = {
  encode(message: ComfyMessage_QueueStatus, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.sid !== undefined) {
      writer.uint32(10).string(message.sid);
    }
    if (message.queue_remaining !== 0) {
      writer.uint32(16).uint32(message.queue_remaining);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): ComfyMessage_QueueStatus {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseComfyMessage_QueueStatus();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 10) {
            break;
          }

          message.sid = reader.string();
          continue;
        case 2:
          if (tag !== 16) {
            break;
          }

          message.queue_remaining = reader.uint32();
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  create(base?: DeepPartial<ComfyMessage_QueueStatus>): ComfyMessage_QueueStatus {
    return ComfyMessage_QueueStatus.fromPartial(base ?? {});
  },
  fromPartial(object: DeepPartial<ComfyMessage_QueueStatus>): ComfyMessage_QueueStatus {
    const message = createBaseComfyMessage_QueueStatus();
    message.sid = object.sid ?? undefined;
    message.queue_remaining = object.queue_remaining ?? 0;
    return message;
  },
};

function createBaseComfyMessage_ExecutionStart(): ComfyMessage_ExecutionStart {
  return {};
}

export const ComfyMessage_ExecutionStart = {
  encode(_: ComfyMessage_ExecutionStart, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): ComfyMessage_ExecutionStart {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseComfyMessage_ExecutionStart();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  create(base?: DeepPartial<ComfyMessage_ExecutionStart>): ComfyMessage_ExecutionStart {
    return ComfyMessage_ExecutionStart.fromPartial(base ?? {});
  },
  fromPartial(_: DeepPartial<ComfyMessage_ExecutionStart>): ComfyMessage_ExecutionStart {
    const message = createBaseComfyMessage_ExecutionStart();
    return message;
  },
};

function createBaseComfyMessage_Executing(): ComfyMessage_Executing {
  return { node_id: "" };
}

export const ComfyMessage_Executing = {
  encode(message: ComfyMessage_Executing, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.node_id !== "") {
      writer.uint32(10).string(message.node_id);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): ComfyMessage_Executing {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseComfyMessage_Executing();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 10) {
            break;
          }

          message.node_id = reader.string();
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  create(base?: DeepPartial<ComfyMessage_Executing>): ComfyMessage_Executing {
    return ComfyMessage_Executing.fromPartial(base ?? {});
  },
  fromPartial(object: DeepPartial<ComfyMessage_Executing>): ComfyMessage_Executing {
    const message = createBaseComfyMessage_Executing();
    message.node_id = object.node_id ?? "";
    return message;
  },
};

function createBaseComfyMessage_Progress(): ComfyMessage_Progress {
  return { max: 0, value: 0 };
}

export const ComfyMessage_Progress = {
  encode(message: ComfyMessage_Progress, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.max !== 0) {
      writer.uint32(8).uint32(message.max);
    }
    if (message.value !== 0) {
      writer.uint32(16).uint32(message.value);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): ComfyMessage_Progress {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseComfyMessage_Progress();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 8) {
            break;
          }

          message.max = reader.uint32();
          continue;
        case 2:
          if (tag !== 16) {
            break;
          }

          message.value = reader.uint32();
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  create(base?: DeepPartial<ComfyMessage_Progress>): ComfyMessage_Progress {
    return ComfyMessage_Progress.fromPartial(base ?? {});
  },
  fromPartial(object: DeepPartial<ComfyMessage_Progress>): ComfyMessage_Progress {
    const message = createBaseComfyMessage_Progress();
    message.max = object.max ?? 0;
    message.value = object.value ?? 0;
    return message;
  },
};

function createBaseComfyMessage_ExecutionError(): ComfyMessage_ExecutionError {
  return {
    currentInputs: undefined,
    currentOutputs: undefined,
    execution_message: "",
    exception_type: "",
    executed: [],
    node_id: "",
    node_type: "",
    traceback: [],
  };
}

export const ComfyMessage_ExecutionError = {
  encode(message: ComfyMessage_ExecutionError, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.currentInputs !== undefined) {
      Struct.encode(Struct.wrap(message.currentInputs), writer.uint32(10).fork()).ldelim();
    }
    if (message.currentOutputs !== undefined) {
      Struct.encode(Struct.wrap(message.currentOutputs), writer.uint32(18).fork()).ldelim();
    }
    if (message.execution_message !== "") {
      writer.uint32(26).string(message.execution_message);
    }
    if (message.exception_type !== "") {
      writer.uint32(34).string(message.exception_type);
    }
    for (const v of message.executed) {
      writer.uint32(42).string(v!);
    }
    if (message.node_id !== "") {
      writer.uint32(50).string(message.node_id);
    }
    if (message.node_type !== "") {
      writer.uint32(58).string(message.node_type);
    }
    for (const v of message.traceback) {
      writer.uint32(66).string(v!);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): ComfyMessage_ExecutionError {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseComfyMessage_ExecutionError();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 10) {
            break;
          }

          message.currentInputs = Struct.unwrap(Struct.decode(reader, reader.uint32()));
          continue;
        case 2:
          if (tag !== 18) {
            break;
          }

          message.currentOutputs = Struct.unwrap(Struct.decode(reader, reader.uint32()));
          continue;
        case 3:
          if (tag !== 26) {
            break;
          }

          message.execution_message = reader.string();
          continue;
        case 4:
          if (tag !== 34) {
            break;
          }

          message.exception_type = reader.string();
          continue;
        case 5:
          if (tag !== 42) {
            break;
          }

          message.executed.push(reader.string());
          continue;
        case 6:
          if (tag !== 50) {
            break;
          }

          message.node_id = reader.string();
          continue;
        case 7:
          if (tag !== 58) {
            break;
          }

          message.node_type = reader.string();
          continue;
        case 8:
          if (tag !== 66) {
            break;
          }

          message.traceback.push(reader.string());
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  create(base?: DeepPartial<ComfyMessage_ExecutionError>): ComfyMessage_ExecutionError {
    return ComfyMessage_ExecutionError.fromPartial(base ?? {});
  },
  fromPartial(object: DeepPartial<ComfyMessage_ExecutionError>): ComfyMessage_ExecutionError {
    const message = createBaseComfyMessage_ExecutionError();
    message.currentInputs = object.currentInputs ?? undefined;
    message.currentOutputs = object.currentOutputs ?? undefined;
    message.execution_message = object.execution_message ?? "";
    message.exception_type = object.exception_type ?? "";
    message.executed = object.executed?.map((e) => e) || [];
    message.node_id = object.node_id ?? "";
    message.node_type = object.node_type ?? "";
    message.traceback = object.traceback?.map((e) => e) || [];
    return message;
  },
};

function createBaseComfyMessage_ExecutionInterrupted(): ComfyMessage_ExecutionInterrupted {
  return { executed: [], node_id: "", node_type: "" };
}

export const ComfyMessage_ExecutionInterrupted = {
  encode(message: ComfyMessage_ExecutionInterrupted, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    for (const v of message.executed) {
      writer.uint32(10).string(v!);
    }
    if (message.node_id !== "") {
      writer.uint32(18).string(message.node_id);
    }
    if (message.node_type !== "") {
      writer.uint32(26).string(message.node_type);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): ComfyMessage_ExecutionInterrupted {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseComfyMessage_ExecutionInterrupted();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 10) {
            break;
          }

          message.executed.push(reader.string());
          continue;
        case 2:
          if (tag !== 18) {
            break;
          }

          message.node_id = reader.string();
          continue;
        case 3:
          if (tag !== 26) {
            break;
          }

          message.node_type = reader.string();
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  create(base?: DeepPartial<ComfyMessage_ExecutionInterrupted>): ComfyMessage_ExecutionInterrupted {
    return ComfyMessage_ExecutionInterrupted.fromPartial(base ?? {});
  },
  fromPartial(object: DeepPartial<ComfyMessage_ExecutionInterrupted>): ComfyMessage_ExecutionInterrupted {
    const message = createBaseComfyMessage_ExecutionInterrupted();
    message.executed = object.executed?.map((e) => e) || [];
    message.node_id = object.node_id ?? "";
    message.node_type = object.node_type ?? "";
    return message;
  },
};

function createBaseComfyMessage_ExecutionCached(): ComfyMessage_ExecutionCached {
  return { node_ids: [] };
}

export const ComfyMessage_ExecutionCached = {
  encode(message: ComfyMessage_ExecutionCached, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    for (const v of message.node_ids) {
      writer.uint32(10).string(v!);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): ComfyMessage_ExecutionCached {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseComfyMessage_ExecutionCached();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 10) {
            break;
          }

          message.node_ids.push(reader.string());
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  create(base?: DeepPartial<ComfyMessage_ExecutionCached>): ComfyMessage_ExecutionCached {
    return ComfyMessage_ExecutionCached.fromPartial(base ?? {});
  },
  fromPartial(object: DeepPartial<ComfyMessage_ExecutionCached>): ComfyMessage_ExecutionCached {
    const message = createBaseComfyMessage_ExecutionCached();
    message.node_ids = object.node_ids?.map((e) => e) || [];
    return message;
  },
};

function createBaseComfyMessage_Output(): ComfyMessage_Output {
  return { node_id: "", files: [] };
}

export const ComfyMessage_Output = {
  encode(message: ComfyMessage_Output, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.node_id !== "") {
      writer.uint32(10).string(message.node_id);
    }
    for (const v of message.files) {
      WorkflowFile.encode(v!, writer.uint32(18).fork()).ldelim();
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): ComfyMessage_Output {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseComfyMessage_Output();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 10) {
            break;
          }

          message.node_id = reader.string();
          continue;
        case 2:
          if (tag !== 18) {
            break;
          }

          message.files.push(WorkflowFile.decode(reader, reader.uint32()));
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  create(base?: DeepPartial<ComfyMessage_Output>): ComfyMessage_Output {
    return ComfyMessage_Output.fromPartial(base ?? {});
  },
  fromPartial(object: DeepPartial<ComfyMessage_Output>): ComfyMessage_Output {
    const message = createBaseComfyMessage_Output();
    message.node_id = object.node_id ?? "";
    message.files = object.files?.map((e) => WorkflowFile.fromPartial(e)) || [];
    return message;
  },
};

function createBaseComfyMessage_CustomMessage(): ComfyMessage_CustomMessage {
  return { type: "", data: undefined };
}

export const ComfyMessage_CustomMessage = {
  encode(message: ComfyMessage_CustomMessage, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.type !== "") {
      writer.uint32(10).string(message.type);
    }
    if (message.data !== undefined) {
      Struct.encode(Struct.wrap(message.data), writer.uint32(18).fork()).ldelim();
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): ComfyMessage_CustomMessage {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseComfyMessage_CustomMessage();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 10) {
            break;
          }

          message.type = reader.string();
          continue;
        case 2:
          if (tag !== 18) {
            break;
          }

          message.data = Struct.unwrap(Struct.decode(reader, reader.uint32()));
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  create(base?: DeepPartial<ComfyMessage_CustomMessage>): ComfyMessage_CustomMessage {
    return ComfyMessage_CustomMessage.fromPartial(base ?? {});
  },
  fromPartial(object: DeepPartial<ComfyMessage_CustomMessage>): ComfyMessage_CustomMessage {
    const message = createBaseComfyMessage_CustomMessage();
    message.type = object.type ?? "";
    message.data = object.data ?? undefined;
    return message;
  },
};

function createBaseMessageFilter(): MessageFilter {
  return { outputs_only: false, include_temp_files: undefined, include_latent_previews: undefined };
}

export const MessageFilter = {
  encode(message: MessageFilter, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.outputs_only === true) {
      writer.uint32(8).bool(message.outputs_only);
    }
    if (message.include_temp_files !== undefined) {
      writer.uint32(16).bool(message.include_temp_files);
    }
    if (message.include_latent_previews !== undefined) {
      writer.uint32(24).bool(message.include_latent_previews);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): MessageFilter {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseMessageFilter();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 8) {
            break;
          }

          message.outputs_only = reader.bool();
          continue;
        case 2:
          if (tag !== 16) {
            break;
          }

          message.include_temp_files = reader.bool();
          continue;
        case 3:
          if (tag !== 24) {
            break;
          }

          message.include_latent_previews = reader.bool();
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  create(base?: DeepPartial<MessageFilter>): MessageFilter {
    return MessageFilter.fromPartial(base ?? {});
  },
  fromPartial(object: DeepPartial<MessageFilter>): MessageFilter {
    const message = createBaseMessageFilter();
    message.outputs_only = object.outputs_only ?? false;
    message.include_temp_files = object.include_temp_files ?? undefined;
    message.include_latent_previews = object.include_latent_previews ?? undefined;
    return message;
  },
};

function createBaseRoomStreamRequest(): RoomStreamRequest {
  return { session_id: "", filter: undefined };
}

export const RoomStreamRequest = {
  encode(message: RoomStreamRequest, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.session_id !== "") {
      writer.uint32(10).string(message.session_id);
    }
    if (message.filter !== undefined) {
      MessageFilter.encode(message.filter, writer.uint32(18).fork()).ldelim();
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): RoomStreamRequest {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseRoomStreamRequest();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 10) {
            break;
          }

          message.session_id = reader.string();
          continue;
        case 2:
          if (tag !== 18) {
            break;
          }

          message.filter = MessageFilter.decode(reader, reader.uint32());
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  create(base?: DeepPartial<RoomStreamRequest>): RoomStreamRequest {
    return RoomStreamRequest.fromPartial(base ?? {});
  },
  fromPartial(object: DeepPartial<RoomStreamRequest>): RoomStreamRequest {
    const message = createBaseRoomStreamRequest();
    message.session_id = object.session_id ?? "";
    message.filter = (object.filter !== undefined && object.filter !== null)
      ? MessageFilter.fromPartial(object.filter)
      : undefined;
    return message;
  },
};

function createBaseJobStreamRequest(): JobStreamRequest {
  return { job_id: "", filter: undefined };
}

export const JobStreamRequest = {
  encode(message: JobStreamRequest, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.job_id !== "") {
      writer.uint32(10).string(message.job_id);
    }
    if (message.filter !== undefined) {
      MessageFilter.encode(message.filter, writer.uint32(18).fork()).ldelim();
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): JobStreamRequest {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseJobStreamRequest();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 10) {
            break;
          }

          message.job_id = reader.string();
          continue;
        case 2:
          if (tag !== 18) {
            break;
          }

          message.filter = MessageFilter.decode(reader, reader.uint32());
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  create(base?: DeepPartial<JobStreamRequest>): JobStreamRequest {
    return JobStreamRequest.fromPartial(base ?? {});
  },
  fromPartial(object: DeepPartial<JobStreamRequest>): JobStreamRequest {
    const message = createBaseJobStreamRequest();
    message.job_id = object.job_id ?? "";
    message.filter = (object.filter !== undefined && object.filter !== null)
      ? MessageFilter.fromPartial(object.filter)
      : undefined;
    return message;
  },
};

export type ComfyDefinition = typeof ComfyDefinition;
export const ComfyDefinition = {
  name: "Comfy",
  fullName: "comfy_request.v1.Comfy",
  methods: {
    /** Queue a workflow and receive the job id */
    runWorkflow: {
      name: "RunWorkflow",
      requestType: ComfyRequest,
      requestStream: false,
      responseType: JobCreated,
      responseStream: false,
      options: {},
    },
    /** Queue a workflow and wait for the final outputs */
    runWorkflowSync: {
      name: "RunWorkflowSync",
      requestType: ComfyRequest,
      requestStream: false,
      responseType: JobOutput,
      responseStream: false,
      options: {},
    },
    /** Cancels a specific job (regardless if it's running or queued) */
    cancel: {
      name: "Cancel",
      requestType: CancelJob,
      requestStream: false,
      responseType: Empty,
      responseStream: false,
      options: {},
    },
    /** Cancels all queued jobs owned by the user in a given session-id */
    purgeQueue: {
      name: "PurgeQueue",
      requestType: PurgeRoomQueue,
      requestStream: false,
      responseType: Empty,
      responseStream: false,
      options: {},
    },
    /** Server-side stream of all jobs in a given session-id */
    streamRoom: {
      name: "StreamRoom",
      requestType: RoomStreamRequest,
      requestStream: false,
      responseType: ComfyMessage,
      responseStream: true,
      options: {},
    },
    /** Server-side stream of a specific job-id */
    streamJob: {
      name: "StreamJob",
      requestType: JobStreamRequest,
      requestStream: false,
      responseType: ComfyMessage,
      responseStream: true,
      options: {},
    },
  },
} as const;

export interface ComfyServiceImplementation<CallContextExt = {}> {
  /** Queue a workflow and receive the job id */
  runWorkflow(request: ComfyRequest, context: CallContext & CallContextExt): Promise<DeepPartial<JobCreated>>;
  /** Queue a workflow and wait for the final outputs */
  runWorkflowSync(request: ComfyRequest, context: CallContext & CallContextExt): Promise<DeepPartial<JobOutput>>;
  /** Cancels a specific job (regardless if it's running or queued) */
  cancel(request: CancelJob, context: CallContext & CallContextExt): Promise<DeepPartial<Empty>>;
  /** Cancels all queued jobs owned by the user in a given session-id */
  purgeQueue(request: PurgeRoomQueue, context: CallContext & CallContextExt): Promise<DeepPartial<Empty>>;
  /** Server-side stream of all jobs in a given session-id */
  streamRoom(
    request: RoomStreamRequest,
    context: CallContext & CallContextExt,
  ): ServerStreamingMethodResult<DeepPartial<ComfyMessage>>;
  /** Server-side stream of a specific job-id */
  streamJob(
    request: JobStreamRequest,
    context: CallContext & CallContextExt,
  ): ServerStreamingMethodResult<DeepPartial<ComfyMessage>>;
}

export interface ComfyClient<CallOptionsExt = {}> {
  /** Queue a workflow and receive the job id */
  runWorkflow(request: DeepPartial<ComfyRequest>, options?: CallOptions & CallOptionsExt): Promise<JobCreated>;
  /** Queue a workflow and wait for the final outputs */
  runWorkflowSync(request: DeepPartial<ComfyRequest>, options?: CallOptions & CallOptionsExt): Promise<JobOutput>;
  /** Cancels a specific job (regardless if it's running or queued) */
  cancel(request: DeepPartial<CancelJob>, options?: CallOptions & CallOptionsExt): Promise<Empty>;
  /** Cancels all queued jobs owned by the user in a given session-id */
  purgeQueue(request: DeepPartial<PurgeRoomQueue>, options?: CallOptions & CallOptionsExt): Promise<Empty>;
  /** Server-side stream of all jobs in a given session-id */
  streamRoom(
    request: DeepPartial<RoomStreamRequest>,
    options?: CallOptions & CallOptionsExt,
  ): AsyncIterable<ComfyMessage>;
  /** Server-side stream of a specific job-id */
  streamJob(
    request: DeepPartial<JobStreamRequest>,
    options?: CallOptions & CallOptionsExt,
  ): AsyncIterable<ComfyMessage>;
}

type Builtin = Date | Function | Uint8Array | string | number | boolean | undefined;

export type DeepPartial<T> = T extends Builtin ? T
  : T extends globalThis.Array<infer U> ? globalThis.Array<DeepPartial<U>>
  : T extends ReadonlyArray<infer U> ? ReadonlyArray<DeepPartial<U>>
  : T extends {} ? { [K in keyof T]?: DeepPartial<T[K]> }
  : Partial<T>;

export type ServerStreamingMethodResult<Response> = { [Symbol.asyncIterator](): AsyncIterator<Response, void> };
