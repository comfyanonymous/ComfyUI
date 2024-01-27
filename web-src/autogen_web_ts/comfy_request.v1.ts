/* eslint-disable */
import type { CallContext, CallOptions } from "nice-grpc-common";
import * as _m0 from "protobufjs/minimal";
import { Empty } from "./google/empty";
import { Struct } from "./google/struct";

export const protobufPackage = "comfy_request.v1";

/** These are more direct client-created workflows for client -> server -> worker */

/** Message definition for WorkflowStep */
export interface WorkflowStep {
  classType: string;
  /** Inputs are too idiosyncratic to define ahead of time */
  inputs: { [key: string]: any } | undefined;
}

/**
 * keys are node_ids
 * Unfortunately 'steps' is necessary because we cannot have a protobuf definition like
 * repeated map<A, B>, so we have to wrap it this message
 */
export interface Workflow {
  steps: { [key: string]: WorkflowStep };
}

export interface Workflow_StepsEntry {
  key: string;
  value: WorkflowStep | undefined;
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
  fileHash: string;
  /** ComfyUI terminology: key 'format' */
  mimeType: string;
  reference?: WorkflowFile_FileReference | undefined;
  data?: Uint8Array | undefined;
}

export interface WorkflowFile_FileReference {
  /** string must be a valid url */
  url: string;
  /** Comfy UI terminology: key 'type', values 'temp' | 'output' */
  isTemp: boolean;
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
  saveOutputs: boolean;
  sendLatentPreviews: boolean;
}

/** client -> server message */
export interface ComfyRequest {
  userUid: string;
  workflows: Workflow[];
  inputFiles: WorkflowFile[];
  outputConfig: OutputConfig | undefined;
  workerWaitDuration?:
    | number
    | undefined;
  /** redis channel name to publish results to */
  sessionId?: string | undefined;
}

/**
 * ComfyUI has 'delete' (remove specific non-running items), and 'interrupt'
 * (stop currently running process) commands. We roll them all into a single endpoint here.
 */
export interface CancelJob {
  jobId: string;
}

/** ComfyUI calls this 'clear' (remove all queued items owned by the user) */
export interface PurgeRoomQueue {
  sessionId: string;
}

export interface JobCreated {
  /** created by the server; id of the job in the queue */
  jobId: string;
  /** redis channel to subscribe to for updates */
  sessionId: string;
  queueSeconds: number;
  executionSeconds: number;
}

/** Temp-files and latent-previews are not included */
export interface JobOutput {
  jobId: string;
  sessionId: string;
  files: WorkflowFile[];
}

/** It's assumed that the consumer knows what session_id it's watching */
export interface ComfyMessage {
  jobId: string;
  userId: string;
  queueStatus?: ComfyMessage_QueueStatus | undefined;
  executionStart?: ComfyMessage_ExecutionStart | undefined;
  executing?: ComfyMessage_Executing | undefined;
  progress?: ComfyMessage_Progress | undefined;
  executionError?: ComfyMessage_ExecutionError | undefined;
  executionInterrupted?: ComfyMessage_ExecutionInterrupted | undefined;
  executionCached?: ComfyMessage_ExecutionCached | undefined;
  output?: ComfyMessage_Output | undefined;
  customMessage?: ComfyMessage_CustomMessage | undefined;
}

/**
 * updates queue-display on client. SID's purpose is unknown
 * ComfyUI terminology: 'Status'
 */
export interface ComfyMessage_QueueStatus {
  /** looks like: "99506f0d89b64dbdb09ae567274fb078" */
  sid?: string | undefined;
  queueRemaining: number;
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
  nodeId: string;
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
  executionMessage: string;
  exceptionType: string;
  /** list of nodes executed */
  executed: string[];
  /** node id that threw the error */
  nodeId: string;
  nodeType: string;
  traceback: string[];
}

export interface ComfyMessage_ExecutionInterrupted {
  /** node-ids that already finished */
  executed: string[];
  nodeId: string;
  nodeType: string;
}

/** This specifies nodes that were skipped due to their output being cached */
export interface ComfyMessage_ExecutionCached {
  nodeIds: string[];
}

/**
 * A node produced an output (temp or saved); display it
 * In the original ComfyUI, it's like output.images[0] = { filename, subfolder, type }, or output.gifs[0]
 * There is also output.animated; an array whose indices are bools corresponding to the images array
 * We simplify all of this
 * ComfyUI terinology; this is called 'Executed', which was confusing
 */
export interface ComfyMessage_Output {
  nodeId: string;
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
  outputsOnly: boolean;
  includeTempFiles?: boolean | undefined;
  includeLatentPreviews?: boolean | undefined;
}

/** By default, all message-types will be returned, unless a filter is applied */
export interface RoomStreamRequest {
  sessionId: string;
  filter?: MessageFilter | undefined;
}

export interface JobStreamRequest {
  jobId: string;
  filter?: MessageFilter | undefined;
}

function createBaseWorkflowStep(): WorkflowStep {
  return { classType: "", inputs: undefined };
}

export const WorkflowStep = {
  encode(message: WorkflowStep, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.classType !== "") {
      writer.uint32(10).string(message.classType);
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

          message.classType = reader.string();
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
    message.classType = object.classType ?? "";
    message.inputs = object.inputs ?? undefined;
    return message;
  },
};

function createBaseWorkflow(): Workflow {
  return { steps: {} };
}

export const Workflow = {
  encode(message: Workflow, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    Object.entries(message.steps).forEach(([key, value]) => {
      Workflow_StepsEntry.encode({ key: key as any, value }, writer.uint32(10).fork()).ldelim();
    });
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): Workflow {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseWorkflow();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 10) {
            break;
          }

          const entry1 = Workflow_StepsEntry.decode(reader, reader.uint32());
          if (entry1.value !== undefined) {
            message.steps[entry1.key] = entry1.value;
          }
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  create(base?: DeepPartial<Workflow>): Workflow {
    return Workflow.fromPartial(base ?? {});
  },
  fromPartial(object: DeepPartial<Workflow>): Workflow {
    const message = createBaseWorkflow();
    message.steps = Object.entries(object.steps ?? {}).reduce<{ [key: string]: WorkflowStep }>((acc, [key, value]) => {
      if (value !== undefined) {
        acc[key] = WorkflowStep.fromPartial(value);
      }
      return acc;
    }, {});
    return message;
  },
};

function createBaseWorkflow_StepsEntry(): Workflow_StepsEntry {
  return { key: "", value: undefined };
}

export const Workflow_StepsEntry = {
  encode(message: Workflow_StepsEntry, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.key !== "") {
      writer.uint32(10).string(message.key);
    }
    if (message.value !== undefined) {
      WorkflowStep.encode(message.value, writer.uint32(18).fork()).ldelim();
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): Workflow_StepsEntry {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseWorkflow_StepsEntry();
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

  create(base?: DeepPartial<Workflow_StepsEntry>): Workflow_StepsEntry {
    return Workflow_StepsEntry.fromPartial(base ?? {});
  },
  fromPartial(object: DeepPartial<Workflow_StepsEntry>): Workflow_StepsEntry {
    const message = createBaseWorkflow_StepsEntry();
    message.key = object.key ?? "";
    message.value = (object.value !== undefined && object.value !== null)
      ? WorkflowStep.fromPartial(object.value)
      : undefined;
    return message;
  },
};

function createBaseWorkflowFile(): WorkflowFile {
  return { fileHash: "", mimeType: "", reference: undefined, data: undefined };
}

export const WorkflowFile = {
  encode(message: WorkflowFile, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.fileHash !== "") {
      writer.uint32(10).string(message.fileHash);
    }
    if (message.mimeType !== "") {
      writer.uint32(18).string(message.mimeType);
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

          message.fileHash = reader.string();
          continue;
        case 2:
          if (tag !== 18) {
            break;
          }

          message.mimeType = reader.string();
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
    message.fileHash = object.fileHash ?? "";
    message.mimeType = object.mimeType ?? "";
    message.reference = (object.reference !== undefined && object.reference !== null)
      ? WorkflowFile_FileReference.fromPartial(object.reference)
      : undefined;
    message.data = object.data ?? undefined;
    return message;
  },
};

function createBaseWorkflowFile_FileReference(): WorkflowFile_FileReference {
  return { url: "", isTemp: false };
}

export const WorkflowFile_FileReference = {
  encode(message: WorkflowFile_FileReference, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.url !== "") {
      writer.uint32(10).string(message.url);
    }
    if (message.isTemp === true) {
      writer.uint32(16).bool(message.isTemp);
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

          message.isTemp = reader.bool();
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
    message.isTemp = object.isTemp ?? false;
    return message;
  },
};

function createBaseOutputConfig(): OutputConfig {
  return { saveOutputs: false, sendLatentPreviews: false };
}

export const OutputConfig = {
  encode(message: OutputConfig, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.saveOutputs === true) {
      writer.uint32(8).bool(message.saveOutputs);
    }
    if (message.sendLatentPreviews === true) {
      writer.uint32(16).bool(message.sendLatentPreviews);
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

          message.saveOutputs = reader.bool();
          continue;
        case 2:
          if (tag !== 16) {
            break;
          }

          message.sendLatentPreviews = reader.bool();
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
    message.saveOutputs = object.saveOutputs ?? false;
    message.sendLatentPreviews = object.sendLatentPreviews ?? false;
    return message;
  },
};

function createBaseComfyRequest(): ComfyRequest {
  return {
    userUid: "",
    workflows: [],
    inputFiles: [],
    outputConfig: undefined,
    workerWaitDuration: undefined,
    sessionId: undefined,
  };
}

export const ComfyRequest = {
  encode(message: ComfyRequest, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.userUid !== "") {
      writer.uint32(10).string(message.userUid);
    }
    for (const v of message.workflows) {
      Workflow.encode(v!, writer.uint32(18).fork()).ldelim();
    }
    for (const v of message.inputFiles) {
      WorkflowFile.encode(v!, writer.uint32(26).fork()).ldelim();
    }
    if (message.outputConfig !== undefined) {
      OutputConfig.encode(message.outputConfig, writer.uint32(34).fork()).ldelim();
    }
    if (message.workerWaitDuration !== undefined) {
      writer.uint32(40).uint32(message.workerWaitDuration);
    }
    if (message.sessionId !== undefined) {
      writer.uint32(50).string(message.sessionId);
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

          message.userUid = reader.string();
          continue;
        case 2:
          if (tag !== 18) {
            break;
          }

          message.workflows.push(Workflow.decode(reader, reader.uint32()));
          continue;
        case 3:
          if (tag !== 26) {
            break;
          }

          message.inputFiles.push(WorkflowFile.decode(reader, reader.uint32()));
          continue;
        case 4:
          if (tag !== 34) {
            break;
          }

          message.outputConfig = OutputConfig.decode(reader, reader.uint32());
          continue;
        case 5:
          if (tag !== 40) {
            break;
          }

          message.workerWaitDuration = reader.uint32();
          continue;
        case 6:
          if (tag !== 50) {
            break;
          }

          message.sessionId = reader.string();
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
    message.userUid = object.userUid ?? "";
    message.workflows = object.workflows?.map((e) => Workflow.fromPartial(e)) || [];
    message.inputFiles = object.inputFiles?.map((e) => WorkflowFile.fromPartial(e)) || [];
    message.outputConfig = (object.outputConfig !== undefined && object.outputConfig !== null)
      ? OutputConfig.fromPartial(object.outputConfig)
      : undefined;
    message.workerWaitDuration = object.workerWaitDuration ?? undefined;
    message.sessionId = object.sessionId ?? undefined;
    return message;
  },
};

function createBaseCancelJob(): CancelJob {
  return { jobId: "" };
}

export const CancelJob = {
  encode(message: CancelJob, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.jobId !== "") {
      writer.uint32(10).string(message.jobId);
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

          message.jobId = reader.string();
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
    message.jobId = object.jobId ?? "";
    return message;
  },
};

function createBasePurgeRoomQueue(): PurgeRoomQueue {
  return { sessionId: "" };
}

export const PurgeRoomQueue = {
  encode(message: PurgeRoomQueue, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.sessionId !== "") {
      writer.uint32(10).string(message.sessionId);
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

          message.sessionId = reader.string();
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
    message.sessionId = object.sessionId ?? "";
    return message;
  },
};

function createBaseJobCreated(): JobCreated {
  return { jobId: "", sessionId: "", queueSeconds: 0, executionSeconds: 0 };
}

export const JobCreated = {
  encode(message: JobCreated, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.jobId !== "") {
      writer.uint32(10).string(message.jobId);
    }
    if (message.sessionId !== "") {
      writer.uint32(18).string(message.sessionId);
    }
    if (message.queueSeconds !== 0) {
      writer.uint32(24).uint32(message.queueSeconds);
    }
    if (message.executionSeconds !== 0) {
      writer.uint32(32).uint32(message.executionSeconds);
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

          message.jobId = reader.string();
          continue;
        case 2:
          if (tag !== 18) {
            break;
          }

          message.sessionId = reader.string();
          continue;
        case 3:
          if (tag !== 24) {
            break;
          }

          message.queueSeconds = reader.uint32();
          continue;
        case 4:
          if (tag !== 32) {
            break;
          }

          message.executionSeconds = reader.uint32();
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
    message.jobId = object.jobId ?? "";
    message.sessionId = object.sessionId ?? "";
    message.queueSeconds = object.queueSeconds ?? 0;
    message.executionSeconds = object.executionSeconds ?? 0;
    return message;
  },
};

function createBaseJobOutput(): JobOutput {
  return { jobId: "", sessionId: "", files: [] };
}

export const JobOutput = {
  encode(message: JobOutput, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.jobId !== "") {
      writer.uint32(10).string(message.jobId);
    }
    if (message.sessionId !== "") {
      writer.uint32(18).string(message.sessionId);
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

          message.jobId = reader.string();
          continue;
        case 2:
          if (tag !== 18) {
            break;
          }

          message.sessionId = reader.string();
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
    message.jobId = object.jobId ?? "";
    message.sessionId = object.sessionId ?? "";
    message.files = object.files?.map((e) => WorkflowFile.fromPartial(e)) || [];
    return message;
  },
};

function createBaseComfyMessage(): ComfyMessage {
  return {
    jobId: "",
    userId: "",
    queueStatus: undefined,
    executionStart: undefined,
    executing: undefined,
    progress: undefined,
    executionError: undefined,
    executionInterrupted: undefined,
    executionCached: undefined,
    output: undefined,
    customMessage: undefined,
  };
}

export const ComfyMessage = {
  encode(message: ComfyMessage, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.jobId !== "") {
      writer.uint32(10).string(message.jobId);
    }
    if (message.userId !== "") {
      writer.uint32(18).string(message.userId);
    }
    if (message.queueStatus !== undefined) {
      ComfyMessage_QueueStatus.encode(message.queueStatus, writer.uint32(26).fork()).ldelim();
    }
    if (message.executionStart !== undefined) {
      ComfyMessage_ExecutionStart.encode(message.executionStart, writer.uint32(34).fork()).ldelim();
    }
    if (message.executing !== undefined) {
      ComfyMessage_Executing.encode(message.executing, writer.uint32(42).fork()).ldelim();
    }
    if (message.progress !== undefined) {
      ComfyMessage_Progress.encode(message.progress, writer.uint32(50).fork()).ldelim();
    }
    if (message.executionError !== undefined) {
      ComfyMessage_ExecutionError.encode(message.executionError, writer.uint32(58).fork()).ldelim();
    }
    if (message.executionInterrupted !== undefined) {
      ComfyMessage_ExecutionInterrupted.encode(message.executionInterrupted, writer.uint32(66).fork()).ldelim();
    }
    if (message.executionCached !== undefined) {
      ComfyMessage_ExecutionCached.encode(message.executionCached, writer.uint32(74).fork()).ldelim();
    }
    if (message.output !== undefined) {
      ComfyMessage_Output.encode(message.output, writer.uint32(82).fork()).ldelim();
    }
    if (message.customMessage !== undefined) {
      ComfyMessage_CustomMessage.encode(message.customMessage, writer.uint32(90).fork()).ldelim();
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

          message.jobId = reader.string();
          continue;
        case 2:
          if (tag !== 18) {
            break;
          }

          message.userId = reader.string();
          continue;
        case 3:
          if (tag !== 26) {
            break;
          }

          message.queueStatus = ComfyMessage_QueueStatus.decode(reader, reader.uint32());
          continue;
        case 4:
          if (tag !== 34) {
            break;
          }

          message.executionStart = ComfyMessage_ExecutionStart.decode(reader, reader.uint32());
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

          message.executionError = ComfyMessage_ExecutionError.decode(reader, reader.uint32());
          continue;
        case 8:
          if (tag !== 66) {
            break;
          }

          message.executionInterrupted = ComfyMessage_ExecutionInterrupted.decode(reader, reader.uint32());
          continue;
        case 9:
          if (tag !== 74) {
            break;
          }

          message.executionCached = ComfyMessage_ExecutionCached.decode(reader, reader.uint32());
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

          message.customMessage = ComfyMessage_CustomMessage.decode(reader, reader.uint32());
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
    message.jobId = object.jobId ?? "";
    message.userId = object.userId ?? "";
    message.queueStatus = (object.queueStatus !== undefined && object.queueStatus !== null)
      ? ComfyMessage_QueueStatus.fromPartial(object.queueStatus)
      : undefined;
    message.executionStart = (object.executionStart !== undefined && object.executionStart !== null)
      ? ComfyMessage_ExecutionStart.fromPartial(object.executionStart)
      : undefined;
    message.executing = (object.executing !== undefined && object.executing !== null)
      ? ComfyMessage_Executing.fromPartial(object.executing)
      : undefined;
    message.progress = (object.progress !== undefined && object.progress !== null)
      ? ComfyMessage_Progress.fromPartial(object.progress)
      : undefined;
    message.executionError = (object.executionError !== undefined && object.executionError !== null)
      ? ComfyMessage_ExecutionError.fromPartial(object.executionError)
      : undefined;
    message.executionInterrupted = (object.executionInterrupted !== undefined && object.executionInterrupted !== null)
      ? ComfyMessage_ExecutionInterrupted.fromPartial(object.executionInterrupted)
      : undefined;
    message.executionCached = (object.executionCached !== undefined && object.executionCached !== null)
      ? ComfyMessage_ExecutionCached.fromPartial(object.executionCached)
      : undefined;
    message.output = (object.output !== undefined && object.output !== null)
      ? ComfyMessage_Output.fromPartial(object.output)
      : undefined;
    message.customMessage = (object.customMessage !== undefined && object.customMessage !== null)
      ? ComfyMessage_CustomMessage.fromPartial(object.customMessage)
      : undefined;
    return message;
  },
};

function createBaseComfyMessage_QueueStatus(): ComfyMessage_QueueStatus {
  return { sid: undefined, queueRemaining: 0 };
}

export const ComfyMessage_QueueStatus = {
  encode(message: ComfyMessage_QueueStatus, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.sid !== undefined) {
      writer.uint32(10).string(message.sid);
    }
    if (message.queueRemaining !== 0) {
      writer.uint32(16).uint32(message.queueRemaining);
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

          message.queueRemaining = reader.uint32();
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
    message.queueRemaining = object.queueRemaining ?? 0;
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
  return { nodeId: "" };
}

export const ComfyMessage_Executing = {
  encode(message: ComfyMessage_Executing, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.nodeId !== "") {
      writer.uint32(10).string(message.nodeId);
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

          message.nodeId = reader.string();
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
    message.nodeId = object.nodeId ?? "";
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
    executionMessage: "",
    exceptionType: "",
    executed: [],
    nodeId: "",
    nodeType: "",
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
    if (message.executionMessage !== "") {
      writer.uint32(26).string(message.executionMessage);
    }
    if (message.exceptionType !== "") {
      writer.uint32(34).string(message.exceptionType);
    }
    for (const v of message.executed) {
      writer.uint32(42).string(v!);
    }
    if (message.nodeId !== "") {
      writer.uint32(50).string(message.nodeId);
    }
    if (message.nodeType !== "") {
      writer.uint32(58).string(message.nodeType);
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

          message.executionMessage = reader.string();
          continue;
        case 4:
          if (tag !== 34) {
            break;
          }

          message.exceptionType = reader.string();
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

          message.nodeId = reader.string();
          continue;
        case 7:
          if (tag !== 58) {
            break;
          }

          message.nodeType = reader.string();
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
    message.executionMessage = object.executionMessage ?? "";
    message.exceptionType = object.exceptionType ?? "";
    message.executed = object.executed?.map((e) => e) || [];
    message.nodeId = object.nodeId ?? "";
    message.nodeType = object.nodeType ?? "";
    message.traceback = object.traceback?.map((e) => e) || [];
    return message;
  },
};

function createBaseComfyMessage_ExecutionInterrupted(): ComfyMessage_ExecutionInterrupted {
  return { executed: [], nodeId: "", nodeType: "" };
}

export const ComfyMessage_ExecutionInterrupted = {
  encode(message: ComfyMessage_ExecutionInterrupted, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    for (const v of message.executed) {
      writer.uint32(10).string(v!);
    }
    if (message.nodeId !== "") {
      writer.uint32(18).string(message.nodeId);
    }
    if (message.nodeType !== "") {
      writer.uint32(26).string(message.nodeType);
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

          message.nodeId = reader.string();
          continue;
        case 3:
          if (tag !== 26) {
            break;
          }

          message.nodeType = reader.string();
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
    message.nodeId = object.nodeId ?? "";
    message.nodeType = object.nodeType ?? "";
    return message;
  },
};

function createBaseComfyMessage_ExecutionCached(): ComfyMessage_ExecutionCached {
  return { nodeIds: [] };
}

export const ComfyMessage_ExecutionCached = {
  encode(message: ComfyMessage_ExecutionCached, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    for (const v of message.nodeIds) {
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

          message.nodeIds.push(reader.string());
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
    message.nodeIds = object.nodeIds?.map((e) => e) || [];
    return message;
  },
};

function createBaseComfyMessage_Output(): ComfyMessage_Output {
  return { nodeId: "", files: [] };
}

export const ComfyMessage_Output = {
  encode(message: ComfyMessage_Output, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.nodeId !== "") {
      writer.uint32(10).string(message.nodeId);
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

          message.nodeId = reader.string();
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
    message.nodeId = object.nodeId ?? "";
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
  return { outputsOnly: false, includeTempFiles: undefined, includeLatentPreviews: undefined };
}

export const MessageFilter = {
  encode(message: MessageFilter, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.outputsOnly === true) {
      writer.uint32(8).bool(message.outputsOnly);
    }
    if (message.includeTempFiles !== undefined) {
      writer.uint32(16).bool(message.includeTempFiles);
    }
    if (message.includeLatentPreviews !== undefined) {
      writer.uint32(24).bool(message.includeLatentPreviews);
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

          message.outputsOnly = reader.bool();
          continue;
        case 2:
          if (tag !== 16) {
            break;
          }

          message.includeTempFiles = reader.bool();
          continue;
        case 3:
          if (tag !== 24) {
            break;
          }

          message.includeLatentPreviews = reader.bool();
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
    message.outputsOnly = object.outputsOnly ?? false;
    message.includeTempFiles = object.includeTempFiles ?? undefined;
    message.includeLatentPreviews = object.includeLatentPreviews ?? undefined;
    return message;
  },
};

function createBaseRoomStreamRequest(): RoomStreamRequest {
  return { sessionId: "", filter: undefined };
}

export const RoomStreamRequest = {
  encode(message: RoomStreamRequest, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.sessionId !== "") {
      writer.uint32(10).string(message.sessionId);
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

          message.sessionId = reader.string();
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
    message.sessionId = object.sessionId ?? "";
    message.filter = (object.filter !== undefined && object.filter !== null)
      ? MessageFilter.fromPartial(object.filter)
      : undefined;
    return message;
  },
};

function createBaseJobStreamRequest(): JobStreamRequest {
  return { jobId: "", filter: undefined };
}

export const JobStreamRequest = {
  encode(message: JobStreamRequest, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.jobId !== "") {
      writer.uint32(10).string(message.jobId);
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

          message.jobId = reader.string();
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
    message.jobId = object.jobId ?? "";
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
    /** Server-side stream of a specific job */
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
  /** Server-side stream of a specific job */
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
  /** Server-side stream of a specific job */
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
