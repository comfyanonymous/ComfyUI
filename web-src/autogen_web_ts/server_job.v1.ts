/* eslint-disable */
import type { CallContext, CallOptions } from "nice-grpc-common";
import * as _m0 from "protobufjs/minimal";

export const protobufPackage = "server_job.v1";

/** These are requests for client -> server to create jobs */

export enum JobStatus {
  IN_QUEUE = "IN_QUEUE",
  IN_PROGRESS = "IN_PROGRESS",
  FAILED = "FAILED",
  COMPLETED = "COMPLETED",
  UNRECOGNIZED = "UNRECOGNIZED",
}

export function jobStatusFromJSON(object: any): JobStatus {
  switch (object) {
    case 0:
    case "IN_QUEUE":
      return JobStatus.IN_QUEUE;
    case 1:
    case "IN_PROGRESS":
      return JobStatus.IN_PROGRESS;
    case 2:
    case "FAILED":
      return JobStatus.FAILED;
    case 3:
    case "COMPLETED":
      return JobStatus.COMPLETED;
    case -1:
    case "UNRECOGNIZED":
    default:
      return JobStatus.UNRECOGNIZED;
  }
}

export function jobStatusToNumber(object: JobStatus): number {
  switch (object) {
    case JobStatus.IN_QUEUE:
      return 0;
    case JobStatus.IN_PROGRESS:
      return 1;
    case JobStatus.FAILED:
      return 2;
    case JobStatus.COMPLETED:
      return 3;
    case JobStatus.UNRECOGNIZED:
    default:
      return -1;
  }
}

export interface JobResponse {
  /** Firestore doc-id that can be used to watch status */
  uid: string;
  status: JobStatus;
}

export interface CanvasBox {
  text: string;
  width: number;
  height: number;
  x: number;
  y: number;
}

export interface BasicJob {
  /** we use strings for model-names rather than enums */
  modelName: string[];
  positivePrompt: string;
  negativePrompt: string;
  aspectRatio: string;
  numImages: number;
  seed: number;
}

export interface UpscaleJob {
  upscaleModel: string;
  imageUid: string;
  denoise: number;
}

export interface GligenBox {
  modelName: string;
  boxes: CanvasBox[];
  positivePrompt: string;
  negativePrompt: string;
  aspectRatio: string;
  numImages: number;
  seed: number;
}

export interface BasicSVD {
  imageUid: string;
  imageUrl: string;
  motion: number;
  seed: number;
  modelName: string;
  positivePrompt: string;
  negativePrompt: string;
  aspectRatio: string;
}

export interface JobType {
  basicJob?: BasicJob | undefined;
  upscaleJob?: UpscaleJob | undefined;
  gligenBox?: GligenBox | undefined;
  basicSVD?: BasicSVD | undefined;
}

function createBaseJobResponse(): JobResponse {
  return { uid: "", status: JobStatus.IN_QUEUE };
}

export const JobResponse = {
  encode(message: JobResponse, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.uid !== "") {
      writer.uint32(10).string(message.uid);
    }
    if (message.status !== JobStatus.IN_QUEUE) {
      writer.uint32(16).int32(jobStatusToNumber(message.status));
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): JobResponse {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseJobResponse();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 10) {
            break;
          }

          message.uid = reader.string();
          continue;
        case 2:
          if (tag !== 16) {
            break;
          }

          message.status = jobStatusFromJSON(reader.int32());
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  create(base?: DeepPartial<JobResponse>): JobResponse {
    return JobResponse.fromPartial(base ?? {});
  },
  fromPartial(object: DeepPartial<JobResponse>): JobResponse {
    const message = createBaseJobResponse();
    message.uid = object.uid ?? "";
    message.status = object.status ?? JobStatus.IN_QUEUE;
    return message;
  },
};

function createBaseCanvasBox(): CanvasBox {
  return { text: "", width: 0, height: 0, x: 0, y: 0 };
}

export const CanvasBox = {
  encode(message: CanvasBox, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.text !== "") {
      writer.uint32(10).string(message.text);
    }
    if (message.width !== 0) {
      writer.uint32(16).uint32(message.width);
    }
    if (message.height !== 0) {
      writer.uint32(24).uint32(message.height);
    }
    if (message.x !== 0) {
      writer.uint32(32).int32(message.x);
    }
    if (message.y !== 0) {
      writer.uint32(40).int32(message.y);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): CanvasBox {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseCanvasBox();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 10) {
            break;
          }

          message.text = reader.string();
          continue;
        case 2:
          if (tag !== 16) {
            break;
          }

          message.width = reader.uint32();
          continue;
        case 3:
          if (tag !== 24) {
            break;
          }

          message.height = reader.uint32();
          continue;
        case 4:
          if (tag !== 32) {
            break;
          }

          message.x = reader.int32();
          continue;
        case 5:
          if (tag !== 40) {
            break;
          }

          message.y = reader.int32();
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  create(base?: DeepPartial<CanvasBox>): CanvasBox {
    return CanvasBox.fromPartial(base ?? {});
  },
  fromPartial(object: DeepPartial<CanvasBox>): CanvasBox {
    const message = createBaseCanvasBox();
    message.text = object.text ?? "";
    message.width = object.width ?? 0;
    message.height = object.height ?? 0;
    message.x = object.x ?? 0;
    message.y = object.y ?? 0;
    return message;
  },
};

function createBaseBasicJob(): BasicJob {
  return { modelName: [], positivePrompt: "", negativePrompt: "", aspectRatio: "", numImages: 0, seed: 0 };
}

export const BasicJob = {
  encode(message: BasicJob, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    for (const v of message.modelName) {
      writer.uint32(10).string(v!);
    }
    if (message.positivePrompt !== "") {
      writer.uint32(18).string(message.positivePrompt);
    }
    if (message.negativePrompt !== "") {
      writer.uint32(26).string(message.negativePrompt);
    }
    if (message.aspectRatio !== "") {
      writer.uint32(34).string(message.aspectRatio);
    }
    if (message.numImages !== 0) {
      writer.uint32(40).int32(message.numImages);
    }
    if (message.seed !== 0) {
      writer.uint32(48).int32(message.seed);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): BasicJob {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseBasicJob();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 10) {
            break;
          }

          message.modelName.push(reader.string());
          continue;
        case 2:
          if (tag !== 18) {
            break;
          }

          message.positivePrompt = reader.string();
          continue;
        case 3:
          if (tag !== 26) {
            break;
          }

          message.negativePrompt = reader.string();
          continue;
        case 4:
          if (tag !== 34) {
            break;
          }

          message.aspectRatio = reader.string();
          continue;
        case 5:
          if (tag !== 40) {
            break;
          }

          message.numImages = reader.int32();
          continue;
        case 6:
          if (tag !== 48) {
            break;
          }

          message.seed = reader.int32();
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  create(base?: DeepPartial<BasicJob>): BasicJob {
    return BasicJob.fromPartial(base ?? {});
  },
  fromPartial(object: DeepPartial<BasicJob>): BasicJob {
    const message = createBaseBasicJob();
    message.modelName = object.modelName?.map((e) => e) || [];
    message.positivePrompt = object.positivePrompt ?? "";
    message.negativePrompt = object.negativePrompt ?? "";
    message.aspectRatio = object.aspectRatio ?? "";
    message.numImages = object.numImages ?? 0;
    message.seed = object.seed ?? 0;
    return message;
  },
};

function createBaseUpscaleJob(): UpscaleJob {
  return { upscaleModel: "", imageUid: "", denoise: 0 };
}

export const UpscaleJob = {
  encode(message: UpscaleJob, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.upscaleModel !== "") {
      writer.uint32(10).string(message.upscaleModel);
    }
    if (message.imageUid !== "") {
      writer.uint32(18).string(message.imageUid);
    }
    if (message.denoise !== 0) {
      writer.uint32(24).int32(message.denoise);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): UpscaleJob {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseUpscaleJob();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 10) {
            break;
          }

          message.upscaleModel = reader.string();
          continue;
        case 2:
          if (tag !== 18) {
            break;
          }

          message.imageUid = reader.string();
          continue;
        case 3:
          if (tag !== 24) {
            break;
          }

          message.denoise = reader.int32();
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  create(base?: DeepPartial<UpscaleJob>): UpscaleJob {
    return UpscaleJob.fromPartial(base ?? {});
  },
  fromPartial(object: DeepPartial<UpscaleJob>): UpscaleJob {
    const message = createBaseUpscaleJob();
    message.upscaleModel = object.upscaleModel ?? "";
    message.imageUid = object.imageUid ?? "";
    message.denoise = object.denoise ?? 0;
    return message;
  },
};

function createBaseGligenBox(): GligenBox {
  return { modelName: "", boxes: [], positivePrompt: "", negativePrompt: "", aspectRatio: "", numImages: 0, seed: 0 };
}

export const GligenBox = {
  encode(message: GligenBox, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.modelName !== "") {
      writer.uint32(10).string(message.modelName);
    }
    for (const v of message.boxes) {
      CanvasBox.encode(v!, writer.uint32(18).fork()).ldelim();
    }
    if (message.positivePrompt !== "") {
      writer.uint32(26).string(message.positivePrompt);
    }
    if (message.negativePrompt !== "") {
      writer.uint32(34).string(message.negativePrompt);
    }
    if (message.aspectRatio !== "") {
      writer.uint32(42).string(message.aspectRatio);
    }
    if (message.numImages !== 0) {
      writer.uint32(48).int32(message.numImages);
    }
    if (message.seed !== 0) {
      writer.uint32(56).int32(message.seed);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): GligenBox {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseGligenBox();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 10) {
            break;
          }

          message.modelName = reader.string();
          continue;
        case 2:
          if (tag !== 18) {
            break;
          }

          message.boxes.push(CanvasBox.decode(reader, reader.uint32()));
          continue;
        case 3:
          if (tag !== 26) {
            break;
          }

          message.positivePrompt = reader.string();
          continue;
        case 4:
          if (tag !== 34) {
            break;
          }

          message.negativePrompt = reader.string();
          continue;
        case 5:
          if (tag !== 42) {
            break;
          }

          message.aspectRatio = reader.string();
          continue;
        case 6:
          if (tag !== 48) {
            break;
          }

          message.numImages = reader.int32();
          continue;
        case 7:
          if (tag !== 56) {
            break;
          }

          message.seed = reader.int32();
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  create(base?: DeepPartial<GligenBox>): GligenBox {
    return GligenBox.fromPartial(base ?? {});
  },
  fromPartial(object: DeepPartial<GligenBox>): GligenBox {
    const message = createBaseGligenBox();
    message.modelName = object.modelName ?? "";
    message.boxes = object.boxes?.map((e) => CanvasBox.fromPartial(e)) || [];
    message.positivePrompt = object.positivePrompt ?? "";
    message.negativePrompt = object.negativePrompt ?? "";
    message.aspectRatio = object.aspectRatio ?? "";
    message.numImages = object.numImages ?? 0;
    message.seed = object.seed ?? 0;
    return message;
  },
};

function createBaseBasicSVD(): BasicSVD {
  return {
    imageUid: "",
    imageUrl: "",
    motion: 0,
    seed: 0,
    modelName: "",
    positivePrompt: "",
    negativePrompt: "",
    aspectRatio: "",
  };
}

export const BasicSVD = {
  encode(message: BasicSVD, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.imageUid !== "") {
      writer.uint32(10).string(message.imageUid);
    }
    if (message.imageUrl !== "") {
      writer.uint32(18).string(message.imageUrl);
    }
    if (message.motion !== 0) {
      writer.uint32(24).uint32(message.motion);
    }
    if (message.seed !== 0) {
      writer.uint32(32).int32(message.seed);
    }
    if (message.modelName !== "") {
      writer.uint32(42).string(message.modelName);
    }
    if (message.positivePrompt !== "") {
      writer.uint32(50).string(message.positivePrompt);
    }
    if (message.negativePrompt !== "") {
      writer.uint32(58).string(message.negativePrompt);
    }
    if (message.aspectRatio !== "") {
      writer.uint32(66).string(message.aspectRatio);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): BasicSVD {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseBasicSVD();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 10) {
            break;
          }

          message.imageUid = reader.string();
          continue;
        case 2:
          if (tag !== 18) {
            break;
          }

          message.imageUrl = reader.string();
          continue;
        case 3:
          if (tag !== 24) {
            break;
          }

          message.motion = reader.uint32();
          continue;
        case 4:
          if (tag !== 32) {
            break;
          }

          message.seed = reader.int32();
          continue;
        case 5:
          if (tag !== 42) {
            break;
          }

          message.modelName = reader.string();
          continue;
        case 6:
          if (tag !== 50) {
            break;
          }

          message.positivePrompt = reader.string();
          continue;
        case 7:
          if (tag !== 58) {
            break;
          }

          message.negativePrompt = reader.string();
          continue;
        case 8:
          if (tag !== 66) {
            break;
          }

          message.aspectRatio = reader.string();
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  create(base?: DeepPartial<BasicSVD>): BasicSVD {
    return BasicSVD.fromPartial(base ?? {});
  },
  fromPartial(object: DeepPartial<BasicSVD>): BasicSVD {
    const message = createBaseBasicSVD();
    message.imageUid = object.imageUid ?? "";
    message.imageUrl = object.imageUrl ?? "";
    message.motion = object.motion ?? 0;
    message.seed = object.seed ?? 0;
    message.modelName = object.modelName ?? "";
    message.positivePrompt = object.positivePrompt ?? "";
    message.negativePrompt = object.negativePrompt ?? "";
    message.aspectRatio = object.aspectRatio ?? "";
    return message;
  },
};

function createBaseJobType(): JobType {
  return { basicJob: undefined, upscaleJob: undefined, gligenBox: undefined, basicSVD: undefined };
}

export const JobType = {
  encode(message: JobType, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.basicJob !== undefined) {
      BasicJob.encode(message.basicJob, writer.uint32(10).fork()).ldelim();
    }
    if (message.upscaleJob !== undefined) {
      UpscaleJob.encode(message.upscaleJob, writer.uint32(18).fork()).ldelim();
    }
    if (message.gligenBox !== undefined) {
      GligenBox.encode(message.gligenBox, writer.uint32(26).fork()).ldelim();
    }
    if (message.basicSVD !== undefined) {
      BasicSVD.encode(message.basicSVD, writer.uint32(34).fork()).ldelim();
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): JobType {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseJobType();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 10) {
            break;
          }

          message.basicJob = BasicJob.decode(reader, reader.uint32());
          continue;
        case 2:
          if (tag !== 18) {
            break;
          }

          message.upscaleJob = UpscaleJob.decode(reader, reader.uint32());
          continue;
        case 3:
          if (tag !== 26) {
            break;
          }

          message.gligenBox = GligenBox.decode(reader, reader.uint32());
          continue;
        case 4:
          if (tag !== 34) {
            break;
          }

          message.basicSVD = BasicSVD.decode(reader, reader.uint32());
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  create(base?: DeepPartial<JobType>): JobType {
    return JobType.fromPartial(base ?? {});
  },
  fromPartial(object: DeepPartial<JobType>): JobType {
    const message = createBaseJobType();
    message.basicJob = (object.basicJob !== undefined && object.basicJob !== null)
      ? BasicJob.fromPartial(object.basicJob)
      : undefined;
    message.upscaleJob = (object.upscaleJob !== undefined && object.upscaleJob !== null)
      ? UpscaleJob.fromPartial(object.upscaleJob)
      : undefined;
    message.gligenBox = (object.gligenBox !== undefined && object.gligenBox !== null)
      ? GligenBox.fromPartial(object.gligenBox)
      : undefined;
    message.basicSVD = (object.basicSVD !== undefined && object.basicSVD !== null)
      ? BasicSVD.fromPartial(object.basicSVD)
      : undefined;
    return message;
  },
};

/** Used by server */
export type JobServiceDefinition = typeof JobServiceDefinition;
export const JobServiceDefinition = {
  name: "JobService",
  fullName: "server_job.v1.JobService",
  methods: {
    runJob: {
      name: "RunJob",
      requestType: JobType,
      requestStream: false,
      responseType: JobResponse,
      responseStream: true,
      options: {},
    },
  },
} as const;

export interface JobServiceImplementation<CallContextExt = {}> {
  runJob(
    request: JobType,
    context: CallContext & CallContextExt,
  ): ServerStreamingMethodResult<DeepPartial<JobResponse>>;
}

export interface JobServiceClient<CallOptionsExt = {}> {
  runJob(request: DeepPartial<JobType>, options?: CallOptions & CallOptionsExt): AsyncIterable<JobResponse>;
}

type Builtin = Date | Function | Uint8Array | string | number | boolean | undefined;

export type DeepPartial<T> = T extends Builtin ? T
  : T extends globalThis.Array<infer U> ? globalThis.Array<DeepPartial<U>>
  : T extends ReadonlyArray<infer U> ? ReadonlyArray<DeepPartial<U>>
  : T extends {} ? { [K in keyof T]?: DeepPartial<T[K]> }
  : Partial<T>;

export type ServerStreamingMethodResult<Response> = { [Symbol.asyncIterator](): AsyncIterator<Response, void> };
