/* eslint-disable */
import type { CallContext, CallOptions } from "nice-grpc-common";
import * as _m0 from "protobufjs/minimal";

export const protobufPackage = "grpc.health.v1";

export interface HealthCheckRequest {
  service: string;
}

export interface HealthCheckResponse {
  status: HealthCheckResponse_ServingStatus;
}

export enum HealthCheckResponse_ServingStatus {
  UNKNOWN = "UNKNOWN",
  SERVING = "SERVING",
  NOT_SERVING = "NOT_SERVING",
  /** SERVICE_UNKNOWN - Used only by the Watch method. */
  SERVICE_UNKNOWN = "SERVICE_UNKNOWN",
  UNRECOGNIZED = "UNRECOGNIZED",
}

export function healthCheckResponse_ServingStatusFromJSON(object: any): HealthCheckResponse_ServingStatus {
  switch (object) {
    case 0:
    case "UNKNOWN":
      return HealthCheckResponse_ServingStatus.UNKNOWN;
    case 1:
    case "SERVING":
      return HealthCheckResponse_ServingStatus.SERVING;
    case 2:
    case "NOT_SERVING":
      return HealthCheckResponse_ServingStatus.NOT_SERVING;
    case 3:
    case "SERVICE_UNKNOWN":
      return HealthCheckResponse_ServingStatus.SERVICE_UNKNOWN;
    case -1:
    case "UNRECOGNIZED":
    default:
      return HealthCheckResponse_ServingStatus.UNRECOGNIZED;
  }
}

export function healthCheckResponse_ServingStatusToNumber(object: HealthCheckResponse_ServingStatus): number {
  switch (object) {
    case HealthCheckResponse_ServingStatus.UNKNOWN:
      return 0;
    case HealthCheckResponse_ServingStatus.SERVING:
      return 1;
    case HealthCheckResponse_ServingStatus.NOT_SERVING:
      return 2;
    case HealthCheckResponse_ServingStatus.SERVICE_UNKNOWN:
      return 3;
    case HealthCheckResponse_ServingStatus.UNRECOGNIZED:
    default:
      return -1;
  }
}

function createBaseHealthCheckRequest(): HealthCheckRequest {
  return { service: "" };
}

export const HealthCheckRequest = {
  encode(message: HealthCheckRequest, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.service !== "") {
      writer.uint32(10).string(message.service);
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): HealthCheckRequest {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseHealthCheckRequest();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 10) {
            break;
          }

          message.service = reader.string();
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  create(base?: DeepPartial<HealthCheckRequest>): HealthCheckRequest {
    return HealthCheckRequest.fromPartial(base ?? {});
  },
  fromPartial(object: DeepPartial<HealthCheckRequest>): HealthCheckRequest {
    const message = createBaseHealthCheckRequest();
    message.service = object.service ?? "";
    return message;
  },
};

function createBaseHealthCheckResponse(): HealthCheckResponse {
  return { status: HealthCheckResponse_ServingStatus.UNKNOWN };
}

export const HealthCheckResponse = {
  encode(message: HealthCheckResponse, writer: _m0.Writer = _m0.Writer.create()): _m0.Writer {
    if (message.status !== HealthCheckResponse_ServingStatus.UNKNOWN) {
      writer.uint32(8).int32(healthCheckResponse_ServingStatusToNumber(message.status));
    }
    return writer;
  },

  decode(input: _m0.Reader | Uint8Array, length?: number): HealthCheckResponse {
    const reader = input instanceof _m0.Reader ? input : _m0.Reader.create(input);
    let end = length === undefined ? reader.len : reader.pos + length;
    const message = createBaseHealthCheckResponse();
    while (reader.pos < end) {
      const tag = reader.uint32();
      switch (tag >>> 3) {
        case 1:
          if (tag !== 8) {
            break;
          }

          message.status = healthCheckResponse_ServingStatusFromJSON(reader.int32());
          continue;
      }
      if ((tag & 7) === 4 || tag === 0) {
        break;
      }
      reader.skipType(tag & 7);
    }
    return message;
  },

  create(base?: DeepPartial<HealthCheckResponse>): HealthCheckResponse {
    return HealthCheckResponse.fromPartial(base ?? {});
  },
  fromPartial(object: DeepPartial<HealthCheckResponse>): HealthCheckResponse {
    const message = createBaseHealthCheckResponse();
    message.status = object.status ?? HealthCheckResponse_ServingStatus.UNKNOWN;
    return message;
  },
};

export type HealthDefinition = typeof HealthDefinition;
export const HealthDefinition = {
  name: "Health",
  fullName: "grpc.health.v1.Health",
  methods: {
    /**
     * If the requested service is unknown, the call will fail with status
     * NOT_FOUND.
     */
    check: {
      name: "Check",
      requestType: HealthCheckRequest,
      requestStream: false,
      responseType: HealthCheckResponse,
      responseStream: false,
      options: {},
    },
    /**
     * Performs a watch for the serving status of the requested service.
     * The server will immediately send back a message indicating the current
     * serving status.  It will then subsequently send a new message whenever
     * the service's serving status changes.
     *
     * If the requested service is unknown when the call is received, the
     * server will send a message setting the serving status to
     * SERVICE_UNKNOWN but will *not* terminate the call.  If at some
     * future point, the serving status of the service becomes known, the
     * server will send a new message with the service's serving status.
     *
     * If the call terminates with status UNIMPLEMENTED, then clients
     * should assume this method is not supported and should not retry the
     * call.  If the call terminates with any other status (including OK),
     * clients should retry the call with appropriate exponential backoff.
     */
    watch: {
      name: "Watch",
      requestType: HealthCheckRequest,
      requestStream: false,
      responseType: HealthCheckResponse,
      responseStream: true,
      options: {},
    },
  },
} as const;

export interface HealthServiceImplementation<CallContextExt = {}> {
  /**
   * If the requested service is unknown, the call will fail with status
   * NOT_FOUND.
   */
  check(request: HealthCheckRequest, context: CallContext & CallContextExt): Promise<DeepPartial<HealthCheckResponse>>;
  /**
   * Performs a watch for the serving status of the requested service.
   * The server will immediately send back a message indicating the current
   * serving status.  It will then subsequently send a new message whenever
   * the service's serving status changes.
   *
   * If the requested service is unknown when the call is received, the
   * server will send a message setting the serving status to
   * SERVICE_UNKNOWN but will *not* terminate the call.  If at some
   * future point, the serving status of the service becomes known, the
   * server will send a new message with the service's serving status.
   *
   * If the call terminates with status UNIMPLEMENTED, then clients
   * should assume this method is not supported and should not retry the
   * call.  If the call terminates with any other status (including OK),
   * clients should retry the call with appropriate exponential backoff.
   */
  watch(
    request: HealthCheckRequest,
    context: CallContext & CallContextExt,
  ): ServerStreamingMethodResult<DeepPartial<HealthCheckResponse>>;
}

export interface HealthClient<CallOptionsExt = {}> {
  /**
   * If the requested service is unknown, the call will fail with status
   * NOT_FOUND.
   */
  check(request: DeepPartial<HealthCheckRequest>, options?: CallOptions & CallOptionsExt): Promise<HealthCheckResponse>;
  /**
   * Performs a watch for the serving status of the requested service.
   * The server will immediately send back a message indicating the current
   * serving status.  It will then subsequently send a new message whenever
   * the service's serving status changes.
   *
   * If the requested service is unknown when the call is received, the
   * server will send a message setting the serving status to
   * SERVICE_UNKNOWN but will *not* terminate the call.  If at some
   * future point, the serving status of the service becomes known, the
   * server will send a new message with the service's serving status.
   *
   * If the call terminates with status UNIMPLEMENTED, then clients
   * should assume this method is not supported and should not retry the
   * call.  If the call terminates with any other status (including OK),
   * clients should retry the call with appropriate exponential backoff.
   */
  watch(
    request: DeepPartial<HealthCheckRequest>,
    options?: CallOptions & CallOptionsExt,
  ): AsyncIterable<HealthCheckResponse>;
}

type Builtin = Date | Function | Uint8Array | string | number | boolean | undefined;

export type DeepPartial<T> = T extends Builtin ? T
  : T extends globalThis.Array<infer U> ? globalThis.Array<DeepPartial<U>>
  : T extends ReadonlyArray<infer U> ? ReadonlyArray<DeepPartial<U>>
  : T extends {} ? { [K in keyof T]?: DeepPartial<T[K]> }
  : Partial<T>;

export type ServerStreamingMethodResult<Response> = { [Symbol.asyncIterator](): AsyncIterator<Response, void> };
