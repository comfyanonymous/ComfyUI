declare module "stream/consumers" {
    import { Blob as NodeBlob } from "node:buffer";
    import { Readable } from "node:stream";
    function buffer(stream: NodeJS.ReadableStream | Readable | AsyncIterable<any>): Promise<Buffer>;
    function text(stream: NodeJS.ReadableStream | Readable | AsyncIterable<any>): Promise<string>;
    function arrayBuffer(stream: NodeJS.ReadableStream | Readable | AsyncIterable<any>): Promise<ArrayBuffer>;
    function blob(stream: NodeJS.ReadableStream | Readable | AsyncIterable<any>): Promise<NodeBlob>;
    function json(stream: NodeJS.ReadableStream | Readable | AsyncIterable<any>): Promise<unknown>;
}
declare module "node:stream/consumers" {
    export * from "stream/consumers";
}
