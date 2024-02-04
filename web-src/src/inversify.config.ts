import 'reflect-metadata';
import { Container } from 'inversify';
import { ComfyGraph, SerializeGraph } from './litegraph/comfyGraph.ts';

const container = new Container();
container.bind<ComfyGraph>('ComfyGraph').to(ComfyGraph);
container.bind<SerializeGraph>('SerializeGraph').to(SerializeGraph);

export { container };
