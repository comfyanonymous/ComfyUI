import 'reflect-metadata';
import { Container } from 'inversify';
import { ComfyGraph } from './litegraph/comfyGraph.ts';

const container = new Container();
container.bind<ComfyGraph>(ComfyGraph).to(ComfyGraph);

export { container };
