// This module binds the plugins (dependencies) to their interfaces.

import 'reflect-metadata';
import { Container } from 'inversify';
import { ISerializeGraph } from '../types/interfaces';
import { SerializeGraph } from '../litegraph/comfyGraph';

const container = new Container();
container.bind<ISerializeGraph>('ISerializeGraph').to(SerializeGraph);

// Use the lines below instead if you want to use the alternative implementation
// import { AlternativeSerializeGraph } from '../litegraph/alternativeSerializeGraph';
// container.bind<ISerializeGraph>("ISerializeGraph").to(AlternativeSerializeGraph);

export { container };
