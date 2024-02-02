// This is a list of all default tokens defined within the Comfy Creator.
// If you're building a plugin and want to over-ride a default implementation, import one of these tokens
// and list it as 'provides' in your comfy-plugin definition.
// Plugins can also define their own new tokens.
// Tokens provie unique identifiers; they bind to an interface, rather than implementation.

import { Token } from '../types/interfaces';

import { ISerializeGraph } from '../types/interfaces';
export const SerializeGraphToken = new Token<ISerializeGraph>('serialize-graph');
