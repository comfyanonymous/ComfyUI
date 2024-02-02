import React from 'react';
import { PluginStore } from './PluginStore';

const PluginStoreContext = React.createContext(new PluginStore());

export default PluginStoreContext;
