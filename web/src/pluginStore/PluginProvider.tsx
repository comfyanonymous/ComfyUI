import React from 'react';
import { PluginStore } from './PluginStore';
import PluginStoreContext from './PluginStoreContext';

export const PluginProvider: React.FC<{
    pluginStore: PluginStore;
    children: React.ReactNode;
}> = ({ pluginStore, children }) => {
    return <PluginStoreContext.Provider value={pluginStore}>{children}</PluginStoreContext.Provider>;
};
