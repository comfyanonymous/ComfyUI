import { useContext } from 'react';
import PluginStoreContext from '../PluginStoreContext';

export function usePluginStore() {
    return useContext(PluginStoreContext);
}
