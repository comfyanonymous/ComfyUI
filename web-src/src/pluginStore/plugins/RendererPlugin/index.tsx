import { IComfyPlugin } from '../../../types/interfaces';
import { PluginStore } from '../../PluginStore';
import { Renderer } from './components/Renderer';
import ComponentUpdatedEvent from './events/ComponentUpdatedEvent';
import randomString from './randomString';

export class RendererPlugin implements IComfyPlugin<void> {
    private pluginStore: PluginStore;

    private componentMap = new Map<
        string,
        Array<{
            component: React.ComponentClass;
            key?: string;
        }>
    >();

    id = 'core:Renderer';
    autoStart = true;
    dependencies = [];

    constructor(pluginStore: PluginStore) {
        this.pluginStore = pluginStore;
    }

    addToComponentMap(position: string, component: React.ComponentClass, key?: string) {
        let array = this.componentMap.get(position);
        const componentKey = key ? key : randomString(8);
        if (!array) {
            array = [{ component, key: componentKey }];
        } else {
            array.push({ component, key: componentKey });
        }
        this.componentMap.set(position, array);
        this.pluginStore.dispatchEvent(new ComponentUpdatedEvent('Renderer.componentUpdated', position));
    }

    removeFromComponentMap(position: string, component: React.ComponentClass) {
        const array = this.componentMap.get(position);
        if (array) {
            array.splice(
                array.findIndex(item => item.component === component),
                1
            );
        }
        this.pluginStore.dispatchEvent(new ComponentUpdatedEvent('Renderer.componentUpdated', position));
    }

    getRendererComponent() {
        return Renderer;
    }

    getComponentsInPosition(position: string) {
        const componentArray = this.componentMap.get(position);
        if (!componentArray) return [];

        return componentArray;
    }

    activate() {
        this.pluginStore.addFunction('Renderer.add', this.addToComponentMap.bind(this));

        this.pluginStore.addFunction('Renderer.getComponentsInPosition', this.getComponentsInPosition.bind(this));

        this.pluginStore.addFunction('Renderer.getRendererComponent', this.getRendererComponent.bind(this));

        this.pluginStore.addFunction('Renderer.remove', this.removeFromComponentMap.bind(this));
    }

    deactivate() {
        this.pluginStore.removeFunction('Renderer.add');

        this.pluginStore.removeFunction('Renderer.getComponentsInPosition');

        this.pluginStore.removeFunction('Renderer.getRendererComponent');

        this.pluginStore.removeFunction('Renderer.remove');
    }
}

export type PluginStoreRenderer = {
    executeFunction(functionName: 'Renderer.getComponentsInPosition', position: string): Array<React.Component>;
};
