// plugin-IDs should be globally unique
// Token-names are only for debug purposes and can be anything

import { Application, IComfyPlugin } from '../../types/interfaces';

interface IPrintText {
    printText: (text: string) => void;
}

// This token unique identifies an interface.
// Other plugins should import this token and add it to their dependencies if they want to use it.
// Other plugins can duplicate this token and create an alternative implementation of the interface.
// export const printTextToken = new Token<IPrintText>('my-extension:IPrintText');

const printerPlugin: IComfyPlugin<IPrintText> = {
    id: 'my-extension:text-printer',
    autoStart: true,
    provides: 'printText',
    activate: (app: Application): IPrintText => {
        console.log('Test extension activated!');

        return {
            printText: (text: string) => {
                console.log('Test extension printing text: ', text);
            },
        };
    },
    deactivate: (app: Application) => {
        console.log('Test extension deactivated!');
    },
};

const popupPlugin: IComfyPlugin<IPrintText> = {
    id: 'my-extension:text-popup',
    autoStart: true,
    provides: 'printText',
    activate: (app: Application): IPrintText => {
        alert('Test extension activated!');

        return {
            printText: (text: string) => {
                alert('Test extension printing text: ' + text);
            },
        };
    },
    deactivate: (app: Application) => {
        alert('Test extension deactivated!');
    },
};

export default [printerPlugin, popupPlugin];
