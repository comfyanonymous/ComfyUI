// plugin-IDs should be globally unique
// Token-names are only for debug purposes and can be anything

import { IComfyPlugin, Token, Application } from '../types/interfaces';

interface IPrintText {
    printText: (text: string) => void;
}

// Unique identifier token. Other plugins should import this and list it as a dependency if they want to use it.
export const printTextToken = new Token<IPrintText>('my-extension:IPrintText');

const myPlugin: IComfyPlugin<IPrintText> = {
    id: 'my-extension:text-printer',
    autoStart: true,
    provides: printTextToken,
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

export default myPlugin;
