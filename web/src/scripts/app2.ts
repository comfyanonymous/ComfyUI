// // This class acts as a gateway for plugins
//
// import {IComfyApp} from '../types/interfaces';
// import {ComfyGraph} from "../litegraph/comfyGraph.ts";
//
// // Single class
// export class ComfyApp implements IComfyApp {
//     private static instance: IComfyApp;
//
//     private saveInterval: NodeJS.Timeout | null = null;
//
//     public static getInstance() {
//         if (!ComfyApp.instance) {
//             ComfyApp.instance = new ComfyApp();
//         }
//         return ComfyApp.instance;
//     }
//
//
//     clean() {
//         this.disableWorkflowAutoSave();
//     }
//
//     getWidgetType(inputData: any, inputName: string): string | null {
//         return '';
//     }
//
//     // LiteGraph
//     // the api
//     // ui components?
// }
