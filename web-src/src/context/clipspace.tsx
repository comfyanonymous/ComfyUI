import { createContext, ReactNode, useContext, useState } from 'react';
import type { ComfyGraph } from '../litegraph/comfyGraph.ts';
import { SerializedNodeObject } from '../types/interfaces.ts';
import type { ComfyNode } from '../litegraph/comfyNode.ts';
import { ComfyFile } from '../types/many.ts';
import type { ComfyWidget } from '../types/comfyWidget.ts';

interface ClipspaceContextType {
    graph: ComfyGraph | null;
    clipspaceReturnNode: ComfyNode | null;
    clipspace: SerializedNodeObject | null;
    setClipspaceInvalidateHandler: (handler: ClipspaceInvalidateHandler | null) => void;
}

type ClipspaceInvalidateHandler = () => void;

const ClipspaceContext = createContext<ClipspaceContextType | null>(null);

export function useClipspace() {
    const context = useContext(ClipspaceContext);
    if (!context) {
        throw new Error('useClipspace must be used within a ClipspaceProvider');
    }

    return context;
}

export function ClipspaceProvider({ children }: { children: ReactNode }) {
    const [graph, setGraph] = useState<ComfyGraph | null>(null);
    const [clipspace, setClipspace] = useState<SerializedNodeObject | null>(null);
    const [clipspaceReturnNode, setClipspaceReturnNode] = useState<ComfyNode | null>(null);
    const [clipspaceInvalidateHandler, setClipspaceInvalidateHandler] = useState<ClipspaceInvalidateHandler | null>(
        null
    );

    const onClipspaceEditorSave = () => {
        if (clipspaceReturnNode) {
            pasteFromClipspace(clipspaceReturnNode);
        }
    };

    const onClipspaceEditorClosed = () => {
        setClipspaceReturnNode(null);
    };

    const copyToClipspace = (node: ComfyNode) => {
        let widgets = null;
        if (node.widgets) {
            widgets = node.widgets.map(({ type, name, value }) => ({
                type,
                name,
                value,
            })) as ComfyWidget[];
        }

        let imgs = undefined;
        let orig_imgs = undefined;
        if (node.imgs != undefined) {
            imgs = [];
            orig_imgs = [];

            for (let i = 0; i < node.imgs.length; i++) {
                imgs[i] = new Image();
                imgs[i].src = (node.imgs[i] as HTMLImageElement).src;
                orig_imgs[i] = imgs[i];
            }
        }

        let selectedIndex = 0;
        if (node.imageIndex) {
            selectedIndex = node.imageIndex;
        }

        const newClipspace = {
            widgets: widgets,
            imgs: imgs,
            original_imgs: orig_imgs,
            images: node.images,
            selectedIndex: selectedIndex,
            img_paste_mode: 'selected', // reset to default imf_paste_mode state on copy action
        };

        setClipspace(prev => ({
            ...prev,
            ...newClipspace,
        }));
        setClipspaceReturnNode(null);

        if (clipspaceInvalidateHandler) {
            clipspaceInvalidateHandler();
        }
    };

    const pasteFromClipspace = (node: ComfyNode, outputs: Record<string, any> = {}) => {
        if (clipspace) {
            // image paste
            if (clipspace.imgs && node.imgs) {
                if (node.images && clipspace.images) {
                    if (clipspace['img_paste_mode'] == 'selected') {
                        node.images = [clipspace.images[clipspace['selectedIndex']] as HTMLImageElement];
                    } else {
                        node.images = clipspace.images;
                    }

                    if (outputs[node.id + '']) outputs[node.id + ''].images = node.images;
                }

                if (clipspace.imgs) {
                    // deep-copy to cut link with clipspace
                    if (clipspace['img_paste_mode'] == 'selected') {
                        const img = new Image();
                        img.src = (clipspace.imgs[clipspace['selectedIndex']] as HTMLImageElement).src;
                        node.imgs = [img];
                        node.imageIndex = 0;
                    } else {
                        const imgs = [];
                        for (let i = 0; i < clipspace.imgs.length; i++) {
                            imgs[i] = new Image();
                            imgs[i].src = (clipspace.imgs[i] as HTMLImageElement).src;
                            node.imgs = imgs;
                        }
                    }
                }
            }

            if (node.widgets) {
                if (clipspace.images) {
                    const clip_image = clipspace.images[clipspace['selectedIndex']] as ComfyFile;
                    const index = node.widgets.findIndex(obj => obj.name === 'image');
                    if (index >= 0) {
                        if (
                            node.widgets[index].type != 'image' &&
                            typeof node.widgets[index].value == 'string' &&
                            clip_image.filename
                        ) {
                            node.widgets[index].value =
                                (clip_image.subfolder ? clip_image.subfolder + '/' : '') +
                                clip_image.filename +
                                (clip_image.type ? ` [${clip_image.type}]` : '');
                        } else {
                            node.widgets[index].value = clip_image;
                        }
                    }
                }
                if (clipspace.widgets) {
                    clipspace.widgets.forEach(({ type, name, value }) => {
                        const prop = Object.values(node.widgets).find(obj => obj.type === type && obj.name === name);
                        if (prop && prop.type != 'button') {
                            value = value as ComfyFile;
                            if (prop.type != 'image' && typeof prop.value == 'string' && value.filename) {
                                prop.value =
                                    (value.subfolder ? value.subfolder + '/' : '') +
                                    value.filename +
                                    (value.type ? ` [${value.type}]` : '');
                            } else {
                                prop.value = value;
                                prop.callback?.(value);
                            }
                        }
                    });
                }
            }

            graph?.setDirtyCanvas(true, true);
        }
    };

    return (
        <ClipspaceContext.Provider
            value={{
                graph,
                clipspace,
                clipspaceReturnNode,
                setClipspaceInvalidateHandler,
            }}
        >
            {children}
        </ClipspaceContext.Provider>
    );
}
