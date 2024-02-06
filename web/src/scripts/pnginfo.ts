import { api } from './api.js';
import { LiteGraph } from 'litegraph.js';
import { LatentInfo, PngInfo } from '../types/many';
import { ComfyGraph } from '../litegraph/comfyGraph.js';
import { ComfyNode } from '../litegraph/comfyNode.js';

export function getPngMetadata(file: File) {
    return new Promise<PngInfo | Record<string, string> | void>(r => {
        const reader = new FileReader();
        reader.onload = event => {
            // Get the PNG data as a Uint8Array
            if (!event.target) return;
            const pngData = new Uint8Array(
                typeof event.target.result === 'string' ? Buffer.from(event.target.result) : event.target.result!
            );
            const dataView = new DataView(pngData.buffer);

            // Check that the PNG signature is present
            if (dataView.getUint32(0) !== 0x89504e47) {
                console.error('Not a valid PNG file');
                r();
                return;
            }

            // Start searching for chunks after the PNG signature
            let offset = 8;
            let txt_chunks: Record<string, string> = {};
            // Loop through the chunks in the PNG file
            while (offset < pngData.length) {
                // Get the length of the chunk
                const length = dataView.getUint32(offset);
                // Get the chunk type
                const type = String.fromCharCode(...pngData.slice(offset + 4, offset + 8));
                if (type === 'tEXt' || type == 'comf') {
                    // Get the keyword
                    let keyword_end = offset + 8;
                    while (pngData[keyword_end] !== 0) {
                        keyword_end++;
                    }
                    const keyword = String.fromCharCode(...pngData.slice(offset + 8, keyword_end));
                    // Get the text
                    const contentArraySegment = pngData.slice(keyword_end + 1, offset + 8 + length);
                    const contentJson = Array.from(contentArraySegment)
                        .map(s => String.fromCharCode(s))
                        .join('');
                    txt_chunks[keyword] = contentJson;
                }

                offset += 12 + length;
            }

            r(txt_chunks);
        };

        reader.readAsArrayBuffer(file);
    });
}

function parseExifData(exifData: Uint8Array) {
    // Check for the correct TIFF header (0x4949 for little-endian or 0x4D4D for big-endian)
    const isLittleEndian = new Uint16Array(exifData.slice(0, 2))[0] === 0x4949;

    // Function to read 16-bit and 32-bit integers from binary data
    function readInt(offset: number, isLittleEndian: boolean, length: number) {
        let arr = exifData.slice(offset, offset + length);
        if (length === 2) {
            return new DataView(arr.buffer, arr.byteOffset, arr.byteLength).getUint16(0, isLittleEndian);
        } else if (length === 4) {
            return new DataView(arr.buffer, arr.byteOffset, arr.byteLength).getUint32(0, isLittleEndian);
        }
    }

    // Read the offset to the first IFD (Image File Directory)
    const ifdOffset = readInt(4, isLittleEndian, 4);

    function parseIFD(offset: number) {
        const numEntries = readInt(offset, isLittleEndian, 2);
        const result: Record<number, string> = {};
        if (!numEntries) return result;

        for (let i = 0; i < numEntries; i++) {
            const entryOffset = offset + 2 + i * 12;
            const tag = readInt(entryOffset, isLittleEndian, 2);
            const type = readInt(entryOffset + 2, isLittleEndian, 2);
            const numValues = readInt(entryOffset + 4, isLittleEndian, 4);
            const valueOffset = readInt(entryOffset + 8, isLittleEndian, 4);

            // Read the value(s) based on the data type
            let value;
            if (type === 2 && valueOffset && numValues) {
                // ASCII string
                value = String.fromCharCode(...exifData.slice(valueOffset, valueOffset + numValues - 1));
            }

            if (tag && value) {
                result[tag] = value;
            }
        }

        return result;
    }

    // Parse the first IFD
    const ifdData = parseIFD(ifdOffset!);
    return ifdData;
}

function splitValues(input: Record<string, string>) {
    var output: Record<string, string> = {};

    for (var key in input) {
        var value = input[key];
        var splitValues = value.split(':', 2);
        output[splitValues[0]] = splitValues[1];
    }
    return output;
}

export function getWebpMetadata(file: File) {
    return new Promise<PngInfo | Record<string, string> | void>(r => {
        const reader = new FileReader();
        reader.onload = event => {
            const encoded =
                typeof event.target?.result === 'string'
                    ? new TextEncoder().encode(event.target.result)
                    : event.target?.result!;

            const webp = new Uint8Array(encoded);
            const dataView = new DataView(webp.buffer);

            // Check that the WEBP signature is present
            if (dataView.getUint32(0) !== 0x52494646 || dataView.getUint32(8) !== 0x57454250) {
                console.error('Not a valid WEBP file');
                r();
                return;
            }

            // Start searching for chunks after the WEBP signature
            let offset = 12;
            let txt_chunks: Record<string, string> = {};
            // Loop through the chunks in the WEBP file
            while (offset < webp.length) {
                const chunk_length = dataView.getUint32(offset + 4, true);
                const chunk_type = String.fromCharCode(...webp.slice(offset, offset + 4));
                if (chunk_type === 'EXIF') {
                    if (String.fromCharCode(...webp.slice(offset + 8, offset + 8 + 6)) == 'Exif\0\0') {
                        offset += 6;
                    }
                    let data = parseExifData(webp.slice(offset + 8, offset + 8 + chunk_length));
                    for (var key in data) {
                        var value = data[key];
                        let index = value.indexOf(':');
                        txt_chunks[value.slice(0, index)] = value.slice(index + 1);
                    }
                }

                offset += 8 + chunk_length;
            }

            r(txt_chunks);
        };

        reader.readAsArrayBuffer(file);
    });
}

export function getLatentMetadata(file: File) {
    return new Promise<LatentInfo>(r => {
        const reader = new FileReader();
        reader.onload = event => {
            const arrBuffer =
                typeof event.target?.result === 'string' ? Buffer.from(event.target.result) : event.target?.result!;
            const safetensorsData = new Uint8Array(arrBuffer);
            const dataView = new DataView(safetensorsData.buffer);
            let header_size = dataView.getUint32(0, true);
            let offset = 8;
            let header = JSON.parse(new TextDecoder().decode(safetensorsData.slice(offset, offset + header_size)));
            r(header.__metadata__);
        };

        var slice = file.slice(0, 1024 * 1024 * 4);
        reader.readAsArrayBuffer(slice);
    });
}

export async function importA1111(graph: ComfyGraph, parameters: string) {
    const p = parameters.lastIndexOf('\nSteps:');
    if (p > -1) {
        const embeddings = await api.getEmbeddings();
        const opts = parameters
            .substr(p)
            .split('\n')[1]
            .split(',')
            .reduce((p: Record<string, string>, n) => {
                const s = n.split(':');
                p[s[0].trim().toLowerCase()] = s[1].trim();
                return p;
            }, {});
        const p2 = parameters.lastIndexOf('\nNegative prompt:', p);
        if (p2 > -1) {
            let positive = parameters.substr(0, p2).trim();
            let negative = parameters.substring(p2 + 18, p).trim();

            const ckptNode = LiteGraph.createNode<ComfyNode>('CheckpointLoaderSimple');
            const clipSkipNode = LiteGraph.createNode<ComfyNode>('CLIPSetLastLayer');
            const positiveNode = LiteGraph.createNode<ComfyNode>('CLIPTextEncode');
            const negativeNode = LiteGraph.createNode<ComfyNode>('CLIPTextEncode');
            const samplerNode = LiteGraph.createNode<ComfyNode>('KSampler');
            const imageNode = LiteGraph.createNode<ComfyNode>('EmptyLatentImage');
            const vaeNode = LiteGraph.createNode<ComfyNode>('VAEDecode');
            const vaeLoaderNode = LiteGraph.createNode<ComfyNode>('VAELoader');
            const saveNode = LiteGraph.createNode<ComfyNode>('SaveImage');
            let hrSamplerNode: ComfyNode | null = null;

            const ceil64 = (v: number) => Math.ceil(v / 64) * 64;

            function getWidget(node: ComfyNode, name: string) {
                return node.widgets.find(w => w.name === name);
            }

            function setWidgetValue(node: ComfyNode, name: string, value: any, isOptionPrefix?: boolean) {
                const w = getWidget(node, name);
                if (!w) return;

                if (isOptionPrefix) {
                    const o = w.options.values.find((w: string) => w.startsWith(value));
                    if (o) {
                        w.value = o;
                    } else {
                        console.warn(`Unknown value '${value}' for widget '${name}'`, node);
                        w.value = value;
                    }
                } else {
                    w.value = value;
                }
            }

            function createLoraNodes(clipNode: ComfyNode, text: string, prevClip: any, prevModel: any) {
                const loras: { name: string; weight: number }[] = [];
                text = text.replace(/<lora:([^:]+:[^>]+)>/g, function (m, c) {
                    const s = c.split(':');
                    const weight = parseFloat(s[1]);
                    if (isNaN(weight)) {
                        console.warn('Invalid LORA', m);
                    } else {
                        loras.push({ name: s[0], weight });
                    }
                    return '';
                });

                for (const l of loras) {
                    const loraNode = LiteGraph.createNode<ComfyNode>('LoraLoader');
                    graph.add(loraNode);
                    setWidgetValue(loraNode, 'lora_name', l.name, true);
                    setWidgetValue(loraNode, 'strength_model', l.weight);
                    setWidgetValue(loraNode, 'strength_clip', l.weight);
                    prevModel.node.connect(prevModel.index, loraNode, 0);
                    prevClip.node.connect(prevClip.index, loraNode, 1);
                    prevModel = { node: loraNode, index: 0 };
                    prevClip = { node: loraNode, index: 1 };
                }

                prevClip.node.connect(1, clipNode, 0);
                prevModel.node.connect(0, samplerNode, 0);
                if (hrSamplerNode) {
                    prevModel.node.connect(0, hrSamplerNode, 0);
                }

                return { text, prevModel, prevClip };
            }

            function replaceEmbeddings(text: string) {
                if (!embeddings.length) return text;
                return text.replaceAll(
                    new RegExp(
                        '\\b(' + embeddings.map(e => e.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('\\b|\\b') + ')\\b',
                        'ig'
                    ),
                    'embedding:$1'
                );
            }

            function popOpt(name: string) {
                const v = opts[name];
                delete opts[name];
                return v;
            }

            graph.clear();
            graph.add(ckptNode);
            graph.add(clipSkipNode);
            graph.add(positiveNode);
            graph.add(negativeNode);
            graph.add(samplerNode);
            graph.add(imageNode);
            graph.add(vaeNode);
            graph.add(vaeLoaderNode);
            graph.add(saveNode);

            ckptNode.connect(1, clipSkipNode, 0);
            clipSkipNode.connect(0, positiveNode, 0);
            clipSkipNode.connect(0, negativeNode, 0);
            ckptNode.connect(0, samplerNode, 0);
            positiveNode.connect(0, samplerNode, 1);
            negativeNode.connect(0, samplerNode, 2);
            imageNode.connect(0, samplerNode, 3);
            vaeNode.connect(0, saveNode, 0);
            samplerNode.connect(0, vaeNode, 0);
            vaeLoaderNode.connect(0, vaeNode, 1);

            const handlers = {
                model(v: string) {
                    setWidgetValue(ckptNode, 'ckpt_name', v, true);
                },
                'cfg scale'(v: number) {
                    setWidgetValue(samplerNode, 'cfg', +v);
                },
                'clip skip'(v: number) {
                    setWidgetValue(clipSkipNode, 'stop_at_clip_layer', -v);
                },
                sampler(v: string) {
                    let name = v.toLowerCase().replace('++', 'pp').replaceAll(' ', '_');
                    if (name.includes('karras')) {
                        name = name.replace('karras', '').replace(/_+$/, '');
                        setWidgetValue(samplerNode, 'scheduler', 'karras');
                    } else {
                        setWidgetValue(samplerNode, 'scheduler', 'normal');
                    }
                    const w = getWidget(samplerNode, 'sampler_name');
                    const o = w?.options.values.find((w: string) => w === name || w === 'sample_' + name);
                    if (o) {
                        setWidgetValue(samplerNode, 'sampler_name', o);
                    }
                },
                size(v: string) {
                    const wxh = v.split('x');
                    const w = ceil64(+wxh[0]);
                    const h = ceil64(+wxh[1]);
                    const hrUp = popOpt('hires upscale');
                    const hrSz = popOpt('hires resize');
                    let hrMethod = popOpt('hires upscaler');

                    setWidgetValue(imageNode, 'width', w);
                    setWidgetValue(imageNode, 'height', h);

                    if (hrUp || hrSz) {
                        let uw, uh;
                        if (hrUp) {
                            uw = w * Number(hrUp);
                            uh = h * Number(hrUp);
                        } else {
                            const s = hrSz.split('x');
                            uw = +s[0];
                            uh = +s[1];
                        }

                        let upscaleNode;
                        let latentNode;

                        if (hrMethod.startsWith('Latent')) {
                            latentNode = upscaleNode = LiteGraph.createNode<ComfyNode>('LatentUpscale');
                            graph.add(upscaleNode);
                            samplerNode.connect(0, upscaleNode, 0);

                            switch (hrMethod) {
                                case 'Latent (nearest-exact)':
                                    hrMethod = 'nearest-exact';
                                    break;
                            }
                            setWidgetValue(upscaleNode, 'upscale_method', hrMethod, true);
                        } else {
                            const decode = LiteGraph.createNode<ComfyNode>('VAEDecodeTiled');
                            graph.add(decode);
                            samplerNode.connect(0, decode, 0);
                            vaeLoaderNode.connect(0, decode, 1);

                            const upscaleLoaderNode = LiteGraph.createNode<ComfyNode>('UpscaleModelLoader');
                            graph.add(upscaleLoaderNode);
                            setWidgetValue(upscaleLoaderNode, 'model_name', hrMethod, true);

                            const modelUpscaleNode = LiteGraph.createNode<ComfyNode>('ImageUpscaleWithModel');
                            graph.add(modelUpscaleNode);
                            decode.connect(0, modelUpscaleNode, 1);
                            upscaleLoaderNode.connect(0, modelUpscaleNode, 0);

                            upscaleNode = LiteGraph.createNode<ComfyNode>('ImageScale');
                            graph.add(upscaleNode);
                            modelUpscaleNode.connect(0, upscaleNode, 0);

                            const vaeEncodeNode = (latentNode = LiteGraph.createNode<ComfyNode>('VAEEncodeTiled'));
                            graph.add(vaeEncodeNode);
                            upscaleNode.connect(0, vaeEncodeNode, 0);
                            vaeLoaderNode.connect(0, vaeEncodeNode, 1);
                        }

                        setWidgetValue(upscaleNode, 'width', ceil64(uw));
                        setWidgetValue(upscaleNode, 'height', ceil64(uh));

                        hrSamplerNode = LiteGraph.createNode<ComfyNode>('KSampler');
                        graph.add(hrSamplerNode);
                        ckptNode.connect(0, hrSamplerNode, 0);
                        positiveNode.connect(0, hrSamplerNode, 1);
                        negativeNode.connect(0, hrSamplerNode, 2);
                        latentNode.connect(0, hrSamplerNode, 3);
                        hrSamplerNode?.connect(0, vaeNode, 0);
                    }
                },
                steps(v: number) {
                    setWidgetValue(samplerNode, 'steps', +v);
                },
                seed(v: number) {
                    setWidgetValue(samplerNode, 'seed', +v);
                },
            };

            for (const opt in opts) {
                if (opt in handlers) {
                    handlers[opt as keyof typeof handlers](popOpt(opt) as never);
                }
            }

            if (hrSamplerNode) {
                setWidgetValue(hrSamplerNode, 'steps', getWidget(samplerNode, 'steps')?.value);
                setWidgetValue(hrSamplerNode, 'cfg', getWidget(samplerNode, 'cfg')?.value);
                setWidgetValue(hrSamplerNode, 'scheduler', getWidget(samplerNode, 'scheduler')?.value);
                setWidgetValue(hrSamplerNode, 'sampler_name', getWidget(samplerNode, 'sampler_name')?.value);
                setWidgetValue(hrSamplerNode, 'denoise', +(popOpt('denoising strength') || '1'));
            }

            let n = createLoraNodes(
                positiveNode,
                positive,
                { node: clipSkipNode, index: 0 },
                { node: ckptNode, index: 0 }
            );
            positive = n.text;
            n = createLoraNodes(negativeNode, negative, n.prevClip, n.prevModel);
            negative = n.text;

            setWidgetValue(positiveNode, 'text', replaceEmbeddings(positive));
            setWidgetValue(negativeNode, 'text', replaceEmbeddings(negative));

            graph.arrange();

            for (const opt of ['model hash', 'ensd']) {
                delete opts[opt];
            }

            console.warn('Unhandled parameters:', opts);
        }
    }
}
