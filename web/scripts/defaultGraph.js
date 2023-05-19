export const defaultGraph = {
        last_node_id: 14,
        last_link_id: 17,
        nodes: [
                {
                        id: 6,
                        type: "CLIPTextEncode",
                        pos: [440, 180],
                        size: [260, 100],
                        flags: {},
                        order: 3,
                        mode: 0,
                        inputs: [{ name: "clip", type: "CLIP", link: 3 }],
                        outputs: [{ name: "CONDITIONING", type: "CONDITIONING", links: [4], slot_index: 0 }],
                        properties: {},
                        widgets_values: ["beautiful scenery nature\nglass bottle landscape,\n, purple galaxy bottle,"]
                },
                {
                        id: 7,
                        type: "CLIPTextEncode",
                        pos: [440, 320],
                        size: [260, 100],
                        flags: {},
                        order: 4,
                        mode: 0,
                        inputs: [{ name: "clip", type: "CLIP", link: 5 }],
                        outputs: [{ name: "CONDITIONING", type: "CONDITIONING", links: [6], slot_index: 0 }],
                        properties: {},
                        widgets_values: ["text, watermark"]
                },
                {
                        id: 11,
                        type: "Reroute",
                        pos: [440, 100],
                        size: [82, 26],
                        flags: {},
                        order: 2,
                        mode: 0,
                        inputs: [{ name: "", type: "*", link: 10 }],
                        outputs: [{ name: "MODEL", type: "MODEL", links: [11], slot_index: 0 }],
                        properties: { showOutputText: true, horizontal: false }
                },
                {
                        id: 13,
                        type: "Reroute",
                        pos: [440, 580],
                        size: [75, 26],
                        flags: {},
                        order: 5,
                        mode: 0,
                        inputs: [{ name: "", type: "*", link: 13 }],
                        outputs: [{ name: "VAE", type: "VAE", links: [14], slot_index: 0 }],
                        properties: { showOutputText: true, horizontal: false }
                },
                {
                        id: 3,
                        type: "KSampler",
                        pos: [780, 180],
                        size: [280, 260],
                        flags: {},
                        order: 8,
                        mode: 0,
                        inputs: [
                                { name: "model", type: "MODEL", link: 12 },
                                { name: "positive", type: "CONDITIONING", link: 4 },
                                { name: "negative", type: "CONDITIONING", link: 6 },
                                { name: "latent_image", type: "LATENT", link: 2 }
                        ],
                        outputs: [{ name: "LATENT", type: "LATENT", links: [7], slot_index: 0 }],
                        properties: {},
                        widgets_values: [156680208700286, "increment", 20, 8, "euler", "normal", 1]
                },
                {
                        id: 10,
                        type: "PreviewImage",
                        pos: [1140, 320],
                        size: [210, 246],
                        flags: {},
                        order: 10,
                        mode: 0,
                        inputs: [{ name: "images", type: "IMAGE", link: 16 }],
                        properties: {}
                },
                {
                        id: 14,
                        type: "Reroute",
                        pos: [980, 580],
                        size: [75, 26],
                        flags: {},
                        order: 7,
                        mode: 0,
                        inputs: [{ name: "", type: "*", link: 14 }],
                        outputs: [{ name: "VAE", type: "VAE", links: [15], slot_index: 0 }],
                        properties: { showOutputText: true, horizontal: false }
                },
                {
                        id: 12,
                        type: "Reroute",
                        pos: [620, 100],
                        size: [82, 26],
                        flags: {},
                        order: 6,
                        mode: 0,
                        inputs: [{ name: "", type: "*", link: 11 }],
                        outputs: [{ name: "MODEL", type: "MODEL", links: [12], slot_index: 0 }],
                        properties: { showOutputText: true, horizontal: false }
                },
                {
                        id: 5,
                        type: "EmptyLatentImage",
                        pos: [480, 460],
                        size: [220, 100],
                        flags: {},
                        order: 0,
                        mode: 0,
                        outputs: [{ name: "LATENT", type: "LATENT", links: [2], slot_index: 0 }],
                        properties: {},
                        widgets_values: [512, 512, 1]
                },
                {
                        id: 4,
                        type: "CheckpointLoaderSimple",
                        pos: [20, 160],
                        size: [340, 100],
                        flags: {},
                        order: 1,
                        mode: 0,
                        outputs: [
                                { name: "MODEL", type: "MODEL", links: [10], slot_index: 0 },
                                { name: "CLIP", type: "CLIP", links: [3, 5], slot_index: 1 },
                                { name: "VAE", type: "VAE", links: [13], slot_index: 2 }
                        ],
                        properties: {},
                        widgets_values: ["v1-5-pruned-emaonly.ckpt"]
                },
                {
                        id: 9,
                        type: "SaveImage",
                        pos: [1420, 180],
                        size: [220, 280],
                        flags: {},
                        order: 11,
                        mode: 0,
                        inputs: [{ name: "images", type: "IMAGE", link: 17 }],
                        properties: {},
                        widgets_values: ["ComfyUI"]
                },
                {
                        id: 8,
                        type: "VAEDecode",
                        pos: [1140, 180],
                        size: [140, 46],
                        flags: {},
                        order: 9,
                        mode: 0,
                        inputs: [
                                { name: "samples", type: "LATENT", link: 7 },
                                { name: "vae", type: "VAE", link: 15 }
                        ],
                        outputs: [{ name: "IMAGE", type: "IMAGE", links: [16, 17], slot_index: 0 }],
                        properties: {}
                }
        ],
        links: [
                [2, 5, 0, 3, 3, "LATENT"],
                [3, 4, 1, 6, 0, "CLIP"],
                [4, 6, 0, 3, 1, "CONDITIONING"],
                [5, 4, 1, 7, 0, "CLIP"],
                [6, 7, 0, 3, 2, "CONDITIONING"],
                [7, 3, 0, 8, 0, "LATENT"],
                [10, 4, 0, 11, 0, "*"],
                [11, 11, 0, 12, 0, "*"],
                [12, 12, 0, 3, 0, "MODEL"],
                [13, 4, 2, 13, 0, "*"],
                [14, 13, 0, 14, 0, "*"],
                [15, 14, 0, 8, 1, "VAE"],
                [16, 8, 0, 10, 0, "IMAGE"]
        ],
        groups: [],
        config: {},
        extra: {},
        version: 0.4
};
