import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "i18next",
    addCustomNodeDefs(defs) {
        for (const k in defs) {
            defs[k].display_name = i18next.t(`node.title.${k}`)

            if ("input" in defs[k] && defs[k].input.required) {
                for (const i in defs[k].input.required) {
                    if (defs[k].input.required[i].length > 1) {
                        defs[k].input.required[i][1].label = i18next.t(`node.input.${k}.${i}`)
                    } else {
                        defs[k].input.required[i].push({ label: i18next.t(`node.input.${k}.${i}`) })
                    }
                }
            }
        }
    },
    nodeCreated(node) {
        if ("inputs" in node) {
            for (const item of node.inputs) {
                item.label = i18next.t(`node.input.${node.comfyClass}.${item.name}`)
            }
        }

        if ("widgets" in node) {
            for (const item of node.widgets) {
                item.label = i18next.t(`node.input.${node.comfyClass}.${item.name}`)
            }
        }

        if ("outputs" in node) {
            for (const item of node.outputs) {
                item.label = i18next.t(`node.output.${node.comfyClass}.${item.name}`)
            }
        }
    },
    afterNodesRegistrations() {
        const defs = LiteGraph.registered_node_types

        for (const k in defs) {
            defs[k].category = i18next.t(`category.${defs[k].category}`)
        }
    }
})