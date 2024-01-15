// In the orignal ComfyUI, Litegraph's classes are monkey-patched with new functionality,
// which is a difficult-to-maintain pattern.
// Instead, in ComfyTS we use class-inheritance to extend the functionality of Litegraph's
// original base classes. This is simpler and more maintanable.
import { LiteGraph, LGraphCanvas } from 'litegraph.js';
import { ComfyNode } from './comfyNode';
import { ComfyApp } from './app';
import { ComfyGraph } from './comfyGraph';

// TO DO: list all hot keys this class has and what they do

export class ComfyCanvas extends LGraphCanvas {
    app: ComfyApp; // reference to the app this canvas is inside of
    selected_group_moving: boolean = false;
    
    constructor(app: ComfyApp, canvas: HTMLCanvasElement | string, graph?: ComfyGraph, options?: { skip_render?: boolean; autoresize?: boolean; }) {
        super(canvas, graph, options);
        this.app = app;
    }

    /** Draws group header bar */
    drawGroups(canvas: HTMLCanvasElement | string, ctx: CanvasRenderingContext2D): void {
        if (!this.graph) {
            return;
        }

        const groups = this.graph.groups;

        ctx.save();
        ctx.globalAlpha = 0.7 * this.editor_alpha;

        for (let i = 0; i < groups.length; ++i) {
            const group = groups[i];

            if (!LiteGraph.overlapBounding(this.visible_area, group.bounding)) {
                continue;
            } // Out of the visible area

            ctx.fillStyle = group.color || '#335';
            ctx.strokeStyle = group.color || '#335';

            const pos = group.pos;
            const size = group.size;

            ctx.globalAlpha = 0.25 * this.editor_alpha;
            ctx.beginPath();
            const font_size = group.font_size || LiteGraph.DEFAULT_GROUP_FONT_SIZE;
            ctx.rect(pos[0] + 0.5, pos[1] + 0.5, size[0], font_size * 1.4);
            ctx.fill();
            ctx.globalAlpha = this.editor_alpha;
        }

        ctx.restore();
        super.drawGroups(canvas, ctx);
    }

    /** Draws node highlights (executing, drag drop) and progress bar */
    drawNode(node: ComfyNode, ctx: CanvasRenderingContext2D): void {
            var editor_alpha = this.editor_alpha;
            var old_color = node.bgcolor;

            if (node.mode === 2) {
                // never
                this.editor_alpha = 0.4;
            }

            // if (node.mode === 4) {
            if (node.mode === LiteGraph.NEVER) {
                // never
                node.bgcolor = '#FF00FF';
                this.editor_alpha = 0.2;
            }

            const res = super.drawNode(node, ctx);

            this.editor_alpha = editor_alpha;
            node.bgcolor = old_color;

            return res;
        };

    drawNodeShape(node: ComfyNode, ctx: CanvasRenderingContext2D, size: [number, number], fgcolor: string, bgcolor: string, selected: boolean, mouse_over: boolean): void {
        const res = super.drawNodeShape(node, ctx, size, fgcolor, bgcolor, selected, mouse_over);
        const self = this.app;

            const nodeErrors = self.lastNodeErrors?.[node.id];

            let color = null;
            let lineWidth = 1;
            if (node.id === +(self.runningNodeId ?? 0)) {
                color = '#0f0';
            } else if (self.dragOverNode && node.id === self.dragOverNode.id) {
                color = 'dodgerblue';
            } else if (nodeErrors?.errors) {
                color = 'red';
                lineWidth = 2;
            } else if (self.lastExecutionError && +self.lastExecutionError.node_id === node.id) {
                color = '#f0f';
                lineWidth = 2;
            }

            if (color) {
                // const shape = node._shape || node.constructor.shape || LiteGraph.ROUND_SHAPE;
                const shape = node.shape || LiteGraph.ROUND_SHAPE;
                ctx.lineWidth = lineWidth;
                ctx.globalAlpha = 0.8;
                ctx.beginPath();
                if (shape == LiteGraph.BOX_SHAPE)
                    ctx.rect(
                        -6,
                        -6 - LiteGraph.NODE_TITLE_HEIGHT,
                        12 + size[0] + 1,
                        12 + size[1] + LiteGraph.NODE_TITLE_HEIGHT
                    );
                else if (shape == LiteGraph.ROUND_SHAPE || (shape == LiteGraph.CARD_SHAPE && node.flags.collapsed))
                    ctx.roundRect(
                        -6,
                        -6 - LiteGraph.NODE_TITLE_HEIGHT,
                        12 + size[0] + 1,
                        12 + size[1] + LiteGraph.NODE_TITLE_HEIGHT,
                        this.round_radius * 2
                    );
                else if (shape == LiteGraph.CARD_SHAPE)
                    ctx.roundRect(
                        -6,
                        -6 - LiteGraph.NODE_TITLE_HEIGHT,
                        12 + size[0] + 1,
                        12 + size[1] + LiteGraph.NODE_TITLE_HEIGHT,
                        [this.round_radius * 2, this.round_radius * 2, 2, 2]
                    );
                else if (shape == LiteGraph.CIRCLE_SHAPE)
                    ctx.arc(size[0] * 0.5, size[1] * 0.5, size[0] * 0.5 + 6, 0, Math.PI * 2);
                ctx.strokeStyle = color;
                ctx.stroke();
                ctx.strokeStyle = fgcolor;
                ctx.globalAlpha = 1;
            }

            if (self.progress && node.id === +(self.runningNodeId ?? 0)) {
                ctx.fillStyle = 'green';
                ctx.fillRect(0, 0, size[0] * (self.progress.value / self.progress.max), 6);
                ctx.fillStyle = bgcolor;
            }

            // Highlight inputs that failed validation
            if (nodeErrors) {
                ctx.lineWidth = 2;
                ctx.strokeStyle = 'red';
                for (const error of nodeErrors.errors) {
                    if (error.extra_info && error.extra_info.input_name) {
                        const inputIndex = node.findInputSlot(error.extra_info.input_name);
                        if (inputIndex !== -1) {
                            let pos = node.getConnectionPos(true, inputIndex);
                            ctx.beginPath();
                            ctx.arc(pos[0] - node.pos[0], pos[1] - node.pos[1], 12, 0, 2 * Math.PI, false);
                            ctx.stroke();
                        }
                    }
                }
            }

            return res;
    };

    /** Handle keypress, Ctrl + M mute/unmute selected nodes, and other hot keys */
    processKey(e: KeyboardEvent): boolean | undefined {
        if (!this.graph) {
            return false;
        }

        let block_default = false;

        if (e.target?.localName === 'input') {
            return false;
        }

        if (e.type === 'keydown' && !e.repeat) {
            // Ctrl + M mute/unmute
            if (e.key === 'm' && e.ctrlKey) {
                if (this.selected_nodes) {
                    for (const i in this.selected_nodes) {
                        const node = this.selected_nodes[i];
                        node.mode = node.mode === LiteGraph.NEVER ? LiteGraph.ALWAYS : LiteGraph.NEVER;
                    }
                }
                block_default = true;
            }

            // Ctrl + B bypass
            if (e.key === 'b' && e.ctrlKey) {
                if (this.selected_nodes) {
                    for (const i in this.selected_nodes) {
                        const node = this.selected_nodes[i];
                        node.mode = node.mode === LiteGraph.NEVER ? LiteGraph.ALWAYS : LiteGraph.NEVER;
                    }
                }
                block_default = true;
            }

            // Alt + C collapse/uncollapse
            if (e.key === 'c' && e.altKey) {
                if (this.selected_nodes) {
                    for (const i in this.selected_nodes) {
                        const node = this.selected_nodes[i];
                        node.collapse(false);
                    }
                }
                block_default = true;
            }

            // Ctrl+C Copy
            if (e.key === 'c' && (e.metaKey || e.ctrlKey)) {
                // Trigger onCopy
                return true;
            }

            // Ctrl+V Paste
            if ((e.key === 'v' || e.key === 'V') && (e.metaKey || e.ctrlKey) && !e.shiftKey) {
                // Trigger onPaste
                return true;
            }

            this.graph.change();

            if (block_default) {
                e.preventDefault();
                e.stopImmediatePropagation();
                return false;
            }
        }

        return super.processKey(e);
    }

    /** handle mouse, move group by header */
    processMouseDown(e: MouseEvent): boolean | undefined {
        const res = super.processMouseDown(e);
        this.selected_group_moving = false;

        if (this.selected_group && !this.selected_group_resizing) {
            const font_size = this.selected_group.font_size || LiteGraph.DEFAULT_GROUP_FONT_SIZE;
            const height = font_size * 1.4;

            // Move group by header
            if (
                LiteGraph.isInsideRectangle(
                    e.canvasX,
                    e.canvasY,
                    this.selected_group.pos[0],
                    this.selected_group.pos[1],
                    this.selected_group.size[0],
                    height
                )
            ) {
                this.selected_group_moving = true;
            }
        }

        return res;
    }


    processMouseMove(e: MouseEvent): boolean | undefined {
        const orig_selected_group = this.selected_group;

        if (this.selected_group && !this.selected_group_resizing && !this.selected_group_moving) {
            this.selected_group = null;
        }

        const res = super.processMouseMove(e); // original event-handler

        if (orig_selected_group && !this.selected_group_resizing && !this.selected_group_moving) {
            this.selected_group = orig_selected_group;
        }

        return res;
    }
}

