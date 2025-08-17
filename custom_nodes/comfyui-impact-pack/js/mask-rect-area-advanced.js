import { app } from "/scripts/app.js";
function showPreviewCanvas(node, app) {

    const widget = {
        type: "customCanvas",
        name: "mask-rect-area-canvas",
        get value() {
            return this.canvas.value;
        },
        set value(x) {
            this.canvas.value = x;
        },
        draw: function (ctx, node, widgetWidth, widgetY) {

            // If we are initially offscreen when created we wont have received a resize event
            // Calculate it here instead
            if (!node.canvasHeight) {
                computeCanvasSize(node, node.size);
            }

            const visible = true;
            const t = ctx.getTransform();
            const margin = 12;
            const border = 2;
            const widgetHeight = node.canvasHeight;
            const width = Math.round(node.properties["width"]);
            const height = Math.round(node.properties["height"]);
            const scale = Math.min((widgetWidth - margin * 3) / width, (widgetHeight - margin * 3) / height);
            const blurRadius = node.properties["blur_radius"] || 0;
            const index = 0;

            Object.assign(this.canvas.style, {
                left: `${t.e}px`,
                top: `${t.f + (widgetY * t.d)}px`,
                width: `${widgetWidth * t.a}px`,
                height: `${widgetHeight * t.d}px`,
                position: "absolute",
                zIndex: 1,
                fontSize: `${t.d * 10.0}px`,
                pointerEvents: "none"
            });

            this.canvas.hidden = !visible;

            let backgroundWidth = width * scale;
            let backgroundHeight = height * scale;

            let xOffset = margin;
            if (backgroundWidth < widgetWidth) {
                xOffset += (widgetWidth - backgroundWidth) / 2 - margin;
            }
            let yOffset = (margin / 2);
            if (backgroundHeight < widgetHeight) {
                yOffset += (widgetHeight - backgroundHeight) / 2 - margin;
            }

            let widgetX = xOffset;
            widgetY = widgetY + yOffset;

            // Draw the background border
            ctx.fillStyle = globalThis.LiteGraph.WIDGET_OUTLINE_COLOR;
            ctx.fillRect(widgetX - border, widgetY - border, backgroundWidth + border * 2, backgroundHeight + border * 2)

            // Draw the main background area 
            ctx.fillStyle = globalThis.LiteGraph.WIDGET_BGCOLOR;
            ctx.fillRect(widgetX, widgetY, backgroundWidth, backgroundHeight);

            // Draw the conditioning zone
            let [x, y, w, h] = getDrawArea(node, backgroundWidth, backgroundHeight);

            ctx.fillStyle = getDrawColor(0, "80");
            ctx.fillRect(widgetX + x, widgetY + y, w, h);
            ctx.beginPath();
            ctx.lineWidth = 1;

            // Draw grid lines
            for (let x = 0; x <= width / 64; x += 1) {
                ctx.moveTo(widgetX + x * 64 * scale, widgetY);
                ctx.lineTo(widgetX + x * 64 * scale, widgetY + backgroundHeight);
            }

            for (let y = 0; y <= height / 64; y += 1) {
                ctx.moveTo(widgetX, widgetY + y * 64 * scale);
                ctx.lineTo(widgetX + backgroundWidth, widgetY + y * 64 * scale);
            }

            ctx.strokeStyle = "#66666650";
            ctx.stroke();
            ctx.closePath();

            // Draw current zone
            let [sx, sy, sw, sh] = getDrawArea(node, backgroundWidth, backgroundHeight);

            ctx.fillStyle = getDrawColor(0, "80");
            ctx.fillRect(widgetX + sx, widgetY + sy, sw, sh);

            ctx.fillStyle = getDrawColor(0, "40");
            ctx.fillRect(widgetX + sx + border, widgetY + sy + border, sw - border * 2, sh - border * 2);

            // Draw white border around the current zone
            ctx.strokeStyle = globalThis.LiteGraph.NODE_SELECTED_TITLE_COLOR;
            ctx.lineWidth = 2;
            ctx.strokeRect(widgetX + sx, widgetY + sy, sw, sh);

            // Display
            ctx.beginPath();

            ctx.arc(LiteGraph.NODE_SLOT_HEIGHT * 0.5, LiteGraph.NODE_SLOT_HEIGHT * (index + 0.5) + 4, 4, 0, Math.PI * 2);
            ctx.fill();

            ctx.lineWidth = 1;
            ctx.strokeStyle = "white";
            ctx.stroke();

            ctx.lineWidth = 1;
            ctx.closePath();

            // Draw progress bar canvas
            if (backgroundWidth < widgetWidth) {
                xOffset += (widgetWidth - backgroundWidth) / 2 - margin;
            }

            // Ajustar las coordenadas X e Y
            const barHeight = 8;
            let widgetYBar = widgetY + backgroundHeight + margin;

            // Dibujar el borde negro alrededor de la barra
            ctx.fillStyle = globalThis.LiteGraph.WIDGET_OUTLINE_COLOR;
            ctx.fillRect(
                    widgetX - border,
                    widgetYBar - border,
                    backgroundWidth + border * 2,
                    barHeight + border * 2
                    );

            // Dibujar el área principal de la barra (fondo)
            ctx.fillStyle = globalThis.LiteGraph.WIDGET_BGCOLOR; // Mismo color de fondo que el canvas
            ctx.fillRect(
                    widgetX,
                    widgetYBar,
                    backgroundWidth,
                    barHeight
                    );


            // Draw progress bar grid
            ctx.beginPath();
            ctx.lineWidth = 1;
            ctx.strokeStyle = "#66666650";

            // Calcular el número de líneas en función del tamaño de la barra
            const numLines = Math.floor(backgroundWidth / 64);

            // Dibujar líneas del grid
            for (let x = 0; x <= width / 64; x += 1) {
                ctx.moveTo(widgetX + x * 64 * scale, widgetYBar);
                ctx.lineTo(widgetX + x * 64 * scale, widgetYBar + barHeight);
            }
            ctx.stroke();
            ctx.closePath();

            // Dibujar progreso (basado en blur_radius)
            const progress = Math.min(blurRadius / 255, 1);
            ctx.fillStyle = "rgba(0, 120, 255, 0.5)";

            ctx.fillRect(
                    widgetX,
                    widgetYBar,
                    backgroundWidth * progress,
                    barHeight
                    );
        }
    };

    widget.canvas = document.createElement("canvas");
    widget.canvas.className = "mask-rect-area-canvas";
    widget.parent = node;

    document.body.appendChild(widget.canvas);
    node.addCustomWidget(widget);

    app.canvas.onDrawBackground = function () {
        // Draw node isnt fired once the node is off the screen
        // if it goes off screen quickly, the input may not be removed
        // this shifts it off screen so it can be moved back if the node is visible.
        for (let n in app.graph._nodes) {
            n = app.graph._nodes[n];
            for (let w in n.widgets) {
                let wid = n.widgets[w];
                if (Object.hasOwn(wid, "canvas")) {
                    wid.canvas.style.left = -8000 + "px";
                    wid.canvas.style.position = "absolute";
                }
            }
        }
    };

    node.onResize = function (size) {
        computeCanvasSize(node, size);
    };

    return {minWidth: 200, minHeight: 200, widget};
}

app.registerExtension({
    name: 'drltdata.MaskRectAreaAdvanced',
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "MaskRectAreaAdvanced") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                this.setProperty("width", 512);
                this.setProperty("height", 512);
                this.setProperty("x", 0);
                this.setProperty("y", 0);
                this.setProperty("w", 256);
                this.setProperty("h", 256);
                this.setProperty("blur_radius", 0);

                this.selected = false;
                this.index = 3;
                this.serialize_widgets = true;

                CUSTOM_INT(this, "x", 0, function (v, _, node) {
                    const s = this.options.step / 10;
                    this.value = Math.round(v / s) * s;
                    node.properties["x"] = this.value;
                });
                CUSTOM_INT(this, "y", 0, function (v, _, node) {
                    const s = this.options.step / 10;
                    this.value = Math.round(v / s) * s;
                    node.properties["y"] = this.value;
                });
                CUSTOM_INT(this, "width", 256, function (v, _, node) {
                    const s = this.options.step / 10;
                    this.value = Math.round(v / s) * s;
                    node.properties["w"] = this.value;
                });
                CUSTOM_INT(this, "height", 256, function (v, _, node) {
                    const s = this.options.step / 10;
                    this.value = Math.round(v / s) * s;
                    node.properties["h"] = this.value;
                });
                CUSTOM_INT(this, "image_width", 512, function (v, _, node) {
                    const s = this.options.step / 10;
                    this.value = Math.round(v / s) * s;
                    node.properties["width"] = this.value;
                });
                CUSTOM_INT(this, "image_height", 512, function (v, _, node) {
                    const s = this.options.step / 10;
                    this.value = Math.round(v / s) * s;
                    node.properties["height"] = this.value;
                });
                CUSTOM_INT(this, "blur_radius", 0, function (v, _, node) {
                    this.value = Math.round(v) || 0;
                    node.properties["blur_radius"] = this.value;
                },
                        {"min": 0, "max": 255, "step": 10}
                );

                showPreviewCanvas(this, app);

                this.onSelected = function () {
                    this.selected = true;
                };
                this.onDeselected = function () {
                    this.selected = false;
                };

                return r;
            };
        }
    }
});

// Calculate the drawing area using individual properties.
function getDrawArea(node, backgroundWidth, backgroundHeight) {
    let x = node.properties["x"] * backgroundWidth / node.properties["width"];
    let y = node.properties["y"] * backgroundHeight / node.properties["height"];
    let w = node.properties["w"] * backgroundWidth / node.properties["width"];
    let h = node.properties["h"] * backgroundHeight / node.properties["height"];

    if (x > backgroundWidth) {
        x = backgroundWidth;
    }
    if (y > backgroundHeight) {
        y = backgroundHeight;
    }

    if (x + w > backgroundWidth) {
        w = Math.max(0, backgroundWidth - x);
    }

    if (y + h > backgroundHeight) {
        h = Math.max(0, backgroundHeight - y);
    }

    return [x, y, w, h];
}

function CUSTOM_INT(node, inputName, val, func, config = {}) {
    return {
        widget: node.addWidget(
                "number",
                inputName,
                val,
                func,
                Object.assign({}, {min: 0, max: 4096, step: 640, precision: 0}, config)
                )
    };
}

function getDrawColor(percent, alpha) {
    let h = 360 * percent;
    let s = 50;
    let l = 50;
    l /= 100;
    const a = s * Math.min(l, 1 - l) / 100;
    const f = n => {
        const k = (n + h / 30) % 12;
        const color = l - a * Math.max(Math.min(k - 3, 9 - k, 1), -1);
        return Math.round(255 * color).toString(16).padStart(2, '0');   // convert to Hex and prefix "0" if needed
    };
    return `#${f(0)}${f(8)}${f(4)}${alpha}`;
}

function computeCanvasSize(node, size) {
    if (node.widgets[0].last_y == null) {
        return;
    }

    const MIN_HEIGHT = 220;
    const MIN_WIDTH = 240;

    let y = LiteGraph.NODE_WIDGET_HEIGHT * Math.max(node.inputs.length, node.outputs.length) + 5;
    let freeSpace = size[1] - y;

    // Compute the height of all non-customCanvas widgets
    let widgetHeight = 0;
    for (let i = 0; i < node.widgets.length; i++) {
        const w = node.widgets[i];
        if (w.type !== "customCanvas") {
            if (w.computeSize) {
                widgetHeight += w.computeSize()[1] + 4;
            } else {
                widgetHeight += LiteGraph.NODE_WIDGET_HEIGHT + 5;
            }
        }
    }

    // Ensure there is enough vertical space
    freeSpace -= widgetHeight;

    // Adjust the height of the node if needed
    if (freeSpace < MIN_HEIGHT) {
        freeSpace = MIN_HEIGHT;
        node.size[1] = y + widgetHeight + freeSpace;
        node.graph.setDirtyCanvas(true);
    }

    // Ensure the node width meets the minimum width requirement
    if (node.size[0] < MIN_WIDTH) {
        node.size[0] = MIN_WIDTH;
        node.graph.setDirtyCanvas(true);
    }

    // Position each of the widgets
    for (const w of node.widgets) {
        w.y = y;
        if (w.type === "customCanvas") {
            y += freeSpace;
        } else if (w.computeSize) {
            y += w.computeSize()[1] + 4;
        } else {
            y += LiteGraph.NODE_WIDGET_HEIGHT + 4;
        }
    }

    node.canvasHeight = freeSpace;
}
