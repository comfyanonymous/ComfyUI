import { app } from "/scripts/app.js";

// Show grids from combinatorial outputs

async function loadImageAsync(imageURL) {
    return new Promise((resolve) => {
        const e = new Image();
        e.setAttribute('crossorigin', 'anonymous');
        e.addEventListener("load", () => { resolve(e); });
        e.src = imageURL;
        return e;
    });
}

app.registerExtension({
	name: "Comfy.ShowGrid",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (!(nodeData.name === "SaveImage" || nodeData.name === "PreviewImage")) {
			return
		}

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

			this.showGridWidget = this.addWidget("button", "Show Grid", "Show Grid", () => {
				const grid = app.nodeGrids[this.id];
				if (grid == null) {
					console.warn("No grid to show!");
					return;
				}

				const graphCanvas = LiteGraph.LGraphCanvas.active_canvas
				if (graphCanvas == null)
					return;

				if (this._gridPanel != null)
					return

				this._gridPanel = graphCanvas.createPanel("Grid", { closable: true });
				this._gridPanel.onClose = () => {
					this._gridPanel = null;
				}
				this._gridPanel.node = this;
				this._gridPanel.classList.add("grid_dialog");

				const rootHtml = `
<div class="axis-selectors">
</div>
<table class="image-table">
</table>
`;
				const rootElem = this._gridPanel.addHTML(rootHtml, "grid-root");
				const axisSelectors = rootElem.querySelector(".axis-selectors");
				const imageTable = rootElem.querySelector(".image-table");

				this.imageSize = 512;
				this.imageWidth = this.imageSize
				this.imageHeight = this.imageSize
				this.naturalWidth = this.imageSize
				this.naturalHeight = this.imageSize

				const footerHtml = `
<label for="image-size">Image size</label>
<input class="image-size" id="image-size" type="range" min="64" max="1024" step="1" value="${this.imageSize}">
</input>
`
				const footerElem = this._gridPanel.addHTML(footerHtml, "grid-footer", true);
				const imageSizeInput = footerElem.querySelector(".image-size");

				const frozenCoords = Array.from({length: grid.axes.length}, (v, i) => 0)

				const getAxisData = (index) => {
					let data = grid.axes[index];
					if (data == null) {
						data = {
							nodeID: null,
							id: "none",
							label: "(Nothing)",
							values: ["(None)"]
						}
					}
					return data;
				}

				const selectAxis = (isY, axisID, change) => {
					const axisName = isY ? "y" : "x";
					const group = axisSelectors.querySelector(`.${axisName}-axis-selector`);

					for (const input of group.querySelectorAll(`input#${axisName}-${axisID}`)) {
						input.checked = true;
						if (change) {
							input.dispatchEvent(new Event('change'));
						}
					}
				}

				const getImagesAt = (x, y) => {
					return grid.images.filter(image => {
						for (let i = 0; i < grid.axes.length; i++) {
							if (i === this.xAxis) {
								if (image.coords[this.xAxis] !== x)
									return false;
							}
							else if (i === this.yAxis) {
								if (image.coords[this.yAxis] !== y)
									return false;
							}
							else {
								if (image.coords[i] !== frozenCoords[i])
									return false;
							}
						}
						return true;
					});
				}

				const refreshGrid = async (xAxis, yAxis) => {
					this.xAxis = xAxis;
					this.yAxis = yAxis;
					this.xAxisData = getAxisData(this.xAxis);
					this.yAxisData = getAxisData(this.yAxis);

					selectAxis(false, this.xAxisData.selectorID)
					selectAxis(true, this.yAxisData.selectorID)

					if (xAxis === yAxis) {
						this.yAxisData = getAxisData(-1);
					}

					this.imageWidth = this.imageSize
					this.imageHeight = this.imageSize
					this.naturalWidth = this.imageSize
					this.naturalHeight = this.imageSize

					const firstImages = getImagesAt(0, 0);
					if (firstImages.length > 0) {
						const src = "/view?" + new URLSearchParams(firstImages[0].image).toString() + app.getPreviewFormatParam();
						const imgElem = await loadImageAsync(src);
						this.naturalWidth = imgElem.naturalWidth
						this.naturalHeight = imgElem.naturalHeight

						const ratio = Math.min(this.imageSize / this.naturalWidth, this.imageSize / this.naturalHeight);
						const newWidth = this.naturalWidth * ratio;
						const newHeight = this.naturalHeight * ratio;
						this.imageWidth = newWidth;
						this.imageHeight = newHeight;
					}

					imageTable.innerHTML = "";

					const thead = document.createElement("thead")

					const trXAxisLabel = document.createElement("tr");
					const thXAxisLabel = document.createElement("th");
					thXAxisLabel.setAttribute("colspan", String(this.xAxisData.values.length + 2))
					thXAxisLabel.classList.add("axis", "x-axis")
					thXAxisLabel.innerHTML = "<span>" + this.xAxisData.label + "</span>";
					trXAxisLabel.appendChild(thXAxisLabel);
					thead.appendChild(trXAxisLabel);

					const trLabel = document.createElement("tr");
					trLabel.appendChild(document.createElement("th")) // blank
					trLabel.appendChild(document.createElement("th")) // blank
					for (const xValue of this.xAxisData.values) {
						const th = document.createElement("th");
						th.classList.add("label", "x-label");
						th.innerHTML = "<span>" + String(xValue) + "</span>";
						trLabel.appendChild(th);
					}
					thead.appendChild(trLabel)

					imageTable.appendChild(thead)

					const tableBody = document.createElement("tbody");
					imageTable.appendChild(tableBody);

					const trYAxisLabel = document.createElement("tr");
					const thYAxisLabel = document.createElement("th");
					thYAxisLabel.setAttribute("rowspan", String(this.yAxisData.values.length + 1))
					thYAxisLabel.classList.add("axis", "y-axis")
					thYAxisLabel.style.textAlign = "center"
					thYAxisLabel.style.textAlign = "center"
					thYAxisLabel.innerHTML = "<span>" + this.yAxisData.label + "</span>";
					trYAxisLabel.appendChild(thYAxisLabel);
					tableBody.appendChild(trYAxisLabel);

					for (const [y, yValue] of this.yAxisData.values.entries()) {
						const tr = document.createElement("tr");

						const tdLabel = document.createElement("td");
						tdLabel.innerHTML = "<span>" + String(yValue) + "</span>";
						tdLabel.classList.add("label", "y-label")
						tr.append(tdLabel);

						for (const [x, xValue] of this.xAxisData.values.entries()) {
							const td = document.createElement("td");

							const img = document.createElement("img");
							img.style.width = `${this.imageWidth}px`
							img.style.height = `${this.imageHeight}px`
							const gridImages = getImagesAt(x, y);
							if (gridImages.length > 0) {
								img.src = "/view?" + new URLSearchParams(gridImages[0].image).toString() + app.getPreviewFormatParam();
							}
							td.append(img);

							tr.append(td);
						}
						tableBody.appendChild(tr);
					}
				}

				for (let i = 0; i < 2; i++) {
					const axisName = i === 0 ? "x" : "y";
					const isY = i === 1;

					const group = document.createElement("div")
					group.setAttribute("role", "group");
					group.classList.add("axis-selector", `${axisName}-axis-selector`)

					group.innerHTML = `${axisName.toUpperCase()} Axis:&nbsp `;

					const addAxis = (index, axis) => {
						const axisID = `${axisName}-${axis.selectorID}`;

						const input = document.createElement("input")
						input.setAttribute("type", "radio")
						input.setAttribute("name", `${axisName}-axis-selector`)
						input.setAttribute("id", axisID)
						input.classList.add("axis-radio")
						input.addEventListener("change", () => {
							if (input.checked) {
								if (isY)
									this.yAxis = index;
								else
									this.xAxis = index;
							}

							refreshGrid(this.xAxis, this.yAxis);
						})

						const label = document.createElement("label")
						label.setAttribute("for", axisID)
						label.classList.add("axis-label")
						label.innerHTML = String(axis.label);
						label.addEventListener("click", () => {
							console.warn("SETAXIS", axis);
							selectAxis(isY, axis.selectorID, true);
						})

						group.appendChild(input)
						group.appendChild(label)
					}

					// Add "None" entry
					addAxis(-1, getAxisData(-1));

					for (const [index, axis] of grid.axes.entries()) {
						addAxis(index, axis);
					}

					axisSelectors.appendChild(group);
				}

				imageSizeInput.addEventListener("input", () => {
					this.imageSize = parseInt(imageSizeInput.value);

					const ratio = Math.min(this.imageSize / this.naturalWidth, this.imageSize / this.naturalHeight);
					const newWidth = this.naturalWidth * ratio;
					const newHeight = this.naturalHeight * ratio;
					this.imageWidth = newWidth;
					this.imageHeight = newHeight;

					for (const img of imageTable.querySelectorAll("img")) {
						img.style.width = `${this.imageWidth}px`
						img.style.height = `${this.imageHeight}px`
					}
				})

				refreshGrid(0, Math.min(1, grid.axes.length));

				document.body.appendChild(this._gridPanel);
			})

			this.showGridWidget.disabled = true;
		}

		const onExecuted = nodeType.prototype.onExecuted;
		nodeType.prototype.onExecuted = function (output) {
			const r = onExecuted ? onExecuted.apply(this, arguments) : undefined;
			this.showGridWidget.disabled = app.nodeGrids[this.id] == null;
		}
	}
})
