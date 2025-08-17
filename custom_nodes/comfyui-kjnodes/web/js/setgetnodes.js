import { app } from "../../../scripts/app.js";

//based on diffus3's SetGet: https://github.com/diffus3/ComfyUI-extensions

// Nodes that allow you to tunnel connections for cleaner graphs
function setColorAndBgColor(type) {
    const colorMap = {
        "MODEL": LGraphCanvas.node_colors.blue,
        "LATENT": LGraphCanvas.node_colors.purple,
        "VAE": LGraphCanvas.node_colors.red,
        "CONDITIONING": LGraphCanvas.node_colors.brown,
        "IMAGE": LGraphCanvas.node_colors.pale_blue,
        "CLIP": LGraphCanvas.node_colors.yellow,
        "FLOAT": LGraphCanvas.node_colors.green,
		"MASK": { color: "#1c5715", bgcolor: "#1f401b"},
		"INT": { color: "#1b4669", bgcolor: "#29699c"},
		"CONTROL_NET": { color: "#156653", bgcolor: "#1c453b"},
		"NOISE": { color: "#2e2e2e", bgcolor: "#242121"},
		"GUIDER": { color: "#3c7878", bgcolor: "#1c453b"},
		"SAMPLER": { color: "#614a4a", bgcolor: "#3b2c2c"},
		"SIGMAS": { color: "#485248", bgcolor: "#272e27"},

    };

    const colors = colorMap[type];
    if (colors) {
        this.color = colors.color;
        this.bgcolor = colors.bgcolor;
    }
}
let disablePrefix = app.ui.settings.getSettingValue("KJNodes.disablePrefix")
const LGraphNode = LiteGraph.LGraphNode

function showAlert(message) {
  app.extensionManager.toast.add({
    severity: 'warn',
    summary: "KJ Get/Set",
    detail: `${message}. Most likely you're missing custom nodes`,
    life: 5000,
  })
}
app.registerExtension({
	name: "SetNode",
	registerCustomNodes() {
		class SetNode extends LGraphNode {
			defaultVisibility = true;
			serialize_widgets = true;
			drawConnection = false;
			currentGetters = null;
			slotColor = "#FFF";
			canvas = app.canvas;
			menuEntry = "Show connections";

			constructor(title) {
				super(title)
				if (!this.properties) {
					this.properties = {
						"previousName": ""
					};
				}
				this.properties.showOutputText = SetNode.defaultVisibility;

				const node = this;

				this.addWidget(
					"text", 
					"Constant", 
					'', 
					(s, t, u, v, x) => {
						node.validateName(node.graph);
						if(this.widgets[0].value !== ''){
							this.title = (!disablePrefix ? "Set_" : "") + this.widgets[0].value;
						}
						this.update();
						this.properties.previousName = this.widgets[0].value;
					}, 
					{}
				)
				
				this.addInput("*", "*");
				this.addOutput("*", '*');

				this.onConnectionsChange = function(
					slotType,	//1 = input, 2 = output
					slot,
					isChangeConnect,
                    link_info,
                    output
				) {
					//On Disconnect
					if (slotType == 1 && !isChangeConnect) {
						if(this.inputs[slot].name === ''){
							this.inputs[slot].type = '*';
							this.inputs[slot].name = '*';
							this.title = "Set"
						}
					}
					if (slotType == 2 && !isChangeConnect) {
						if (this.outputs && this.outputs[slot]) {
							this.outputs[slot].type = '*';
							this.outputs[slot].name = '*';
						}
					}
					//On Connect
					if (link_info && node.graph && slotType == 1 && isChangeConnect) {
						const fromNode = node.graph._nodes.find((otherNode) => otherNode.id == link_info.origin_id);
						
						if (fromNode && fromNode.outputs && fromNode.outputs[link_info.origin_slot]) {
							const type = fromNode.outputs[link_info.origin_slot].type;
						
							if (this.title === "Set"){
								this.title = (!disablePrefix ? "Set_" : "") + type;
							}
							if (this.widgets[0].value === '*'){
								this.widgets[0].value = type	
							}
							
							this.validateName(node.graph);
							this.inputs[0].type = type;
							this.inputs[0].name = type;
							
							if (app.ui.settings.getSettingValue("KJNodes.nodeAutoColor")){
								setColorAndBgColor.call(this, type);	
							}
						} else {
                showAlert("node input undefined.")
						}
					}
					if (link_info && node.graph && slotType == 2 && isChangeConnect) {
						const fromNode = node.graph._nodes.find((otherNode) => otherNode.id == link_info.origin_id);
						
						if (fromNode && fromNode.inputs && fromNode.inputs[link_info.origin_slot]) {
							const type = fromNode.inputs[link_info.origin_slot].type;
							
							this.outputs[0].type = type;
							this.outputs[0].name = type;
						} else {
							showAlert('node output undefined');
						}
					}
					

					//Update either way
					this.update();
				}

				this.validateName = function(graph) {
					let widgetValue = node.widgets[0].value;
				
					if (widgetValue !== '') {
						let tries = 0;
						const existingValues = new Set();
				
						graph._nodes.forEach(otherNode => {
							if (otherNode !== this && otherNode.type === 'SetNode') {
								existingValues.add(otherNode.widgets[0].value);
							}
						});
				
						while (existingValues.has(widgetValue)) {
							widgetValue = node.widgets[0].value + "_" + tries;
							tries++;
						}
				
						node.widgets[0].value = widgetValue;
						this.update();
					}
				}

				this.clone = function () {
					const cloned = SetNode.prototype.clone.apply(this);
					cloned.inputs[0].name = '*';
					cloned.inputs[0].type = '*';
					cloned.value = '';
					cloned.properties.previousName = '';
					cloned.size = cloned.computeSize();
					return cloned;
				};

				this.onAdded = function(graph) {
					this.validateName(graph);
				}


				this.update = function() {
					if (!node.graph) {
						return;
					}
				
					const getters = this.findGetters(node.graph);
					getters.forEach(getter => {
						getter.setType(this.inputs[0].type);
					});
				
					if (this.widgets[0].value) {
						const gettersWithPreviousName = this.findGetters(node.graph, true);
						gettersWithPreviousName.forEach(getter => {
							getter.setName(this.widgets[0].value);
						});
					}
				
					const allGetters = node.graph._nodes.filter(otherNode => otherNode.type === "GetNode");
					allGetters.forEach(otherNode => {
						if (otherNode.setComboValues) {
							otherNode.setComboValues();
						}
					});
				}


				this.findGetters = function(graph, checkForPreviousName) {
					const name = checkForPreviousName ? this.properties.previousName : this.widgets[0].value;
					return graph._nodes.filter(otherNode => otherNode.type === 'GetNode' && otherNode.widgets[0].value === name && name !== '');
				}

				
				// This node is purely frontend and does not impact the resulting prompt so should not be serialized
				this.isVirtualNode = true;
			}
				

			onRemoved() {
				const allGetters = this.graph._nodes.filter((otherNode) => otherNode.type == "GetNode");
				allGetters.forEach((otherNode) => {
					if (otherNode.setComboValues) {
						otherNode.setComboValues([this]);
					}
				})
			}
			getExtraMenuOptions(_, options) {
				this.menuEntry = this.drawConnection ? "Hide connections" : "Show connections";
				options.unshift(
					{
						content: this.menuEntry,
						callback: () => {
							this.currentGetters = this.findGetters(this.graph);								
							if (this.currentGetters.length == 0) return;
							let linkType = (this.currentGetters[0].outputs[0].type);	
							this.slotColor = this.canvas.default_connection_color_byType[linkType]
							this.menuEntry = this.drawConnection ? "Hide connections" : "Show connections";
							this.drawConnection = !this.drawConnection;
							this.canvas.setDirty(true, true);
							
						},
						has_submenu: true,
						submenu: {
							title: "Color",
                            options: [ 
								{
								content: "Highlight",
								callback: () => {
									this.slotColor = "orange"
									this.canvas.setDirty(true, true);
									}
								}
							],
						},
					},
					{
						content: "Hide all connections",
						callback: () => {
							const allGetters = this.graph._nodes.filter(otherNode => otherNode.type === "GetNode" || otherNode.type === "SetNode");
							allGetters.forEach(otherNode => {
								otherNode.drawConnection = false;
								console.log(otherNode);
							});
							
							this.menuEntry = "Show connections";
							this.drawConnection = false
							this.canvas.setDirty(true, true);
							
						},
					
					},
				);
				// Dynamically add a submenu for all getters
				this.currentGetters = this.findGetters(this.graph);
				if (this.currentGetters) {
					
					let gettersSubmenu = this.currentGetters.map(getter => ({
						
						content: `${getter.title} id: ${getter.id}`,
						callback: () => {
							this.canvas.centerOnNode(getter);
							this.canvas.selectNode(getter, false);
							this.canvas.setDirty(true, true);
							
						},
					}));
			
					options.unshift({
						content: "Getters",
						has_submenu: true,
						submenu: {
							title: "GetNodes",
                            options: gettersSubmenu,
						}
					});
				}
			}
			
			
			onDrawForeground(ctx, lGraphCanvas) {
				if (this.drawConnection) {
					this._drawVirtualLinks(lGraphCanvas, ctx);
				}
			}
			// onDrawCollapsed(ctx, lGraphCanvas) {
			// 	if (this.drawConnection) {
			// 		this._drawVirtualLinks(lGraphCanvas, ctx);
			// 	}
			// }
			_drawVirtualLinks(lGraphCanvas, ctx) {
				if (!this.currentGetters?.length) return;
				var title = this.getTitle ? this.getTitle() : this.title;
				var title_width = ctx.measureText(title).width;
				if (!this.flags.collapsed) {
					var start_node_slotpos = [
						this.size[0],
						LiteGraph.NODE_TITLE_HEIGHT * 0.5,
						];
				}
				else {
					
					var start_node_slotpos = [
						title_width + 55,
						-15,

						];
				}
				// Provide a default link object with necessary properties, to avoid errors as link can't be null anymore
				const defaultLink = { type: 'default', color: this.slotColor };

				for (const getter of this.currentGetters) {
					if (!this.flags.collapsed) {
					var end_node_slotpos = this.getConnectionPos(false, 0);
					end_node_slotpos = [
						getter.pos[0] - end_node_slotpos[0] + this.size[0],
						getter.pos[1] - end_node_slotpos[1]
						];
					}
					else {
						var end_node_slotpos = this.getConnectionPos(false, 0);
						end_node_slotpos = [
						getter.pos[0] - end_node_slotpos[0] + title_width + 50,
						getter.pos[1] - end_node_slotpos[1] - 30
						];
					}
					lGraphCanvas.renderLink(
						ctx,
						start_node_slotpos,
						end_node_slotpos,
						defaultLink,
						false,
						null,
						this.slotColor,
						LiteGraph.RIGHT,
						LiteGraph.LEFT
					);
				}
			}
		}

		LiteGraph.registerNodeType(
			"SetNode",
			Object.assign(SetNode, {
				title: "Set",
			})
		);

		SetNode.category = "KJNodes";
	},
});

app.registerExtension({
	name: "GetNode",
	registerCustomNodes() {
		class GetNode extends LGraphNode {

			defaultVisibility = true;
			serialize_widgets = true;
			drawConnection = false;
			slotColor = "#FFF";
			currentSetter = null;
			canvas = app.canvas;

			constructor(title) {
				super(title)
				if (!this.properties) {
					this.properties = {};
				}
				this.properties.showOutputText = GetNode.defaultVisibility;
				const node = this;
				this.addWidget(
					"combo",
					"Constant",
					"",
					(e) => {
						this.onRename();
					},
					{
						values: () => {
                            const setterNodes = node.graph._nodes.filter((otherNode) => otherNode.type == 'SetNode');
                            return setterNodes.map((otherNode) => otherNode.widgets[0].value).sort();
                        }
					}
				)

				this.addOutput("*", '*');			
				this.onConnectionsChange = function(
					slotType,	//0 = output, 1 = input
					slot,	//self-explanatory
					isChangeConnect,
                    link_info,
                    output
				) {
					this.validateLinks();	
				}

				this.setName = function(name) {
					node.widgets[0].value = name;
					node.onRename();
					node.serialize();
				}
				
				this.onRename = function() {
					const setter = this.findSetter(node.graph);
					if (setter) {
						let linkType = (setter.inputs[0].type);
						
						this.setType(linkType);
						this.title = (!disablePrefix ? "Get_" : "") + setter.widgets[0].value;
						
						if (app.ui.settings.getSettingValue("KJNodes.nodeAutoColor")){
							setColorAndBgColor.call(this, linkType);	
						}

					} else {
						this.setType('*');
					}
				}

				this.clone = function () {
					const cloned = GetNode.prototype.clone.apply(this);
					cloned.size = cloned.computeSize();
					return cloned;
				};

				this.validateLinks = function() {
					if (this.outputs[0].type !== '*' && this.outputs[0].links) {
						this.outputs[0].links.filter(linkId => {
							const link = node.graph.links[linkId];
							return link && (!link.type.split(",").includes(this.outputs[0].type) && link.type !== '*');
						}).forEach(linkId => {
							node.graph.removeLink(linkId);
						});
					}
				};

				this.setType = function(type) {
					this.outputs[0].name = type;
					this.outputs[0].type = type;
					this.validateLinks();
				}

				this.findSetter = function(graph) {
					const name = this.widgets[0].value;
					const foundNode = graph._nodes.find(otherNode => otherNode.type === 'SetNode' && otherNode.widgets[0].value === name && name !== '');
					return foundNode;
				};

				this.goToSetter = function() {
					const setter = this.findSetter(this.graph);	
					this.canvas.centerOnNode(setter);
					this.canvas.selectNode(setter, false);
				};
				
				// This node is purely frontend and does not impact the resulting prompt so should not be serialized
				this.isVirtualNode = true;
			}
			
			getInputLink(slot) {
				const setter = this.findSetter(this.graph);
			
				if (setter) {
					const slotInfo = setter.inputs[slot];
					const link = this.graph.links[slotInfo.link];
					return link;
				} else {
					const errorMessage = "No SetNode found for " + this.widgets[0].value + "(" + this.type + ")";
					showAlert(errorMessage);
					//throw new Error(errorMessage);
				}
			}
			onAdded(graph) {
			}
			getExtraMenuOptions(_, options) {
				let menuEntry = this.drawConnection ? "Hide connections" : "Show connections";
				
				options.unshift(
					{
						content: "Go to setter",
						callback: () => {
							this.goToSetter();
						},
					},
					{
						content: menuEntry,
						callback: () => {
							this.currentSetter = this.findSetter(this.graph);
							if (this.currentSetter.length == 0) return;
							let linkType = (this.currentSetter.inputs[0].type);	
							this.drawConnection = !this.drawConnection;
							this.slotColor = this.canvas.default_connection_color_byType[linkType]
							menuEntry = this.drawConnection ? "Hide connections" : "Show connections";
							this.canvas.setDirty(true, true);
						},
					},
				);
			}

			onDrawForeground(ctx, lGraphCanvas) {
				if (this.drawConnection) {
					this._drawVirtualLink(lGraphCanvas, ctx);
				}
			}
			// onDrawCollapsed(ctx, lGraphCanvas) {
			// 	if (this.drawConnection) {
			// 		this._drawVirtualLink(lGraphCanvas, ctx);
			// 	}
			// }
			_drawVirtualLink(lGraphCanvas, ctx) {
				if (!this.currentSetter) return;

				// Provide a default link object with necessary properties, to avoid errors as link can't be null anymore
				const defaultLink = { type: 'default', color: this.slotColor };
				
				let start_node_slotpos = this.currentSetter.getConnectionPos(false, 0);
				start_node_slotpos = [
					start_node_slotpos[0] - this.pos[0],
					start_node_slotpos[1] - this.pos[1],
				];
				let end_node_slotpos = [0, -LiteGraph.NODE_TITLE_HEIGHT * 0.5];
				lGraphCanvas.renderLink(
					ctx,
					start_node_slotpos,
					end_node_slotpos,
					defaultLink,
					false,
					null,
					this.slotColor
				);
			}
		}

		LiteGraph.registerNodeType(
			"GetNode",
			Object.assign(GetNode, {
				title: "Get",
			})
		);

		GetNode.category = "KJNodes";
	},
});
