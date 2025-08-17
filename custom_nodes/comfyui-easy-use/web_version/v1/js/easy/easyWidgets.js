import { app } from "../../../../scripts/app.js";
import { ComfyWidgets } from "../../../../scripts/widgets.js";

const KEY_CODES = { ENTER: 13, ESC: 27, ARROW_DOWN: 40, ARROW_UP: 38 };
const WIDGET_GAP = -4;

function hideInfoWidget(e, node, widget) {
    let dropdownShouldBeRemoved = false;
    let selectionIndex = -1;

    if (e) {
        e.preventDefault();
        e.stopPropagation();
        displayDropdown(widget);
    } else {
        hideWidget(widget, node);
    }

    function createDropdownElement() {
        const dropdown = document.createElement('ul');
        dropdown.id = 'hideinfo-dropdown';
        dropdown.setAttribute('role', 'listbox');
        dropdown.classList.add('hideInfo-dropdown');
        return dropdown;
    }

    function createDropdownItem(textContent, action) {
        const listItem = document.createElement('li');
        listItem.id = `hideInfo-item-${textContent.replace(/ /g, '')}`;
        listItem.classList.add('hideInfo-item');
        listItem.setAttribute('role', 'option');
        listItem.textContent = textContent;
        listItem.addEventListener('mousedown', (event) => {
            event.preventDefault();
            action(widget, node); // perform the action when dropdown item is clicked
            removeDropdown();
            dropdownShouldBeRemoved = false;
        });
        listItem.dataset.action = textContent.replace(/ /g, ''); // store the action in a data attribute
        return listItem;
    }

    function displayDropdown(widget) {
        removeDropdown();

        const dropdown = createDropdownElement();
        const listItemHide = createDropdownItem('Hide info Widget', hideWidget);
        const listItemHideAll = createDropdownItem('Hide for all of this node-type', hideWidgetForNodetype);

        dropdown.appendChild(listItemHide);
        dropdown.appendChild(listItemHideAll);

        const inputRect = widget.inputEl.getBoundingClientRect();
        dropdown.style.top = `${inputRect.top + inputRect.height}px`;
        dropdown.style.left = `${inputRect.left}px`;
        dropdown.style.width = `${inputRect.width}px`;

        document.body.appendChild(dropdown);
        dropdownShouldBeRemoved = true;

        widget.inputEl.removeEventListener('keydown', handleKeyDown);
        widget.inputEl.addEventListener('keydown', handleKeyDown);
        document.addEventListener('click', handleDocumentClick);
    }

    function removeDropdown() {
        const dropdown = document.getElementById('hideinfo-dropdown');
        if (dropdown) {
            dropdown.remove();
            widget.inputEl.removeEventListener('keydown', handleKeyDown);
        }
        document.removeEventListener('click', handleDocumentClick);
    
    }

    function handleKeyDown(event) {
        const dropdownItems = document.querySelectorAll('.hideInfo-item');

        if (event.keyCode === KEY_CODES.ENTER && dropdownShouldBeRemoved) {
            event.preventDefault();
            if (selectionIndex !== -1) {
                const selectedAction = dropdownItems[selectionIndex].dataset.action;
                if (selectedAction === 'HideinfoWidget') {
                    hideWidget(widget, node);
                } else if (selectedAction === 'Hideforall') {
                    hideWidgetForNodetype(widget, node);
                }
                removeDropdown();
                dropdownShouldBeRemoved = false;
            }
        } else if (event.keyCode === KEY_CODES.ARROW_DOWN && dropdownShouldBeRemoved) {
            event.preventDefault();
            if (selectionIndex !== -1) {
                dropdownItems[selectionIndex].classList.remove('selected');
            }
            selectionIndex = (selectionIndex + 1) % dropdownItems.length;
            dropdownItems[selectionIndex].classList.add('selected');
        } else if (event.keyCode === KEY_CODES.ARROW_UP && dropdownShouldBeRemoved) {
            event.preventDefault();
            if (selectionIndex !== -1) {
                dropdownItems[selectionIndex].classList.remove('selected');
            }
            selectionIndex = (selectionIndex - 1 + dropdownItems.length) % dropdownItems.length;
            dropdownItems[selectionIndex].classList.add('selected');
        } else if (event.keyCode === KEY_CODES.ESC && dropdownShouldBeRemoved) {
            event.preventDefault();
            removeDropdown();
        }
    }

    function hideWidget(widget, node) {
        node.properties['infoWidgetHidden'] = true;
        widget.type = "esayHidden";
        widget.computeSize = () => [0, WIDGET_GAP];
        node.setSize([node.size[0], node.size[1]]);
    }

    function hideWidgetForNodetype(widget, node) {
        hideWidget(widget, node)
        const hiddenNodeTypes = JSON.parse(localStorage.getItem('hiddenWidgetNodeTypes') || "[]");
        if (!hiddenNodeTypes.includes(node.constructor.type)) {
            hiddenNodeTypes.push(node.constructor.type);
        }
        localStorage.setItem('hiddenWidgetNodeTypes', JSON.stringify(hiddenNodeTypes));
    }

    function handleDocumentClick(event) {
        const dropdown = document.getElementById('hideinfo-dropdown');

        // If the click was outside the dropdown and the dropdown should be removed, remove it
        if (dropdown && !dropdown.contains(event.target) && dropdownShouldBeRemoved) {
            removeDropdown();
            dropdownShouldBeRemoved = false;
        }
    }
}


var styleElement = document.createElement("style");
const cssCode = `
.easy-info_widget {
	background-color: var(--comfy-input-bg);
	color: var(--input-text);
	overflow: hidden;
	padding: 2px;
	resize: none;
	border: none;
	box-sizing: border-box;
	font-size: 10px;
	border-radius: 7px;
	text-align: center;
	text-wrap: balance;
}
.hideInfo-dropdown {
	position: absolute;
	box-sizing: border-box;
	background-color: #121212;
	border-radius: 7px;
	box-shadow: 0 2px 4px rgba(255, 255, 255, .25);
	padding: 0;
	margin: 0;
	list-style: none;
	z-index: 1000;
	overflow: auto;
	max-height: 200px;
}
	
.hideInfo-dropdown li {
	padding: 4px 10px;
	cursor: pointer;
	font-family: system-ui;
	font-size: 0.7rem;
}
	
.hideInfo-dropdown li:hover,
.hideInfo-dropdown li.selected {
	background-color: #e5e5e5;
	border-radius: 7px;
}
`
styleElement.innerHTML = cssCode
document.head.appendChild(styleElement);

const InfoSymbol = Symbol();
const InfoResizeSymbol = Symbol();




// WIDGET FUNCTIONS
function addInfoWidget(node, name, opts, app) {
	const INFO_W_SIZE = 50;

    node.addProperty('infoWidgetHidden', false)

	function computeSize(size) {
		if (node.widgets[0].last_y == null) return;
	
		let y = node.widgets[0].last_y;
	
		// Compute the height of all non easyInfo widgets
		let widgetHeight = 0;
		const infoWidges = [];
		for (let i = 0; i < node.widgets.length; i++) {
			const w = node.widgets[i];
			if (w.type === "easyInfo") {
				infoWidges.push(w);
			} else {
				if (w.computeSize) {
					widgetHeight += w.computeSize()[1] + 4;
				} else {
					widgetHeight += LiteGraph.NODE_WIDGET_HEIGHT + 4;
				}
			}
		}
	
		let infoWidgetSpace = infoWidges.length * INFO_W_SIZE; // Height for all info widgets
	
		// Check if there's enough space for all widgets
		if (size[1] < y + widgetHeight + infoWidgetSpace) {
			// There isn't enough space for all the widgets, increase the size of the node
			node.size[1] = y + widgetHeight + infoWidgetSpace;
			node.graph.setDirtyCanvas(true);
		}
	
		// Position each of the widgets
		for (const w of node.widgets) {
			w.y = y;
			if (w.type === "easyInfo") {
				y += INFO_W_SIZE;
			} else if (w.computeSize) {
				y += w.computeSize()[1] + 4;
			} else {
				y += LiteGraph.NODE_WIDGET_HEIGHT + 4;
			}
		}
	}
	
	const widget = {
		type: "easyInfo",
		name,
		get value() {
			return this.inputEl.value;
		},
		set value(x) {
			this.inputEl.value = x;
		},
		draw: function (ctx, _, widgetWidth, y, widgetHeight) {
			if (!this.parent.inputHeight) {
				// If we are initially offscreen when created we wont have received a resize event
				// Calculate it here instead
				computeSize(node.size);
			}
			const visible = app.canvas.ds.scale > 0.5 && this.type === "easyInfo";
			const margin = 10;
			const elRect = ctx.canvas.getBoundingClientRect();
			const transform = new DOMMatrix()
				.scaleSelf(elRect.width / ctx.canvas.width, elRect.height / ctx.canvas.height)
				.multiplySelf(ctx.getTransform())
				.translateSelf(margin, margin + y);

			Object.assign(this.inputEl.style, {
				transformOrigin: "0 0",
				transform: transform,
				left: "0px",
				top: "0px",
				width: `${widgetWidth - (margin * 2)}px`,
				height: `${this.parent.inputHeight - (margin * 2)}px`,
				position: "absolute",
				background: (!node.color)?'':node.color,
				color: (!node.color)?'':'white',
				zIndex: app.graph._nodes.indexOf(node),
			});
			this.inputEl.hidden = !visible;
		},
	};
	widget.inputEl = document.createElement("textarea");
	widget.inputEl.className = "easy-info_widget";
	widget.inputEl.value = opts.defaultVal;
	widget.inputEl.placeholder = opts.placeholder || "";
	widget.inputEl.readOnly = true;
	widget.parent = node;

	document.body.appendChild(widget.inputEl);

	node.addCustomWidget(widget);

	app.canvas.onDrawBackground = function () {
		// Draw node isnt fired once the node is off the screen
		// if it goes off screen quickly, the input may not be removed
		// this shifts it off screen so it can be moved back if the node is visible.
		for (let n in app.graph._nodes) {
			n = app.graph._nodes[n];
			for (let w in n.widgets) {
				let wid = n.widgets[w];
				if (Object.hasOwn(wid, "inputEl")) {
					wid.inputEl.style.left = -8000 + "px";
					wid.inputEl.style.position = "absolute";
				}
			}
		}
	};

	node.onRemoved = function () {
		// When removing this node we need to remove the input from the DOM
		for (let y in this.widgets) {
			if (this.widgets[y].inputEl) {
				this.widgets[y].inputEl.remove();
			}
		}
	};

	widget.onRemove = () => {
		widget.inputEl?.remove();

		// Restore original size handler if we are the last
		if (!--node[InfoSymbol]) {
			node.onResize = node[InfoResizeSymbol];
			delete node[InfoSymbol];
			delete node[InfoResizeSymbol];
		}
	};

	if (node[InfoSymbol]) {
		node[InfoSymbol]++;
	} else {
		node[InfoSymbol] = 1;
		const onResize = (node[InfoResizeSymbol] = node.onResize);

		node.onResize = function (size) {
			computeSize(size);

			// Call original resizer handler
			if (onResize) {
				console.log(this, arguments)
				onResize.apply(this, arguments);
			}
		};
	}

	return { widget };
}

// WIDGETS
const easyCustomWidgets = {
	INFO(node, inputName, inputData, app) {
		const defaultVal = inputData[1].default || "";
		return addInfoWidget(node, inputName, { defaultVal, ...inputData[1] }, app);
	},
}



app.registerExtension({
    name: "comfy.easy.widgets",
    getCustomWidgets(app) {
        return easyCustomWidgets;
    },
    nodeCreated(node) {
        if (node.widgets) {
            // Locate info widgets
            const widgets = node.widgets.filter((n) => (n.type === "easyInfo"));
            for (const widget of widgets) {
                    widget.inputEl.addEventListener('contextmenu', function(e) {
                        hideInfoWidget(e, node, widget);
                    });
                    widget.inputEl.addEventListener('click', function(e) {
                        hideInfoWidget(e, node, widget);
                    });
            }
        }
    },
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        const hiddenNodeTypes = JSON.parse(localStorage.getItem('hiddenWidgetNodeTypes') || "[]");
        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = origOnConfigure ? origOnConfigure.apply(this, arguments) : undefined;
            if (this.properties['infoWidgetHidden']) {
                for (let i in this.widgets) {
                    if (this.widgets[i].type == "easyInfo") {
                        hideInfoWidget(null, this, this.widgets[i]);
                    }
                }
            }
            return r;
        };
        const origOnAdded = nodeType.prototype.onAdded;
        nodeType.prototype.onAdded = function () {
            const r = origOnAdded ? origOnAdded.apply(this, arguments) : undefined;
            if (hiddenNodeTypes.includes(this.type)) {
                for (let i in this.widgets) {
                    if (this.widgets[i].type == "easyInfo") {
                        this.properties['infoWidgetHidden'] = true;
                    }
                }
            }
            return r;
        }
    }
}); 