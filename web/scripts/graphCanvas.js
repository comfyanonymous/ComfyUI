import { LGraphCanvas } from "../lib/litegraph.core.js"

export default class ComfyGraphCanvas extends LGraphCanvas {
  processKey(e) {
	const res = super.processKey(e);

	if (res === false) {
	  return res;
	}

	if (!this.graph) {
	  return;
	}

	var block_default = false;

	if (e.target.localName == "input") {
	  return;
	}

	if (e.type == "keydown") {
	  // Ctrl + M mute/unmute
	  if (e.keyCode == 77 && e.ctrlKey) {
		if (this.selected_nodes) {
		  for (var i in this.selected_nodes) {
			if (this.selected_nodes[i].mode === 2) { // never
			  this.selected_nodes[i].mode = 0; // always
			} else {
			  this.selected_nodes[i].mode = 2; // never
			}
		  }
		}
		block_default = true;
	  }

	  if (e.keyCode == 66 && e.ctrlKey) {
		if (this.selected_nodes) {
		  for (var i in this.selected_nodes) {
			if (this.selected_nodes[i].mode === 4) { // never
			  this.selected_nodes[i].mode = 0; // always
			} else {
			  this.selected_nodes[i].mode = 4; // never
			}
		  }
		}
		block_default = true;
	  }
	}

	this.graph.change();

	if (block_default) {
	  e.preventDefault();
	  e.stopImmediatePropagation();
	  return false;
	}

	return res;
  }

  processMouseDown(e) {
    const res = super.processMouseDown(e);

	this.selected_group_moving = false;

	if (this.selected_group && !this.selected_group_resizing) {
	  var font_size =
		  this.selected_group.font_size || LiteGraph.DEFAULT_GROUP_FONT_SIZE;
	  var height = font_size * 1.4;

	  // Move group by header
	  if (LiteGraph.isInsideRectangle(e.canvasX, e.canvasY, this.selected_group.pos[0], this.selected_group.pos[1], this.selected_group.size[0], height)) {
		this.selected_group_moving = true;
	  }
	}

	return res;
  }

  processMouseMove(e) {
	const orig_selected_group = this.selected_group;

	if (this.selected_group && !this.selected_group_resizing && !this.selected_group_moving) {
	  this.selected_group = null;
	}

	const res = super.processMouseMove(e);

	if (orig_selected_group && !this.selected_group_resizing && !this.selected_group_moving) {
	  this.selected_group = orig_selected_group;
	}

	return res;
  }

  /**
   * Draws group header bar
   */
  drawGroups(canvas, ctx) {
    if (!this.graph) {
      return;
    }

	var groups = this.graph._groups;

	ctx.save();
	ctx.globalAlpha = 0.7 * this.editor_alpha;

	for (var i = 0; i < groups.length; ++i) {
	  var group = groups[i];

	  if (!LiteGraph.overlapBounding(this.visible_area, group._bounding)) {
		continue;
	  } //out of the visible area

	  ctx.fillStyle = group.color || "#335";
	  ctx.strokeStyle = group.color || "#335";
	  var pos = group._pos;
	  var size = group._size;
	  ctx.globalAlpha = 0.25 * this.editor_alpha;
	  ctx.beginPath();
	  var font_size =
		  group.font_size || LiteGraph.DEFAULT_GROUP_FONT_SIZE;
	  ctx.rect(pos[0] + 0.5, pos[1] + 0.5, size[0], font_size * 1.4);
	  ctx.fill();
	  ctx.globalAlpha = this.editor_alpha;
	}

	ctx.restore();

	  const res = super.drawGroups(canvas, ctx);
	return res;
  }

  /**
   * Draws node highlights (executing, drag drop) and progress bar
   */
  drawNodeShape(node, ctx, size, fgcolor, bgcolor, selected, mouse_over) {
    const res = super.drawNodeShape(node, ctx, size, fgcolor, bgcolor, selected, mouse_over);

	const nodeErrors = self.lastNodeErrors?.[node.id];

	let color = null;
	let lineWidth = 1;
	if (node.id === +self.runningNodeId) {
	  color = "#0f0";
	} else if (self.dragOverNode && node.id === self.dragOverNode.id) {
	  color = "dodgerblue";
	}
	else if (nodeErrors?.errors) {
	  color = "red";
	  lineWidth = 2;
	}
	else if (self.lastExecutionError && +self.lastExecutionError.node_id === node.id) {
	  color = "#f0f";
	  lineWidth = 2;
	}

	if (color) {
	  const shape = node._shape || node.constructor.shape || LiteGraph.ROUND_SHAPE;
	  ctx.lineWidth = lineWidth;
	  ctx.globalAlpha = 0.8;
	  ctx.beginPath();
	  if (shape == LiteGraph.BOX_SHAPE)
		ctx.rect(-6, -6 - LiteGraph.NODE_TITLE_HEIGHT, 12 + size[0] + 1, 12 + size[1] + LiteGraph.NODE_TITLE_HEIGHT);
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

	if (self.progress && node.id === +self.runningNodeId) {
	  ctx.fillStyle = "green";
	  ctx.fillRect(0, 0, size[0] * (self.progress.value / self.progress.max), 6);
	  ctx.fillStyle = bgcolor;
	}

	// Highlight inputs that failed validation
	if (nodeErrors) {
	  ctx.lineWidth = 2;
	  ctx.strokeStyle = "red";
	  for (const error of nodeErrors.errors) {
		if (error.extra_info && error.extra_info.input_name) {
		  const inputIndex = node.findInputSlot(error.extra_info.input_name)
		  if (inputIndex !== -1) {
			let pos = node.getConnectionPos(true, inputIndex);
			ctx.beginPath();
			ctx.arc(pos[0] - node.pos[0], pos[1] - node.pos[1], 12, 0, 2 * Math.PI, false)
			ctx.stroke();
		  }
		}
	  }
	}

	return res;
  }

  drawNode(node, ctx) {
	var editor_alpha = this.editor_alpha;
	var old_color = node.bgcolor;

	if (node.mode === 2) { // never
	  this.editor_alpha = 0.4;
	}

	if (node.mode === 4) { // never
	  node.bgcolor = "#FF00FF";
	  this.editor_alpha = 0.2;
	}

	  const res = super.drawNode(node, ctx);

	this.editor_alpha = editor_alpha;
	node.bgcolor = old_color;

	return res;
  }
}
