import { app } from "../../scripts/app.js";

// Allows for simple dynamic prompt replacement
// Inputs in the format {a|b} will have a random value of a or b chosen when the prompt is queued.

/*
 * Strips C-style line and block comments from a string
 */
function stripComments(str) {
	return str.replace(/\/\*[\s\S]*?\*\/|\/\/.*/g,'');
}

app.registerExtension({
	name: "Comfy.DynamicPrompts",
	nodeCreated(node) {
		if (node.widgets) {
			// Locate dynamic prompt text widgets
			// Include any widgets with dynamicPrompts set to true, and customtext
			const widgets = node.widgets.filter(
				(n) => n.dynamicPrompts
			);
			for (const widget of widgets) {
				// Override the serialization of the value to resolve dynamic prompts for all widgets supporting it in this node
				widget.serializeValue = (workflowNode, widgetIndex) => {
					let prompt = stripComments(widget.value);
					while (prompt.replace("\\{", "").includes("{") && prompt.replace("\\}", "").includes("}")) {
						const startIndex = prompt.replace("\\{", "00").indexOf("{");
						const endIndex = prompt.replace("\\}", "00").indexOf("}");

						const optionsString = prompt.substring(startIndex + 1, endIndex);
						const options = optionsString.split("|");

						const randomIndex = Math.floor(Math.random() * options.length);
						const randomOption = options[randomIndex];

						prompt = prompt.substring(0, startIndex) + randomOption + prompt.substring(endIndex + 1);
					}

					// Overwrite the value in the serialized workflow pnginfo
					if (workflowNode?.widgets_values)
						workflowNode.widgets_values[widgetIndex] = prompt;

					return prompt;
				};
			}
		}
	},
});
