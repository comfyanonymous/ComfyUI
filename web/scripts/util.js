// If there are files with specific extensions in the array inside the config of nodes/inputs/widget in the workflow, empty that array.
const excludeExtensions = new Set(["png", "jpg", "webp", "jpeg", "safetensors", "ckpt", "pt", "pth"]);

function getFileExtension(filename) {
	return filename.slice((filename.lastIndexOf('.') - 1 >>> 0) + 2);
}

export class Util {
	static workflow_security_filter(workflow) {
		workflow.nodes.forEach((node) => {
			// filter for 0 weighted LoraLoader
			if(node.widgets_values && node.widgets_values.length == 3){
				let wv = node.widgets_values;
				if(typeof(wv[0]) == "string" && wv[1] == 0 && wv[2] == 0){
					if(excludeExtensions.has(getFileExtension(wv[0])))
						wv[0] = "";
				}
			}

			if (node.inputs) {
			node.inputs.forEach((input) => {
				if (input.widget && input.widget.config) {
				const configArray = input.widget.config[0];
				if (Array.isArray(configArray) && configArray.every((filename) => excludeExtensions.has(getFileExtension(filename)))) {
					input.widget.config[0] = [];
				}
				}
			});
			}
			if (node.outputs) {
			node.outputs.forEach((output) => {
				if (output.widget && output.widget.config) {
				const configArray = output.widget.config[0];
				if (Array.isArray(configArray) && configArray.every((filename) => excludeExtensions.has(getFileExtension(filename)))) {
					output.widget.config[0] = [];
				}
				}
			});
			}
		});

		return workflow;
	}
};
