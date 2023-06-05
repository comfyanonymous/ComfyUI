// If there are files with specific extensions in the array inside the config of nodes/inputs/widget in the workflow, empty that array.
const excludeExtensions = new Set(["png", "jpg", "webp", "jpeg", "safetensors", "ckpt", "pt", "pth"]);

function getFileExtension(filename) {
  return filename.slice((filename.lastIndexOf('.') - 1 >>> 0) + 2);
}

export class Util {
	static workflow_security_filter(workflow) {
	  workflow.nodes.forEach((node) => {
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
