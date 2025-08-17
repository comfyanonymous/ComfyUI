import { app } from "../../../../scripts/app.js";
import {removeDropdown, createDropdown} from "../common/dropdown.js";

function generateNumList(dictionary) {
  const minimum = dictionary["min"] || 0;
  const maximum = dictionary["max"] || 0;
  const step = dictionary["step"] || 1;

  if (step === 0) {
    return [];
  }

  const result = [];
  let currentValue = minimum;

  while (currentValue <= maximum) {
    if (Number.isInteger(step)) {
      result.push(Math.round(currentValue) + '; ');
    } else {
      let formattedValue = currentValue.toFixed(3);
      if(formattedValue == -0.000){
        formattedValue = '0.000';
      }
      if (!/\.\d{3}$/.test(formattedValue)) {
        formattedValue += "0";
      }
      result.push(formattedValue + "; ");
    }
    currentValue += step;
  }

  if (maximum >= 0 && minimum >= 0) {
	//low to high
	return result;
  }
  else {
	//high to low
    return result.reverse();
  }
}

let plotDict = {};
let currentOptionsDict = {};

function getCurrentOptionLists(node, widget) {
	const nodeId = String(node.id);
	const widgetName = widget.name;
	const widgetValue = widget.value.replace(/^(loader|preSampling):\s/, '');

	if (!currentOptionsDict[widgetName]) {
	  currentOptionsDict = {...currentOptionsDict, [widgetName]: plotDict[widgetValue]};
	}  else if (currentOptionsDict[widgetName] != plotDict[widgetValue]) {
	  currentOptionsDict[widgetName] = plotDict[widgetValue];
	}
}

function addGetSetters(node) {
	if (node.widgets)
		for (const w of node.widgets) {
			if (w.name === "x_axis" ||
					w.name === "y_axis") {
				let widgetValue = w.value;

				// Define getters and setters for widget values
				Object.defineProperty(w, 'value', {

					get() {
						return widgetValue;
					},
					set(newVal) {
						if (newVal !== widgetValue) {
							widgetValue = newVal;
							getCurrentOptionLists(node, w);
						}
					}
				});
			}
		}
}

function dropdownCreator(node) {
	if (node.widgets) {
		const widgets = node.widgets.filter(
			(n) => (n.type === "customtext" && n.dynamicPrompts !== false) || n.dynamicPrompts
		);

		for (const w of widgets) {
			function replaceOptionSegments(selectedOption, inputSegments, cursorSegmentIndex, optionsList) {
				if (selectedOption) {
					inputSegments[cursorSegmentIndex] = selectedOption;
				}

				return inputSegments.map(segment => verifySegment(segment, optionsList))
									 .filter(item => item !== '')
									 .join('');
			}

			function verifySegment(segment, optionsList) {
				segment = cleanSegment(segment);

				if (isInOptionsList(segment, optionsList)) {
					return segment + '; ';
				}

				let matchedOptions = findMatchedOptions(segment, optionsList);

				if (matchedOptions.length === 1 || matchedOptions.length === 2) {
					return matchedOptions[0];
				}

				if (isInOptionsList(formatNumberSegment(segment), optionsList)) {
					return formatNumberSegment(segment) + '; ';
				}

				return '';
			}

			function cleanSegment(segment) {
				return segment.replace(/(\n|;| )/g, '');
			}

			function isInOptionsList(segment, optionsList) {
				return optionsList.includes(segment + '; ');
			}

			function findMatchedOptions(segment, optionsList) {
				return optionsList.filter(option => option.toLowerCase().includes(segment.toLowerCase()));
			}

			function formatNumberSegment(segment) {
				if (Number(segment)) {
					return Number(segment).toFixed(3);
				}

				if (['0', '0.', '0.0', '0.00', '00'].includes(segment)) {
					return '0.000';
				}
				return segment;
			}


			const onInput = function () {
				const axisWidgetName = w.name[0] + '_axis';
				let optionsList = currentOptionsDict?.[axisWidgetName] || [];
				if (optionsList.length === 0) {return}

				const inputText = w.inputEl.value;
				const cursorPosition = w.inputEl.selectionStart;
				let inputSegments = inputText.split('; ');

				const cursorSegmentIndex = inputText.substring(0, cursorPosition).split('; ').length - 1;
				const currentSegment = inputSegments[cursorSegmentIndex];
				const currentSegmentLower = currentSegment.replace(/\n/g, '').toLowerCase();
				const filteredOptionsList = optionsList.filter(option => option.toLowerCase().includes(currentSegmentLower)).map(option => option.replace(/; /g, ''));

				if (filteredOptionsList.length > 0) {
					createDropdown(w.inputEl, filteredOptionsList, (selectedOption) => {
						const verifiedText = replaceOptionSegments(selectedOption, inputSegments, cursorSegmentIndex, optionsList);
						w.inputEl.value = verifiedText;
					});
				}
				else {
					removeDropdown();
					const verifiedText = replaceOptionSegments(null, inputSegments, cursorSegmentIndex, optionsList);
					w.inputEl.value = verifiedText;
				}
			};

			w.inputEl.removeEventListener('input', onInput);
			w.inputEl.addEventListener('input', onInput);
			w.inputEl.removeEventListener('mouseup', onInput);
			w.inputEl.addEventListener('mouseup', onInput);
		}
	}
}

app.registerExtension({
	name: "comfy.easy.xyPlot",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "easy XYPlot") {
			plotDict = nodeData.input.hidden.plot_dict[0];

			for (const key in plotDict) {
				const value = plotDict[key];
				if (Array.isArray(value)) {
					let updatedValues = [];
					for (const v of value) {
						updatedValues.push(v + '; ');
					}
					plotDict[key] = updatedValues;
				} else if (typeof(value) === 'object') {
					if(key == 'seed'){
						plotDict[key] = value + '; ';
					}
					else {
						plotDict[key] = generateNumList(value);
					}
				} else {
					plotDict[key] = value + '; ';
				}
			}
			plotDict["None"] = [];
			plotDict["---------------------"] = [];
		}
	},
	nodeCreated(node) {
		if (node.comfyClass === "easy XYPlot") {
			addGetSetters(node);
			dropdownCreator(node);
		}
	}
});