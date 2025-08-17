import { app } from "../../../scripts/app.js";
import { applyTextReplacements } from "../../../scripts/utils.js";

app.registerExtension({
	name: "KJNodes.jsnodes",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if(!nodeData?.category?.startsWith("KJNodes")) {
			return;
		  }
		switch (nodeData.name) {
			case "ConditioningMultiCombine":
				nodeType.prototype.onNodeCreated = function () {
				this._type = "CONDITIONING"
				this.inputs_offset = nodeData.name.includes("selective")?1:0
				this.addWidget("button", "Update inputs", null, () => {
					if (!this.inputs) {
						this.inputs = [];
					}
					const target_number_of_inputs = this.widgets.find(w => w.name === "inputcount")["value"];
					const num_inputs = this.inputs.filter(input => input.type === this._type).length
					if(target_number_of_inputs===num_inputs)return; // already set, do nothing

					if(target_number_of_inputs < num_inputs){
						const inputs_to_remove = num_inputs - target_number_of_inputs;
						for(let i = 0; i < inputs_to_remove; i++) {
							this.removeInput(this.inputs.length - 1);
						}
					}
					else{
						for(let i = num_inputs+1; i <= target_number_of_inputs; ++i)
							this.addInput(`conditioning_${i}`, this._type)
					}
					});
				}
				break;
			case "ImageBatchMulti":
			case "ImageAddMulti":
			case "ImageConcatMulti":
			case "CrossFadeImagesMulti":
			case "TransitionImagesMulti":
				nodeType.prototype.onNodeCreated = function () {
				this._type = "IMAGE"
				this.addWidget("button", "Update inputs", null, () => {
					if (!this.inputs) {
						this.inputs = [];
					}
					const target_number_of_inputs = this.widgets.find(w => w.name === "inputcount")["value"];
					const num_inputs = this.inputs.filter(input => input.type === this._type).length
					if(target_number_of_inputs===num_inputs)return; // already set, do nothing

					if(target_number_of_inputs < num_inputs){
						const inputs_to_remove = num_inputs - target_number_of_inputs;
						for(let i = 0; i < inputs_to_remove; i++) {
							this.removeInput(this.inputs.length - 1);
						}
					}
					else{
						for(let i = num_inputs+1; i <= target_number_of_inputs; ++i)
							this.addInput(`image_${i}`, this._type)
					}
					
					});
				}
				break;
			case "MaskBatchMulti":
				nodeType.prototype.onNodeCreated = function () {
				this._type = "MASK"
				this.addWidget("button", "Update inputs", null, () => {
					if (!this.inputs) {
						this.inputs = [];
					}
					const target_number_of_inputs = this.widgets.find(w => w.name === "inputcount")["value"];
					const num_inputs = this.inputs.filter(input => input.type === this._type).length
					if(target_number_of_inputs===num_inputs)return; // already set, do nothing

					if(target_number_of_inputs < num_inputs){
						const inputs_to_remove = num_inputs - target_number_of_inputs;
						for(let i = 0; i < inputs_to_remove; i++) {
							this.removeInput(this.inputs.length - 1);
						}
					}
					else{
						for(let i = num_inputs+1; i <= target_number_of_inputs; ++i)
							this.addInput(`mask_${i}`, this._type)
					}
					});
					}
					break;
			
			case "FluxBlockLoraSelect":
			case "HunyuanVideoBlockLoraSelect":
			case "Wan21BlockLoraSelect":
				nodeType.prototype.onNodeCreated = function () {
					this.addWidget("button", "Set all", null, () => {
						const userInput = prompt("Enter the values to set for widgets (e.g., s0,1,2-7=2.0, d0,1,2-7=2.0, or 1.0):", "");
						if (userInput) {
							const regex = /([sd])?(\d+(?:,\d+|-?\d+)*?)?=(\d+(\.\d+)?)/;
							const match = userInput.match(regex);
							if (match) {
								const type = match[1];
								const indicesPart = match[2];
								const value = parseFloat(match[3]);
			
								let targetWidgets = [];
								if (type === 's') {
									targetWidgets = this.widgets.filter(widget => widget.name.includes("single"));
								} else if (type === 'd') {
									targetWidgets = this.widgets.filter(widget => widget.name.includes("double"));
								} else {
									targetWidgets = this.widgets; // No type specified, all widgets
								}
			
								if (indicesPart) {
									const indices = indicesPart.split(',').flatMap(part => {
										if (part.includes('-')) {
											const [start, end] = part.split('-').map(Number);
											return Array.from({ length: end - start + 1 }, (_, i) => start + i);
										}
										return Number(part);
									});
			
									for (const index of indices) {
										if (index < targetWidgets.length) {
											targetWidgets[index].value = value;
										}
									}
								} else {
									// No indices provided, set value for all target widgets
									for (const widget of targetWidgets) {
										widget.value = value;
									}
								}
							} else if (!isNaN(parseFloat(userInput))) {
								// Single value provided, set it for all widgets
								const value = parseFloat(userInput);
								for (const widget of this.widgets) {
									widget.value = value;
								}
							} else {
								alert("Invalid input format. Please use the format s0,1,2-7=2.0, d0,1,2-7=2.0, or 1.0");
							}
						} else {
							alert("Invalid input. Please enter a value.");
						}
					});
				};
				break;

			case "GetMaskSizeAndCount":
				const onGetMaskSizeConnectInput = nodeType.prototype.onConnectInput;
				nodeType.prototype.onConnectInput = function (targetSlot, type, output, originNode, originSlot) {
					const v = onGetMaskSizeConnectInput? onGetMaskSizeConnectInput.apply(this, arguments): undefined
					this.outputs[1]["label"] = "width"
					this.outputs[2]["label"] = "height" 
					this.outputs[3]["label"] = "count"
					return v;
				}
				const onGetMaskSizeExecuted = nodeType.prototype.onAfterExecuteNode;
				nodeType.prototype.onExecuted = function(message) {
					const r = onGetMaskSizeExecuted? onGetMaskSizeExecuted.apply(this,arguments): undefined
					let values = message["text"].toString().split('x').map(Number);
					this.outputs[1]["label"] = values[1] + " width"
					this.outputs[2]["label"] = values[2] + " height" 
					this.outputs[3]["label"] = values[0] + " count" 
					return r
				}
				break;
			
			case "GetImageSizeAndCount":
				const onGetImageSizeConnectInput = nodeType.prototype.onConnectInput;
				nodeType.prototype.onConnectInput = function (targetSlot, type, output, originNode, originSlot) {
					console.log(this)
					const v = onGetImageSizeConnectInput? onGetImageSizeConnectInput.apply(this, arguments): undefined
					//console.log(this)
					this.outputs[1]["label"] = "width"
					this.outputs[2]["label"] = "height" 
					this.outputs[3]["label"] = "count"
					return v;
				}
				//const onGetImageSizeExecuted = nodeType.prototype.onExecuted;
				const onGetImageSizeExecuted = nodeType.prototype.onAfterExecuteNode;
				nodeType.prototype.onExecuted = function(message) {
					console.log(this)
					const r = onGetImageSizeExecuted? onGetImageSizeExecuted.apply(this,arguments): undefined
					let values = message["text"].toString().split('x').map(Number);
					console.log(values)
					this.outputs[1]["label"] = values[1] + " width"
					this.outputs[2]["label"] = values[2] + " height" 
					this.outputs[3]["label"] = values[0] + " count" 
					return r
				}
				break;

			case "PreviewAnimation":
				const onPreviewAnimationConnectInput = nodeType.prototype.onConnectInput;
				nodeType.prototype.onConnectInput = function (targetSlot, type, output, originNode, originSlot) {
					const v = onPreviewAnimationConnectInput? onPreviewAnimationConnectInput.apply(this, arguments): undefined
					this.title = "Preview Animation"
					return v;
				}
				const onPreviewAnimationExecuted = nodeType.prototype.onAfterExecuteNode;
				nodeType.prototype.onExecuted = function(message) {
					const r = onPreviewAnimationExecuted? onPreviewAnimationExecuted.apply(this,arguments): undefined
					let values = message["text"].toString();
					this.title = "Preview Animation " + values
					return r
				}
				break;

			case "VRAM_Debug":
				const onVRAM_DebugConnectInput = nodeType.prototype.onConnectInput;
				nodeType.prototype.onConnectInput = function (targetSlot, type, output, originNode, originSlot) {
					const v = onVRAM_DebugConnectInput? onVRAM_DebugConnectInput.apply(this, arguments): undefined
					this.outputs[3]["label"] = "freemem_before"
					this.outputs[4]["label"] = "freemem_after" 
					return v;
				}
				const onVRAM_DebugExecuted = nodeType.prototype.onAfterExecuteNode;
				nodeType.prototype.onExecuted = function(message) {
					const r = onVRAM_DebugExecuted? onVRAM_DebugExecuted.apply(this,arguments): undefined
					let values = message["text"].toString().split('x');
					this.outputs[3]["label"] = values[0] + "   freemem_before"
					this.outputs[4]["label"] = values[1] + "      freemem_after" 
					return r
				}
				break;

			case "JoinStringMulti":
				const originalOnNodeCreated = nodeType.prototype.onNodeCreated || function() {};
				nodeType.prototype.onNodeCreated = function () {
					originalOnNodeCreated.apply(this, arguments);
			
					this._type = "STRING";
					this.addWidget("button", "Update inputs", null, () => {
						if (!this.inputs) {
							this.inputs = [];
						}
						const target_number_of_inputs = this.widgets.find(w => w.name === "inputcount")["value"];
						const num_inputs = this.inputs.filter(input => input.name && input.name.toLowerCase().includes("string_")).length
						if (target_number_of_inputs === num_inputs) return; // already set, do nothing
			
						if(target_number_of_inputs < num_inputs){
							const inputs_to_remove = num_inputs - target_number_of_inputs;
							for(let i = 0; i < inputs_to_remove; i++) {
								this.removeInput(this.inputs.length - 1);
							}
						}
						else{
							for(let i = num_inputs+1; i <= target_number_of_inputs; ++i)
								this.addInput(`string_${i}`, this._type)
						}
					});
				}
				break;
			case "SoundReactive":
				nodeType.prototype.onNodeCreated = function () {
					let audioContext;
					let microphoneStream;
					let animationFrameId;
					let analyser;
					let dataArray;
					let startRangeHz;
    				let endRangeHz;
					let smoothingFactor = 0.5;
					let smoothedSoundLevel = 0;
				
					// Function to update the widget value in real-time
					const updateWidgetValueInRealTime = () => {
						// Ensure analyser and dataArray are defined before using them
						if (analyser && dataArray) {
							analyser.getByteFrequencyData(dataArray);

							const startRangeHzWidget = this.widgets.find(w => w.name === "start_range_hz");
							if (startRangeHzWidget) startRangeHz = startRangeHzWidget.value;
							const endRangeHzWidget = this.widgets.find(w => w.name === "end_range_hz");
							if (endRangeHzWidget) endRangeHz = endRangeHzWidget.value;
							const smoothingFactorWidget = this.widgets.find(w => w.name === "smoothing_factor");
							if (smoothingFactorWidget) smoothingFactor = smoothingFactorWidget.value;

							// Calculate frequency bin width (frequency resolution)
							const frequencyBinWidth = audioContext.sampleRate / analyser.fftSize;	
							// Convert the widget values from Hz to indices
							const startRangeIndex = Math.floor(startRangeHz / frequencyBinWidth);
							const endRangeIndex = Math.floor(endRangeHz / frequencyBinWidth);

							// Function to calculate the average value for a frequency range
							const calculateAverage = (start, end) => {
								const sum = dataArray.slice(start, end).reduce((acc, val) => acc + val, 0);
								const average = sum / (end - start);

								// Apply exponential moving average smoothing
    							smoothedSoundLevel = (average * (1 - smoothingFactor)) + (smoothedSoundLevel * smoothingFactor);
								return smoothedSoundLevel;
							};
							// Calculate the average levels for each frequency range
							const soundLevel = calculateAverage(startRangeIndex, endRangeIndex);
							
							// Update the widget values

							const lowLevelWidget = this.widgets.find(w => w.name === "sound_level");
							if (lowLevelWidget) lowLevelWidget.value = soundLevel;

							animationFrameId = requestAnimationFrame(updateWidgetValueInRealTime);
						}
					};
				
					// Function to start capturing audio from the microphone
					const startMicrophoneCapture = () => {
						// Only create the audio context and analyser once
						if (!audioContext) {
							audioContext = new (window.AudioContext || window.webkitAudioContext)();
							// Access the sample rate of the audio context
							console.log(`Sample rate: ${audioContext.sampleRate}Hz`);
							analyser = audioContext.createAnalyser();
							analyser.fftSize = 2048;
							dataArray = new Uint8Array(analyser.frequencyBinCount);
							// Get the range values from widgets (assumed to be in Hz)
							const lowRangeWidget = this.widgets.find(w => w.name === "low_range_hz");
							if (lowRangeWidget) startRangeHz = lowRangeWidget.value;
				
							const midRangeWidget = this.widgets.find(w => w.name === "mid_range_hz");
							if (midRangeWidget) endRangeHz = midRangeWidget.value;
						}
						
						navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
							microphoneStream = stream;
							const microphone = audioContext.createMediaStreamSource(stream);
							microphone.connect(analyser);
							updateWidgetValueInRealTime();
						}).catch(error => {
							console.error('Access to microphone was denied or an error occurred:', error);
						});
					};
				
					// Function to stop capturing audio from the microphone
					const stopMicrophoneCapture = () => {
						if (animationFrameId) {
							cancelAnimationFrame(animationFrameId);
						}
						if (microphoneStream) {
							microphoneStream.getTracks().forEach(track => track.stop());
						}
						if (audioContext) {
							audioContext.close();
							// Reset audioContext to ensure it can be created again when starting
							audioContext = null;
						}
					};
				
					// Add start button
					this.addWidget("button", "Start mic capture", null, startMicrophoneCapture);
				
					// Add stop button
					this.addWidget("button", "Stop mic capture", null, stopMicrophoneCapture);
				};
			break;
		case "SaveImageKJ":
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function() {
				const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : void 0;
				const widget = this.widgets.find((w) => w.name === "filename_prefix");
				widget.serializeValue = () => {
				return applyTextReplacements(app, widget.value);
				};
				return r;
			};
			break;
			
		}	
		
	},
	async setup() {
		// to keep Set/Get node virtual connections visible when offscreen
		const originalComputeVisibleNodes = LGraphCanvas.prototype.computeVisibleNodes;
		LGraphCanvas.prototype.computeVisibleNodes = function () {
			const visibleNodesSet = new Set(originalComputeVisibleNodes.apply(this, arguments));
			for (const node of this.graph._nodes) {
				if ((node.type === "SetNode" || node.type === "GetNode") && node.drawConnection) {
					visibleNodesSet.add(node);
				}
			}
			return Array.from(visibleNodesSet);
		};

	}
});