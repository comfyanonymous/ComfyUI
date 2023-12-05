// @ts-check
/// <reference path="../node_modules/@types/jest/index.d.ts" />
const { start } = require("../utils");
const lg = require("../utils/litegraph");

describe("extensions", () => {
	beforeEach(() => {
		lg.setup(global);
	});

	afterEach(() => {
		lg.teardown(global);
	});

	it("calls each extension hook", async () => {
		const mockExtension = {
			name: "TestExtension",
			init: jest.fn(),
			setup: jest.fn(),
			addCustomNodeDefs: jest.fn(),
			getCustomWidgets: jest.fn(),
			beforeRegisterNodeDef: jest.fn(),
			registerCustomNodes: jest.fn(),
			loadedGraphNode: jest.fn(),
			nodeCreated: jest.fn(),
			beforeConfigureGraph: jest.fn(),
			afterConfigureGraph: jest.fn(),
		};

		const { app, ez, graph } = await start({
			async preSetup(app) {
				app.registerExtension(mockExtension);
			},
		});

		// Basic initialisation hooks should be called once, with app
		expect(mockExtension.init).toHaveBeenCalledTimes(1);
		expect(mockExtension.init).toHaveBeenCalledWith(app);

		// Adding custom node defs should be passed the full list of nodes
		expect(mockExtension.addCustomNodeDefs).toHaveBeenCalledTimes(1);
		expect(mockExtension.addCustomNodeDefs.mock.calls[0][1]).toStrictEqual(app);
		const defs = mockExtension.addCustomNodeDefs.mock.calls[0][0];
		expect(defs).toHaveProperty("KSampler");
		expect(defs).toHaveProperty("LoadImage");

		// Get custom widgets is called once and should return new widget types
		expect(mockExtension.getCustomWidgets).toHaveBeenCalledTimes(1);
		expect(mockExtension.getCustomWidgets).toHaveBeenCalledWith(app);

		// Before register node def will be called once per node type
		const nodeNames = Object.keys(defs);
		const nodeCount = nodeNames.length;
		expect(mockExtension.beforeRegisterNodeDef).toHaveBeenCalledTimes(nodeCount);
		for (let i = 0; i < 10; i++) {
			// It should be send the JS class and the original JSON definition
			const nodeClass = mockExtension.beforeRegisterNodeDef.mock.calls[i][0];
			const nodeDef = mockExtension.beforeRegisterNodeDef.mock.calls[i][1];

			expect(nodeClass.name).toBe("ComfyNode");
			expect(nodeClass.comfyClass).toBe(nodeNames[i]);
			expect(nodeDef.name).toBe(nodeNames[i]);
			expect(nodeDef).toHaveProperty("input");
			expect(nodeDef).toHaveProperty("output");
		}

		// Register custom nodes is called once after registerNode defs to allow adding other frontend nodes
		expect(mockExtension.registerCustomNodes).toHaveBeenCalledTimes(1);

		// Before configure graph will be called here as the default graph is being loaded
		expect(mockExtension.beforeConfigureGraph).toHaveBeenCalledTimes(1);
		// it gets sent the graph data that is going to be loaded
		const graphData = mockExtension.beforeConfigureGraph.mock.calls[0][0];

		// A node created is fired for each node constructor that is called
		expect(mockExtension.nodeCreated).toHaveBeenCalledTimes(graphData.nodes.length);
		for (let i = 0; i < graphData.nodes.length; i++) {
			expect(mockExtension.nodeCreated.mock.calls[i][0].type).toBe(graphData.nodes[i].type);
		}

		// Each node then calls loadedGraphNode to allow them to be updated
		expect(mockExtension.loadedGraphNode).toHaveBeenCalledTimes(graphData.nodes.length);
		for (let i = 0; i < graphData.nodes.length; i++) {
			expect(mockExtension.loadedGraphNode.mock.calls[i][0].type).toBe(graphData.nodes[i].type);
		}

		// After configure is then called once all the setup is done
		expect(mockExtension.afterConfigureGraph).toHaveBeenCalledTimes(1);

		expect(mockExtension.setup).toHaveBeenCalledTimes(1);
		expect(mockExtension.setup).toHaveBeenCalledWith(app);

		// Ensure hooks are called in the correct order
		const callOrder = [
			"init",
			"addCustomNodeDefs",
			"getCustomWidgets",
			"beforeRegisterNodeDef",
			"registerCustomNodes",
			"beforeConfigureGraph",
			"nodeCreated",
			"loadedGraphNode",
			"afterConfigureGraph",
			"setup",
		];
		for (let i = 1; i < callOrder.length; i++) {
			const fn1 = mockExtension[callOrder[i - 1]];
			const fn2 = mockExtension[callOrder[i]];
			expect(fn1.mock.invocationCallOrder[0]).toBeLessThan(fn2.mock.invocationCallOrder[0]);
		}

		graph.clear();

		// Ensure adding a new node calls the correct callback
		ez.LoadImage();
		expect(mockExtension.loadedGraphNode).toHaveBeenCalledTimes(graphData.nodes.length);
		expect(mockExtension.nodeCreated).toHaveBeenCalledTimes(graphData.nodes.length + 1);
		expect(mockExtension.nodeCreated.mock.lastCall[0].type).toBe("LoadImage");

		// Reload the graph to ensure correct hooks are fired
		await graph.reload();

		// These hooks should not be fired again
		expect(mockExtension.init).toHaveBeenCalledTimes(1);
		expect(mockExtension.addCustomNodeDefs).toHaveBeenCalledTimes(1);
		expect(mockExtension.getCustomWidgets).toHaveBeenCalledTimes(1);
		expect(mockExtension.registerCustomNodes).toHaveBeenCalledTimes(1);
		expect(mockExtension.beforeRegisterNodeDef).toHaveBeenCalledTimes(nodeCount);
		expect(mockExtension.setup).toHaveBeenCalledTimes(1);

		// These should be called again
		expect(mockExtension.beforeConfigureGraph).toHaveBeenCalledTimes(2);
		expect(mockExtension.nodeCreated).toHaveBeenCalledTimes(graphData.nodes.length + 2);
		expect(mockExtension.loadedGraphNode).toHaveBeenCalledTimes(graphData.nodes.length + 1);
		expect(mockExtension.afterConfigureGraph).toHaveBeenCalledTimes(2);
	}, 15000);

	it("allows custom nodeDefs and widgets to be registered", async () => {
		const widgetMock = jest.fn((node, inputName, inputData, app) => {
			expect(node.constructor.comfyClass).toBe("TestNode");
			expect(inputName).toBe("test_input");
			expect(inputData[0]).toBe("CUSTOMWIDGET");
			expect(inputData[1]?.hello).toBe("world");
			expect(app).toStrictEqual(app);

			return {
				widget: node.addWidget("button", inputName, "hello", () => {}),
			};
		});

		// Register our extension that adds a custom node + widget type
		const mockExtension = {
			name: "TestExtension",
			addCustomNodeDefs: (nodeDefs) => {
				nodeDefs["TestNode"] = {
					output: [],
					output_name: [],
					output_is_list: [],
					name: "TestNode",
					display_name: "TestNode",
					category: "Test",
					input: {
						required: {
							test_input: ["CUSTOMWIDGET", { hello: "world" }],
						},
					},
				};
			},
			getCustomWidgets: jest.fn(() => {
				return {
					CUSTOMWIDGET: widgetMock,
				};
			}),
		};

		const { graph, ez } = await start({
			async preSetup(app) {
				app.registerExtension(mockExtension);
			},
		});

		expect(mockExtension.getCustomWidgets).toBeCalledTimes(1);

		graph.clear();
		expect(widgetMock).toBeCalledTimes(0);
		const node = ez.TestNode();
		expect(widgetMock).toBeCalledTimes(1);

		// Ensure our custom widget is created
		expect(node.inputs.length).toBe(0);
		expect(node.widgets.length).toBe(1);
		const w = node.widgets[0].widget;
		expect(w.name).toBe("test_input");
		expect(w.type).toBe("button");
	});
});
