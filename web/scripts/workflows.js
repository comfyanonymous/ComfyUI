import { $el } from "./ui.js";
import { addStylesheet } from "./utils.js";

addStylesheet("https://cdn.jsdelivr.net/npm/@mdi/font@7.4.47/css/materialdesignicons.min.css");

const $i = (n, s = 24, o) => $el(`span.mdi.mdi-${n}${s ? `.mdi-${s}px` : ""}`, o ?? {});

$el(
	"style",
	{ parent: document.head },
	`
	.comfy-nav {
		position: absolute;
		left: 0;
		top: 450px;
		z-index: 99;
		background: var(--comfy-menu-bg);
		font-family: Arial;
		padding: 4px 8px;
		display: flex;
		gap: 10px;
		align-items: center;
		font-size: 14px;
		width: calc(100vw - 16px);
		height: 30px;
		color: var(--fg-color);

		h1 {
			font-size: 16px;
		}

		.comfy-nav-split-btn {
			padding: 0;
			border: 0;
			height: 100%;
			color: var(--fg-color);
			display: flex;
			align-items: stretch;
			border-radius: 4px;
			background: var(--comfy-input-bg);
			position: relative;
			.mdi {
				display: flex;
				align-items: center;
			}
			.mdi:hover {
				background: var(--border-color);
			}
			.btn-icon {
				display: flex;
				border-top-left-radius: 4px;
				border-bottom-left-radius: 4px;
				padding: 0 8px;
			}
			.mdi-chevron-down {
				border-left: 1px solid var(--border-color);
				border-top-right-radius: 4px;
				border-bottom-right-radius: 4px;
			}
			.mdi-chevron-down:hover .comfy-nav-split-dropdown {
				display: block;
			}

			.comfy-nav-split-dropdown {
				display: none;
				background: var(--border-color);
				border-top-left-radius: 4px;
				border-bottom-right-radius: 4px;
				border-bottom-left-radius: 4px;

				li {
					white-space: nowrap;
					padding: 8px 10px;
					text-align: left;
					display: flex;
					gap: 5px;
					align-items: center;
				}
				li:hover {
					background: var(--comfy-input-bg);
				}
			}
		}

		.mdi {
			cursor: pointer;
		}

		.comfy-workflow-nav {
			border-radius: 4px;
			height: 100%;
			padding: 0 5px 0 10px;
			background: var(--comfy-input-bg);
			display: flex;
			gap: 10px;
			align-items: center;
			cursor: pointer;
			.comfy-workflow-list {
				display: none
			}
		}
		.comfy-workflow-nav:hover {
			background: var(--border-color);
		}

		.unsaved {
			font-style: italic;
		}
	}
`
);

export class Workflows {
	/** @type { boolean } */
	#autoSave;

	/** @type { string | undefined } */
	name;

	get autoSave() {
		return this.#autoSave;
	}

	set autoSave(value) {
		this.#autoSave = value;
	}

	async loadWorkflows() {}

	async saveWorkflow() {}

	async newWorkflow(name) {}

	constructor() {
		const el = $el(
			"nav.comfy-nav",
			{
				style: {},
			},
			[
				$el("span.mdi.mdi-graph-outline.mdi-24px", { style: { marginRight: "-5px", transform: "rotate(-90deg)" } }),
				$el("h1", "ComfyUI"),
				$el("div.comfy-workflow-nav", [
					$el("span.unsaved", "Unsaved Workflow*"),
					$i("chevron-down", 18),
					$el("div.comfy-workflow-list", []),
				]),
				$el("button.comfy-nav-split-btn", {}, [
					$i("content-save.btn-icon", 18),
					$el("span.mdi.mdi-chevron-down.mdi-18px", [
						$el(
							"div",
							{
								style: {
									position: "absolute",
									top: "calc(100% - 2px)",
									right: 0,
									background: "var(--comfy-menu-bg)",
								},
							},
							$el(
								"ul.comfy-nav-split-dropdown",
								{
									style: {
										listStyle: "none",
										padding: 0,
										margin: 0,
										whiteSpace: "no-wrap",
									},
								},
								[
									$el("li", [$i("content-save", 18), $el("span", "Save")]),
									$el("li", [$i("content-save-edit", 18), $el("span", "Save as")]),
									$el("li", [$i("api", 18), $el("span", "Save (API format)")]),
									$el("li", [$i("download", 18), $el("span", "Download")]),
									// $el("li", {
									// 	style: {
									// 		borderTop: "1px solid rgba(255,255,255,0.5)"
									// 	}
									// }, [$i("checkbox-outline", 18), $el("span", "Auto save (on)")]),
								]
							)
						),
					]),
				]),
				$el(
					"div",
					{
						style: {
							height: "100%",
							background: "var(--comfy-input-bg)",
							borderRadius: "4px",
							display: "flex",
							alignItems: "center",
						},
					},
					[
						$i("refresh", 24, { style: { background: "var(--comfy-input-bg)", borderRadius: "4px", padding: "0 10px" } }),
						$el("div", { style: { width: "1px", height: "90%", background: "var(--border-color", opacity: "0.5" } }),
						$i("clipboard-edit-outline", 24, { style: { background: "var(--comfy-input-bg)", borderRadius: "4px", padding: "0 10px" } }),
						$el("div", { style: { width: "1px", height: "90%", background: "var(--border-color", opacity: "0.5" } }),
						$i("cancel", 24, { style: { background: "var(--comfy-input-bg)", borderRadius: "4px", padding: "0 10px" } }),
					]
				),
				$el(
					"div",
					{
						style: {
							marginLeft: "auto",
							height: "100%",
							background: "var(--comfy-input-bg)",
							borderRadius: "4px",
							display: "flex",
							alignItems: "center",
						},
					},
					[
						$i("history", 24, { style: { background: "var(--comfy-input-bg)", borderRadius: "4px", padding: "0 10px" } }),
						$el("div", { style: { width: "1px", height: "90%", background: "var(--border-color", opacity: "0.5" } }),
						$i("format-list-numbered", 24, { style: { background: "var(--comfy-input-bg)", borderRadius: "4px", padding: "0 10px" } }),
					]
				),
				$el(
					"button.comfy-nav-split-btn",
					{
						
					},
					[
						$i("play.btn-icon"),
						$el(
							"span.btn-icon",
							{
								style: {
									width: "60px",
									position: "relative",
									top: "7px",
									left: "-10px",
								},
							},
							[
								$el("span", "Queue"),
								$el(
									"span",
									{
										style: {
											background: "dodgerblue",
											color: "#fff",
											height: "16px",
											borderRadius: "25%",
											marginLeft: "5px",
											padding: "0 2px",
										},
									},
									"182"
								),
							]
						),
						$el("span.mdi.mdi-chevron-down.mdi-18px", [
							$el(
								"div",
								{
									style: {
										position: "absolute",
										top: "calc(100% - 2px)",
										right: 0,
										background: "var(--comfy-menu-bg)",
									},
								},
								$el(
									"ul.comfy-nav-split-dropdown",
									{
										style: {
											listStyle: "none",
											padding: 0,
											margin: 0,
											whiteSpace: "no-wrap",
										},
									},
									[
										$el("li", [$i("numeric-1-box-outline", 18), $el("span", "Queue front")]),
										$el("li", [$el("div", {}, [
											
										])])
										// $el("li", {
										// 	style: {
										// 		borderTop: "1px solid rgba(255,255,255,0.5)"
										// 	}
										// }, [$i("checkbox-outline", 18), $el("span", "Auto save (on)")]),
									]
								)
							),
						]),
					]
				),
				$i("cog", 24, { style: {  } }),
			]
		);

		document.body.append(el);
	}
}
