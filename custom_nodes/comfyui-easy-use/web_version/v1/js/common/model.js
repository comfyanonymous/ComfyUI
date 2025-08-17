import { $el, ComfyDialog } from "../../../../scripts/ui.js";
import { api } from "../../../../scripts/api.js";
import {formatTime} from './utils.js';
import {$t} from "./i18n.js";
import {toast} from "./toast.js";

class MetadataDialog extends ComfyDialog {
	constructor() {
		super();
		this.element.classList.add("easyuse-model-metadata");
	}
	show(metadata) {
		super.show(
			$el(
				"div",
				Object.keys(metadata).map((k) =>
					$el("div", [$el("label", { textContent: k }), $el("span", { textContent: metadata[k] })])
				)
			)
		);
	}
}

export class ModelInfoDialog extends ComfyDialog {
	constructor(name) {
		super();
		this.name = name;
		this.element.classList.add("easyuse-model-info");
	}

	get customNotes() {
		return this.metadata["easyuse.notes"];
	}

	set customNotes(v) {
		this.metadata["easyuse.notes"] = v;
	}

	get hash() {
		return this.metadata["easyuse.sha256"];
	}

	async show(type, value) {
		this.type = type;

		const req = api.fetchApi("/easyuse/metadata/" + encodeURIComponent(`${type}/${value}`));
		this.info = $el("div", { style: { flex: "auto" } });
		// this.img = $el("img", { style: { display: "none" } });
		this.imgCurrent = 0
		this.imgList = $el("div.easyuse-preview-list",{
			style: { display: "none" }
		})
		this.imgWrapper = $el("div.easyuse-preview", [
			$el("div.easyuse-preview-group",[
				this.imgList
			]),
		]);
		this.main = $el("main", { style: { display: "flex" } }, [this.imgWrapper, this.info]);
		this.content = $el("div.easyuse-model-content", [
			$el("div.easyuse-model-header",[$el("h2", { textContent: this.name })])
			, this.main]);

		const loading = $el("div", { textContent: "ℹ️ Loading...", parent: this.content });

		super.show(this.content);

		this.metadata = await (await req).json();
		this.viewMetadata.style.cursor = this.viewMetadata.style.opacity = "";
		this.viewMetadata.removeAttribute("disabled");

		loading.remove();
		this.addInfo();
	}

	createButtons() {
		const btns = super.createButtons();
		this.viewMetadata = $el("button", {
			type: "button",
			textContent: "View raw metadata",
			disabled: "disabled",
			style: {
				opacity: 0.5,
				cursor: "not-allowed",
			},
			onclick: (e) => {
				if (this.metadata) {
					new MetadataDialog().show(this.metadata);
				}
			},
		});

		btns.unshift(this.viewMetadata);
		return btns;
	}

	parseNote() {
		if (!this.customNotes) return [];

		let notes = [];
		// Extract links from notes
		const r = new RegExp("(\\bhttps?:\\/\\/[^\\s]+)", "g");
		let end = 0;
		let m;
		do {
			m = r.exec(this.customNotes);
			let pos;
			let fin = 0;
			if (m) {
				pos = m.index;
				fin = m.index + m[0].length;
			} else {
				pos = this.customNotes.length;
			}

			let pre = this.customNotes.substring(end, pos);
			if (pre) {
				pre = pre.replaceAll("\n", "<br>");
				notes.push(
					$el("span", {
						innerHTML: pre,
					})
				);
			}
			if (m) {
				notes.push(
					$el("a", {
						href: m[0],
						textContent: m[0],
						target: "_blank",
					})
				);
			}

			end = fin;
		} while (m);
		return notes;
	}

	addInfoEntry(name, value) {
		return $el(
			"p",
			{
				parent: this.info,
			},
			[
				typeof name === "string" ? $el("label", { textContent: name + ": " }) : name,
				typeof value === "string" ? $el("span", { textContent: value }) : value,
			]
		);
	}

	async getCivitaiDetails() {
		const req = await fetch("https://civitai.com/api/v1/model-versions/by-hash/" + this.hash);
		if (req.status === 200) {
			return await req.json();
		} else if (req.status === 404) {
			throw new Error("Model not found");
		} else {
			throw new Error(`Error loading info (${req.status}) ${req.statusText}`);
		}
	}

	addCivitaiInfo() {
		const promise = this.getCivitaiDetails();
		const content = $el("span", { textContent: "ℹ️ Loading..." });

		this.addInfoEntry(
			$el("label", [
				$el("img", {
					style: {
						width: "18px",
						position: "relative",
						top: "3px",
						margin: "0 5px 0 0",
					},
					src: "https://civitai.com/favicon.ico",
				}),
				$el("span", { textContent: "Civitai: " }),
			]),
			content
		);

		return promise
			.then((info) => {
				this.imgWrapper.style.display = 'block'
				// 变更标题信息
				let header = this.element.querySelector('.easyuse-model-header')
				if(header){
					header.replaceChildren(
						$el("h2", { textContent: this.name }),
						$el("div.easyuse-model-header-remark",[
							$el("h5", { textContent:  $t("Updated At:") + formatTime(new Date(info.updatedAt),'yyyy/MM/dd')}),
							$el("h5", { textContent: $t("Created At:") + formatTime(new Date(info.updatedAt),'yyyy/MM/dd')}),
						])
					)
				}
				// 替换内容
				let textarea = null
				let notes = this.parseNote.call(this)
				let editText = $t("✏️ Edit")
				console.log(notes)
				let textarea_div = $el("div.easyuse-model-detail-textarea",[
					$el("p",notes?.length>0 ? notes : {textContent:$t('No notes')}),
				])
				if(!notes || notes.length == 0) textarea_div.classList.add('empty')
				else textarea_div.classList.remove('empty')
				this.info.replaceChildren(
					$el("div.easyuse-model-detail",[
						$el("div.easyuse-model-detail-head.flex-b",[
							$el('span',$t("Notes")),
							$el("a", {
								textContent: editText,
								href: "#",
								style: {
									fontSize: "12px",
									float: "right",
									color: "var(--warning-color)",
									textDecoration: "none",
								},
								onclick: async (e) => {
									e.preventDefault();

									if (textarea) {
										if(textarea.value != this.customNotes){
											toast.showLoading($t('Saving Notes...'))
											this.customNotes = textarea.value;
											const resp = await api.fetchApi(
												"/easyuse/metadata/notes/" + encodeURIComponent(`${this.type}/${this.name}`),
												{
													method: "POST",
													body: this.customNotes,
												}
											);
											toast.hideLoading()
											if (resp.status !== 200) {
												toast.error($t('Saving Failed'))
												console.error(resp);
												alert(`Error saving notes (${resp.status}) ${resp.statusText}`);
												return;
											}
											toast.success($t('Saving Succeed'))
											notes = this.parseNote.call(this)
											console.log(notes)
											textarea_div.replaceChildren($el("p",notes?.length>0 ? notes : {textContent:$t('No notes')}));
											if(textarea.value) textarea_div.classList.remove('empty')
											else textarea_div.classList.add('empty')
										}else {
											textarea_div.replaceChildren($el("p",{textContent:$t('No notes')}));
											textarea_div.classList.add('empty')
										}
										e.target.textContent = editText;
										textarea.remove();
										textarea = null;

									} else {
										e.target.textContent = "💾 Save";
										textarea = $el("textarea", {
											placeholder: $t("Type your notes here"),
											style: {
												width: "100%",
												minWidth: "200px",
												minHeight: "50px",
												height:"100px"
											},
											textContent: this.customNotes,
										});
										textarea_div.replaceChildren(textarea);
										textarea.focus()
									}
								}
							})
						]),
						textarea_div
					]),
					$el("div.easyuse-model-detail",[
						$el("div.easyuse-model-detail-head",{textContent:$t("Details")}),
						$el("div.easyuse-model-detail-body",[
							$el("div.easyuse-model-detail-item",[
								$el("div.easyuse-model-detail-item-label",{textContent:$t("Type")}),
								$el("div.easyuse-model-detail-item-value",{textContent:info.model.type}),
							]),
							$el("div.easyuse-model-detail-item",[
								$el("div.easyuse-model-detail-item-label",{textContent:$t("BaseModel")}),
								$el("div.easyuse-model-detail-item-value",{textContent:info.baseModel}),
							]),
							$el("div.easyuse-model-detail-item",[
								$el("div.easyuse-model-detail-item-label",{textContent:$t("Download")}),
								$el("div.easyuse-model-detail-item-value",{textContent:info.stats?.downloadCount || 0}),
							]),
							$el("div.easyuse-model-detail-item",[
								$el("div.easyuse-model-detail-item-label",{textContent:$t("Trained Words")}),
								$el("div.easyuse-model-detail-item-value",{textContent:info?.trainedWords.join(',') || '-'}),
							]),
							$el("div.easyuse-model-detail-item",[
								$el("div.easyuse-model-detail-item-label",{textContent:$t("Source")}),
								$el("div.easyuse-model-detail-item-value",[
									$el("label", [
										$el("img", {
											style: {
												width: "14px",
												position: "relative",
												top: "3px",
												margin: "0 5px 0 0",
											},
											src: "https://civitai.com/favicon.ico",
										}),
										$el("a", {
											href: "https://civitai.com/models/" + info.modelId,
											textContent: "View " + info.model.name,
											target: "_blank",
										})
									])
								]),
							])
						]),
					])
				);

				if (info.images?.length) {
					this.imgCurrent = 0
					this.isSaving = false
					info.images.map(cate=>
						cate.url &&
						this.imgList.appendChild(
							$el('div.easyuse-preview-slide',[
								$el('div.easyuse-preview-slide-content',[
									$el('img',{src:(cate.url)}),
									$el("div.save", {
										textContent: "Save as preview",
										onclick: async () => {
											if(this.isSaving) return
											this.isSaving = true
											toast.showLoading($t('Saving Preview...'))
											// Convert the preview to a blob
											const blob = await (await fetch(cate.url)).blob();

											// Store it in temp
											const name = "temp_preview." + new URL(cate.url).pathname.split(".")[1];
											const body = new FormData();
											body.append("image", new File([blob], name));
											body.append("overwrite", "true");
											body.append("type", "temp");

											const resp = await api.fetchApi("/upload/image", {
												method: "POST",
												body,
											});

											if (resp.status !== 200) {
												this.isSaving = false
												toast.error($t('Saving Failed'))
												toast.hideLoading()
												console.error(resp);
												alert(`Error saving preview (${req.status}) ${req.statusText}`);
												return;
											}

											// Use as preview
											await api.fetchApi("/easyuse/save/" + encodeURIComponent(`${this.type}/${this.name}`), {
												method: "POST",
												body: JSON.stringify({
													filename: name,
													type: "temp",
												}),
												headers: {
													"content-type": "application/json",
												},
											}).then(_=>{
												toast.success($t('Saving Succeed'))
												toast.hideLoading()
											});
											this.isSaving = false
											app.refreshComboInNodes();
										},
									})
								])
							])
						)
					)
					let _this = this
					this.imgDistance = (-660 * this.imgCurrent).toString()
					this.imgList.style.display = ''
					this.imgList.style.transform = 'translate3d(' + this.imgDistance +'px, 0px, 0px)'
					this.slides = this.imgList.querySelectorAll('.easyuse-preview-slide')
					// 添加按钮
					this.slideLeftButton = $el("button.left",{
						parent: this.imgWrapper,
						style:{
							display:info.images.length <= 2 ? 'none' : 'block'
						},
						innerHTML:`<svg viewBox="0 0 15 15" fill="none" xmlns="http://www.w3.org/2000/svg" width="16" height="16" style="transform: rotate(90deg);"><path d="M3.13523 6.15803C3.3241 5.95657 3.64052 5.94637 3.84197 6.13523L7.5 9.56464L11.158 6.13523C11.3595 5.94637 11.6759 5.95657 11.8648 6.15803C12.0536 6.35949 12.0434 6.67591 11.842 6.86477L7.84197 10.6148C7.64964 10.7951 7.35036 10.7951 7.15803 10.6148L3.15803 6.86477C2.95657 6.67591 2.94637 6.35949 3.13523 6.15803Z" fill="currentColor" fill-rule="evenodd" clip-rule="evenodd"></path></svg>`,
						onclick: ()=>{
							if(info.images.length <= 2) return
							_this.imgList.classList.remove("no-transition")
							if(_this.imgCurrent == 0){
								_this.imgCurrent = (info.images.length/2)-1
								this.slides[this.slides.length-1].style.transform = 'translate3d(' + (-660 * (this.imgCurrent+1)).toString()+'px, 0px, 0px)'
								this.slides[this.slides.length-2].style.transform = 'translate3d(' + (-660 * (this.imgCurrent+1)).toString()+'px, 0px, 0px)'
								_this.imgList.style.transform = 'translate3d(660px, 0px, 0px)'
								setTimeout(_=>{
									this.slides[this.slides.length-1].style.transform = 'translate3d(0px, 0px, 0px)'
									this.slides[this.slides.length-2].style.transform = 'translate3d(0px, 0px, 0px)'
									_this.imgDistance = (-660 * this.imgCurrent).toString()
									_this.imgList.style.transform = 'translate3d(' + _this.imgDistance +'px, 0px, 0px)'
									_this.imgList.classList.add("no-transition")
								},500)
							}
							else {
								_this.imgCurrent = _this.imgCurrent-1
								_this.imgDistance = (-660 * this.imgCurrent).toString()
								_this.imgList.style.transform = 'translate3d(' + _this.imgDistance +'px, 0px, 0px)'
							}
						}
					})
					this.slideRightButton = $el("button.right",{
						parent: this.imgWrapper,
						style:{
							display:info.images.length <= 2 ? 'none' : 'block'
						},
						innerHTML:`<svg viewBox="0 0 15 15" fill="none" xmlns="http://www.w3.org/2000/svg" width="16" height="16" style="transform: rotate(-90deg);"><path d="M3.13523 6.15803C3.3241 5.95657 3.64052 5.94637 3.84197 6.13523L7.5 9.56464L11.158 6.13523C11.3595 5.94637 11.6759 5.95657 11.8648 6.15803C12.0536 6.35949 12.0434 6.67591 11.842 6.86477L7.84197 10.6148C7.64964 10.7951 7.35036 10.7951 7.15803 10.6148L3.15803 6.86477C2.95657 6.67591 2.94637 6.35949 3.13523 6.15803Z" fill="currentColor" fill-rule="evenodd" clip-rule="evenodd"></path></svg>`,
						onclick: ()=>{
							if(info.images.length <= 2) return
							_this.imgList.classList.remove("no-transition")

							if( _this.imgCurrent >= (info.images.length/2)-1){
								_this.imgCurrent = 0
								const max = info.images.length/2
								this.slides[0].style.transform = 'translate3d(' + (660 * max).toString()+'px, 0px, 0px)'
								this.slides[1].style.transform = 'translate3d(' + (660 * max).toString()+'px, 0px, 0px)'
								_this.imgList.style.transform = 'translate3d(' + (-660 * max).toString()+'px, 0px, 0px)'
								setTimeout(_=>{
									this.slides[0].style.transform = 'translate3d(0px, 0px, 0px)'
									this.slides[1].style.transform = 'translate3d(0px, 0px, 0px)'
									_this.imgDistance = (-660 * this.imgCurrent).toString()
									_this.imgList.style.transform = 'translate3d(' + _this.imgDistance +'px, 0px, 0px)'
									_this.imgList.classList.add("no-transition")
								},500)
							}
							else  {
								_this.imgCurrent = _this.imgCurrent+1
								_this.imgDistance = (-660 * this.imgCurrent).toString()
								_this.imgList.style.transform = 'translate3d(' + _this.imgDistance +'px, 0px, 0px)'
							}

						}
					})

				}

				if(info.description){
					$el("div", {
						parent: this.content,
						innerHTML: info.description,
						style: {
							marginTop: "10px",
						},
					});
				}

				return info;
			})
			.catch((err) => {
				this.imgWrapper.style.display = 'none'
				content.textContent = "⚠️ " + err.message;
			})
			.finally(_=>{
			})
	}
}


export class CheckpointInfoDialog extends ModelInfoDialog {
    async addInfo() {
        // super.addInfo();
        await this.addCivitaiInfo();
    }
}

const MAX_TAGS = 500
export class LoraInfoDialog extends ModelInfoDialog {
	getTagFrequency() {
		if (!this.metadata.ss_tag_frequency) return [];

		const datasets = JSON.parse(this.metadata.ss_tag_frequency);
		const tags = {};
		for (const setName in datasets) {
			const set = datasets[setName];
			for (const t in set) {
				if (t in tags) {
					tags[t] += set[t];
				} else {
					tags[t] = set[t];
				}
			}
		}

		return Object.entries(tags).sort((a, b) => b[1] - a[1]);
	}

	getResolutions() {
		let res = [];
		if (this.metadata.ss_bucket_info) {
			const parsed = JSON.parse(this.metadata.ss_bucket_info);
			if (parsed?.buckets) {
				for (const { resolution, count } of Object.values(parsed.buckets)) {
					res.push([count, `${resolution.join("x")} * ${count}`]);
				}
			}
		}
		res = res.sort((a, b) => b[0] - a[0]).map((a) => a[1]);
		let r = this.metadata.ss_resolution;
		if (r) {
			const s = r.split(",");
			const w = s[0].replace("(", "");
			const h = s[1].replace(")", "");
			res.push(`${w.trim()}x${h.trim()} (Base res)`);
		} else if ((r = this.metadata["modelspec.resolution"])) {
			res.push(r + " (Base res");
		}
		if (!res.length) {
			res.push("⚠️ Unknown");
		}
		return res;
	}

	getTagList(tags) {
		return tags.map((t) =>
			$el(
				"li.easyuse-model-tag",
				{
					dataset: {
						tag: t[0],
					},
					$: (el) => {
						el.onclick = () => {
							el.classList.toggle("easyuse-model-tag--selected");
						};
					},
				},
				[
					$el("p", {
						textContent: t[0],
					}),
					$el("span", {
						textContent: t[1],
					}),
				]
			)
		);
	}

	addTags() {
		let tags = this.getTagFrequency();
		let hasMore;
		if (tags?.length) {
			const c = tags.length;
			let list;
			if (c > MAX_TAGS) {
				tags = tags.slice(0, MAX_TAGS);
				hasMore = $el("p", [
					$el("span", { textContent: `⚠️ Only showing first ${MAX_TAGS} tags ` }),
					$el("a", {
						href: "#",
						textContent: `Show all ${c}`,
						onclick: () => {
							list.replaceChildren(...this.getTagList(this.getTagFrequency()));
							hasMore.remove();
						},
					}),
				]);
			}
			list = $el("ol.easyuse-model-tags-list", this.getTagList(tags));
			this.tags = $el("div", [list]);
		} else {
			this.tags = $el("p", { textContent: "⚠️ No tag frequency metadata found" });
		}

		this.content.append(this.tags);

		if (hasMore) {
			this.content.append(hasMore);
		}
	}

	async addInfo() {
		// this.addInfoEntry("Name", this.metadata.ss_output_name || "⚠️ Unknown");
		// this.addInfoEntry("Base Model", this.metadata.ss_sd_model_name || "⚠️ Unknown");
		// this.addInfoEntry("Clip Skip", this.metadata.ss_clip_skip || "⚠️ Unknown");
		//
		// this.addInfoEntry(
		// 	"Resolution",
		// 	$el(
		// 		"select",
		// 		this.getResolutions().map((r) => $el("option", { textContent: r }))
		// 	)
		// );

		// super.addInfo();
		const p = this.addCivitaiInfo();
		this.addTags();

		const info = await p;
		if (info) {
			// $el(
			// 	"p",
			// 	{
			// 		parent: this.content,
			// 		textContent: "Trained Words: ",
			// 	},
			// 	[
			// 		$el("pre", {
			// 			textContent: info.trainedWords.join(", "),
			// 			style: {
			// 				whiteSpace: "pre-wrap",
			// 				margin: "10px 0",
			// 				background: "#222",
			// 				padding: "5px",
			// 				borderRadius: "5px",
			// 				maxHeight: "250px",
			// 				overflow: "auto",
			// 			},
			// 		}),
			// 	]
			// );
			$el("div", {
				parent: this.content,
				innerHTML: info.description,
				style: {
					maxHeight: "250px",
					overflow: "auto",
				},
			});
		}
	}

	createButtons() {
		const btns = super.createButtons();

		function copyTags(e, tags) {
			const textarea = $el("textarea", {
				parent: document.body,
				style: {
					position: "fixed",
				},
				textContent: tags.map((el) => el.dataset.tag).join(", "),
			});
			textarea.select();
			try {
				document.execCommand("copy");
				if (!e.target.dataset.text) {
					e.target.dataset.text = e.target.textContent;
				}
				e.target.textContent = "Copied " + tags.length + " tags";
				setTimeout(() => {
					e.target.textContent = e.target.dataset.text;
				}, 1000);
			} catch (ex) {
				prompt("Copy to clipboard: Ctrl+C, Enter", text);
			} finally {
				document.body.removeChild(textarea);
			}
		}

		btns.unshift(
			$el("button", {
				type: "button",
				textContent: "Copy Selected",
				onclick: (e) => {
					copyTags(e, [...this.tags.querySelectorAll(".easyuse-model-tag--selected")]);
				},
			}),
			$el("button", {
				type: "button",
				textContent: "Copy All",
				onclick: (e) => {
					copyTags(e, [...this.tags.querySelectorAll(".easyuse-model-tag")]);
				},
			})
		);

		return btns;
	}
}