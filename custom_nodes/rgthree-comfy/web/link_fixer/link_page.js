import { WorkflowLinkFixer } from "../common/link_fixer.js";
import { getPngMetadata } from "../common/comfyui_shim.js";
function wait(ms = 16, value) {
    return new Promise((resolve) => {
        setTimeout(() => {
            resolve(value);
        }, ms);
    });
}
const logger = {
    logTo: console,
    log: (...args) => {
        logger.logTo === console
            ? console.log(...args)
            : (logger.logTo.innerText += args.join(",") + "\n");
    },
};
export class LinkPage {
    constructor() {
        this.fixer = null;
        this.containerEl = document.querySelector(".box");
        this.figcaptionEl = document.querySelector("figcaption");
        this.outputeMessageEl = document.querySelector(".output");
        this.outputImageEl = document.querySelector(".output-image");
        this.btnFix = document.querySelector(".btn-fix");
        document.addEventListener("dragover", (e) => {
            e.preventDefault();
        }, false);
        document.addEventListener("drop", (e) => {
            this.onDrop(e);
        });
        this.btnFix.addEventListener("click", (e) => {
            this.onFixClick(e);
        });
    }
    async onFixClick(e) {
        var _a;
        if (!((_a = this.fixer) === null || _a === void 0 ? void 0 : _a.checkedData) || !this.graph) {
            this.updateUi("⛔ Fix button click without results.");
            return;
        }
        this.graphFinalResults = this.fixer.fix();
        if (this.graphFinalResults.hasBadLinks) {
            this.updateUi("⛔ Hmm... Still detecting bad links. Can you file an issue at https://github.com/rgthree/rgthree-comfy/issues with your image/workflow.");
        }
        else {
            this.updateUi("✅ Workflow fixed.<br><br><small>Please load new saved workflow json and double check linking and execution.</small>");
        }
        await wait(16);
        await this.saveFixedWorkflow();
    }
    async onDrop(event) {
        var _a, _b, _c, _d;
        if (!event.dataTransfer) {
            return;
        }
        this.reset();
        event.preventDefault();
        event.stopPropagation();
        if (event.dataTransfer.files.length && ((_b = (_a = event.dataTransfer.files) === null || _a === void 0 ? void 0 : _a[0]) === null || _b === void 0 ? void 0 : _b.type) !== "image/bmp") {
            await this.handleFile(event.dataTransfer.files[0]);
            return;
        }
        const validTypes = ["text/uri-list", "text/x-moz-url"];
        const match = [...event.dataTransfer.types].find((t) => validTypes.find((v) => t === v));
        if (match) {
            const uri = (_d = (_c = event.dataTransfer.getData(match)) === null || _c === void 0 ? void 0 : _c.split("\n")) === null || _d === void 0 ? void 0 : _d[0];
            if (uri) {
                await this.handleFile(await (await fetch(uri)).blob());
            }
        }
    }
    reset() {
        this.file = undefined;
        this.graph = undefined;
        this.graphResults = undefined;
        this.graphFinalResults = undefined;
        this.updateUi();
    }
    updateUi(msg) {
        this.outputeMessageEl.innerHTML = "";
        if (this.file && !this.containerEl.classList.contains("-has-file")) {
            this.containerEl.classList.add("-has-file");
            this.figcaptionEl.innerHTML = this.file.name || this.file.type;
            if (this.file.type === "application/json") {
                this.outputImageEl.src = "icon_file_json.png";
            }
            else {
                const reader = new FileReader();
                reader.onload = () => (this.outputImageEl.src = reader.result);
                reader.readAsDataURL(this.file);
            }
        }
        else if (!this.file && this.containerEl.classList.contains("-has-file")) {
            this.containerEl.classList.remove("-has-file");
            this.outputImageEl.src = "";
            this.outputImageEl.removeAttribute("src");
        }
        if (this.graphResults) {
            this.containerEl.classList.add("-has-results");
            if (!this.graphResults.patches && !this.graphResults.deletes) {
                this.outputeMessageEl.innerHTML = "✅ No bad links detected in the workflow.";
            }
            else {
                this.containerEl.classList.add("-has-fixable-results");
                this.outputeMessageEl.innerHTML = `⚠️ Found ${this.graphResults.patches} links to fix, and ${this.graphResults.deletes} to be removed.`;
            }
        }
        else {
            this.containerEl.classList.remove("-has-results");
            this.containerEl.classList.remove("-has-fixable-results");
        }
        if (msg) {
            this.outputeMessageEl.innerHTML = msg;
        }
    }
    async handleFile(file) {
        this.file = file;
        this.updateUi();
        let workflow = null;
        if (file.type.startsWith("image/")) {
            const pngInfo = await getPngMetadata(file);
            workflow = pngInfo === null || pngInfo === void 0 ? void 0 : pngInfo.workflow;
        }
        else if (file.type === "application/json" ||
            (file instanceof File && file.name.endsWith(".json"))) {
            workflow = await new Promise((resolve) => {
                const reader = new FileReader();
                reader.onload = () => {
                    resolve(reader.result);
                };
                reader.readAsText(file);
            });
        }
        if (!workflow) {
            this.updateUi("⛔ No workflow found in dropped item.");
        }
        else {
            try {
                this.graph = JSON.parse(workflow);
            }
            catch (e) {
                this.graph = undefined;
            }
            if (!this.graph) {
                this.updateUi("⛔ Invalid workflow found in dropped item.");
            }
            else {
                this.loadGraphData(this.graph);
            }
        }
    }
    async loadGraphData(graphData) {
        this.fixer = WorkflowLinkFixer.create(graphData);
        this.graphResults = this.fixer.check();
        this.updateUi();
    }
    async saveFixedWorkflow() {
        if (!this.graphFinalResults) {
            this.updateUi("⛔ Save w/o final graph patched.");
            return false;
        }
        let filename = this.file.name || "workflow.json";
        let filenames = filename.split(".");
        filenames.pop();
        filename = filenames.join(".");
        filename += "_fixed.json";
        filename = prompt("Save workflow as:", filename);
        if (!filename)
            return false;
        if (!filename.toLowerCase().endsWith(".json")) {
            filename += ".json";
        }
        const json = JSON.stringify(this.graphFinalResults.graph, null, 2);
        const blob = new Blob([json], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement("a");
        anchor.download = filename;
        anchor.href = url;
        anchor.style.display = "none";
        document.body.appendChild(anchor);
        await wait();
        anchor.click();
        await wait();
        anchor.remove();
        window.URL.revokeObjectURL(url);
        return true;
    }
}
