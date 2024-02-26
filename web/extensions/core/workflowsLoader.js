import { app } from "../../scripts/app.js";
import { ComfyDialog, $el } from "../../scripts/ui.js";
import { ComfyApp } from "../../scripts/app.js";

export class WorkflowsChoosingDialog extends ComfyDialog {
    static instance = null;

    workflows = []
    select_workflow = null;

    workflowList;
    constructor() {
        super();
        const fileInput = $el("input", {
            id: "comfy-workflows-file-input",
            type: "file",
            accept: ".json,image/png,image/webp",
            style: { display: "none" },
            parent: document.body,
            onchange: () => {
                this.close();
                app.handleFile(fileInput.files[0]);
            },
        });
        const buttons = [];
        buttons.push($el("button", {
            type: "button",
            textContent: "Choose",
            onclick: () => {
                this.close();
                if (this.select_workflow) {
                    const xhr = new XMLHttpRequest();
                    xhr.open("GET", "workflows/" + this.select_workflow, false);
                    xhr.send();
                    const file = new File([new Blob([xhr.response], { type: "application/json" })], this.select_workflow);
                    app.handleFile(file)
                }
            }
        }));
        buttons.push($el("button", {
            type: "button",
            textContent: "Browse..",
            onclick: () => { fileInput.click(); }
        }));
        buttons.push($el("button", {
            type: "button",
            textContent: "Close",
            onclick: () => { this.close(); }
        }));
        this.workflowList = $el("select", {
            style: {
                marginBottom: "0.15rem",
                width: "100%",
            },
            onchange: (e) => {
                this.select_workflow = e.target.value
            }
        }, [])
        const children = $el("div.comfy-modal-content", [fileInput, this.workflowList, ...buttons]);
        this.element = $el("div.comfy-modal", { parent: document.body }, [children,]);
    }

    display() {
        const xhr = new XMLHttpRequest();
        xhr.open("GET", "workflows", false);
        xhr.send();
        this.workflows = JSON.parse(xhr.response);
        this.select_workflow = this.workflows.length == 0 ? null : this.workflows[0];
        this.workflowList.options.length = 0;
        for (var workflow of this.workflows) {
            this.workflowList.add(new Option(workflow, workflow));
        }
        this.element.style.display = "block";
    }

    static show() {
        if (!WorkflowsChoosingDialog.instance) {
            WorkflowsChoosingDialog.instance = new WorkflowsChoosingDialog();
        }


        WorkflowsChoosingDialog.instance.display();
    }
}
app.registerExtension({
    name: "Comfy.WorkflowsLoader",
    init(app) {
        app.loadWorkflows =
            function () {
                WorkflowsChoosingDialog.show();
            };
    }
});