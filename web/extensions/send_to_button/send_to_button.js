import { app } from "/scripts/app.js";

app.registerExtension({
    name: "SendToLoadImage",
    async nodeCreated(node) {
        if (node.comfyClass === "SaveImage") {
            node.onExecuted = (output) => {
                if (!output?.images) return;
                
                // Crée un bouton une seule fois
                if (!node.sendBtn) {
                    const btn = document.createElement("button");
                    btn.innerText = "Send To...";
                    btn.style.margin = "5px";
                    btn.onclick = () => {
                        alert("Ici tu pourras choisir vers quel TaggedLoadImage envoyer l’image");
                    };
                    node.addDOMElement(btn);
                    node.sendBtn = btn;
                }
            };
        }
    }
});
