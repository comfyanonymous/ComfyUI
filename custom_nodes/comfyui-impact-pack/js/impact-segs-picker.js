import { ComfyApp, app } from "../../scripts/app.js";
import { ComfyDialog, $el } from "../../scripts/ui.js";
import { api } from "../../scripts/api.js";

async function open_picker(node) {
    const resp = await api.fetchApi(`/impact/segs/picker/count?id=${node.id}`);
    const body = await resp.text();

    let cnt = parseInt(body);

    var existingPicker = document.getElementById('impact-picker');
    if (existingPicker) {
        existingPicker.parentNode.removeChild(existingPicker);
    }

    var gallery = document.createElement('div');
    gallery.id = 'impact-picker';

    gallery.style.position = "absolute";
    gallery.style.height = "80%";
    gallery.style.width = "80%";
    gallery.style.top = "10%";
    gallery.style.left = "10%";
    gallery.style.display = 'flex';
    gallery.style.flexWrap = 'wrap';
    gallery.style.maxHeight = '600px';
    gallery.style.overflow = 'auto';
    gallery.style.backgroundColor = 'rgba(0,0,0,0.3)';
    gallery.style.padding = '20px';
    gallery.draggable = false;
    gallery.style.zIndex = 5000;

    var doneButton = document.createElement('button');
    doneButton.textContent = 'Done';
    doneButton.style.padding = '10px 10px';
    doneButton.style.border = 'none';
    doneButton.style.borderRadius = '5px';
    doneButton.style.fontFamily = 'Arial, sans-serif';
    doneButton.style.fontSize = '16px';
    doneButton.style.fontWeight = 'bold';
    doneButton.style.color = '#fff';
    doneButton.style.background = 'linear-gradient(to bottom, #0070B8, #003D66)';
    doneButton.style.boxShadow = '0 2px 4px rgba(0, 0, 0, 0.4)';
    doneButton.style.margin = "20px";
    doneButton.style.height = "40px";

    var cancelButton = document.createElement('button');
    cancelButton.textContent = 'Cancel';
    cancelButton.style.padding = '10px 10px';
    cancelButton.style.border = 'none';
    cancelButton.style.borderRadius = '5px';
    cancelButton.style.fontFamily = 'Arial, sans-serif';
    cancelButton.style.fontSize = '16px';
    cancelButton.style.fontWeight = 'bold';
    cancelButton.style.color = '#fff';
    cancelButton.style.background = 'linear-gradient(to bottom, #ff70B8, #ff3D66)';
    cancelButton.style.boxShadow = '0 2px 4px rgba(0, 0, 0, 0.4)';
    cancelButton.style.margin = "20px";
    cancelButton.style.height = "40px";

    const w = node.widgets.find((w) => w.name == 'picks');
    let prev_selected = w.value.split(',').map(function(item) {
      return parseInt(item, 10);
    });

    let images = [];
    doneButton.onclick = () => {
        var result = '';
        for(let i in images) {
            if(images[i].isSelected) {
                if(result != '')
                    result += ', ';

                result += (parseInt(i)+1);
            }
        }

        w.value = result;

        gallery.parentNode.removeChild(gallery);
    }

    cancelButton.onclick = () => {
        gallery.parentNode.removeChild(gallery);
    }

    var panel = document.createElement('div');
    panel.style.clear = 'both';
    panel.style.width = '100%';
    panel.style.height = '40px';
    panel.style.justifyContent = 'center';
    panel.style.alignItems = 'center';
    panel.style.display = 'flex';
    panel.appendChild(doneButton);
    panel.appendChild(cancelButton);
    gallery.appendChild(panel);

    var hint = document.createElement('label');
    hint.style.position = 'absolute';
    hint.innerHTML = 'Click: Toggle Selection<BR>Ctrl-click: Single Selection';
    gallery.appendChild(hint);

    let max_size = 300;

    for(let i=0; i<cnt; i++) {
        let image = new Image();
        image.src = `/impact/segs/picker/view?id=${node.id}&idx=${i}`;
        image.style.margin = '10px';
        image.draggable = false;
        images.push(image);
        image.isSelected = prev_selected.includes(i + 1);
        if(image.isSelected) {
            image.style.border = '2px solid #006699';
        }

        image.onload = function() {
            var ratio = 1.0;
            if(image.naturalWidth > image.naturalHeight) {
                ratio = max_size/image.naturalWidth;
            }
            else {
                ratio = max_size/image.naturalHeight;
            }

            let width = image.naturalWidth * ratio;
            let height = image.naturalHeight * ratio;

            if(width < height) {
                this.style.marginLeft = (200-width)/2+"px";
            }
            else{
                this.style.marginTop = (200-height)/2+"px";
            }

            this.style.width = width+"px";
            this.style.height = height+"px";
            this.style.objectFit = 'cover';
        }

        image.addEventListener('click', function(event) {
            if(event.ctrlKey) {
                for(let i in images) {
                    if(images[i].isSelected) {
                        images[i].style.border = 'none';
                        images[i].isSelected = false;
                    }
                }

                image.style.border = '2px solid #006699';
                image.isSelected = true;

                return;
            }

            if(image.isSelected) {
                image.style.border = 'none';
                image.isSelected = false;
            }
            else {
                image.style.border = '2px solid #006699';
                image.isSelected = true;
            }
        });

        gallery.appendChild(image);
    }

    document.body.appendChild(gallery);
}


app.registerExtension({
	name: "Comfy.Impack.Picker",

	nodeCreated(node, app) {
		if(node.comfyClass == "ImpactSEGSPicker") {
		    node.addWidget("button", "pick", "image", () => {
		        open_picker(node);
		    });
        }
    }
});