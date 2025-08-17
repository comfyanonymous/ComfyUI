import {app} from "../../../../scripts/app.js";
import {$el} from "../../../../scripts/ui.js";
import {$t} from "../common/i18n.js";
import {findWidgetByName, toggleWidget} from "../common/utils.js";


const tags = {
    "selfie_multiclass_256x256": ["Background", "Hair", "Body", "Face", "Clothes", "Others",],
    "human_parsing_lip":["Background","Hat","Hair","Glove","Sunglasses","Upper-clothes","Dress","Coat","Socks","Pants","Jumpsuits","Scarf","Skirt","Face","Left-arm","Right-arm","Left-leg","Right-leg","Left-shoe","Right-shoe"],
}
function getTagList(tags) {
    let rlist=[]
    tags.forEach((k,i) => {
        rlist.push($el(
            "label.easyuse-prompt-styles-tag",
            {
                dataset: {
                    tag: i,
                    name: $t(k),
                    index: i
                },
                $: (el) => {
                    el.children[0].onclick = () => {
                        el.classList.toggle("easyuse-prompt-styles-tag-selected");
                    };
                },
            },
            [
                $el("input",{
                    type: 'checkbox',
                    name: i
                }),
                $el("span",{
                    textContent: $t(k),
                })
            ]
        ))
    });
    return rlist
}


app.registerExtension({
    name: 'comfy.easyUse.seg',
    async beforeRegisterNodeDef(nodeType, nodeData, app) {

        if (nodeData.name == 'easy humanSegmentation') {
            // 创建时
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated ? onNodeCreated?.apply(this, arguments) : undefined;
                const method = this.widgets.findIndex((w) => w.name == 'method');
                const list = $el("ul.easyuse-prompt-styles-list.no-top", []);
                let method_values = ''
                this.setProperty("values", [])

                let selector = this.addDOMWidget('mask_components',"btn",$el('div.easyuse-prompt-styles',[list]))

                Object.defineProperty(this.widgets[method],'value',{
                    set:(value)=>{
                        method_values = value
                        if(method_values){
                            selector.element.children[0].innerHTML = ''
                            if(method_values == 'selfie_multiclass_256x256'){
                                toggleWidget(this, findWidgetByName(this, 'confidence'), true)
                                this.setSize([300, 260]);
                            }else{
                                toggleWidget(this, findWidgetByName(this, 'confidence'))
                                this.setSize([300, 500]);
                            }
                            let list = getTagList(tags[method_values]);
                            selector.element.children[0].append(...list)
                        }
                    },
                    get: () => {
                        return method_values
                    }
                })

                let mask_select_values = ''

                Object.defineProperty(selector, "value", {
                    set: (value) => {
                        setTimeout(_=>{
                            selector.element.children[0].querySelectorAll(".easyuse-prompt-styles-tag").forEach(el => {
                                let arr = value.split(',')
                                if (arr.includes(el.dataset.tag)) {
                                    el.classList.add("easyuse-prompt-styles-tag-selected");
                                    el.children[0].checked = true
                                }
                            })
                        },100)
                    },
                    get: () => {
                        selector.element.children[0].querySelectorAll(".easyuse-prompt-styles-tag").forEach(el => {
                            if(el.classList.value.indexOf("easyuse-prompt-styles-tag-selected")>=0){
                                if(!this.properties["values"].includes(el.dataset.tag)){
                                    this.properties["values"].push(el.dataset.tag);
                                }
                            }else{
                                if(this.properties["values"].includes(el.dataset.tag)){
                                    this.properties["values"]= this.properties["values"].filter(v=>v!=el.dataset.tag);
                                }
                            }
                        });
                        mask_select_values = this.properties["values"].join(',');
                        return mask_select_values;
                    }
                });

                let old_values = ''
                let mask_lists_dom = selector.element.children[0]

                // 初始化
                setTimeout(_=>{
                    if(!method_values) {
                        method_values = 'selfie_multiclass_256x256'
                        selector.element.children[0].innerHTML = ''
                        // 重新排序
                        let list = getTagList(tags[method_values]);
                        selector.element.children[0].append(...list)
                    }
                    if(method_values == 'selfie_multiclass_256x256'){
                        toggleWidget(this, findWidgetByName(this, 'confidence'), true)
                        this.setSize([300, 260]);
                    }else{
                        toggleWidget(this, findWidgetByName(this, 'confidence'))
                        this.setSize([300, 500]);
                    }
                },1)

                return onNodeCreated;
            }
        }
    }
})