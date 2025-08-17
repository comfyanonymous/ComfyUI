// 1.0.3
import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";
import { $el } from "../../../../scripts/ui.js";
import { $t } from "../common/i18n.js";

// 获取风格列表
let styles_list_cache = {}
let styles_image_cache = {}
async function getStylesList(name){
    if(styles_list_cache[name]) return styles_list_cache[name]
    else{
       const resp = await api.fetchApi(`/easyuse/prompt/styles?name=${name}`);
       if (resp.status === 200) {
            let data = await resp.json();
            styles_list_cache[name] = data;
            return data;
       }
       return undefined;
    }
}
async function getStylesImage(name, styles_name){
    if(!styles_image_cache[styles_name]) styles_image_cache[styles_name] = {}
    if(styles_image_cache[styles_name][name]) return styles_image_cache[styles_name][name]
    else{
       const resp = await api.fetchApi(`/easyuse/prompt/styles/image?name=${name}&styles_name=${styles_name}`);
       if (resp.status === 200) {
           const text = await resp.text()
           if(text.startsWith('http')){
               styles_image_cache[styles_name][name] = text
               return text
           }
           const url = `/easyuse/prompt/styles/image?name=${name}&styles_name=${styles_name}`
           styles_image_cache[styles_name][name] = url
           return url
       }
       return undefined;
    }
}

function getTagList(tags, styleName, language='en-US') {
    let rlist=[]
    tags.forEach((k,i) => {
        rlist.push($el(
            "label.easyuse-prompt-styles-tag",
            {
                dataset: {
                    tag: k['name'],
                    name: language == 'zh-CN' && k['name_cn'] ? k['name_cn'] : k['name'],
                    imgName: k['imgName'],
                    index: i
                },
                $: (el) => {
                    el.children[0].onclick = () => {
                        el.classList.toggle("easyuse-prompt-styles-tag-selected");
                    };
                    el.onmousemove = (e) => {
                        displayImage(el.dataset.imgName, styleName, e)
                    };
                    el.onmouseout = () => {
                        hiddenImage()
                    };
                    el.onmouseover = (e) => {
                        displayImage(el.dataset.imgName, styleName)
                    };
                },
            },
            [
                $el("input",{
                    type: 'checkbox',
                    name: k['name']
                }),
                $el("span",{
                    textContent: language == 'zh-CN' && k['name_cn'] ? k['name_cn'] : k['name'],
                })
            ]
        ))
    });
    return rlist
}

const foocus_base_path = "https://raw.githubusercontent.com/lllyasviel/Fooocus/main/sdxl_styles/samples/"
const empty_img = "data:image/jpeg;base64,/9j/4QAYRXhpZgAASUkqAAgAAAAAAAAAAAAAAP/sABFEdWNreQABAAQAAAA8AAD/4QNLaHR0cDovL25zLmFkb2JlLmNvbS94YXAvMS4wLwA8P3hwYWNrZXQgYmVnaW49Iu+7vyIgaWQ9Ilc1TTBNcENlaGlIenJlU3pOVGN6a2M5ZCI/PiA8eDp4bXBtZXRhIHhtbG5zOng9ImFkb2JlOm5zOm1ldGEvIiB4OnhtcHRrPSJBZG9iZSBYTVAgQ29yZSA5LjEtYzAwMSA3OS4xNDYyODk5Nzc3LCAyMDIzLzA2LzI1LTIzOjU3OjE0ICAgICAgICAiPiA8cmRmOlJERiB4bWxuczpyZGY9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkvMDIvMjItcmRmLXN5bnRheC1ucyMiPiA8cmRmOkRlc2NyaXB0aW9uIHJkZjphYm91dD0iIiB4bWxuczp4bXA9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC8iIHhtbG5zOnhtcE1NPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvbW0vIiB4bWxuczpzdFJlZj0iaHR0cDovL25zLmFkb2JlLmNvbS94YXAvMS4wL3NUeXBlL1Jlc291cmNlUmVmIyIgeG1wOkNyZWF0b3JUb29sPSJBZG9iZSBQaG90b3Nob3AgMjUuMSAoMjAyMzA5MDUubS4yMzE2IDk3OWM4NmQpICAoV2luZG93cykiIHhtcE1NOkluc3RhbmNlSUQ9InhtcC5paWQ6RjA3NEU1QzNCNUJBMTFFRUExMUVDNkZDRjI0NzlBN0QiIHhtcE1NOkRvY3VtZW50SUQ9InhtcC5kaWQ6RjA3NEU1QzRCNUJBMTFFRUExMUVDNkZDRjI0NzlBN0QiPiA8eG1wTU06RGVyaXZlZEZyb20gc3RSZWY6aW5zdGFuY2VJRD0ieG1wLmlpZDpGMDc0RTVDMUI1QkExMUVFQTExRUM2RkNGMjQ3OUE3RCIgc3RSZWY6ZG9jdW1lbnRJRD0ieG1wLmRpZDpGMDc0RTVDMkI1QkExMUVFQTExRUM2RkNGMjQ3OUE3RCIvPiA8L3JkZjpEZXNjcmlwdGlvbj4gPC9yZGY6UkRGPiA8L3g6eG1wbWV0YT4gPD94cGFja2V0IGVuZD0iciI/Pv/uAA5BZG9i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"
async function displayImage(imgName, styleName) {
    var e = event || window.event;
    var img = document.getElementById("show_image_id");
    var pxy= img.parentElement.getBoundingClientRect();
    if(imgName) {
        const url = await getStylesImage(imgName, styleName)
        img.src = url
        img.onerror = _ =>{
            img.src = empty_img
        }
    }
    var scale = app?.canvas?.ds?.scale || 1;
    var x = (e.pageX-pxy.x-100)/scale;
    var y = (e.pageY-pxy.y+25)/scale;
    img.style.left = x+"px";
    img.style.top = y+"px";
    img.style.display = "block";
    img.style.borderRadius = "10px";
    img.style.borderColor = "var(--fg-color)"
    img.style.borderWidth = "1px";
    img.style.borderStyle = "solid";
}
function hiddenImage(){ //theEvent用来传入事件，Firefox的方式
    var img = document.getElementById('show_image_id');
    img.style.display = "none";
}

// StylePromptSelector
app.registerExtension({
    name: 'comfy.easyUse.styleSelector',
    async beforeRegisterNodeDef(nodeType, nodeData, app) {

        if(nodeData.name == 'easy stylesSelector'){
            // 创建时
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                onNodeCreated ? onNodeCreated?.apply(this, arguments) : undefined;
                const styles_id =  this.widgets.findIndex((w) => w.name == 'styles');
                const language = localStorage['AGL.Locale'] || localStorage['Comfy.Settings.AGL.Locale'] || 'en-US'
                const list = $el("ul.easyuse-prompt-styles-list",[]);
                let styles_values = ''
                this.setProperty("values", [])

                let selector = this.addDOMWidget('select_styles',"btn",$el('div.easyuse-prompt-styles',[$el('div.tools', [
                        $el('button.delete',{
                            textContent: $t('Empty All'),
                            style:{},
                            onclick:()=>{
                                selector.element.children[0].querySelectorAll(".search").forEach(el=>{
                                    el.value = ''
                                })
                                selector.element.children[1].querySelectorAll(".easyuse-prompt-styles-tag-selected").forEach(el => {
                                    el.classList.remove("easyuse-prompt-styles-tag-selected");
                                    el.children[0].checked = false
                                })
                                selector.element.children[1].querySelectorAll(".easyuse-prompt-styles-tag").forEach(el => {
                                    el.classList.remove('hide')
                                })
                                this.setProperty("values", [])
                            }}
                        ),
                        $el('textarea.search',{
                            dir:"ltr",
                            style:{"overflow-y": "scroll"},
                            rows:1,
                            placeholder:$t("🔎 Type here to search styles ..."),
                            oninput:(e)=>{
                                let value = e.target.value
                                selector.element.children[1].querySelectorAll(".easyuse-prompt-styles-tag").forEach(el => {
                                    const name = el.dataset.name.toLowerCase()
                                    const tag = el.dataset.tag.toLowerCase()
                                    const lower_value = value.toLowerCase()
                                    if(name.indexOf(lower_value) != -1 || tag.indexOf(lower_value) != -1  || el.classList.value.indexOf("easyuse-prompt-styles-tag-selected")!=-1){
                                        el.classList.remove('hide')
                                    }
                                    else{
                                        el.classList.add('hide')
                                    }
                                })
                            }
                        })
                    ]),list,
                    $el('img',{id:'show_image_id',
                        style:{display:'none',position:'absolute'},
                        src:``,
                        onerror:()=>{
                            this.src = empty_img
                        }
                    })
                ]));

                Object.defineProperty(this.widgets[styles_id],'value',{
                    set:(value)=>{
                        styles_values = value
                        if(styles_values){
                            getStylesList(styles_values).then(_=>{
                                selector.element.children[1].innerHTML=''
                                if(styles_list_cache[styles_values]){
                                    let tags = styles_list_cache[styles_values]
                                    // 重新排序
                                    if(selector.value) tags = tags.sort((a,b)=> selector.value.includes(b.name) - selector.value.includes(a.name))
                                    this.properties["values"] = []
                                    let list = getTagList(tags, value, language);
                                    selector.element.children[1].append(...list)
                                    selector.element.children[1].querySelectorAll(".easyuse-prompt-styles-tag").forEach(el => {
                                        if (this.properties["values"].includes(el.dataset.tag)) {
                                            el.classList.add("easyuse-prompt-styles-tag-selected");
                                        }
                                        if(this.size?.[0]<150 || this.size?.[1]<150) this.setSize([425, 500]);
                                    })
                                }
                            })
                        }
                    },
                    get: () => {
                        return styles_values
                    }
                })

                
                let style_select_values = ''
                Object.defineProperty(selector, "value", {
                    set: (value) => {
                        setTimeout(_=>{
                            selector.element.children[1].querySelectorAll(".easyuse-prompt-styles-tag").forEach(el => {
                                let arr = value.split(',')
                                if (arr.includes(el.dataset.tag)) {
                                    el.classList.add("easyuse-prompt-styles-tag-selected");
                                    el.children[0].checked = true
                                }
                            })
                        },300)
                    },
                    get: () => {
                        selector.element.children[1].querySelectorAll(".easyuse-prompt-styles-tag").forEach(el => {
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
                        style_select_values = this.properties["values"].join(',');
                        return style_select_values;
                    }
                });

                let old_values = ''
                let style_lists_dom = selector.element.children[1]
                style_lists_dom.addEventListener('mouseenter', function (e) {
                    let value = ''
                    style_lists_dom.querySelectorAll(".easyuse-prompt-styles-tag-selected").forEach(el=> value+=el.dataset.tag)
                    old_values = value
                })
                style_lists_dom.addEventListener('mouseleave', function (e) {
                    let value = ''
                    style_lists_dom.querySelectorAll(".easyuse-prompt-styles-tag-selected").forEach(el=> value+=el.dataset.tag)
                    let new_values = value
                    if(old_values != new_values){
                        // console.log("选项发生了变化")
                        // 获取搜索值
                        const search_value = document.getElementsByClassName('search')[0]['value']
                        // 重新排序
                        const tags = styles_list_cache[styles_values].sort((a,b)=> new_values.includes(b.name) - new_values.includes(a.name))
                        style_lists_dom.innerHTML = ''
                        let list = getTagList(tags, styles_values, language);
                        style_lists_dom.append(...list)
                        style_lists_dom.querySelectorAll(".easyuse-prompt-styles-tag").forEach(el => {
                            if (new_values.includes(el.dataset.tag)) {
                                el.classList.add("easyuse-prompt-styles-tag-selected");
                                el.children[0].checked = true;
                            }
                            if(search_value){
                                if(el.dataset.name.indexOf(search_value) != -1 || el.dataset.tag.indexOf(search_value) != -1  || el.classList.value.indexOf("easyuse-prompt-styles-tag-selected")!=-1){
                                    el.classList.remove('hide')
                                }
                                else{
                                    el.classList.add('hide')
                                }
                            }

                        })
                    }
                })


                // 初始化
                setTimeout(_=>{
                    if(!styles_values) {
                        styles_values = 'fooocus_styles'
                        getStylesList(styles_values).then(_=>{
                            selector.element.children[1].innerHTML=''
                            if(styles_list_cache[styles_values]){
                                let tags = styles_list_cache[styles_values]
                                // 重新排序
                                if(selector.value) tags = tags.sort((a,b)=> selector.value.includes(b.name) - selector.value.includes(a.name))
                                let list = getTagList(tags, styles_values, language);
                                selector.element.children[1].append(...list)
                            }
                        })
                    }
                    if(this.size?.[0]<150 || this.size?.[1]<150) this.setSize([425, 500]);
                    //
                },100)

                return onNodeCreated;
            }
        }
    }
})