import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";
import { $el } from "../../../../scripts/ui.js";
import { $t } from "../common/i18n.js";
import { sleep } from "../common/utils.js";


const calculatePercent = (value, min, max) => ((value-min)/(max-min)*100)

const getLayerDefaultValue = (index) => {
    switch (index){
        case 3:
            return 2.5
        case 6:
            return 1
        default:
            return 0
    }
}

const addLayer = (_this, layer_total, arrays, sliders, i) => {
    let scroll = $el('div.easyuse-slider-item-scroll')
    let value = $el('div.easyuse-slider-item-input', {textContent: arrays[i]['value']})
    let label = $el('div.easyuse-slider-item-label', {textContent: 'L'+i})
    let girdTotal = (arrays[i]['max'] - arrays[i]['min']) / arrays[i]['step']
    let area = $el('div.easyuse-slider-item-area', {style:{ height: calculatePercent(arrays[i]['default'],arrays[i]['min'],arrays[i]['max']) + '%'}})
    let bar = $el('div.easyuse-slider-item-bar', {
        style:{ top: (100-calculatePercent(arrays[i]['default'],arrays[i]['min'],arrays[i]['max'])) + '%'},
        onmousedown: (e) => {
            let event = e || window.event;
            var y = event.clientY - bar.offsetTop;
            document.onmousemove = (e) => {
                let event = e || window.event;
                let top = event.clientY - y;
                if(top < 0){
                    top = 0;
                }
                else if(top > scroll.offsetHeight - bar.offsetHeight){
                    top = scroll.offsetHeight - bar.offsetHeight;
                }
                // top到最近的girdHeight值
                let girlHeight = (scroll.offsetHeight - bar.offsetHeight)/ girdTotal
                top = Math.round(top / girlHeight) * girlHeight;
                bar.style.top = Math.floor(top/(scroll.offsetHeight - bar.offsetHeight)* 100) + '%';
                area.style.height = Math.floor((scroll.offsetHeight - bar.offsetHeight - top)/(scroll.offsetHeight - bar.offsetHeight)* 100) + '%';
                value.innerText = parseFloat(parseFloat(arrays[i]['max'] - (arrays[i]['max']-arrays[i]['min']) * (top/(scroll.offsetHeight - bar.offsetHeight))).toFixed(2))
                arrays[i]['value'] = value.innerText
                _this.properties['values'][i] = i+':'+value.innerText
                window.getSelection ? window.getSelection().removeAllRanges() : document.selection.empty();
            }
        },
        ondblclick:_=>{
            bar.style.top = (100-calculatePercent(arrays[i]['default'],arrays[i]['min'],arrays[i]['max'])) + '%'
            area.style.height = calculatePercent(arrays[i]['default'],arrays[i]['min'],arrays[i]['max']) + '%'
            value.innerText = arrays[i]['default']
            arrays[i]['value'] = arrays[i]['default']
            _this.properties['values'][i] = i+':'+value.innerText
        }
    })
    document.onmouseup = _=> document.onmousemove = null;

    scroll.replaceChildren(bar,area)
    let item_div = $el('div.easyuse-slider-item',[
        value,
        scroll,
        label
    ])
    if(i == 3 ) layer_total == 12 ? item_div.classList.add('negative') : item_div.classList.remove('negative')
    else if(i == 6) layer_total == 12 ?  item_div.classList.add('positive') : item_div.classList.remove('positive')
    sliders.push(item_div)
    return item_div
}

const setSliderValue = (_this, type, refresh=false, values_div, sliders_value) => {
    let layer_total = type == 'sdxl' ? 12 : 16
    let sliders = []
    let arrays = Array.from({length: layer_total}, (v, i) => ({default: layer_total == 12 ? getLayerDefaultValue(i) : 0, min: -1, max: 3, step: 0.05, value:layer_total == 12 ? getLayerDefaultValue(i) : 0}))
    _this.setProperty("values", Array.from({length: layer_total}, (v, i) => i+':'+arrays[i]['value']))
    for (let i = 0; i < layer_total; i++) {
        addLayer(_this, layer_total, arrays, sliders, i)
    }
    if(refresh) values_div.replaceChildren(...sliders)
    else{
        values_div = $el('div.easyuse-slider', sliders)
        sliders_value = _this.addDOMWidget('values',"btn",values_div)
    }

    Object.defineProperty(sliders_value, 'value', {
        set: function() {},
        get: function() {
            return _this.properties.values.join(',');
        }
    });
    return {sliders, arrays, values_div, sliders_value}
}


app.registerExtension({
    name: 'comfy.easyUse.sliderControl',
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if(nodeData.name == 'easy sliderControl'){
            // 创建时
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                onNodeCreated && onNodeCreated.call(this);
                const mode =  this.widgets[0];
                const model_type = this.widgets[1];
                let layer_total = model_type.value == 'sdxl' ? 12 : 16
                let _this = this
                let values_div = null
                let sliders_value = null
                mode.callback = async()=>{
                    switch (mode.value) {
                        case 'ipadapter layer weights':
                            nodeData.output_name = ['layer_weights']
                            _this.outputs[0]['name'] = 'layer_weights'
                            _this.outputs[0]['label'] = 'layer_weights'
                            break
                    }
                }

                model_type.callback = async()=>{
                    if(values_div) {
                        let r2 = setSliderValue(_this, model_type.value, true, values_div, sliders_value)
                        values_div = r2.values_div
                        sliders_value = r2.sliders_value
                    }
                    _this.setSize(model_type.value == 'sdxl' ? [375,320] : [455,320])
                }

                let r1 =  setSliderValue(_this, model_type.value, false, values_div, sliders_value)
                let sliders = r1.sliders
                let arrays = r1.arrays
                values_div = r1.values_div
                sliders_value = r1.sliders_value
                setTimeout(_=>{
                    let values_widgets_index = this.widgets.findIndex((w) => w.name == 'values');
                    if(values_widgets_index != -1){
                        let old_values_widget = this.widgets[values_widgets_index];
                        let old_value = old_values_widget.value.split(',')
                        let layer_total = _this.widgets[1].value == 'sdxl' ? 12 : 16
                        for (let i = 0; i < layer_total; i++) {
                            let value = parseFloat(parseFloat(old_value[i].split(':')[1]).toFixed(2))
                            let item_div = sliders[i] || null
                             // 存在层即修改
                            if(arrays[i]){
                               arrays[i]['value'] = value
                                _this.properties['values'][i] = old_value[i]
                            }else{
                                arrays.push({default: layer_total == 12 ? getLayerDefaultValue(i) : 0, min: -1, max: 3, step: 0.05, value:layer_total == 12 ? getLayerDefaultValue(i) : 0})
                                _this.properties['values'].push(i+':'+arrays[i]['value'])
                                // 添加缺失层
                                item_div = addLayer(_this, layer_total, arrays, sliders, i)
                                values_div.appendChild(item_div)
                            }
                            // todo: 修改bar位置等
                            let input = item_div.getElementsByClassName('easyuse-slider-item-input')[0]
                            let bar = item_div.getElementsByClassName('easyuse-slider-item-bar')[0]
                            let area = item_div.getElementsByClassName('easyuse-slider-item-area')[0]
                            if(i == 3 ) layer_total == 12 ? item_div.classList.add('negative') : item_div.classList.remove('negative')
                            else if(i == 6) layer_total == 12 ?  item_div.classList.add('positive') : item_div.classList.remove('positive')
                            input.textContent = value
                            bar.style.top = (100-calculatePercent(value,arrays[i]['min'],arrays[i]['max'])) + '%'
                            area.style.height = calculatePercent(value,arrays[i]['min'],arrays[i]['max']) + '%'
                        }
                    }
                    _this.setSize(model_type.value == 'sdxl' ? [375,320] : [455,320])
                },1)
                return onNodeCreated;
            }
        }
    }
})