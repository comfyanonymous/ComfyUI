const WC_VARIABLES = {
	mainWidth : 100,
	selectWidth : 300,
	textColor: 'var(--fg-color)',
	bgMain : 'var(--bg-color)',
	itemColor: 'var(--item-color)',
	borderColor : 'var(--fg-color)',
	selectedColor : 'var(--sel-color)',
}

const OPE_VARIABLES = {
	mainWidth : 100,
	mainHeight: 100,
	topIdent : 20,
	borderColor : 'var(--fg-color)',
	bgMain : 'var(--bg-color)',
	editorURL : 'https://zhuyu1997.github.io/open-pose-editor/'
}

log = console.log;

class WC{
	constructor(TARGET){
		this.mainFrame = $Add('div', TARGET, {className : 'topMenu'});
		
		this.comfyMenuHidden = false;
		
		this.loader = $Add("input", $d.body, {
			type: "file",
			accept: ".json",
			style: { display: "none" },
			onchange: () => {
				this.restore(this.loader.files[0]);
			},
		});
		
		//this.slideFrame = $Add('div', this.mainFrame, {innerHTML : '↔', style : {background : WC_VARIABLES.bgMain, position: 'absolute', left: `${WC_VARIABLES.mainWidth}px`, border : `solid 1px ${WC_VARIABLES.borderColor}`, borderLeft : `solid 0 white`, cursor: 'pointer'}, onclick : ()=>{this.slide()} } );
		this.Title = $Add('span', this.mainFrame, {innerHTML: 'ComfyUI', className : 'title'});
		
		this.queuePanel = $Add('span', this.mainFrame, {style : {position: 'absolute', right: 0} } );
		this.queueInfoButton = $Add('span', this.queuePanel, {innerHTML : '≡', className : 'button', onclick : ()=>{this.showHideComfyMenu()} } );
		this.generateButton = $Add('span', this.queuePanel, {innerHTML : 'GENERATE', className : 'button' , onclick : function(){app.queuePrompt(0, this.batchCount)} } );

		this.controlPanel = $Add('span', this.mainFrame, {className : 'workflowControlPanel'});
		
		$Add('span', this.controlPanel, {innerHTML : '	|	'});
		
		this.itemSelect = $Add('select', this.controlPanel, {className: 'comboBox', onchange : () => {this.select() } } );
		
		this.addWorkflow = $Add('span', this.controlPanel, {type : 'button',  innerHTML : 'Add', className : 'button', onclick : () => {this.add(prompt(), jsonEncode(app.graph.serialize() ) ) } } );
		this.removeWorkflow = $Add('span', this.controlPanel, {type : 'button', innerHTML : 'Remove', className : 'button', onclick : () => {this.remove() } } );
		this.renameWorkflow = $Add('span', this.controlPanel, {type : 'button', innerHTML : 'Rename', className : 'button', onclick : () => {this.rename() } } );
		this.backupWorkflow = $Add('span', this.controlPanel, {type : 'button', innerHTML : 'Backup', className : 'button', onclick : () => {this.backup() } } );
		this.restoreWorkflow = $Add('span', this.controlPanel, {type : 'button', innerHTML : 'Restore', className : 'button', onclick : () => {this.loader.click() } } );
		
		$Add('span', this.controlPanel, {innerHTML : '	|	'});
		
		this.unselectWorkflow = $Add('option', this.itemSelect, {value : -1, style : {}, innerHTML : 'Not selected' } );
		
		this.selected = undefined;
		this.items = [];
		
		this.load();
		
		this.saveInterval = setInterval(() => {
			if (this.selected != undefined){
				this.items[this.selected].save(jsonEncode(app.graph.serialize()));
			}
			this.save();
		}, 1000);
		
		this.opened = true;
		//this.slide();
		//this.showHideComfyMenu();
	}
	showHideComfyMenu(){
		if (this.comfyMenuHidden){
			qs('.comfy-menu').style.display = '';
		}else{
			qs('.comfy-menu').style.display = 'none';
		}
		this.comfyMenuHidden = !this.comfyMenuHidden;
	}
	slide(){
		if (this.opened){
			this.opened = !this.opened;
			this.mainFrame.style.left = `${0 - WC_VARIABLES.mainWidth}px`;
		}else{
			this.opened = !this.opened;
			this.mainFrame.style.left = 0;
		}
	}
	add(NAME, GRAPH){
		if (NAME != null && NAME != '')this.items.push(new WCItem(this, NAME, GRAPH));
	}
	remove(){
		if (this.selected == undefined) return 0;
		this.items[this.selected].remove();
		this.items.splice(this.selected, 1);
		this.unselect();
		for (let i = 0; i < this.items.length; i++){
			this.items[i].id = i;
		}
	}
	rename(){
		if (this.selected == undefined) return 0;
		this.items[this.selected].rename(prompt());
	}
	save(){
		let data = [];
		for(let i = 0; i < this.items.length; i++){
			data.push({name : this.items[i].name, graph : this.items[i].graph});
		}
		lsSet('WorkflowCollection', jsonEncode(data));
	}
	backup(){
		const json = lsGet('WorkflowCollection');
			const blob = new Blob([json], { type: "application/json" });
			const url = URL.createObjectURL(blob);
			const a = $Add("a", $d.body, {
				href: url,
				download: "WorkflowCollection.json",
				style: { display: "none" }
			});
			a.click();
			setTimeout(function () {
				a.remove();
				window.URL.revokeObjectURL(url);
			}, 0);
	}
	restore(FILE){
		if (FILE.type === "application/json" || FILE.name.endsWith(".json")) {
			const reader = new FileReader();
			reader.onload = () => {
				this.load(jsonDecode(reader.result));
			};
			reader.readAsText(FILE);
		}
	}
	load(DATA){
		if (DATA != undefined){
			for (let i = 0; i < this.items.length; i++){
				this.items[i].remove();
			}
			this.items = [];
			lsSet('WorkflowCollection', jsonEncode(DATA));
		}
		if (lsGet('WorkflowCollection') == null) {
			//lsSet('WorkflowCollection', jsonEncode(DEFAULT_COLLECTIONS));
			this.loadDefault();
			return
		}
		let data = jsonDecode(lsGet('WorkflowCollection'));
		if (data == null) {data = []};
		for (let i = 0; i < data.length; i++){
			this.add(data[i].name, data[i].graph);
		}
		this.unselect();
	}
	async loadDefault(){
		let listResp = await fetch('DefaultWorkflows/index.json');
		log(listResp);
		let list = await listResp.json();
		log(list);
		let data = [];
		log(data);
		for (let i = 0; i < list.length; i++){
			log(i);
			let temp = await fetch(`DefaultWorkflows/${list[i]}`);
			log(temp);
			data.push( {name : list[i].replace('.json', ''), graph : jsonEncode(await temp.json()) } );
			log(data);
		}
		this.load(data);
	}
	select(){
		this.selected = (this.itemSelect.value == -1) ? undefined : this.itemSelect.value;
		for (let i = 0; i < this.items.length; i++){
			if (i == this.selected){
				this.items[i].frame.style.background = WC_VARIABLES.selectedColor;
			}else{
				this.items[i].frame.style.background = WC_VARIABLES.itemColor;
			}
		}
		if (this.selected == undefined){
			this.unselectWorkflow.style.background = WC_VARIABLES.selectedColor;
		}else{
			this.unselectWorkflow.style.background = WC_VARIABLES.itemColor;
		}
		if (this.selected != undefined) this.items[Number(this.selected)].select()
	}
	unselect(){
		this.selected = undefined;
		this.select();
	}
}

class OPE{
	constructor(TARGET){
		this.mainFrame = $Add('div', TARGET, {className : 'FullWindow'});
		this.slideFrame = $Add('span', wc.mainFrame, {innerHTML : 'OPE', className : 'button', onclick : ()=>{this.slide()} } );
		
		this.iFrame = $Add('iframe', this.mainFrame, {width: '100%', height: '100%', src : OPE_VARIABLES.editorURL, style : {border : 'none'} } );
		
		this.opened = false;
	}
	slide(){
		if (this.opened){
			this.opened = !this.opened;
			this.mainFrame.style.left = `-${OPE_VARIABLES.mainWidth}%`;
			this.slideFrame.style.background = OPE_VARIABLES.bgMain;
		}else{
			this.opened = !this.opened;
			this.mainFrame.style.left = 0;
			this.slideFrame.style.background = WC_VARIABLES.selectedColor;
		}
	}
}

class WCItem{
	constructor(PARENT, NAME, GRAPH){
		this.parent = PARENT;
		this.name = NAME;
		this.frame = $Add('option', this.parent.itemSelect, {innerHTML: this.name, value: this.parent.items.length, style:{} });
		this.id = this.parent.items.length;
		this.graph = GRAPH;
	}
	remove(){
		this.frame.remove();
		delete this;
	}
	select(){
		//this.parent.selected = this.id;
		//this.parent.select();
		app.graph.clear();
		this.load();
	}
	rename(NAME){
		this.name = NAME;
		this.frame.innerHTML = this.name;
	}
	save(GRAPH){
		this.graph = GRAPH;
	}
	load(){
		app.loadGraphData(jsonDecode(this.graph));
	}
}

class NC{
	constructor(TARGET, WIDGET){
		this.frame = $Add('div', TARGET, {style : {marginTop : '5px'} } );
		this.widget = WIDGET;
		this.type = WIDGET.type;
		
		//this.name = $Add('div', this.frame, {innerHTML : this.widget.name});
		this.addController();
	}
	addController(){
		switch (this.type){
			case 'number':
				this.name = $Add('div', this.frame, {innerHTML : this.widget.name});
				this.controller = $Add('input', this.frame, {type : 'number' , min : this.widget.options.min, max : this.widget.options.max, step : this.widget.options.step/10, value : this.widget.value, onchange : ()=>{this.changeController()}});
				break;
			case 'toggle':
				this.name = $Add('div', this.frame, {innerHTML : this.widget.name});
				this.controller = $Add('input', this.frame, {type : 'checkbox', checked : this.widget.value, onchange : ()=>{this.changeController()}});
				break;
			case 'combo':
				this.name = $Add('div', this.frame, {innerHTML : this.widget.name});
				this.controller = $Add('select', this.frame, {onchange : ()=>{this.changeController()}});
				for (let i = 0; i < this.widget.options.values.length; i++){
					$Add('option', this.controller, {value : this.widget.options.values[i], innerHTML : this.widget.options.values[i]});
				}
				this.controller.value = this.widget.value;
				break;
			case 'text':
				this.name = $Add('div', this.frame, {innerHTML : this.widget.name});
				this.controller = $Add('input', this.frame, {type : 'text', value : this.widget.value, onchange : ()=>{this.changeController()}});
				break;
			case 'customtext':
				this.name = $Add('div', this.frame, {innerHTML : this.widget.name});
				this.controller = $Add('textarea', this.frame, {value : this.widget.value, onchange : ()=>{this.changeController()}});
				break;
			case 'button':
				this.controller = $Add('input', this.frame, {type : 'button', value : this.widget.name, onclick : ()=>{this.widget.callback()} } );
				break;
		}
	}
	changeController(){
		switch(this.type){
			case 'number':
			case 'combo':
			case 'text':
			case 'customtext':
				this.widget.value = this.controller.value;
				break;
			case 'toggle':
				this.widget.value = this.controller.checked;
		}
	}
	update(){
		
	}
}

class NCITEM{
	constructor(TARGET, NODEID, PARENT){
		this.id = NODEID;
		this.parent = PARENT;
		//log(this.id);
		this.editMode = true;
		this.haveImage = false;
		
		let nodes = app.graph._nodes_by_id;
		this.node = nodes[this.id];
		this.controllers = [];
		
		this.frame = $Add('div', TARGET, {className: 'button', style : {position : 'absolute', minHeight: '100px', minWidth : '100px', overflow : 'auto'} } );
		if(this.node.bgcolor != undefined) this.frame.style.background = this.node.bgcolor;
		//title
		this.title = $Add('div', this.frame, {className : 'button', innerHTML : `<b style="pointer-events:none">id:</b>${this.id} <b style="pointer-events:none">title:</b>${this.node.title}`} );
		if (this.node.color != undefined) this.title.style.background = this.node.color;
		
		//config farame
		this.config = $Add('div', this.frame, {style : {position : 'absolute', width: '100%', height : '100%', top : 0/*, background : '#0000'*/} } );
		
		this.drag = $Add('div', this.config, {style : {height : '25px', cursor : 'move', position: 'absolute', top : 0, width : '100%'}, onmousedown : (e)=>{this.mouseDownDrag(e)} } );
		
		//this.haveImageToggle = $Add('input', this.config, {type : 'checkbox', style : {position : 'absolute', top : '75px', right : 0}, onchange : ()=>{this.switchHaveImage()}});
		this.deleteButton = $Add('div', this.config, {innerHTML : 'X', className : 'button', style : {position : 'absolute', top : 0, right : 0} } );
		
		this.setColor = $Add('input', this.config, {type : 'color', className : 'button', style : {position : 'absolute', top : '25px', right : 0}, onchange : (e)=>{this.colorSet(e)} } );
		if (this.node.color != undefined) this.setColor.value = (this.node.color != undefined) ? ((this.node.color.length == 4) ? `#${this.node.color[1]}0${this.node.color[2]}0${this.node.color[3]}0` : this.node.color) : '#000000';
		
		this.setBGColor = $Add('input', this.config, {type : 'color', className : 'button', style : {position : 'absolute', top : '50px', right : 0}, onchange : (e)=>{this.BGcolorSet(e)} } );
		if(this.node.bgcolor != undefined) this.setBGColor.value = (this.node.bgcolor != undefined) ? ((this.node.bgcolor.length == 4) ? `#${this.node.bgcolor[1]}0${this.node.bgcolor[2]}0${this.node.bgcolor[3]}0` : this.node.bgcolor) : '#000000';
		
		this.resize = $Add('div', this.config, {className : 'button', style : {position : 'absolute', bottom : 0, right : 0, height : '25px', width : '25px', cursor : 'nwse-resize'}, onmousedown : (e)=>{this.mouseDownResize(e)} } );
		
		//widgets
		if (this.node.widgets != undefined) for (let i = 0; i < this.node.widgets.length; i++){
			this.controllers.push(new NC(this.frame, this.node.widgets[i]));
		}
		//show images
		if(this.node.imgs != undefined){
			this.img = $Add('img', this.frame, {src : this.node.imgs[0].src, style : {maxWidth : '100%'} } );
		}
		
		this.mouseOffset = [0,0];
		
		this.pos = [250, 100];
		this.size = [this.frame.clientWidth, this.frame.clientHeight];
		this.move();
		this.resi();
	}
	update(){
		if(this.node.bgcolor != undefined) this.frame.style.background = this.node.bgcolor;
		if(this.node.bgcolor != undefined) this.setBGColor.value = (this.node.bgcolor != undefined) ? ((this.node.bgcolor.length == 4) ? `#${this.node.bgcolor[1]}0${this.node.bgcolor[2]}0${this.node.bgcolor[3]}0` : this.node.bgcolor) : '#000000';
		//title
		this.title.innerHTML = `<b style="pointer-events:none">id:</b>${this.id} <b style="pointer-events:none">title:</b>${this.node.title}`;
		if (this.node.color != undefined) this.title.style.background = this.node.color;
		if (this.node.color != undefined) this.setColor.value = (this.node.color != undefined) ? ((this.node.color.length == 4) ? `#${this.node.color[1]}0${this.node.color[2]}0${this.node.color[3]}0` : this.node.color) : '#000000';
		
		for (let i = 0; i < this.controllers.length; i++) this.controllers[i].update();
		
		if(this.node.imgs != undefined){
			if(this.img == undefined){
				this.img = $Add('img', this.frame, {src : this.node.imgs[0].src, style : {maxWidth : '100%'} } );
			}else{
				this.img.src = this.node.imgs[0].src;
			}
		}
	}
	colorSet(EVENT){
		//log(EVENT.target.value);
		this.title.style.background = EVENT.target.value;
		this.node.color = EVENT.target.value;
	}
	BGcolorSet(EVENT){
		this.frame.style.background = EVENT.target.value;
		this.node.bgcolor = EVENT.target.value;
	}
	editModeOff(){
		this.editMode = false;
		this.config.style.display = 'none';
	}
	editModeOn(){
		this.editMode = true;
		this.config.style.display = '';
	}
	resi(OFFSET){
		if(OFFSET == undefined){
			this.frame.style.width = `${this.size[0]}px`;
			this.frame.style.height = `${this.size[1]}px`;
		}else{
			this.size[0] += OFFSET[0];
			this.size[1] += OFFSET[1];
			this.resi();
		}
	}
	move(OFFSET){
		if(OFFSET == undefined){
			this.frame.style.left = `${this.pos[0]}px`;
			this.frame.style.top = `${this.pos[1]}px`;
		}else{
			this.pos[0] += OFFSET[0];
			this.pos[1] += OFFSET[1];
			this.move();
		}
	}
	mouseDownResize(EVENT){
		this.mouseOffset = [0,0];
		this.mousePos = [EVENT.clientX, EVENT.clientY];
		//log(EVENT);
		this.moveMode = false; //true - move mode
		this.parent.mainFrame.onmousemove = (e)=>{this.mouseMove(e)};
		this.parent.mainFrame.onmouseup = (e)=>{this.mouseUp(e)};
	}
	mouseDownDrag(EVENT){
		this.mouseOffset = [0,0];
		this.mousePos = [EVENT.clientX, EVENT.clientY];
		//log(EVENT);
		this.moveMode = true; //flase - resize mode
		this.parent.mainFrame.onmousemove = (e)=>{this.mouseMove(e)};
		this.parent.mainFrame.onmouseup = (e)=>{this.mouseUp(e)};
	}
	mouseMove(EVENT){
		event.preventDefault();
		this.mouseOffset = [EVENT.clientX - this.mousePos[0], EVENT.clientY - this.mousePos[1]];
		//log(this.mouseOffset,this.mousePos,this.mouseOffset);
		if (this.moveMode){
			this.move(this.mouseOffset);
		}else{
			this.resi(this.mouseOffset);
		}
		this.mousePos = [EVENT.clientX, EVENT.clientY];
		this.mouseOffset = [0,0];
	}
	mouseUp(EVENT){

		this.pos[0] = Math.clamp(this.pos[0], 0, 1820);
		this.pos[0] = Math.round(this.pos[0]/25)*25;
		this.pos[1] = Math.clamp(this.pos[1], 0, 10000);
		this.pos[1] = Math.round(this.pos[1]/25)*25;
		this.move();
		
		this.size[0] = Math.clamp(this.size[0], 0, 1920);
		this.size[0] = Math.round(this.size[0]/25)*25;
		this.size[1] = Math.clamp(this.size[1], 0, 10000);
		this.size[1] = Math.round(this.size[1]/25)*25;
		this.resi();
		
		this.parent.mainFrame.onmouseup = undefined;
		this.parent.mainFrame.onmousemove = undefined;
	}
	destructor(){
		
	}
}

class NCP{
	constructor(TARGET){
		this.mainFrame = $Add('div', TARGET, {className : 'FullWindow'});
		this.slideFrame = $Add('span', wc.mainFrame, {innerHTML : 'NCP', className : 'button', onclick : ()=>{this.slide()} } );
		
		this.opened = false;
		this.editMode = false;
		
		$Add('div', this.mainFrame, {style : {height : '25px'} } );
		this.workflow = $Add('div', this.mainFrame, {id : 'NCPWorkflow'} );
		this.addPanel = $Add('div', this.mainFrame, {id : 'addDiv', style : {textAlign : 'center', position : 'absolute', left : 0, top : '50px', display : 'none'} } );
		
		//this.addDivButton = $Add('span', this.addPanel, {innerHTML : '+', className : 'button', onclick : ()=>{this.addNCClick()}});
		
		this.editModeButton = $Add('div', this.mainFrame, {innerHTML : 'Edit', className : 'button', style : {position : 'absolute', left : 0, top: '25px'}, onclick : ()=>{this.editSwitch();} } );
		
		this.NCItems = [];
		this.addedList = [];
		this.selectors = [];
		//on change
		/*this.onChange = new Event('onChange');
		this.createEventOnAfterChange();*/
		document.addEventListener('onImageChange', ()=>{this.nodesUpdate()});
	}
	nodesUpdate(){
		for (let i = 0; i < this.NCItems.length; i++){
			this.NCItems[i].update();
		}
	}
	/*async createEventOnAfterChange(){
		await sleep(1000);
		app.canvas.onAfterChange = ()=>{document.dispatchEvent(this.onAfterChange)};//()=>{this.nodesUpdate()};
		document.addEventListener('onChange', ()=>{this.nodesUpdate()});
	}*/
	addNCClick(EVENT){
		log(app.graph._nodes_by_id);
		//log(EVENT.target);
		let nodeid = EVENT.target.getAttribute('nodeid');
		//log(nodeid);
		this.NCItems.push(new NCITEM(this.workflow, nodeid, this));
		this.addedList.push(nodeid);
		this.clearSelectors();
		this.loadSelectors();
	}
	loadSelectors(){
		let nodes = app.graph._nodes_by_id;
		let j = 0;
		for (let i in nodes){
			this.selectors.push($Add('div', this.addPanel, {innerHTML : `<b style="pointer-events:none">id:</b>${nodes[i].id} <b style="pointer-events:none">title:</b>${nodes[i].title}`, className: 'button', onclick : (e)=>{this.addNCClick(e)}}));
			//log(i);
			this.selectors[j].setAttribute('nodeid', i);
			//log(this.addedList.indexOf(i), i, this.addedList);
			if (this.addedList.indexOf(i) != -1){
				this.selectors[j].style.display = 'none';
			}
			if (nodes[i].color != undefined) this.selectors[j].style.background = nodes[i].color;
			j++;
		}
	}
	clearSelectors(){
		for (let i = 0; i < this.selectors.length; i++){
			this.selectors[i].remove();
		}
		this.selectors = [];
	}
	slide(){
		if (this.opened){
			this.opened = !this.opened;
			this.mainFrame.style.left = `-${OPE_VARIABLES.mainWidth}%`;
			this.slideFrame.style.background = OPE_VARIABLES.bgMain;
		}else{
			this.opened = !this.opened;
			this.mainFrame.style.left = 0;
			this.slideFrame.style.background = WC_VARIABLES.selectedColor;
			this.nodesUpdate();
		}
	}
	editSwitch(){
		if (this.editMode){
			this.editMode = !this.editMode;
			this.editModeButton.style.background = OPE_VARIABLES.bgMain;
			this.addPanel.style.display = 'none';
			this.clearSelectors()
			for (let i = 0; i < this.NCItems.length; i++){
				this.NCItems[i].editModeOff();
			}
		}else{
			this.editMode = !this.editMode;
			this.editModeButton.style.background = WC_VARIABLES.selectedColor;
			this.addPanel.style.display = '';
			this.loadSelectors();
			for (let i = 0; i < this.NCItems.length; i++){
				this.NCItems[i].editModeOn();
			}
		}
	}
}

wc = new WC(qs('body'));
ope = new OPE(qs('body'));
ncp = new NCP(qs('body'));

