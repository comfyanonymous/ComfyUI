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

class WC{
	constructor(TARGET){
		this.mainFrame = $Add('div', TARGET, {style: {position: 'fixed', width: `${WC_VARIABLES.mainWidth}%`, top: 0, left:0, background: WC_VARIABLES.bgMain , borderBottom : `solid 1px ${WC_VARIABLES.borderColor}`, zIndex: 3}});
		
		this.loader = $Add("input", $d.body, {
			type: "file",
			accept: ".json",
			style: { display: "none" },
			onchange: () => {
				this.restore(this.loader.files[0]);
			},
		});
		
		//this.slideFrame = $Add('div', this.mainFrame, {innerHTML : '↔', style : {background : WC_VARIABLES.bgMain, position: 'absolute', left: `${WC_VARIABLES.mainWidth}px`, border : `solid 1px ${WC_VARIABLES.borderColor}`, borderLeft : `solid 0 white`, cursor: 'pointer'}, onclick : ()=>{this.slide()} } );
		this.Title = $Add('span', this.mainFrame, {innerHTML: 'ComfyUI', style : {textAlign: 'center', marginRight: '20px'}});

		this.controlPanel = $Add('span', this.mainFrame, {style : {textAlign: 'center'} } );
		
		$Add('span', this.controlPanel, {innerHTML : '	|	'});
		
		this.itemSelect = $Add('select', this.controlPanel, {style : {width : `${WC_VARIABLES.selectWidth}px`, background : WC_VARIABLES.bgMain, color : WC_VARIABLES.textColor}, onchange : () => {this.select() } } );
		
		this.addWorkflow = $Add('span', this.controlPanel, {type : 'button',  innerHTML : 'Add', style : {cursor : 'pointer', border: `solid 1px ${WC_VARIABLES.textColor}`, borderRadius : '3px', padding : '2px', marginLeft: '5px'}, onclick : () => {this.add(prompt(), jsonEncode(app.graph.serialize() ) ) } } );
		this.removeWorkflow = $Add('span', this.controlPanel, {type : 'button', innerHTML : 'Remove', style : {cursor : 'pointer', border: `solid 1px ${WC_VARIABLES.textColor}`, borderRadius : '3px', padding : '2px', marginLeft: '5px'}, onclick : () => {this.remove() } } );
		this.renameWorkflow = $Add('span', this.controlPanel, {type : 'button', innerHTML : 'Rename', style : {cursor : 'pointer', border: `solid 1px ${WC_VARIABLES.textColor}`, borderRadius : '3px', padding : '2px', marginLeft: '5px'}, onclick : () => {this.rename() } } );
		this.backupWorkflow = $Add('span', this.controlPanel, {type : 'button', innerHTML : 'Backup', style : {cursor : 'pointer', border: `solid 1px ${WC_VARIABLES.textColor}`, borderRadius : '3px', padding : '2px', marginLeft: '5px'}, onclick : () => {this.backup() } } );
		this.restoreWorkflow = $Add('span', this.controlPanel, {type : 'button', innerHTML : 'Restore', style : {cursor : 'pointer', border: `solid 1px ${WC_VARIABLES.textColor}`, borderRadius : '3px', padding : '2px', marginLeft: '5px'}, onclick : () => {this.loader.click() } } );
		
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
			lsSet('WorkflowCollection', jsonEncode(DEFAULT_COLLECTIONS));
		}
		let data = jsonDecode(lsGet('WorkflowCollection'));
		if (data == null) {data = []};
		for (let i = 0; i < data.length; i++){
			this.add(data[i].name, data[i].graph);
		}
		this.unselect();
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

class OPE{
	constructor(TARGET){
		this.mainFrame = $Add('div', TARGET, {style: {position: 'fixed', width: `${OPE_VARIABLES.mainWidth}%`, height: `${OPE_VARIABLES.mainHeight}%`, top: 0, left:`-${OPE_VARIABLES.mainWidth}%`, background: WC_VARIABLES.bgMain, zIndex: 2}});
		this.slideFrame = $Add('span', wc.mainFrame, {innerHTML : 'OPE', style : {background : OPE_VARIABLES.bgMain, /*position: 'fixed', left: `50%`, top: 0,*/ border : `solid 1px ${OPE_VARIABLES.borderColor}`, borderRadius : '3px', padding : '2px', cursor: 'pointer'}, onclick : ()=>{this.slide()} } );
		
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

DEFAULT_COLLECTIONS = null;

if (lsGet('WorkflowCollection') == null) {
	fetch('DefaultWorkflows.json')
  .then(
	function(response) {
	  if (response.status !== 200) {
		console.log('Looks like there was a problem. Status Code: ' +
		  response.status);
		return;
	  }
	  // Examine the text in the response  
	  response.json().then(function(data) {
		DEFAULT_COLLECTIONS = data;
		console.log('test')
			lsSet('WorkflowCollection', jsonEncode(DEFAULT_COLLECTIONS));
		wc.load();
	  });
	}
  )
  .catch(function(err) {  
	console.log('Fetch Error :-S', err);  
  });
}

wc = new WC(qs('body'));
ope = new OPE(qs('body'));

