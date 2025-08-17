import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";
import { $el, ComfyDialog } from "../../../../scripts/ui.js";
import { $t } from '../common/i18n.js'
import { toast } from "../common/toast.js";
import {sleep, accSub} from "../common/utils.js";

let api_keys = []
let api_current = 0
let user_info = {}

const api_cost = {
    'sd3': 6.5,
    'sd3-turbo': 4,
}

class AccountDialog extends ComfyDialog {
    constructor() {
		super();
        this.lists = []
        this.dialog_div = null
        this.user_div = null
	}

    addItem(index, user_div){
        return $el('div.easyuse-account-dialog-item',[
          $el('input',{type:'text',placeholder:'Enter name',oninput: e=>{
              const dataIndex = Array.prototype.indexOf.call(this.dialog_div.querySelectorAll('.easyuse-account-dialog-item'), e.target.parentNode)
              api_keys[dataIndex]['name'] = e.target.value
          },value:api_keys[index]['name']}),
          $el('input.key',{type:'text',oninput: e=>{
              const dataIndex = Array.prototype.indexOf.call(this.dialog_div.querySelectorAll('.easyuse-account-dialog-item'), e.target.parentNode)
              api_keys[dataIndex]['key'] = e.target.value
          },placeholder:'Enter APIKEY', value:api_keys[index]['key']}),
          $el('button.choose',{textContent:$t('Choose'),onclick:async(e)=>{
                const dataIndex = Array.prototype.indexOf.call(this.dialog_div.querySelectorAll('.easyuse-account-dialog-item'), e.target.parentNode)
                let name = api_keys[dataIndex]['name']
                let key = api_keys[dataIndex]['key']
                if(!name){
                    toast.error($t('Please enter the account name'))
                    return
                }
                else if(!key){
                    toast.error($t('Please enter the APIKEY'))
                    return
                }
                let missing = true
                for(let i=0;i<api_keys.length;i++){
                    if(!api_keys[i].key) {
                        missing = false
                        break
                    }
                }
                if(!missing){
                    toast.error($t('APIKEY is not Empty'))
                    return
                }
                // 保存记录
                api_current = dataIndex
                const body = new FormData();
                body.append('api_keys', JSON.stringify(api_keys));
                body.append('current',api_current)
                const res = await api.fetchApi('/easyuse/stability/set_api_keys', {
                    method: 'POST',
                    body
                })
                 if (res.status == 200) {
                    const data = await res.json()
                    if(data?.account && data?.balance){
                        const avatar = data.account?.profile_picture || null
                        const email = data.account?.email || null
                        const credits = data.balance?.credits || 0
                        user_div.replaceChildren(
                            $el('div.easyuse-account-user-info', {
                                onclick:_=>{
                                    new AccountDialog().show(user_div);
                                }
                            },[
                                $el('div.user',[
                                   $el('div.avatar', avatar ? [$el('img',{src:avatar})] : '😀'),
                                   $el('div.info', [
                                    $el('h5.name', email),
                                    $el('h6.remark','Credits: '+ credits)
                                   ])
                                ]),
                                $el('div.edit', {textContent:$t('Edit')})
                            ])
                        )
                        toast.success($t('Save Succeed'))
                    }
                    else toast.success($t('Save Succeed'))
                    this.close()
                } else {
                    toast.error($t('Save Failed'))
                }
          }}),
          $el('button.delete',{textContent:$t('Delete'),onclick:e=>{
              const dataIndex = Array.prototype.indexOf.call(this.dialog_div.querySelectorAll('.easyuse-account-dialog-item'), e.target.parentNode)
              if(api_keys.length<=1){
                toast.error($t('At least one account is required'))
                return
              }
              api_keys.splice(dataIndex,1)
              this.dialog_div.removeChild(e.target.parentNode)
          }}),
      ])
    }

    show(userdiv) {
        api_keys.forEach((item,index)=>{
          this.lists.push(this.addItem(index,userdiv))
        })
        this.dialog_div = $el("div.easyuse-account-dialog", this.lists)
		super.show(
            $el('div.easyuse-account-dialog-main',[
               $el('div',[
                    $el('a',{href:'https://platform.stability.ai/account/keys',target:'_blank',textContent:$t('Getting Your APIKEY')}),
                ]),
                this.dialog_div,
            ])
        );
	}

    createButtons() {
		const btns = super.createButtons();
        btns.unshift($el('button',{
            type:'button',
            textContent:$t('Save Account Info'),
            onclick:_=>{
                let missing = true
                for(let i=0;i<api_keys.length;i++){
                    if(!api_keys[i].key) {
                        missing = false
                        break
                    }
                }
                if(!missing){
                    toast.error($t('APIKEY is not Empty'))
                }
                else {
                    const body = new FormData();
                    body.append('api_keys', JSON.stringify(api_keys));
                    api.fetchApi('/easyuse/stability/set_api_keys', {
                        method: 'POST',
                        body
                    }).then(res => {
                        if (res.status == 200) {
                            toast.success($t('Save Succeed'))

                        } else {
                            toast.error($t('Save Failed'))
                        }
                    })
                }
            }
        }))
        btns.unshift($el('button',{
            type:'button',
            textContent:$t('Add Account'),
            onclick:_=>{
                const name = 'Account '+(api_keys.length).toString()
                api_keys.push({name,key:''})
                const item = this.addItem(api_keys.length - 1)
                this.lists.push(item)
                this.dialog_div.appendChild(item)
            }
        }))
        return btns
    }
}


app.registerExtension({
    name: 'comfy.easyUse.account',
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if(nodeData.name == 'easy stableDiffusion3API'){

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = async function() {
                onNodeCreated ? onNodeCreated?.apply(this, arguments) : undefined;
                const seed_widget = this.widgets.find(w => ['seed_num','seed'].includes(w.name))
				const seed_control = this.widgets.find(w=> ['control_before_generate','control_after_generate'].includes(w.name))
                let model_widget = this.widgets.find(w => w.name == 'model')
                model_widget.callback = value =>{
                    cost_widget.value = '-'+api_cost[value]
                }
                const cost_widget = this.addWidget('text', 'cost_credit', '0', _=>{
                },{
                    serialize:false,
                })
                cost_widget.disabled = true
                setTimeout(_=>{
                    if(seed_control.name == 'control_before_generate' && seed_widget.value === 0){
                        seed_widget.value = Math.floor(Math.random() * 4294967294)
				    }
                    cost_widget.value = '-'+api_cost[model_widget.value]
                },100)
                let user_div = $el('div.easyuse-account-user', [$t('Loading UserInfo...')])
                let account = this.addDOMWidget('account',"btn",$el('div.easyuse-account',user_div));
                // 更新balance信息
                api.addEventListener('stable-diffusion-api-generate-succeed', async ({detail}) => {
                    let remarkDiv = user_div.querySelectorAll('.remark')
                    if(remarkDiv && remarkDiv[0]){
                        const credits = detail?.model ? api_cost[detail.model] : 0
                        if(credits) {
                            let balance = accSub(parseFloat(remarkDiv[0].innerText.replace(/Credits: /g,'')),credits)
                            if(balance>0){
                                remarkDiv[0].innerText = 'Credits: '+ balance.toString()
                            }
                        }
                    }
                    await sleep(10000)
                    const res = await api.fetchApi('/easyuse/stability/balance')
                    if(res.status == 200){
                        const data = await res.json()
                        if(data?.balance){
                            const credits = data.balance?.credits || 0
                            if(remarkDiv && remarkDiv[0]){
                                remarkDiv[0].innerText = 'Credits: ' + credits
                            }
                        }
                    }
                })
                // 获取api_keys
                const res = await api.fetchApi('/easyuse/stability/api_keys')
                if (res.status == 200){
                    let data = await res.json()
                    api_keys = data.keys
                    api_current = data.current
                    if (api_keys.length > 0 && api_current!==undefined){
                        const api_key = api_keys[api_current]['key']
                        const api_name = api_keys[api_current]['name']
                        if(!api_key){
                            user_div.replaceChildren(
                                $el('div.easyuse-account-user-info', {
                                    onclick:_=>{
                                        new AccountDialog().show(user_div);
                                    }
                                },[
                                    $el('div.user',[
                                        $el('div.avatar', '😀'),
                                        $el('div.info', [
                                            $el('h5.name', api_name),
                                            $el('h6.remark',$t('Click to set the APIKEY first'))
                                        ])
                                    ]),
                                    $el('div.edit', {textContent:$t('Edit')})
                                ])
                            )
                        }else{
                            // 获取账号信息
                            const res = await  api.fetchApi('/easyuse/stability/user_info')
                            if(res.status == 200){
                                const data = await res.json()
                                if(data?.account && data?.balance){
                                    const avatar = data.account?.profile_picture || null
                                    const email = data.account?.email || null
                                    const credits = data.balance?.credits || 0
                                    user_div.replaceChildren(
                                        $el('div.easyuse-account-user-info', {
                                            onclick:_=>{
                                                new AccountDialog().show(user_div);
                                            }
                                        },[
                                            $el('div.user',[
                                                $el('div.avatar', avatar ? [$el('img',{src:avatar})] : '😀'),
                                                $el('div.info', [
                                                    $el('h5.name', email),
                                                    $el('h6.remark','Credits: '+ credits)
                                                ])
                                            ]),
                                            $el('div.edit', {textContent:$t('Edit')})
                                        ])
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }
    }
})