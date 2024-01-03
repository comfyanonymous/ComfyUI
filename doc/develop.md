

## Webhook
### on_start
Args: 
task_id, str, 任务的id


### on_end
Args:
code, int, 任务状态，1是成功，其它的是错误码
task_id, str, 任务的id
result, dict, 任务的结果
message, str, 如果是失败，失败的错误描述


### on_processing
Args:
task_id, str, 任务的id
progress, float, 0-1, 任务的进度



## 外部节点
将插件的信息，写到config/default_plugins.json中
这样的话，启动时会自动拉取代码。

建议之间到custom_nodes/ComfyUI-Manager/custom-node-list.json中找对应插件的配置，然后加上自己期望的commit id信息。