


## 关于数据结构的说明

总体而言，其实整个工程在不同的地方会出现下面几种不同的流程数据，包括：
* old-workflow, 原ComfyUI中导出保存的flow结构数据，它不会包括本工程新加的流程控制数据，也不会支持各种流程控制的节点
* compitable-workflow, 是本工程导出的一种流程数据格式，它与原版的ComfyUI兼容，只是在上面新增了一些数据，原则上，一个流程如果不包含流程控制节点，那么这个数据应该是可以直接导入到原ComfuUI中进行使用的。
* workflow, 是本工程网页端的一个中间格式数据，是之间将LGraph 序列化后的数据，在这个数据里，控制流程的输入输出是和普通数据的输入输出一起存放的，没有分开，原则上不兼容原ComfyUI。可以转成compitable-workflow数据。
* prompt, 计算所用的一个流程的最小化数据结构，包含了各个节点的连接信息，不包含超出执行流程所需信息的其他内容（例如group，reroute信息等）
* nodedef, 这个是后端返回的所有节点定义的数据，记录了所有节点的类型、输入输出等信息，不需要太多关注

每个流程保存的workflow实际配置应该是一个compatible-workflow（有的地方我会直接称为CWorkflow）,它可以兼容到普通的ComfyUI和我们的版本上。
这个兼容一方面是为了前端的导入导出，也同时也是为了保证部分需要用到workflow的一些节点能够正确运行，尽管这个会增加我们开发和维护的工作量。


这个文件的格式下，每个流程的Prompt数据将会增加一些相应的字段：
* flow_links, list, 每个元素是长度为5的list，是所有flow的链接数据，含义与普通输入输入link的前几个元素相同：
  * [link id, original node id, original flow output slot, target node id, target flow input id]
  * 需要注意的是，这里的slot和普通输入输出的slot是分开单独计算的，实际显示的时候，会先显示所有的
* support_flow_control, bool, 这个流程是否支持流程控制节点，默认false。


同时每个节点的nodes数据也会增加两个字段：
* "flow_inputs": list, 记录这个节点拥有的控制流程的所有flow输入
* "flow_outputs": list, 记录这个节点拥有的控制流程的所有flow输出

compatible-workflow的一个样例如下：
```
{
    "flow_links": [
        [1, 5, 0, 6, 0],
        ...
    ],
    ...
    "nodes":[
        {
            "id": 5, 
            ...
            "flow_inputs": [
                {
                    "name" : str, the name of the flow input,
                    "links": list, link ids of this input. null if not linked
                },
                ...
            ],
            "flow_outputs":[
                {
                    "name": str, the name of the flow output,
                    "link": int, link id. null if not linked
                },
                ...
            ]
        },
        ...
    ],
    ...
}
```


当要执行一个流程时，前端会需要给后端传一个prompt数据，在新增支持流程控制后，则需要额外传一个flow数据，它的格式如下：
```
flows = [
        'node_id': [
            ['str, target node id of output0', target_flow_input_slot], 
            ['5', 0], 
            ...
        ],
        '33': [None, ['10', 0]],
        ...
    ]
```
记录每个节点的所有输出流程的连接，如果某个流程输出没有连接其它节点（通常意味着不需要继续往下执行），则对应内容为None。



## 流程控制节点

### 通用说明
对于节点，可以通过定义两个额外的类属性来自定义支持的流程输入和流程输出
* FLOW_INPUTS, 流程输入
* FLOW_OUTPUTS, 流程输出
如果节点没有定义这两个，那么默认情况下都会加入一个'FROM'输入和'TO'输出，用来指定，执行的前一个节点和后一个节点。


### IfConotrolNode
if分支节点，可以用bool值来控制后续执行的流程。


### MergeFlowNode
流程合并节点，通常和IfControlNode一起用。
它的作用是，连接两个输入节点和一个输出节点，这两个输入节点不管是哪个节点完成后，都会执行后面流程输出所连接的节点。
（简单理解为IF的右括号，if节点衍生出两个分支之后，可以通过这个回到同一个流程上，之所以需要这个节点是因为，如果节点的输入支持连接多个其他节点的话，改动较大，暂时通过这样的MergeFlowNode节点来实现连接到同一个输入点这个事情。）


### LoopFlowNode
循环节点。


