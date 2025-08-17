import json, sys

INFO = '''
If you saved a json workflow using 'Anything Everywhere?' nodes before the third regex was added, then you may find that when you load it, the Group Regex widget doesn't correctly default to '.*'.

If so, run python workflow_fixer.py filename.json newname.json to fix it.
'''

def convert(oldname, newname):
    with open(oldname) as f: workflow = json.load(f)
    for node in workflow['nodes']:
        if node['type'] == "Anything Everywhere?":
            print(f"Fixing {node['title'] if 'title' in node else 'Untitled AE? node'}...")
            node['widgets_values'][2] = '.*'
    with open(newname,'w') as f: print(json.dumps(workflow, indent=2), file=f)

if __name__=='__main__':
    if len(sys.argv)!=3:
        print(INFO)
    else:
        convert(sys.argv[1], sys.argv[2])

