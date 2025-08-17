import subprocess
import json
import os
import torch
import shutil

import server
import folder_paths

web = server.web

@server.PromptServer.instance.routes.post("/VHS_test")
async def test(request):
    try:
        req_data = await request.json()
        output = req_data['output']['gifs'][0]
        filename = output['filename']
        typ = output['type']
        base_args = ["ffprobe", "-v", "error", '-count_packets', "-show_entries", "stream", "-of", "json"]
        video = folder_paths.get_annotated_filepath(f'{filename} [{typ}]')
        vprobe = json.loads(subprocess.run(base_args + ['-select_streams', 'v:0', video],
                             capture_output=True, check=True).stdout)['streams'][0]
        aprobe = json.loads(subprocess.run(base_args + ['-select_streams', 'a:0', video],
                             capture_output=True, check=True).stdout)['streams']
        probe = {'video': vprobe}
        if len(aprobe) > 0:
            probe['audio'] = aprobe[0]
        errors = []
        compare = None
        for test in req_data['tests']:
            if test['type'] == 'compare':
                compare = test
                continue
            key = test['key']
            expected = test['value']
            actual = probe[test['type']][key]
            if expected != actual:
                #Consider always dumping type?
                errors.append(f'{key}: {expected} != {actual}')
        if len(errors) == 0 and compare is not None:
            if not os.path.exists(compare['filename']):
                os.makedirs(os.path.split(compare['filename'])[0], exist_ok=True)
                shutil.copy(video, compare['filename'])
                print("Missing comparison file has been initialized from output:", os.path.abspath(compare['filename']))
            else:
                #NOTE: This does not include the full memory optimizations of VHS
                #Tests should be small
                #TODO: Figure out way to do opacity comparison. May need to do blending in python
                #(easy, but slower and more memory intensive)
                diff = subprocess.run(['ffmpeg', '-v', 'error', '-i', video, '-i', compare['filename'], '-filter_complex', 'blend=all_mode=grainextract', '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-'], stdout=subprocess.PIPE, check=True).stdout
                diff = torch.frombuffer(diff, dtype=torch.uint8).to(dtype=torch.float32).div_(255)
                #diff = diff.reshape((-1,4))
                d = (diff-0.5).abs().sum()/diff.size(0)
                if d > compare['tolerance']:
                    errors.append(f'Similarity is outside specified tolerance: {d}')
                else:
                    print('d:', d)
        return web.json_response(errors)
    except Exception as e:
        return web.json_response(str(e))
