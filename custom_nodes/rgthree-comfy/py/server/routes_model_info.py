import os
import json
from aiohttp import web

from server import PromptServer
import folder_paths

from ..utils import path_exists
from .utils_server import get_param, is_param_falsy
from .utils_info import delete_model_info, get_model_info, set_model_info_partial, get_file_info

routes = PromptServer.instance.routes


def _check_valid_model_type(request):
  model_type = request.match_info['type']
  if model_type not in ['loras', 'checkpoints']:
    return web.json_response({'status': 404, 'error': f'Invalid model type: {model_type}'})
  return None


@routes.get('/rgthree/api/{type}')
async def api_get_models_list(request):
  """Returns a list of model types from user configuration.

  By default, a list of filenames are provided. If `format=details` is specified, a list of objects
  with additional _file info_ is provided. This includes modigied time, hasInfoFile, and imageLocal
  among others.
  """
  if _check_valid_model_type(request):
    return _check_valid_model_type(request)

  model_type = request.match_info['type']
  files = folder_paths.get_filename_list(model_type)
  format_param = get_param(request, 'format')
  if format_param == 'details':
    response = []
    for file in files:
      response.append(get_file_info(file, model_type))
    return web.json_response(response)

  return web.json_response(list(files))


@routes.get('/rgthree/api/{type}/info')
async def api_get_models_info(request):
  """Returns a list model info; either all or a specific ones if provided a 'files' param.

  If a `light` param is specified and not falsy, no metadata will be fetched.
  """
  if _check_valid_model_type(request):
    return _check_valid_model_type(request)

  model_type = request.match_info['type']
  files_param = get_param(request, 'files')
  maybe_fetch_metadata = files_param is not None
  if not is_param_falsy(request, 'light'):
    maybe_fetch_metadata = False
  api_response = await models_info_response(
    request, model_type, maybe_fetch_metadata=maybe_fetch_metadata
  )
  return web.json_response(api_response)


@routes.get('/rgthree/api/{type}/info/refresh')
async def api_get_refresh_get_models_info(request):
  """Refreshes model info; either all or specific ones if provided a 'files' param. """
  if _check_valid_model_type(request):
    return _check_valid_model_type(request)

  model_type = request.match_info['type']
  api_response = await models_info_response(
    request, model_type, maybe_fetch_civitai=True, maybe_fetch_metadata=True
  )
  return web.json_response(api_response)


@routes.get('/rgthree/api/{type}/info/clear')
async def api_get_delete_model_info(request):
  """Clears model info from the filesystem for the provided file."""
  if _check_valid_model_type(request):
    return _check_valid_model_type(request)

  api_response = {'status': 200}
  model_type = request.match_info['type']
  files_param = get_param(request, 'files')
  if files_param is not None:
    files_param = files_param.split(',')
  del_info = not is_param_falsy(request, 'del_info')
  del_metadata = not is_param_falsy(request, 'del_metadata')
  del_civitai = not is_param_falsy(request, 'del_civitai')
  if not files_param:
    api_response['status'] = '404'
    api_response['error'] = f'No file provided. Please pass files=ALL to clear {model_type} info.'
  else:
    if len(files_param) == 1 and files_param[
      0] == "ALL":  # Force the user to supply files=ALL to trigger all clearing.
      files_param = folder_paths.get_filename_list(model_type)
    for file_param in files_param:
      await delete_model_info(
        file_param,
        model_type,
        del_info=del_info,
        del_metadata=del_metadata,
        del_civitai=del_civitai
      )
  return web.json_response(api_response)


@routes.post('/rgthree/api/{type}/info')
async def api_post_save_model_data(request):
  """Saves data to a model by name. """
  if _check_valid_model_type(request):
    return _check_valid_model_type(request)

  model_type = request.match_info['type']
  api_response = {'status': 200}
  file_param = get_param(request, 'file')
  if file_param is None:
    api_response['status'] = '404'
    api_response['error'] = 'No model found at path'
  else:
    post = await request.post()
    await set_model_info_partial(file_param, model_type, json.loads(post.get("json")))
    info_data = await get_model_info(file_param, model_type)
    api_response['data'] = info_data
  return web.json_response(api_response)


@routes.get('/rgthree/api/{type}/img')
async def api_get_models_info_img(request):
  """ Returns an image response if one exists for the model. """
  if _check_valid_model_type(request):
    return _check_valid_model_type(request)

  model_type = request.match_info['type']
  file_param = get_param(request, 'file')
  file_path = folder_paths.get_full_path(model_type, file_param)
  if not path_exists(file_path):
    file_path = os.path.abspath(file_path)

  img_path = None
  for ext in ['jpg', 'png', 'jpeg']:
    try_path = f'{os.path.splitext(file_path)[0]}.{ext}'
    if path_exists(try_path):
      img_path = try_path
      break

  if not path_exists(img_path):
    api_response = {}
    api_response['status'] = '404'
    api_response['error'] = 'No model found at path'
    return web.json_response(api_response)

  return web.FileResponse(img_path)


async def models_info_response(
  request, model_type, maybe_fetch_civitai=False, maybe_fetch_metadata=False
):
  """Gets model info for all or a single model type."""
  api_response = {'status': 200, 'data': []}
  light = not is_param_falsy(request, 'light')
  files_param = get_param(request, 'files')
  if files_param is not None:
    files_param = files_param.split(',')
  else:
    files_param = folder_paths.get_filename_list(model_type)
  for file_param in files_param:
    info_data = await get_model_info(
      file_param,
      model_type,
      maybe_fetch_civitai=maybe_fetch_civitai,
      maybe_fetch_metadata=maybe_fetch_metadata,
      light=light
    )
    api_response['data'].append(info_data)
  return api_response
