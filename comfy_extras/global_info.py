#
# Global Var
#
# Usage:
#   from comfy_extras import global_info
#   global_info.variables['comfyui.revision'] = 1832
#   print(f"log mode: {global_info.variables['logger.enabled']}")
#
variables = {}


#
# Global API
#
# Usage:
# [register API]
#   from comfy_extras import global_info
#
#   def api_hello(msg):
#       print(f"hello: {msg}")
#       return msg
#
#   global_info.APIs['hello'] = api_hello
#
# [use API]
#   from comfy_extras import global_info
#
#   test = global_info.try_call(api='hello', msg='an example')
#   print(f"'{test}' is returned")
#

APIs = {}

def try_call(**kwargs):
    if 'api' in kwargs:
        api_name = kwargs['api']
        try:
            api = APIs.get(api_name)
            if api is not None:
                del kwargs['api']
                return api(**kwargs)
            else:
                print(f"WARN: The '{kwargs['api']}' API has not been registered.")
        except Exception as e:
            print(f"ERROR: An exception occurred while calling the '{api_name}' API.")
            raise e
    else:
        return None


#
# Extension Info
#
# Usage:
#   from comfy_extras import global_info
#
#   global_info.extension_infos['my_extension'] = {'version': '0.1', 'name': 'me', 'description': 'example extension', }
#
extension_infos = {}
