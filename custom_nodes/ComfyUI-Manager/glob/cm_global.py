import traceback

#
# Global Var
#
# Usage:
#   import cm_global
#   cm_global.variables['comfyui.revision'] = 1832
#   print(f"log mode: {cm_global.variables['logger.enabled']}")
#
variables = {}


#
# Global API
#
# Usage:
# [register API]
#   import cm_global
#
#   def api_hello(msg):
#       print(f"hello: {msg}")
#       return msg
#
#   cm_global.register_api('hello', api_hello)
#
# [use API]
#   import cm_global
#
#   test = cm_global.try_call(api='hello', msg='an example')
#   print(f"'{test}' is returned")
#

APIs = {}


def register_api(k, f):
    global APIs
    APIs[k] = f


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
#   import cm_global
#
#   cm_global.extension_infos['my_extension'] = {'version': [0, 1], 'name': 'me', 'description': 'example extension', }
#
extension_infos = {}

on_extension_registered_handlers = {}


def register_extension(extension_name, v):
    global extension_infos
    global on_extension_registered_handlers
    extension_infos[extension_name] = v

    if extension_name in on_extension_registered_handlers:
        for k, f in on_extension_registered_handlers[extension_name]:
            try:
                f(extension_name, v)
            except Exception:
                print(f"[ERROR] '{k}' on_extension_registered_handlers")
                traceback.print_exc()

        del on_extension_registered_handlers[extension_name]


def add_on_extension_registered(k, extension_name, f):
    global on_extension_registered_handlers
    if extension_name in extension_infos:
        try:
            v = extension_infos[extension_name]
            f(extension_name, v)
        except Exception:
            print(f"[ERROR] '{k}' on_extension_registered_handler")
            traceback.print_exc()
    else:
        if extension_name not in on_extension_registered_handlers:
            on_extension_registered_handlers[extension_name] = []

        on_extension_registered_handlers[extension_name].append((k, f))


def add_on_revision_detected(k, f):
    if 'comfyui.revision' in variables:
        try:
            f(variables['comfyui.revision'])
        except Exception:
            print(f"[ERROR] '{k}' on_revision_detected_handler")
            traceback.print_exc()
    else:
        variables['cm.on_revision_detected_handler'].append((k, f))
