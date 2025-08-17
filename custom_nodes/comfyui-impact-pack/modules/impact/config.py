import configparser
import os
import logging


version_code = [8, 22]
version = f"V{version_code[0]}.{version_code[1]}" + (f'.{version_code[2]}' if len(version_code) > 2 else '')

my_path = os.path.dirname(__file__)
old_config_path = os.path.join(my_path, "impact-pack.ini")
config_path = os.path.join(my_path, "..", "..", "impact-pack.ini")
latent_letter_path = os.path.join(my_path, "..", "..", "latent.png")


def write_config():
    config = configparser.ConfigParser()
    config['default'] = {
                            'sam_editor_cpu': str(get_config()['sam_editor_cpu']),
                            'sam_editor_model': get_config()['sam_editor_model'],
                            'custom_wildcards': get_config()['custom_wildcards'],
                            'disable_gpu_opencv': get_config()['disable_gpu_opencv'],
                        }
    with open(config_path, 'w') as configfile:
        config.write(configfile)


def read_config():
    try:
        config = configparser.ConfigParser()
        config.read(config_path)
        default_conf = config['default']

        if not os.path.exists(default_conf['custom_wildcards']):
            logging.warning(f"[Impact Pack] custom_wildcards path not found: {default_conf['custom_wildcards']}. Using default path.")
            default_conf['custom_wildcards'] = os.path.join(my_path, "..", "..", "custom_wildcards")

        return {
                    'sam_editor_cpu': default_conf['sam_editor_cpu'].lower() == 'true' if 'sam_editor_cpu' in default_conf else False,
                    'sam_editor_model': default_conf['sam_editor_model'].lower() if 'sam_editor_model' else 'sam_vit_b_01ec64.pth',
                    'custom_wildcards': default_conf['custom_wildcards'] if 'custom_wildcards' in default_conf else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "custom_wildcards")),
                    'disable_gpu_opencv': default_conf['disable_gpu_opencv'].lower() == 'true' if 'disable_gpu_opencv' in default_conf else True
               }

    except Exception:
        return {
            'sam_editor_cpu': False,
            'sam_editor_model': 'sam_vit_b_01ec64.pth',
            'custom_wildcards': os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "custom_wildcards")),
            'disable_gpu_opencv': True
        }


cached_config = None


def get_config():
    global cached_config

    if cached_config is None:
        cached_config = read_config()

    return cached_config
