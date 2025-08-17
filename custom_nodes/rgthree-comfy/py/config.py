import os
import json

from .utils import get_dict_value, set_dict_value, dict_has_key, load_json_file
from .pyproject import VERSION


def get_config_value(key):
  return get_dict_value(RGTHREE_CONFIG, key)


def extend_config(default_config, user_config):
  """ Returns a new config dict combining user_config into defined keys for default_config."""
  cfg = {}
  for key, value in default_config.items():
    if key not in user_config:
      cfg[key] = value
    elif isinstance(value, dict):
      cfg[key] = extend_config(value, user_config[key])
    else:
      cfg[key] = user_config[key] if key in user_config else value
  return cfg


def set_user_config(data: dict):
  """ Sets the user configuration."""
  count = 0
  for key, value in data.items():
    if dict_has_key(DEFAULT_CONFIG, key):
      set_dict_value(USER_CONFIG, key, value)
      set_dict_value(RGTHREE_CONFIG, key, value)
      count += 1
  if count > 0:
    write_user_config()


def get_rgthree_default_config():
  """ Gets the default configuration."""
  return load_json_file(DEFAULT_CONFIG_FILE, default={})


def get_rgthree_user_config():
  """ Gets the user configuration."""
  return load_json_file(USER_CONFIG_FILE, default={})


def write_user_config():
  """ Writes the user configuration."""
  with open(USER_CONFIG_FILE, 'w+', encoding='UTF-8') as file:
    json.dump(USER_CONFIG, file, sort_keys=True, indent=2, separators=(",", ": "))


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_FILE = os.path.join(THIS_DIR, '..', 'rgthree_config.json.default')
USER_CONFIG_FILE = os.path.join(THIS_DIR, '..', 'rgthree_config.json')

DEFAULT_CONFIG = {}
USER_CONFIG = {}
RGTHREE_CONFIG = {}


def refresh_config():
  """Refreshes the config."""
  global DEFAULT_CONFIG, USER_CONFIG, RGTHREE_CONFIG
  DEFAULT_CONFIG = get_rgthree_default_config()
  USER_CONFIG = get_rgthree_user_config()

  # Migrate old config options into "features"
  needs_to_write_user_config = False
  if 'patch_recursive_execution' in USER_CONFIG:
    del USER_CONFIG['patch_recursive_execution']
    needs_to_write_user_config = True

  if 'features' in USER_CONFIG and 'patch_recursive_execution' in USER_CONFIG['features']:
    del USER_CONFIG['features']['patch_recursive_execution']
    needs_to_write_user_config = True

  if 'show_alerts_for_corrupt_workflows' in USER_CONFIG:
    if 'features' not in USER_CONFIG:
      USER_CONFIG['features'] = {}
    USER_CONFIG['features']['show_alerts_for_corrupt_workflows'] = USER_CONFIG[
      'show_alerts_for_corrupt_workflows']
    del USER_CONFIG['show_alerts_for_corrupt_workflows']
    needs_to_write_user_config = True

  if 'monitor_for_corrupt_links' in USER_CONFIG:
    if 'features' not in USER_CONFIG:
      USER_CONFIG['features'] = {}
    USER_CONFIG['features']['monitor_for_corrupt_links'] = USER_CONFIG['monitor_for_corrupt_links']
    del USER_CONFIG['monitor_for_corrupt_links']
    needs_to_write_user_config = True

  if needs_to_write_user_config is True:
    print('writing new user config.')
    write_user_config()

  RGTHREE_CONFIG = {"version": VERSION} | extend_config(DEFAULT_CONFIG, USER_CONFIG)

  if "unreleased" in USER_CONFIG and "unreleased" not in RGTHREE_CONFIG:
    RGTHREE_CONFIG["unreleased"] = USER_CONFIG["unreleased"]

  if "debug" in USER_CONFIG and "debug" not in RGTHREE_CONFIG:
    RGTHREE_CONFIG["debug"] = USER_CONFIG["debug"]


def get_config():
  """Returns the congfig."""
  return RGTHREE_CONFIG


refresh_config()
