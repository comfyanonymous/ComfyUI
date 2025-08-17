"""Utilities for Power Prompt nodes."""
import re
import os
import folder_paths

from .log import log_node_warn, log_node_info


def get_and_strip_loras(prompt, silent=False, log_node="Power Prompt"):
  """Collects and strips lora tags from a prompt."""
  pattern = r'<lora:([^:>]*?)(?::(-?\d*(?:\.\d*)?))?>'
  lora_paths = folder_paths.get_filename_list('loras')

  matches = re.findall(pattern, prompt)

  loras = []
  unfound_loras = []
  skipped_loras = []
  for match in matches:
    tag_path = match[0]

    strength = float(match[1] if len(match) > 1 and len(match[1]) else 1.0)
    if strength == 0:
      if not silent:
        log_node_info(log_node, f'Skipping "{tag_path}" with strength of zero')
      skipped_loras.append({'lora': tag_path, 'strength': strength})
      continue

    lora_path = get_lora_by_filename(tag_path, lora_paths, log_node=None if silent else log_node)
    if lora_path is None:
      unfound_loras.append({'lora': tag_path, 'strength': strength})
      continue

    loras.append({'lora': lora_path, 'strength': strength})

  return (re.sub(pattern, '', prompt), loras, skipped_loras, unfound_loras)


# pylint: disable = too-many-return-statements, too-many-branches
def get_lora_by_filename(file_path, lora_paths=None, log_node=None):
  """Returns a lora by filename, looking for exactl paths and then fuzzier matching."""
  lora_paths = lora_paths if lora_paths is not None else folder_paths.get_filename_list('loras')

  if file_path in lora_paths:
    return file_path

  lora_paths_no_ext = [os.path.splitext(x)[0] for x in lora_paths]

  # See if we've entered the exact path, but without the extension
  if file_path in lora_paths_no_ext:
    found = lora_paths[lora_paths_no_ext.index(file_path)]
    return found

  # Same check, but ensure file_path is without extension.
  file_path_force_no_ext = os.path.splitext(file_path)[0]
  if file_path_force_no_ext in lora_paths_no_ext:
    found = lora_paths[lora_paths_no_ext.index(file_path_force_no_ext)]
    return found

  # See if we passed just the name, without paths.
  lora_filenames_only = [os.path.basename(x) for x in lora_paths]
  if file_path in lora_filenames_only:
    found = lora_paths[lora_filenames_only.index(file_path)]
    if log_node is not None:
      log_node_info(log_node, f'Matched Lora input "{file_path}" to "{found}".')
    return found

  # Same, but force the input to be without paths
  file_path_force_filename = os.path.basename(file_path)
  lora_filenames_only = [os.path.basename(x) for x in lora_paths]
  if file_path_force_filename in lora_filenames_only:
    found = lora_paths[lora_filenames_only.index(file_path_force_filename)]
    if log_node is not None:
      log_node_info(log_node, f'Matched Lora input "{file_path}" to "{found}".')
    return found

  # Check the filenames and without extension.
  lora_filenames_and_no_ext = [os.path.splitext(os.path.basename(x))[0] for x in lora_paths]
  if file_path in lora_filenames_and_no_ext:
    found = lora_paths[lora_filenames_and_no_ext.index(file_path)]
    if log_node is not None:
      log_node_info(log_node, f'Matched Lora input "{file_path}" to "{found}".')
    return found

  # And, one last forcing the input to be the same
  file_path_force_filename_and_no_ext = os.path.splitext(os.path.basename(file_path))[0]
  if file_path_force_filename_and_no_ext in lora_filenames_and_no_ext:
    found = lora_paths[lora_filenames_and_no_ext.index(file_path_force_filename_and_no_ext)]
    if log_node is not None:
      log_node_info(log_node, f'Matched Lora input "{file_path}" to "{found}".')
    return found

  # Finally, super fuzzy, we'll just check if the input exists in the path at all.
  for index, lora_path in enumerate(lora_paths):
    if file_path in lora_path:
      found = lora_paths[index]
      if log_node is not None:
        log_node_warn(log_node, f'Fuzzy-matched Lora input "{file_path}" to "{found}".')
      return found

  if log_node is not None:
    log_node_warn(log_node, f'Lora "{file_path}" not found, skipping.')

  return None
