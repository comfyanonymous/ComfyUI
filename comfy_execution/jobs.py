"""
Job utilities for the /api/jobs endpoint.
Provides normalization and helper functions for job status tracking.
"""


class JobStatus:
    """Job status constants."""
    PENDING = 'pending'
    IN_PROGRESS = 'in_progress'
    COMPLETED = 'completed'
    FAILED = 'failed'
    CANCELLED = 'cancelled'

    ALL = [PENDING, IN_PROGRESS, COMPLETED, FAILED, CANCELLED]


# Media types that can be previewed in the frontend
PREVIEWABLE_MEDIA_TYPES = frozenset({'images', 'video', 'audio'})

# 3D file extensions for preview fallback (no dedicated media_type exists)
THREE_D_EXTENSIONS = frozenset({'.obj', '.fbx', '.gltf', '.glb'})


def is_previewable(media_type, item):
    """
    Check if an output item is previewable.
    Matches frontend logic in ComfyUI_frontend/src/stores/queueStore.ts

    Priority:
    1. media_type is 'images', 'video', or 'audio'
    2. format field starts with 'video/' or 'audio/'
    3. filename has a 3D extension (.obj, .fbx, .gltf, .glb)
    """
    if media_type in PREVIEWABLE_MEDIA_TYPES:
        return True

    # Check format field (MIME type)
    fmt = item.get('format', '')
    if fmt and (fmt.startswith('video/') or fmt.startswith('audio/')):
        return True

    # Check for 3D files by extension
    filename = item.get('filename', '').lower()
    if any(filename.endswith(ext) for ext in THREE_D_EXTENSIONS):
        return True

    return False


def normalize_queue_item(item, status):
    """Convert queue item tuple to unified job dict."""
    _, prompt_id, _, extra_data, _ = item[:5]
    create_time = extra_data.get('create_time')
    extra_pnginfo = extra_data.get('extra_pnginfo', {}) or {}
    workflow_id = extra_pnginfo.get('workflow', {}).get('id')

    return {
        'id': prompt_id,
        'status': status,
        'create_time': create_time,
        'error_message': None,
        'execution_error': None,
        'execution_start_time': None,
        'execution_end_time': None,
        'outputs_count': 0,
        'preview_output': None,
        'workflow_id': workflow_id,
    }


def normalize_history_item(prompt_id, history_item, include_outputs=False):
    """Convert history item dict to unified job dict."""
    prompt_tuple = history_item['prompt']
    _, _, prompt, extra_data, _ = prompt_tuple[:5]
    create_time = extra_data.get('create_time')
    extra_pnginfo = extra_data.get('extra_pnginfo', {}) or {}
    workflow_id = extra_pnginfo.get('workflow', {}).get('id')

    status_info = history_item.get('status', {})
    status_str = status_info.get('status_str') if status_info else None
    if status_str == 'success':
        status = JobStatus.COMPLETED
    elif status_str == 'error':
        status = JobStatus.FAILED
    else:
        status = JobStatus.COMPLETED

    outputs = history_item.get('outputs', {})
    outputs_count, preview_output = get_outputs_summary(outputs)

    error_message = None
    execution_error = None
    if status == JobStatus.FAILED and status_info:
        messages = status_info.get('messages', [])
        for entry in messages:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2 and entry[0] == 'execution_error':
                detail = entry[1]
                if isinstance(detail, dict):
                    error_message = str(detail.get('exception_message', ''))
                    execution_error = detail
                break

    execution_duration = history_item.get('execution_duration')
    execution_start_time = None
    execution_end_time = None
    if execution_duration is not None and create_time is not None:
        execution_end_time = create_time + int(execution_duration * 1000)
        execution_start_time = create_time

    job = {
        'id': prompt_id,
        'status': status,
        'create_time': create_time,
        'error_message': error_message,
        'execution_error': execution_error,
        'execution_start_time': execution_start_time,
        'execution_end_time': execution_end_time,
        'outputs_count': outputs_count,
        'preview_output': preview_output,
        'workflow_id': workflow_id,
    }

    if include_outputs:
        job['outputs'] = outputs
        job['execution_status'] = status_info
        job['workflow'] = {
            'prompt': prompt,
            'extra_data': extra_data,
        }

    return job


def get_outputs_summary(outputs):
    """
    Count outputs and find preview in a single pass.
    Returns (outputs_count, preview_output).

    Preview priority (matching frontend):
    1. type="output" with previewable media
    2. Any previewable media
    """
    count = 0
    preview_output = None
    fallback_preview = None

    for node_id, node_outputs in outputs.items():
        for media_type, items in node_outputs.items():
            if media_type == 'animated' or not isinstance(items, list):
                continue

            for item in items:
                count += 1

                if preview_output is None and is_previewable(media_type, item):
                    enriched = {
                        **item,
                        'nodeId': node_id,
                        'mediaType': media_type
                    }
                    if item.get('type') == 'output':
                        preview_output = enriched
                    elif fallback_preview is None:
                        fallback_preview = enriched

    return count, preview_output or fallback_preview


def apply_sorting(jobs, sort_by, sort_order):
    """Sort jobs list by specified field and order."""
    reverse = (sort_order == 'desc')

    if sort_by == 'execution_duration':
        def get_sort_key(job):
            start = job.get('execution_start_time') or 0
            end = job.get('execution_end_time') or 0
            return end - start if end and start else 0
    else:
        def get_sort_key(job):
            return job.get('create_time') or 0

    return sorted(jobs, key=get_sort_key, reverse=reverse)
