#!/usr/bin/env python3
"""
RunPod Serverless Worker Handler for ComfyUI
Optimized for the new ComfyUI features and performance improvements
"""

import os
import time
import logging
import tempfile
import requests
from typing import Dict, Any
import runpod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ComfyUIServerlessHandler:
    def __init__(self):
        self.comfyui_url = "http://127.0.0.1:8000"
        self.client_id = "runpod_serverless_worker"
        self.setup_paths()

    def setup_paths(self):
        """Setup required paths for serverless operation"""
        os.makedirs("/tmp/inputs", exist_ok=True)
        os.makedirs("/tmp/outputs", exist_ok=True)
        os.makedirs("/tmp/comfyui", exist_ok=True)

    def wait_for_comfyui(self, timeout: int = 120) -> bool:
        """Wait for ComfyUI to be ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.comfyui_url}/system_stats")
                if response.status_code == 200:
                    logger.info("ComfyUI is ready")
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)

        logger.error(f"ComfyUI not ready after {timeout} seconds")
        return False

    def download_input_files(self, input_data: Dict[str, Any]) -> Dict[str, str]:
        """Download input files and return local paths"""
        local_files = {}

        if "input_files" in input_data:
            for file_key, file_url in input_data["input_files"].items():
                try:
                    response = requests.get(file_url, timeout=60)
                    response.raise_for_status()

                    # Create temporary file
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        dir="/tmp/inputs",
                        suffix=os.path.splitext(file_url)[1]
                    ) as tmp_file:
                        tmp_file.write(response.content)
                        local_files[file_key] = tmp_file.name

                    logger.info(f"Downloaded {file_key} to {local_files[file_key]}")

                except Exception as e:
                    logger.error(f"Failed to download {file_key}: {str(e)}")
                    raise

        return local_files

    def execute_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Execute ComfyUI workflow"""
        try:
            # Queue the workflow
            queue_response = requests.post(
                f"{self.comfyui_url}/prompt",
                json={
                    "prompt": workflow,
                    "client_id": self.client_id
                },
                timeout=30
            )
            queue_response.raise_for_status()

            prompt_id = queue_response.json()["prompt_id"]
            logger.info(f"Queued workflow with prompt_id: {prompt_id}")

            # Wait for completion
            return self.wait_for_completion(prompt_id)

        except Exception as e:
            logger.error(f"Failed to execute workflow: {str(e)}")
            raise

    def wait_for_completion(self, prompt_id: str, timeout: int = 300) -> Dict[str, Any]:
        """Wait for workflow completion and return results"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                # Check queue status
                queue_response = requests.get(f"{self.comfyui_url}/queue")
                queue_data = queue_response.json()

                # Check if our job is still in queue
                running = any(item[1]["prompt_id"] == prompt_id for item in queue_data.get("queue_running", []))
                pending = any(item[1]["prompt_id"] == prompt_id for item in queue_data.get("queue_pending", []))

                if not running and not pending:
                    # Job completed, get results
                    history_response = requests.get(f"{self.comfyui_url}/history/{prompt_id}")
                    if history_response.status_code == 200:
                        history_data = history_response.json()
                        if prompt_id in history_data:
                            return self.process_results(history_data[prompt_id])

                time.sleep(2)

            except Exception as e:
                logger.error(f"Error checking completion: {str(e)}")
                time.sleep(5)

        raise TimeoutError(f"Workflow {prompt_id} timed out after {timeout} seconds")

    def process_results(self, history_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and upload results"""
        results = {
            "status": "completed",
            "outputs": [],
            "metadata": {}
        }

        if "outputs" in history_data:
            for node_id, node_output in history_data["outputs"].items():
                if "images" in node_output:
                    for image_info in node_output["images"]:
                        # Download image from ComfyUI
                        image_url = f"{self.comfyui_url}/view"
                        params = {
                            "filename": image_info["filename"],
                            "subfolder": image_info.get("subfolder", ""),
                            "type": image_info.get("type", "output")
                        }

                        try:
                            image_response = requests.get(image_url, params=params)
                            image_response.raise_for_status()

                            # Save to temp file for upload
                            output_path = f"/tmp/outputs/{image_info['filename']}"
                            with open(output_path, "wb") as f:
                                f.write(image_response.content)

                            results["outputs"].append({
                                "type": "image",
                                "filename": image_info["filename"],
                                "path": output_path,
                                "node_id": node_id
                            })

                        except Exception as e:
                            logger.error(f"Failed to process image {image_info['filename']}: {str(e)}")

        return results

    def cleanup(self):
        """Clean up temporary files"""
        try:
            import shutil
            shutil.rmtree("/tmp/inputs", ignore_errors=True)
            shutil.rmtree("/tmp/outputs", ignore_errors=True)
            os.makedirs("/tmp/inputs", exist_ok=True)
            os.makedirs("/tmp/outputs", exist_ok=True)
            logger.info("Cleaned up temporary files")
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """Main serverless handler function"""
    handler_instance = ComfyUIServerlessHandler()

    try:
        # Wait for ComfyUI to be ready
        if not handler_instance.wait_for_comfyui():
            return {"error": "ComfyUI failed to start"}

        # Get job input
        job_input = job.get("input", {})

        # Download input files if any
        local_files = handler_instance.download_input_files(job_input)

        # Update workflow with local file paths
        workflow = job_input.get("workflow", {})
        if local_files and "file_mappings" in job_input:
            for node_id, mappings in job_input["file_mappings"].items():
                if node_id in workflow:
                    for input_key, file_key in mappings.items():
                        if file_key in local_files:
                            workflow[node_id]["inputs"][input_key] = local_files[file_key]

        # Execute workflow
        results = handler_instance.execute_workflow(workflow)

        # Upload output files to RunPod storage or return base64
        output_urls = []
        for output in results.get("outputs", []):
            if output["type"] == "image":
                # For serverless, we typically return base64 or upload to storage
                with open(output["path"], "rb") as f:
                    import base64
                    image_data = base64.b64encode(f.read()).decode()
                    output_urls.append({
                        "filename": output["filename"],
                        "data": image_data,
                        "node_id": output["node_id"]
                    })

        return {
            "status": "success",
            "outputs": output_urls,
            "execution_time": time.time() - job.get("start_time", time.time())
        }

    except Exception as e:
        logger.error(f"Handler error: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }

    finally:
        # Always cleanup
        handler_instance.cleanup()

if __name__ == "__main__":
    # Start the serverless worker
    runpod.serverless.start({"handler": handler})
