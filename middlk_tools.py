import os
import aiohttp
from ai_api import get_mime_type_from_binary

import json

def log_parameters_to_json(file_name="parameters.json"):
    def convert_value_to_str(value):
        if hasattr(value, "__dict__"):
            return {attr: str(getattr(value, attr)) for attr in value.__dict__}
        else:
            return str(value)
    
    # Retrieve the local variables from the calling function's scope
    import inspect
    frame = inspect.currentframe().f_back
    local_vars = frame.f_locals

    # Create a list to hold the detailed parameter information
    detailed_params = []

    # Print each parameter and its type, and collect the detailed information
    for key, value in local_vars.items():
        detail = {
            "name": key,
            "value": convert_value_to_str(value),
            "type": type(value).__name__
        }
        detailed_params.append(detail)
        print(f"{key}: {value}, type: {type(value)}")
    
    # Save detailed parameter information to a JSON file
    with open(file_name, 'w') as f:
        json.dump(detailed_params, f, indent=4)


import tracemalloc
def trace_memory():
    # 메모리 추적 시작
    tracemalloc.start()

    # 코드 실행 (예시)
    x = [i for i in range(1000000)]  # 큰 리스트 생성

    # 현재 메모리 스냅샷 생성
    snapshot = tracemalloc.take_snapshot()

    # 가장 많은 메모리를 사용하고 있는 top 10 개의 객체 출력
    top_stats = snapshot.statistics('lineno')

    print("[ Top 10 ]")
    for stat in top_stats[:10]:
        print(stat)

def test(a):
    return a + 1
async def upload_file(file_path, client_id, server_address):
    url = f"http://{server_address}/upload/image?clientId={client_id}"

    with open(file_path, 'rb') as file:
        file_data = file.read()

    async with aiohttp.ClientSession() as session:
        writer = aiohttp.MultipartWriter('form-data')
        writer.append(file_data, headers={'Content-Disposition': 
                                          f'inline; \
                                          filename="{os.path.basename(file_path)}"; \
                                          name="uploaded_content"',
                                          'Content-type':get_mime_type_from_binary(file_data)})
        writer.append(file_data, headers={'Content-Disposition': 
                                          f'inline; \
                                          filename="{os.path.basename(file_path)}"; \
                                          name="uploaded_content2"',
                                          'Content-type':get_mime_type_from_binary(file_data)})

        async with session.post(url, data=writer) as response:
            if response.status == 200:
                response_data = await response.text()
                print(response_data)
            else:
                print(f"Error uploading file: {response.status}")

# 클라이언트 코드에서 사용 예시
async def main():
    client_id = '100000'
    image_path = '/home/mlfavorfit/Downloads/mask.jpg'
    response = await upload_file(image_path, client_id, "0.0.0.0:8080")
    print(response)


import subprocess
import json
from pathlib import Path

class PackageFilter:
    def __init__(self, project_path, necessary_package_list, output_file="requirements_production.txt"):
        self.project_path = project_path
        self.output_file = output_file
        self.necessary_package_list = necessary_package_list

    def run_command(self, command):
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
        if result.returncode != 0:
            raise Exception(f"Command failed: {command}\n{result.stderr}")
        return result.stdout

    def get_installed_packages(self):
        output = self.run_command("pip freeze")
        return {line.split('==')[0]: line for line in output.splitlines()}

    def get_project_dependencies(self):
        self.run_command(f"pipreqs {self.project_path} --force --savepath requirements_temp.txt")
        with open("requirements_temp.txt", "r") as file:
            project_requirements = {line.split('==')[0] for line in file}
        Path("requirements_temp.txt").unlink()
        return project_requirements

    def get_dependency_tree(self):
        output = self.run_command("pipdeptree --json")
        return json.loads(output)

    def filter_needed_packages(self):
        installed_packages = self.get_installed_packages()
        project_dependencies = self.get_project_dependencies()
        dependency_tree = self.get_dependency_tree()

        needed_packages = set()

        def add_package(package_name):
            if package_name in project_dependencies and package_name not in needed_packages:
                needed_packages.add(package_name)
                for dependency in dependency_tree:
                    if dependency['package']['key'] == package_name:
                        for req in dependency.get('dependencies', []):
                            add_package(req['package_name'])

        for package in project_dependencies:
            add_package(package)
        
        return {pkg: installed_packages[pkg] for pkg in needed_packages if pkg in installed_packages}

    def write_requirements_file(self):
        needed_packages = self.filter_needed_packages()

        with open(self.output_file, "w") as file:
            for package in needed_packages.values():
                file.write(f"{package}\n")

        with open(self.output_file, "a") as file:
            for package in self.necessary_package_list:
                file.write(f"{package}\n")

        print(f"Filtered requirements written to {self.output_file}")


# 비동기 이벤트 루프 실행
if __name__ == "__main__":
    # import asyncio
    # asyncio.run(main())
    
    filter_instance = PackageFilter(
        project_path=".", 
        necessary_package_list=[
            "open-clip-torch==2.24.0",
            "lightning==2.2.5",
            "insightface==0.7.3",
            "cupy-cuda12x==12.3.0",
            "python-magic==0.4.27",
            "requests_toolbelt==1.0.0",
        ],
        output_file="requirements_production.txt")
    filter_instance.write_requirements_file()