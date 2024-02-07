import subprocess
import sys
import urllib.request
from os import makedirs
from os.path import join, exists

from importlib_resources import files, as_file

from ..vendor.appdirs import user_cache_dir

_version = "7.2.0"
_openapi_jar_basename = f"openapi-generator-cli-{_version}.jar"
_openapi_jar_url = f"https://repo1.maven.org/maven2/org/openapitools/openapi-generator-cli/{_version}/{_openapi_jar_basename}"


def is_java_installed():
    try:
        command = "java -version"
        result = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True, text=True)
        return "version" in result.lower()
    except subprocess.CalledProcessError:
        return False


def main():
    if not is_java_installed():
        print("java must be installed to generate openapi clients automatically", file=sys.stderr)
        raise FileNotFoundError("java")

    cache_dir = user_cache_dir(appname="comfyui")
    jar = join(cache_dir, _openapi_jar_basename)

    if not exists(jar):
        makedirs(cache_dir, exist_ok=True)
        print(f"downloading {_openapi_jar_basename} to {jar}", file=sys.stderr)
        urllib.request.urlretrieve(_openapi_jar_url, jar)

    with as_file(files('comfy.api').joinpath('openapi.yaml')) as openapi_schema:
        with as_file(files('comfy.api').joinpath('openapi_python_config.yaml')) as python_config:
            cmds = [
                "java",
                "--add-opens", "java.base/java.io=ALL-UNNAMED",
                "--add-opens", "java.base/java.util=ALL-UNNAMED",
                "--add-opens", "java.base/java.lang=ALL-UNNAMED",
                "-jar", str(jar),
                "generate",
                "--input-spec", str(openapi_schema).replace('\\', '/'),
                "--global-property", "models",
                "--config", str(python_config).replace('\\', '/')
            ]
            print(" ".join(cmds), file=sys.stderr)
            subprocess.check_output(cmds)


if __name__ == "__main__":
    main()
