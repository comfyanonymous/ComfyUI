import os
import sys
import subprocess
import webbrowser
import subprocess
import shutil
from static_file_server import serve_react_app
from aiohttp import web

# These will be installed and imported at runtime
inquirer = None
keyring = None
aiohttp = None
asyncio = None

required_packages = {
    'inquirer': 'inquirer',
    'keyring': 'keyring',
    'aiohttp': 'aiohttp',
    'asyncio': 'asyncio'
}

# Attempt to import required packages; install them if necessary
for package, var_name in required_packages.items():
    try:
        # Try to import the package
        globals()[var_name] = __import__(package)
    except ImportError:
        # If import fails, install the package and try again
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        globals()[var_name] = __import__(package)


PORT=8188 # local Comfy Engine port
OAUTH_PORT=3003 # this port is only used temporarily for oAuth-callback
BASE_DIR = "web"
BUILD_DIR = "web/dist"
SRC_DIR = "web/src"


# Determines if we need to rebuild the project based on the last modified source file date
def is_build_up_to_date():
    if not os.path.exists(BUILD_DIR):
        return False

    src_last_modified = max(os.path.getmtime(root) for root, _, _ in os.walk(SRC_DIR))
    build_last_modified = max(os.path.getmtime(root) for root, _, _ in os.walk(BUILD_DIR))
    return src_last_modified <= build_last_modified


def find_package_manager():
    """Check for available package managers and return the first one found."""
    for manager in ['yarn', 'npm', 'pnpm']:
        if shutil.which(manager):
            return manager
    raise RuntimeError("No suitable JavaScript package manager found. Please install yarn, npm, or pnpm so we can build the client.")


def install_dependencies():
    manager = find_package_manager()
    print(f"Installing dependencies with {manager}...")
    subprocess.run([manager, "install"], check=True, cwd='web', shell=True)


def run_js_script(action: str):
    manager = find_package_manager()
    print(f"{action.capitalize()}ing client with {manager}...")
    
    command = [manager]
    if manager in ['npm', 'pnpm']:
        command.append("run")
    command.append(action)
        
    subprocess.run(command, check=True, cwd=BASE_DIR, shell=True)


async def start_login_callback_server():
    event = asyncio.Event()

    async def handle_login(request):
        query_components = request.rel_url.query
        token = query_components.get('token')
        if token:
            keyring.set_password("void.tech", "user_login_token", token)
            print("Token received and stored securely.")
            event.set()  # Signal that the token has been received
        return web.Response(text="Login successful. You can close this window.")

    app = web.Application()
    app.add_routes([web.get('/', handle_login)])
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', OAUTH_PORT)
    await site.start()
    print(f"Local server listening on port {OAUTH_PORT}...")
    webbrowser.open(f"https://void.tech/login?redirect_uri=http://localhost:{OAUTH_PORT}")
    await event.wait()  # Wait for the event to be set
    await runner.cleanup()


async def get_api_key() -> str:
    token = keyring.get_password("void.tech", "user_login_token")
    if not token:
        print("No token found; please login.")
        await start_login_callback_server()
        token = keyring.get_password("void.tech", "user_login_token")
    return token


async def main():
    if not is_build_up_to_date():
        install_dependencies()
        run_js_script('build')

    questions = [
        inquirer.List('server',
                      message="Do you want to use a local or remote server?",
                      choices=['local', 'remote'],
                      ),
    ]
    answers = inquirer.prompt(questions)

    if answers['server'] == 'local':
        # Start the full Comfy Engine locally
        print("Starting local server...")
        subprocess.run(["python", "main.py"], check=True)
        
    else:
        # Start a simple server to serve the react-app
        api_key = await get_api_key()
        app = web.Application()
        serve_react_app(app, BUILD_DIR, f"https://api.void.tech/{api_key}")
        
        runner = aiohttp.web.AppRunner(app)
        await runner.setup()
        
        site = aiohttp.web.TCPSite(runner, 'localhost', PORT)
        await site.start()
        
        print(f"Server started at http://localhost:{PORT}")
        webbrowser.open(f"http://localhost:{PORT}")

        await asyncio.Future() # Keep the server running
    
    pass


if __name__ == "__main__":
    asyncio.run(main())
