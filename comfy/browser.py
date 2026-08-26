import logging
import subprocess
import webbrowser

def open_browser(url: str, browser_path: str = None, browser_profile: str = None) -> None:
    """
    Launches a custom browser if specified via CLI args, 
    otherwise falls back to the system default browser.
    """
    if browser_path:
        cmd = [browser_path]
        if browser_profile:
            cmd.append(f"--profile-directory={browser_profile}")
        cmd.append(url)

        try:
            logging.info(f"Launching custom browser: {browser_path}")
            subprocess.Popen(cmd)
            return
        except Exception as e:
            logging.warning(
                f"Failed to launch custom browser '{browser_path}': {e}. "
                f"Falling back to default browser."
            )

    try:
        webbrowser.open(url)
    except Exception as e:
        logging.error(f"Failed to open web browser: {e}")
