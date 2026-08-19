import logging
import os
import shutil
import subprocess
import webbrowser


def open_browser(
    url: str,
    browser_path: str | None = None,
    browser_profile: str | None = None,
) -> bool:
    """Open a URL in the configured browser, falling back to the system default."""
    if browser_path is None:
        if browser_profile is not None:
            logging.warning("Ignoring browser profile because no browser path was configured")
        return webbrowser.open(url)

    command = os.path.expanduser(browser_path)
    executable = shutil.which(command)
    if executable is None:
        logging.warning("Browser command not found: %s; using the system default", browser_path)
        return webbrowser.open(url)

    popen_kwargs = {}
    if os.name != "nt":
        popen_kwargs["start_new_session"] = True

    browser_args = [executable]
    if browser_profile is not None:
        browser_args.extend(["--profile-directory", browser_profile])
    browser_args.append(url)

    try:
        subprocess.Popen(browser_args, **popen_kwargs)
    except OSError as ex:
        logging.warning("Unable to launch browser %s: %s; using the system default", executable, ex)
        return webbrowser.open(url)

    return True
