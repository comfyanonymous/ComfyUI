Here lies dummy ts files that decalre/export ComfyUI's own scripts files as typed types w/o needing
to symlink to the actual implementation.

Actual code in the comfyui/ directory can import these like `import {app} from "/scripts/app.js"`
and have access to `app` as the fully typed `ComfyApp`. The `__build__.py` script will rewrite these
to the relative browser path.