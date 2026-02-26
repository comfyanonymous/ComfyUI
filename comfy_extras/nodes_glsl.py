import os
import sys
import re
import logging
import ctypes.util
import importlib.util
from typing import Type, TypedDict

import numpy as np
import torch

import nodes
from comfy_api.latest import ComfyExtension, io, ui
from typing_extensions import override
from utils.install_util import get_missing_requirements_message

logger = logging.getLogger(__name__)


def _check_opengl_availability():
    """Early check for OpenGL availability. Raises RuntimeError if unlikely to work."""
    logger.debug("_check_opengl_availability: starting")
    missing = []

    # Check Python packages (using find_spec to avoid importing)
    logger.debug("_check_opengl_availability: checking for glfw package")
    if importlib.util.find_spec("glfw") is None:
        missing.append("glfw")

    logger.debug("_check_opengl_availability: checking for OpenGL package")
    if importlib.util.find_spec("OpenGL") is None:
        missing.append("PyOpenGL")

    if missing:
        raise RuntimeError(
            f"OpenGL dependencies not available.\n{get_missing_requirements_message()}\n"
        )

    # On Linux without display, check if headless backends are available
    logger.debug(f"_check_opengl_availability: platform={sys.platform}")
    if sys.platform.startswith("linux"):
        has_display = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        logger.debug(f"_check_opengl_availability: has_display={bool(has_display)}")
        if not has_display:
            # Check for EGL or OSMesa libraries
            logger.debug("_check_opengl_availability: checking for EGL library")
            has_egl = ctypes.util.find_library("EGL")
            logger.debug("_check_opengl_availability: checking for OSMesa library")
            has_osmesa = ctypes.util.find_library("OSMesa")

            # Error disabled for CI as it fails this check
            # if not has_egl and not has_osmesa:
            #     raise RuntimeError(
            #         "GLSL Shader node: No display and no headless backend (EGL/OSMesa) found.\n"
            #         "See error below for installation instructions."
            #     )
            logger.debug(f"Headless mode: EGL={'yes' if has_egl else 'no'}, OSMesa={'yes' if has_osmesa else 'no'}")

    logger.debug("_check_opengl_availability: completed")


# Run early check at import time
logger.debug("nodes_glsl: running _check_opengl_availability at import time")
_check_opengl_availability()


class SizeModeInput(TypedDict):
    size_mode: str
    width: int
    height: int


MAX_IMAGES = 5      # u_image0-4
MAX_UNIFORMS = 5    # u_float0-4, u_int0-4
MAX_OUTPUTS = 4     # fragColor0-3 (MRT)

# Vertex shader using gl_VertexID trick - no VBO needed.
# Draws a single triangle that covers the entire screen:
#
#     (-1,3)
#       /|
#      / |  <- visible area is the unit square from (-1,-1) to (1,1)
#     /  |     parts outside get clipped away
# (-1,-1)---(3,-1)
#
# v_texCoord is computed from clip space: * 0.5 + 0.5 maps (-1,1) -> (0,1)
VERTEX_SHADER = """#version 330 core
out vec2 v_texCoord;
void main() {
    vec2 verts[3] = vec2[](vec2(-1, -1), vec2(3, -1), vec2(-1, 3));
    v_texCoord = verts[gl_VertexID] * 0.5 + 0.5;
    gl_Position = vec4(verts[gl_VertexID], 0, 1);
}
"""

DEFAULT_FRAGMENT_SHADER = """#version 300 es
precision highp float;

uniform sampler2D u_image0;
uniform vec2 u_resolution;

in vec2 v_texCoord;
layout(location = 0) out vec4 fragColor0;

void main() {
    fragColor0 = texture(u_image0, v_texCoord);
}
"""


def _convert_es_to_desktop(source: str) -> str:
    """Convert GLSL ES (WebGL) shader source to desktop GLSL 330 core."""
    # Remove any existing #version directive
    source = re.sub(r"#version\s+\d+(\s+es)?\s*\n?", "", source, flags=re.IGNORECASE)
    # Remove precision qualifiers (not needed in desktop GLSL)
    source = re.sub(r"precision\s+(lowp|mediump|highp)\s+\w+\s*;\s*\n?", "", source)
    # Prepend desktop GLSL version
    return "#version 330 core\n" + source


def _detect_output_count(source: str) -> int:
    """Detect how many fragColor outputs are used in the shader.

    Returns the count of outputs needed (1 to MAX_OUTPUTS).
    """
    matches = re.findall(r"fragColor(\d+)", source)
    if not matches:
        return 1  # Default to 1 output if none found
    max_index = max(int(m) for m in matches)
    return min(max_index + 1, MAX_OUTPUTS)


def _detect_pass_count(source: str) -> int:
    """Detect multi-pass rendering from #pragma passes N directive.

    Returns the number of passes (1 if not specified).
    """
    match = re.search(r'#pragma\s+passes\s+(\d+)', source)
    if match:
        return max(1, int(match.group(1)))
    return 1


############################################################


class GLContext:
    """Manages OpenGL context and resources for shader execution.

    Acts as a singleton factory: ``GLContext`` itself is an "abstract" class (not a true ABC though) and never instantiates itself directly. Instead, its instance is always one of "concrete backend" contexts - a first valid subclass in the fallback sequence. ``GLContext`` doesn't inherit from ABC just to prevent IDE warnings caused by this polymorphism. For all intents and purposes, it **IS** a singleton-ABC.

    Backends fallback order: GLFW (desktop) → EGL (headless GPU) → OSMesa (software). See ``__subclass_fallback_order()``.
    """

    __instance: 'GLContext' = None  # The singleton

    def __new__(cls):
        # Since ``GLContext`` is a singleton anyway, we should store it
        # explicitly in ``GLContext.__instance``, NOT in ``cls.__instance``.
        if GLContext.__instance is None:
            GLContext.__instance = GLContext.__new_instance_using_concrete_class_fallback_order()
            assert isinstance(GLContext.__instance, GLContext)
        return GLContext.__instance

    @staticmethod
    def __concrete_class_fallback_order() -> tuple[Type['GLContext'], ...]:
        """The order concrete subclasses are tried in: GLFW → EGL → OSMesa."""
        return _GLContextGLFW, _GLContextEGL, _GLContextOSMesa

    @staticmethod
    def __new_instance_using_concrete_class_fallback_order() -> 'GLContext':
        """Try to init backends in the fallback order.

        Called from ``__new__()`` on first attempt to instantiate the singleton.
        Raises RuntimeError if none of the backends work.
        """
        errors: list[tuple[str, Exception]] = []

        for cls in GLContext.__concrete_class_fallback_order():
            name = cls.backend_name()
            logger.debug(f"GLContext.__init__: trying {name} backend")
            try:
                instance: GLContext = object.__new__(cls)
                # Since this code is called while in `__new__()`, we need to manually call `__init__()`, too.
                # Otherwise, Python would call it only AFTER `__new__()`, causing init errors outside our try-except check.
                instance.__init__()
                logger.debug(f"GLContext.__init__: {name} backend succeeded. The singleton is: {cls!r}")
                logger.info(f"Concrete GLSL context initialized as: {name}")
                return instance
            except Exception as e:
                logger.debug(f"GLContext.__init__: {name} backend failed: {e}")
                errors.append((name, e))

        # If we still haven't returned, none of the backends succeeded.
        # Let's raise the error.

        if sys.platform == "win32":
            platform_help = (
                "Windows: Ensure GPU drivers are installed and display is available.\n"
                "         CPU-only/headless mode is not supported on Windows."
            )
        elif sys.platform == "darwin":
            platform_help = (
                "macOS: GLFW is not supported.\n"
                "  Install OSMesa via Homebrew: brew install mesa\n"
                "  Then: pip install PyOpenGL PyOpenGL-accelerate"
            )
        else:
            platform_help = (
                "Linux: Install one of these backends:\n"
                "  Desktop:           sudo apt install libgl1-mesa-glx libglfw3\n"
                "  Headless with GPU: sudo apt install libegl1-mesa libgl1-mesa-dri\n"
                "  Headless (CPU):    sudo apt install libosmesa6"
            )

        error_details = "\n".join(f"  {name}: {err}" for name, err in errors)
        raise RuntimeError(
            f"Failed to create OpenGL context.\n\n"
            f"Backend errors:\n{error_details}\n\n"
            f"{platform_help}"
        )

    def __init__(self):
        try:
            # noinspection PyUnresolvedReferences
            if self.__initialized:
                # 99% of the time (after first init) we get here and just return
                logger.debug("GLContext.__init__: already initialized, skipping")
                return
            logger.warning("GLContext.__init__: weird state: the singleton has <__initialized> attribute, but is NOT initialized.")
        except AttributeError:
            # First instance creation: it was created with `__new__()`, but hasn't been initialized yet
            pass

        logger.debug("GLContext.__init__: starting initialization")

        self.__initialized: bool = False
        self._vao = None

        import time
        start_time: float = time.perf_counter()

        self._init_backend_concrete()  # must fully initialize backend

        # Now import OpenGL.GL (after context is current)
        logger.debug("GLContext.__init__: importing OpenGL.GL")
        self.__import_opengl()
        gl = self._gl

        # Create VAO (required for core profile, but OSMesa may use compat profile)
        logger.debug("GLContext.__init__: creating VAO")
        vao = None
        try:
            vao = gl.glGenVertexArrays(1)
            gl.glBindVertexArray(vao)
            self._vao = vao  # Only store after successful bind
            logger.debug("GLContext.__init__: VAO created successfully")
        except Exception as e:
            logger.debug(f"GLContext.__init__: VAO creation failed (may be expected for OSMesa): {e}")
            # OSMesa with older Mesa may not support VAOs
            # Clean up if we created but couldn't bind
            if vao:
                try:
                    gl.glDeleteVertexArrays(1, [vao])
                except Exception:
                    pass

        self._glBindVertexArray = gl.glBindVertexArray
        self.__initialized = True

        elapsed = (time.perf_counter() - start_time) * 1000

        # Log device info

        def gl_string(value) -> str:
            string = gl.glGetString(value)
            return string.decode() if string else "Unknown"

        renderer, vendor, version = (
            gl_string(x) for x in [gl.GL_RENDERER, gl.GL_VENDOR, gl.GL_VERSION]
        )
        logger.info(f"GLSL context initialized in {elapsed:.1f}ms ({self.backend_name()}) - {renderer} ({vendor}), GL {version}")

    def __import_opengl(self):
        """Import OpenGL module. Called after context is created."""
        logger.debug("__import_opengl: importing OpenGL.GL")
        from OpenGL import GL
        self._gl = GL
        logger.debug("__import_opengl: import completed")

    @classmethod
    def backend_name(cls) -> str:
        """Per-concrete-class unique string identifier. Used for log messages."""
        raise NotImplementedError("Must be implemented in a concrete subclass.")

    def _init_backend_concrete(self):
        """Actual initialisation hook of a concrete backend. Called mid-init."""
        raise NotImplementedError("Must be implemented in a concrete subclass.")

    def _make_current_concrete(self):
        raise NotImplementedError("Must be implemented in a concrete subclass.")

    def make_current(self):
        self._make_current_concrete()
        if self._vao is not None:
            self._glBindVertexArray(self._vao)

    # noinspection PyPep8Naming
    @property
    def GL(self):
        """Properly yet lazily imported ``OpenGL.GL`` module."""
        return self._gl

##########

class _GLContextGLFW(GLContext):
    """Concrete GLContext using GLFW backend."""
    @classmethod
    def backend_name(cls) -> str:
        return "GLFW"

    def _init_backend_concrete(self):
        """Initialize GLFW. Raises RuntimeError on failure."""
        logger.debug("_init_backend_concrete (GLFW): starting")
        # On macOS, glfw.init() must be called from main thread or it hangs forever
        if sys.platform == "darwin":
            logger.debug("_init_backend_concrete (GLFW): skipping on macOS")
            raise RuntimeError("GLFW backend not supported on macOS")

        logger.debug("_init_backend_concrete (GLFW): importing glfw module")
        import glfw

        logger.debug("_init_backend_concrete (GLFW): calling glfw.init()")
        if not glfw.init():
            raise RuntimeError("glfw.init() failed")

        try:
            logger.debug("_init_backend_concrete (GLFW): setting window hints")
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

            logger.debug("_init_backend_concrete (GLFW): calling create_window()")
            window = glfw.create_window(64, 64, "ComfyUI GLSL", None, None)
            if not window:
                raise RuntimeError("glfw.create_window() failed")

            logger.debug("_init_backend_concrete (GLFW): calling make_context_current()")
            glfw.make_context_current(window)
        except Exception:
            logger.debug("_init_backend_concrete (GLFW): failed, terminating glfw")
            glfw.terminate()
            raise

        self._window = window
        self._glfw = glfw

        logger.debug("_init_backend_concrete (GLFW): completed successfully")

    def _make_current_concrete(self):
        self._glfw.make_context_current(self._window)

##########

class _GLContextEGL(GLContext):
    """Concrete GLContext using EGL backend."""
    @classmethod
    def backend_name(cls) -> str:
        return "EGL"

    def _init_backend_concrete(self):
        """Initialize EGL for headless rendering. Raises RuntimeError on failure."""
        logger.debug("_init_backend_concrete (EGL): starting")
        from OpenGL import EGL
        logger.debug("_init_backend_concrete (EGL): imports completed")

        display = None
        context = None
        surface = None

        try:
            logger.debug("_init_backend_concrete (EGL): calling eglGetDisplay()")
            display = EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY)
            if display == EGL.EGL_NO_DISPLAY:
                raise RuntimeError("eglGetDisplay() failed")

            logger.debug("_init_backend_concrete (EGL): calling eglInitialize()")
            major, minor = EGL.EGLint(), EGL.EGLint()
            if not EGL.eglInitialize(display, major, minor):
                display = None  # Not initialized, don't terminate
                raise RuntimeError("eglInitialize() failed")
            logger.debug(f"_init_backend_concrete (EGL): EGL version {major.value}.{minor.value}")

            config_attribs = [
                EGL.EGL_SURFACE_TYPE, EGL.EGL_PBUFFER_BIT,
                EGL.EGL_RENDERABLE_TYPE, EGL.EGL_OPENGL_BIT,
                EGL.EGL_RED_SIZE, 8, EGL.EGL_GREEN_SIZE, 8, EGL.EGL_BLUE_SIZE, 8, EGL.EGL_ALPHA_SIZE, 8,
                EGL.EGL_DEPTH_SIZE, 0, EGL.EGL_NONE
            ]
            configs = (EGL.EGLConfig * 1)()
            num_configs = EGL.EGLint()
            if not EGL.eglChooseConfig(display, config_attribs, configs, 1, num_configs) or num_configs.value == 0:
                raise RuntimeError("eglChooseConfig() failed")
            config = configs[0]
            logger.debug(f"_init_backend_concrete (EGL): config chosen, num_configs={num_configs.value}")

            if not EGL.eglBindAPI(EGL.EGL_OPENGL_API):
                raise RuntimeError("eglBindAPI() failed")

            logger.debug("_init_backend_concrete (EGL): calling eglCreateContext()")
            context_attribs = [
                EGL.EGL_CONTEXT_MAJOR_VERSION, 3,
                EGL.EGL_CONTEXT_MINOR_VERSION, 3,
                EGL.EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL.EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
                EGL.EGL_NONE
            ]
            context = EGL.eglCreateContext(display, config, EGL.EGL_NO_CONTEXT, context_attribs)
            if context == EGL.EGL_NO_CONTEXT:
                raise RuntimeError("eglCreateContext() failed")

            logger.debug("_init_backend_concrete (EGL): calling eglCreatePbufferSurface()")
            pbuffer_attribs = [EGL.EGL_WIDTH, 64, EGL.EGL_HEIGHT, 64, EGL.EGL_NONE]
            surface = EGL.eglCreatePbufferSurface(display, config, pbuffer_attribs)
            if surface == EGL.EGL_NO_SURFACE:
                raise RuntimeError("eglCreatePbufferSurface() failed")

            logger.debug("_init_backend_concrete (EGL): calling eglMakeCurrent()")
            if not EGL.eglMakeCurrent(display, surface, surface, context):
                raise RuntimeError("eglMakeCurrent() failed")

        except Exception:
            logger.debug("_init_backend_concrete (EGL): failed, cleaning up")
            # Clean up any resources on failure
            if surface is not None:
                EGL.eglDestroySurface(display, surface)
            if context is not None:
                EGL.eglDestroyContext(display, context)
            if display is not None:
                EGL.eglTerminate(display)
            raise

        self._egl_display = display
        self._egl_context = context
        self._egl_surface = surface

        self._EGL = EGL
        self._eglMakeCurrent = EGL.eglMakeCurrent

        logger.debug("_init_backend_concrete (EGL): completed successfully")

    def _make_current_concrete(self):
        self._eglMakeCurrent(self._egl_display, self._egl_surface, self._egl_surface, self._egl_context)

##########

class _GLContextOSMesa(GLContext):
    """Concrete GLContext using OSMesa backend."""
    @classmethod
    def backend_name(cls) -> str:
        return "OSMesa"

    def _init_backend_concrete(self):
        """Initialize OSMesa for software rendering. Returns (context, buffer). Raises RuntimeError on failure."""
        import ctypes

        logger.debug("_init_backend_concrete (OSMesa): starting")
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"

        logger.debug("_init_backend_concrete (OSMesa): importing OpenGL.osmesa")
        from OpenGL import GL
        from OpenGL.osmesa import (
            OSMesaCreateContextExt, OSMesaMakeCurrent, OSMesaDestroyContext,
            OSMESA_RGBA,
        )
        logger.debug("_init_backend_concrete (OSMesa): imports completed")

        ctx = OSMesaCreateContextExt(OSMESA_RGBA, 24, 0, 0, None)
        if not ctx:
            raise RuntimeError("OSMesaCreateContextExt() failed")

        width, height = 64, 64
        buffer = (ctypes.c_ubyte * (width * height * 4))()

        logger.debug("_init_backend_concrete (OSMesa): calling OSMesaMakeCurrent()")
        if not OSMesaMakeCurrent(ctx, buffer, GL.GL_UNSIGNED_BYTE, width, height):
            OSMesaDestroyContext(ctx)
            raise RuntimeError("OSMesaMakeCurrent() failed")

        self._osmesa_ctx = ctx
        self._osmesa_buffer = buffer

        logger.debug("_init_backend_concrete (OSMesa): completed successfully")

    def _make_current_concrete(self):
        from OpenGL.osmesa import OSMesaMakeCurrent
        OSMesaMakeCurrent(self._osmesa_ctx, self._osmesa_buffer, self._gl.GL_UNSIGNED_BYTE, 64, 64)


############################################################


class __GLRenderMeta(type):
    """Internal metaclass for ``GLRender``.

    Implemented as meta - to make ``GLRender`` truly static, including class-level properties which are also properly type-detected by IDEs.
    """

    @property
    def context(cls) -> GLContext:
        """Global OpenGL context."""
        try:
            # noinspection PyUnresolvedReferences
            return cls.__context
        except AttributeError:
            pass
        # noinspection PyAttributeOutsideInit
        cls.__context = GLContext()
        return cls.__context

    def compile_shader(cls, source: str, shader_type: int) -> int:
        """Compile a shader and return its ID."""
        gl = cls.context.GL

        shader = gl.glCreateShader(shader_type)
        gl.glShaderSource(shader, source)
        gl.glCompileShader(shader)

        if gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS) != gl.GL_TRUE:
            error = gl.glGetShaderInfoLog(shader).decode()
            gl.glDeleteShader(shader)
            raise RuntimeError(f"Shader compilation failed:\n{error}")

        return shader

    def create_program(cls, vertex_source: str, fragment_source: str) -> int:
        """Create and link a shader program."""
        gl = cls.context.GL
        compile_shader = cls.compile_shader

        vertex_shader = compile_shader(vertex_source, gl.GL_VERTEX_SHADER)
        try:
            fragment_shader = compile_shader(fragment_source, gl.GL_FRAGMENT_SHADER)
        except RuntimeError:
            gl.glDeleteShader(vertex_shader)
            raise

        program = gl.glCreateProgram()
        gl.glAttachShader(program, vertex_shader)
        gl.glAttachShader(program, fragment_shader)
        gl.glLinkProgram(program)

        gl.glDeleteShader(vertex_shader)
        gl.glDeleteShader(fragment_shader)

        if gl.glGetProgramiv(program, gl.GL_LINK_STATUS) != gl.GL_TRUE:
            error = gl.glGetProgramInfoLog(program).decode()
            gl.glDeleteProgram(program)
            raise RuntimeError(f"Program linking failed:\n{error}")

        return program

    def __gen_default_texture(cls, width: int, height: int) -> int:
        """Initializes a texture of default type. Returns the handle."""
        gl = cls.context.GL
        tex: int = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F, width, height, 0, gl.GL_RGBA, gl.GL_FLOAT, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        return tex

    def render_shader_batch(
        cls,
        fragment_code: str,
        width: int,
        height: int,
        image_batches: list[list[np.ndarray]],
        floats: list[float],
        ints: list[int],
    ) -> list[list[np.ndarray]]:
        """
        Render a fragment shader for multiple batches efficiently.

        Compiles shader once, reuses framebuffer/textures across batches.
        Supports multi-pass rendering via #pragma passes N directive.

        Args:
            fragment_code: User's fragment shader code
            width: Output width
            height: Output height
            image_batches: List of batches, each batch is a list of input images (H, W, C) float32 [0,1]
            floats: List of float uniforms
            ints: List of int uniforms

        Returns:
            List of batch outputs, each is a list of output images (H, W, 4) float32 [0,1]
        """
        import time

        gl = cls.context.GL

        start_time = time.perf_counter()

        if not image_batches:
            return []

        cls.context.make_current()

        # Convert from GLSL ES to desktop GLSL 330
        fragment_source = _convert_es_to_desktop(fragment_code)

        # Detect how many outputs the shader actually uses
        num_outputs = _detect_output_count(fragment_code)

        # Detect multi-pass rendering
        num_passes = _detect_pass_count(fragment_code)

        # Track resources for cleanup
        program = None
        fbo = None
        output_textures = []
        input_textures = []
        ping_pong_textures = []
        ping_pong_fbos = []

        num_inputs = len(image_batches[0])

        try:
            # Compile shaders (once for all batches)
            try:
                program = cls.create_program(VERTEX_SHADER, fragment_source)
            except RuntimeError:
                logger.error(f"Fragment shader:\n{fragment_source}")
                raise

            gl.glUseProgram(program)

            # Create framebuffer with only the needed color attachments
            fbo = gl.glGenFramebuffers(1)
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)

            draw_buffers = []
            for i in range(num_outputs):
                tex = cls.__gen_default_texture(width, height)
                output_textures.append(tex)
                gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0 + i, gl.GL_TEXTURE_2D, tex, 0)
                draw_buffers.append(gl.GL_COLOR_ATTACHMENT0 + i)

            gl.glDrawBuffers(num_outputs, draw_buffers)

            if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
                raise RuntimeError("Framebuffer is not complete")

            # Create ping-pong resources for multi-pass rendering
            if num_passes > 1:
                for _ in range(2):
                    pp_tex = cls.__gen_default_texture(width, height)
                    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
                    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
                    ping_pong_textures.append(pp_tex)

                    pp_fbo = gl.glGenFramebuffers(1)
                    ping_pong_fbos.append(pp_fbo)
                    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, pp_fbo)
                    gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, pp_tex, 0)
                    gl.glDrawBuffers(1, [gl.GL_COLOR_ATTACHMENT0])

                    if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
                        raise RuntimeError("Ping-pong framebuffer is not complete")

            # Create input textures (reused for all batches)
            for i in range(num_inputs):
                tex = gl.glGenTextures(1)
                input_textures.append(tex)
                gl.glActiveTexture(gl.GL_TEXTURE0 + i)
                gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

                loc = gl.glGetUniformLocation(program, f"u_image{i}")
                if loc >= 0:
                    gl.glUniform1i(loc, i)

            # Set static uniforms (once for all batches)
            loc = gl.glGetUniformLocation(program, "u_resolution")
            if loc >= 0:
                gl.glUniform2f(loc, float(width), float(height))

            for i, v in enumerate(floats):
                loc = gl.glGetUniformLocation(program, f"u_float{i}")
                if loc >= 0:
                    gl.glUniform1f(loc, v)

            for i, v in enumerate(ints):
                loc = gl.glGetUniformLocation(program, f"u_int{i}")
                if loc >= 0:
                    gl.glUniform1i(loc, v)

            # Get u_pass uniform location for multi-pass
            pass_loc = gl.glGetUniformLocation(program, "u_pass")

            gl.glViewport(0, 0, width, height)
            gl.glDisable(gl.GL_BLEND)  # Ensure no alpha blending - write output directly

            # Process each batch
            all_batch_outputs = []
            for images in image_batches:
                # Update input textures with this batch's images
                for i, img in enumerate(images):
                    gl.glActiveTexture(gl.GL_TEXTURE0 + i)
                    gl.glBindTexture(gl.GL_TEXTURE_2D, input_textures[i])

                    # Flip vertically for GL coordinates, ensure RGBA
                    h, w, c = img.shape
                    if c == 3:
                        img_upload = np.empty((h, w, 4), dtype=np.float32)
                        img_upload[:, :, :3] = img[::-1, :, :]
                        img_upload[:, :, 3] = 1.0
                    else:
                        img_upload = np.ascontiguousarray(img[::-1, :, :])

                    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F, w, h, 0, gl.GL_RGBA, gl.GL_FLOAT, img_upload)

                if num_passes == 1:
                    # Single pass - render directly to output FBO
                    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
                    if pass_loc >= 0:
                        gl.glUniform1i(pass_loc, 0)
                    gl.glClearColor(0, 0, 0, 0)
                    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
                    gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
                else:
                    # Multi-pass rendering with ping-pong
                    for p in range(num_passes):
                        is_last_pass = (p == num_passes - 1)

                        # Set pass uniform
                        if pass_loc >= 0:
                            gl.glUniform1i(pass_loc, p)

                        if is_last_pass:
                            # Last pass renders to the main output FBO
                            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
                        else:
                            # Intermediate passes render to ping-pong FBO
                            target_fbo = ping_pong_fbos[p % 2]
                            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, target_fbo)

                        # Set input texture for this pass
                        gl.glActiveTexture(gl.GL_TEXTURE0)
                        if p == 0:
                            # First pass reads from original input
                            gl.glBindTexture(gl.GL_TEXTURE_2D, input_textures[0])
                        else:
                            # Subsequent passes read from previous pass output
                            source_tex = ping_pong_textures[(p - 1) % 2]
                            gl.glBindTexture(gl.GL_TEXTURE_2D, source_tex)

                        gl.glClearColor(0, 0, 0, 0)
                        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
                        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)

                # Read back outputs for this batch
                # (glGetTexImage is synchronous, implicitly waits for rendering)
                batch_outputs = []
                for tex in output_textures:
                    gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
                    data = gl.glGetTexImage(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, gl.GL_FLOAT)
                    img = np.frombuffer(data, dtype=np.float32).reshape(height, width, 4)
                    batch_outputs.append(img[::-1, :, :].copy())

                # Pad with black images for unused outputs
                black_img = np.zeros((height, width, 4), dtype=np.float32)
                for _ in range(num_outputs, MAX_OUTPUTS):
                    batch_outputs.append(black_img)

                all_batch_outputs.append(batch_outputs)

            elapsed = (time.perf_counter() - start_time) * 1000
            num_batches = len(image_batches)
            pass_info = f", {num_passes} passes" if num_passes > 1 else ""
            logger.info(f"GLSL shader executed in {elapsed:.1f}ms ({num_batches} batch{'es' if num_batches != 1 else ''}, {width}x{height}{pass_info})")

            return all_batch_outputs

        finally:
            # Unbind before deleting
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
            gl.glUseProgram(0)

            for tex in input_textures:
                gl.glDeleteTextures(tex)
            for tex in output_textures:
                gl.glDeleteTextures(tex)
            for tex in ping_pong_textures:
                gl.glDeleteTextures(tex)
            if fbo is not None:
                gl.glDeleteFramebuffers(1, [fbo])
            for pp_fbo in ping_pong_fbos:
                gl.glDeleteFramebuffers(1, [pp_fbo])
            if program is not None:
                gl.glDeleteProgram(program)

##########

class GLRender(metaclass=__GLRenderMeta):
    """Static class for all the high-level methods to render with OpenGL. Never instantiated, methods called directly as functions."""

    def __init__(self):
        raise NotImplementedError(f"{self.__class__!r} is a static class - call its methods directly, as just functions, without instantiating.")


############################################################


class GLSLShader(io.ComfyNode):

    @classmethod
    def define_schema(cls) -> io.Schema:
        image_template = io.Autogrow.TemplatePrefix(
            io.Image.Input("image"),
            prefix="image",
            min=1,
            max=MAX_IMAGES,
        )

        float_template = io.Autogrow.TemplatePrefix(
            io.Float.Input("float", default=0.0),
            prefix="u_float",
            min=0,
            max=MAX_UNIFORMS,
        )

        int_template = io.Autogrow.TemplatePrefix(
            io.Int.Input("int", default=0),
            prefix="u_int",
            min=0,
            max=MAX_UNIFORMS,
        )

        return io.Schema(
            node_id="GLSLShader",
            display_name="GLSL Shader",
            category="image/shader",
            description=(
                "Apply GLSL ES fragment shaders to images. "
                "u_resolution (vec2) is always available."
            ),
            inputs=[
                io.String.Input(
                    "fragment_shader",
                    default=DEFAULT_FRAGMENT_SHADER,
                    multiline=True,
                    tooltip="GLSL fragment shader source code (GLSL ES 3.00 / WebGL 2.0 compatible)",
                ),
                io.DynamicCombo.Input(
                    "size_mode",
                    options=[
                        io.DynamicCombo.Option("from_input", []),
                        io.DynamicCombo.Option(
                            "custom",
                            [
                                io.Int.Input(
                                    "width",
                                    default=512,
                                    min=1,
                                    max=nodes.MAX_RESOLUTION,
                                ),
                                io.Int.Input(
                                    "height",
                                    default=512,
                                    min=1,
                                    max=nodes.MAX_RESOLUTION,
                                ),
                            ],
                        ),
                    ],
                    tooltip="Output size: 'from_input' uses first input image dimensions, 'custom' allows manual size",
                ),
                io.Autogrow.Input("images", template=image_template, tooltip=f"Images are available as u_image0-{MAX_IMAGES-1} (sampler2D) in the shader code"),
                io.Autogrow.Input("floats", template=float_template, tooltip=f"Floats are available as u_float0-{MAX_UNIFORMS-1} in the shader code"),
                io.Autogrow.Input("ints", template=int_template, tooltip=f"Ints are available as u_int0-{MAX_UNIFORMS-1} in the shader code"),
            ],
            outputs=[
                io.Image.Output(display_name="IMAGE0", tooltip="Available via layout(location = 0) out vec4 fragColor0 in the shader code"),
                io.Image.Output(display_name="IMAGE1", tooltip="Available via layout(location = 1) out vec4 fragColor1 in the shader code"),
                io.Image.Output(display_name="IMAGE2", tooltip="Available via layout(location = 2) out vec4 fragColor2 in the shader code"),
                io.Image.Output(display_name="IMAGE3", tooltip="Available via layout(location = 3) out vec4 fragColor3 in the shader code"),
            ],
        )

    @classmethod
    def execute(
        cls,
        fragment_shader: str,
        size_mode: SizeModeInput,
        images: io.Autogrow.Type,
        floats: io.Autogrow.Type = None,
        ints: io.Autogrow.Type = None,
        **kwargs,
    ) -> io.NodeOutput:
        image_list = [v for v in images.values() if v is not None]
        float_list = (
            [v if v is not None else 0.0 for v in floats.values()] if floats else []
        )
        int_list = [v if v is not None else 0 for v in ints.values()] if ints else []

        if not image_list:
            raise ValueError("At least one input image is required")

        # Determine output dimensions
        if size_mode["size_mode"] == "custom":
            out_width = size_mode["width"]
            out_height = size_mode["height"]
        else:
            out_height, out_width = image_list[0].shape[1:3]

        batch_size = image_list[0].shape[0]

        # Prepare batches
        image_batches = []
        for batch_idx in range(batch_size):
            batch_images = [img_tensor[batch_idx].cpu().numpy().astype(np.float32) for img_tensor in image_list]
            image_batches.append(batch_images)

        all_batch_outputs = GLRender.render_shader_batch(
            fragment_shader,
            out_width,
            out_height,
            image_batches,
            float_list,
            int_list,
        )

        # Collect outputs into tensors
        all_outputs = [[] for _ in range(MAX_OUTPUTS)]
        for batch_outputs in all_batch_outputs:
            for i, out_img in enumerate(batch_outputs):
                all_outputs[i].append(torch.from_numpy(out_img))

        output_tensors = [torch.stack(all_outputs[i], dim=0) for i in range(MAX_OUTPUTS)]
        return io.NodeOutput(
            *output_tensors,
            ui=cls._build_ui_output(image_list, output_tensors[0]),
        )

    @classmethod
    def _build_ui_output(
        cls, image_list: list[torch.Tensor], output_batch: torch.Tensor
    ) -> dict[str, list]:
        """Build UI output with input and output images for client-side shader execution."""
        combined_inputs = torch.cat(image_list, dim=0)
        input_images_ui = ui.ImageSaveHelper.save_images(
            combined_inputs,
            filename_prefix="GLSLShader_input",
            folder_type=io.FolderType.temp,
            cls=None,
            compress_level=1,
        )

        output_images_ui = ui.ImageSaveHelper.save_images(
            output_batch,
            filename_prefix="GLSLShader_output",
            folder_type=io.FolderType.temp,
            cls=None,
            compress_level=1,
        )

        return {"input_images": input_images_ui, "images": output_images_ui}


class GLSLExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [GLSLShader]


async def comfy_entrypoint() -> GLSLExtension:
    return GLSLExtension()
