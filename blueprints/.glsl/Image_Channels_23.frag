#version 300 es
precision highp float;

uniform sampler2D u_image0;

in vec2 v_texCoord;
layout(location = 0) out vec4 fragColor0;
layout(location = 1) out vec4 fragColor1;
layout(location = 2) out vec4 fragColor2;
layout(location = 3) out vec4 fragColor3;

void main() {
  vec4 color = texture(u_image0, v_texCoord);
  // Output each channel as grayscale to separate render targets
  fragColor0 = vec4(vec3(color.r), 1.0);  // Red channel
  fragColor1 = vec4(vec3(color.g), 1.0);  // Green channel
  fragColor2 = vec4(vec3(color.b), 1.0);  // Blue channel
  fragColor3 = vec4(vec3(color.a), 1.0);  // Alpha channel
}
