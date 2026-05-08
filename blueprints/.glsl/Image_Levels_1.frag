#version 300 es
precision highp float;

// Levels Adjustment
// u_int0:   channel      (0=RGB, 1=R, 2=G, 3=B)         default: 0
// u_float0: input black  (0-255)                        default: 0
// u_float1: input white  (0-255)                        default: 255
// u_float2: gamma        (0.01-9.99)                    default: 1.0
// u_float3: output black (0-255)                        default: 0
// u_float4: output white (0-255)                        default: 255

uniform sampler2D u_image0;
uniform int u_int0;
uniform float u_float0;
uniform float u_float1;
uniform float u_float2;
uniform float u_float3;
uniform float u_float4;

in vec2 v_texCoord;
out vec4 fragColor;

vec3 applyLevels(vec3 color, float inBlack, float inWhite, float gamma, float outBlack, float outWhite) {
    float inRange = max(inWhite - inBlack, 0.0001);
    vec3 result = clamp((color - inBlack) / inRange, 0.0, 1.0);
    result = pow(result, vec3(1.0 / gamma));
    result = mix(vec3(outBlack), vec3(outWhite), result);
    return result;
}

float applySingleChannel(float value, float inBlack, float inWhite, float gamma, float outBlack, float outWhite) {
    float inRange = max(inWhite - inBlack, 0.0001);
    float result = clamp((value - inBlack) / inRange, 0.0, 1.0);
    result = pow(result, 1.0 / gamma);
    result = mix(outBlack, outWhite, result);
    return result;
}

void main() {
    vec4 texColor = texture(u_image0, v_texCoord);
    vec3 color = texColor.rgb;
    
    float inBlack = u_float0 / 255.0;
    float inWhite = u_float1 / 255.0;
    float gamma = u_float2;
    float outBlack = u_float3 / 255.0;
    float outWhite = u_float4 / 255.0;
    
    vec3 result;
    
    if (u_int0 == 0) {
        result = applyLevels(color, inBlack, inWhite, gamma, outBlack, outWhite);
    }
    else if (u_int0 == 1) {
        result = color;
        result.r = applySingleChannel(color.r, inBlack, inWhite, gamma, outBlack, outWhite);
    }
    else if (u_int0 == 2) {
        result = color;
        result.g = applySingleChannel(color.g, inBlack, inWhite, gamma, outBlack, outWhite);
    }
    else if (u_int0 == 3) {
        result = color;
        result.b = applySingleChannel(color.b, inBlack, inWhite, gamma, outBlack, outWhite);
    }
    else {
        result = color;
    }
    
    fragColor = vec4(result, texColor.a);
}