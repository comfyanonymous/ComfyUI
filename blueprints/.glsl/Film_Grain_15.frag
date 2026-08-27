#version 300 es
precision highp float;

uniform sampler2D u_image0;
uniform vec2 u_resolution;
uniform float u_float0; // grain amount      [0.0 – 1.0]   typical: 0.2–0.8
uniform float u_float1; // grain size        [0.3 – 3.0]   lower = finer grain
uniform float u_float2; // color amount      [0.0 – 1.0]   0 = monochrome, 1 = RGB grain
uniform float u_float3; // luminance bias    [0.0 – 1.0]   0 = uniform, 1 = shadows only
uniform int   u_int0;   // noise mode        [0 or 1]      0 = smooth, 1 = grainy

in vec2 v_texCoord;
layout(location = 0) out vec4 fragColor0;

// High-quality integer hash (pcg-like)
uint pcg(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// 2D -> 1D hash input
uint hash2d(uvec2 p) {
    return pcg(p.x + pcg(p.y));
}

// Hash to float [0, 1]
float hashf(uvec2 p) {
    return float(hash2d(p)) / float(0xffffffffu);
}

// Hash to float with offset (for RGB channels)
float hashf(uvec2 p, uint offset) {
    return float(pcg(hash2d(p) + offset)) / float(0xffffffffu);
}

// Convert uniform [0,1] to roughly Gaussian distribution
// Using simple approximation: average of multiple samples
float toGaussian(uvec2 p) {
    float sum = hashf(p, 0u) + hashf(p, 1u) + hashf(p, 2u) + hashf(p, 3u);
    return (sum - 2.0) * 0.7;  // Centered, scaled
}

float toGaussian(uvec2 p, uint offset) {
    float sum = hashf(p, offset) + hashf(p, offset + 1u) 
              + hashf(p, offset + 2u) + hashf(p, offset + 3u);
    return (sum - 2.0) * 0.7;
}

// Smooth noise with better interpolation
float smoothNoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    
    // Quintic interpolation (less banding than cubic)
    f = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    
    uvec2 ui = uvec2(i);
    float a = toGaussian(ui);
    float b = toGaussian(ui + uvec2(1u, 0u));
    float c = toGaussian(ui + uvec2(0u, 1u));
    float d = toGaussian(ui + uvec2(1u, 1u));
    
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float smoothNoise(vec2 p, uint offset) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    
    f = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    
    uvec2 ui = uvec2(i);
    float a = toGaussian(ui, offset);
    float b = toGaussian(ui + uvec2(1u, 0u), offset);
    float c = toGaussian(ui + uvec2(0u, 1u), offset);
    float d = toGaussian(ui + uvec2(1u, 1u), offset);
    
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

void main() {
    vec4 color = texture(u_image0, v_texCoord);
    
    // Luminance (Rec.709)
    float luma = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));
    
    // Grain UV (resolution-independent)
    vec2 grainUV = v_texCoord * u_resolution / max(u_float1, 0.01);
    uvec2 grainPixel = uvec2(grainUV);
    
    float g;
    vec3 grainRGB;
    
    if (u_int0 == 1) {
        // Grainy mode: pure hash noise (no interpolation = no banding)
        g = toGaussian(grainPixel);
        grainRGB = vec3(
            toGaussian(grainPixel, 100u),
            toGaussian(grainPixel, 200u),
            toGaussian(grainPixel, 300u)
        );
    } else {
        // Smooth mode: interpolated with quintic curve
        g = smoothNoise(grainUV);
        grainRGB = vec3(
            smoothNoise(grainUV, 100u),
            smoothNoise(grainUV, 200u),
            smoothNoise(grainUV, 300u)
        );
    }
    
    // Luminance weighting (less grain in highlights)
    float lumWeight = mix(1.0, 1.0 - luma, clamp(u_float3, 0.0, 1.0));
    
    // Strength
    float strength = u_float0 * 0.15;
    
    // Color vs monochrome grain
    vec3 grainColor = mix(vec3(g), grainRGB, clamp(u_float2, 0.0, 1.0));
    
    color.rgb += grainColor * strength * lumWeight;
    fragColor0 = vec4(clamp(color.rgb, 0.0, 1.0), color.a);
}
