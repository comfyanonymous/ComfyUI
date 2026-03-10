#version 300 es
precision highp float;

uniform sampler2D u_image0;
uniform float u_float0; // temperature (-100 to 100)
uniform float u_float1; // tint (-100 to 100)
uniform float u_float2; // vibrance (-100 to 100)
uniform float u_float3; // saturation (-100 to 100)

in vec2 v_texCoord;
out vec4 fragColor;

const float INPUT_SCALE = 0.01;
const float TEMP_TINT_PRIMARY = 0.3;
const float TEMP_TINT_SECONDARY = 0.15;
const float VIBRANCE_BOOST = 2.0;
const float SATURATION_BOOST = 2.0;
const float SKIN_PROTECTION = 0.5;
const float EPSILON = 0.001;
const vec3 LUMA_WEIGHTS = vec3(0.299, 0.587, 0.114);

void main() {
    vec4 tex = texture(u_image0, v_texCoord);
    vec3 color = tex.rgb;
    
    // Scale inputs: -100/100 → -1/1
    float temperature = u_float0 * INPUT_SCALE;
    float tint = u_float1 * INPUT_SCALE;
    float vibrance = u_float2 * INPUT_SCALE;
    float saturation = u_float3 * INPUT_SCALE;
    
    // Temperature (warm/cool): positive = warm, negative = cool
    color.r += temperature * TEMP_TINT_PRIMARY;
    color.b -= temperature * TEMP_TINT_PRIMARY;
    
    // Tint (green/magenta): positive = green, negative = magenta
    color.g += tint * TEMP_TINT_PRIMARY;
    color.r -= tint * TEMP_TINT_SECONDARY;
    color.b -= tint * TEMP_TINT_SECONDARY;
    
    // Single clamp after temperature/tint
    color = clamp(color, 0.0, 1.0);
    
    // Vibrance with skin protection
    if (vibrance != 0.0) {
        float maxC = max(color.r, max(color.g, color.b));
        float minC = min(color.r, min(color.g, color.b));
        float sat = maxC - minC;
        float gray = dot(color, LUMA_WEIGHTS);
        
        if (vibrance < 0.0) {
            // Desaturate: -100 → gray
            color = mix(vec3(gray), color, 1.0 + vibrance);
        } else {
            // Boost less saturated colors more
            float vibranceAmt = vibrance * (1.0 - sat);
            
            // Branchless skin tone protection
            float isWarmTone = step(color.b, color.g) * step(color.g, color.r);
            float warmth = (color.r - color.b) / max(maxC, EPSILON);
            float skinTone = isWarmTone * warmth * sat * (1.0 - sat);
            vibranceAmt *= (1.0 - skinTone * SKIN_PROTECTION);
            
            color = mix(vec3(gray), color, 1.0 + vibranceAmt * VIBRANCE_BOOST);
        }
    }
    
    // Saturation
    if (saturation != 0.0) {
        float gray = dot(color, LUMA_WEIGHTS);
        float satMix = saturation < 0.0
            ? 1.0 + saturation                      // -100 → gray
            : 1.0 + saturation * SATURATION_BOOST;  // +100 → 3x boost
        color = mix(vec3(gray), color, satMix);
    }
    
    fragColor = vec4(clamp(color, 0.0, 1.0), tex.a);
}