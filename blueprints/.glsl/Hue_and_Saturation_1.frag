#version 300 es
precision highp float;

uniform sampler2D u_image0;
uniform int u_int0;      // Mode: 0=Master, 1=Reds, 2=Yellows, 3=Greens, 4=Cyans, 5=Blues, 6=Magentas, 7=Colorize
uniform int u_int1;      // Color Space: 0=HSL, 1=HSB/HSV
uniform float u_float0;  // Hue (-180 to 180)
uniform float u_float1;  // Saturation (-100 to 100)
uniform float u_float2;  // Lightness/Brightness (-100 to 100)
uniform float u_float3;  // Overlap (0 to 100) - feathering between adjacent color ranges

in vec2 v_texCoord;
out vec4 fragColor;

// Color range modes
const int MODE_MASTER   = 0;
const int MODE_RED      = 1;
const int MODE_YELLOW   = 2;
const int MODE_GREEN    = 3;
const int MODE_CYAN     = 4;
const int MODE_BLUE     = 5;
const int MODE_MAGENTA  = 6;
const int MODE_COLORIZE = 7;

// Color space modes
const int COLORSPACE_HSL = 0;
const int COLORSPACE_HSB = 1;

const float EPSILON = 0.0001;

//=============================================================================
// RGB <-> HSL Conversions
//=============================================================================

vec3 rgb2hsl(vec3 c) {
    float maxC = max(max(c.r, c.g), c.b);
    float minC = min(min(c.r, c.g), c.b);
    float delta = maxC - minC;

    float h = 0.0;
    float s = 0.0;
    float l = (maxC + minC) * 0.5;

    if (delta > EPSILON) {
        s = l < 0.5
            ? delta / (maxC + minC)
            : delta / (2.0 - maxC - minC);

        if (maxC == c.r) {
            h = (c.g - c.b) / delta + (c.g < c.b ? 6.0 : 0.0);
        } else if (maxC == c.g) {
            h = (c.b - c.r) / delta + 2.0;
        } else {
            h = (c.r - c.g) / delta + 4.0;
        }
        h /= 6.0;
    }

    return vec3(h, s, l);
}

float hue2rgb(float p, float q, float t) {
    t = fract(t);
    if (t < 1.0/6.0) return p + (q - p) * 6.0 * t;
    if (t < 0.5)       return q;
    if (t < 2.0/3.0)   return p + (q - p) * (2.0/3.0 - t) * 6.0;
    return p;
}

vec3 hsl2rgb(vec3 hsl) {
    if (hsl.y < EPSILON) return vec3(hsl.z);

    float q = hsl.z < 0.5
        ? hsl.z * (1.0 + hsl.y)
        : hsl.z + hsl.y - hsl.z * hsl.y;
    float p = 2.0 * hsl.z - q;

    return vec3(
        hue2rgb(p, q, hsl.x + 1.0/3.0),
        hue2rgb(p, q, hsl.x),
        hue2rgb(p, q, hsl.x - 1.0/3.0)
    );
}

vec3 rgb2hsb(vec3 c) {
    float maxC = max(max(c.r, c.g), c.b);
    float minC = min(min(c.r, c.g), c.b);
    float delta = maxC - minC;

    float h = 0.0;
    float s = (maxC > EPSILON) ? delta / maxC : 0.0;
    float b = maxC;

    if (delta > EPSILON) {
        if (maxC == c.r) {
            h = (c.g - c.b) / delta + (c.g < c.b ? 6.0 : 0.0);
        } else if (maxC == c.g) {
            h = (c.b - c.r) / delta + 2.0;
        } else {
            h = (c.r - c.g) / delta + 4.0;
        }
        h /= 6.0;
    }

    return vec3(h, s, b);
}

vec3 hsb2rgb(vec3 hsb) {
    vec3 rgb = clamp(abs(mod(hsb.x * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
    return hsb.z * mix(vec3(1.0), rgb, hsb.y);
}

//=============================================================================
// Color Range Weight Calculation
//=============================================================================

float hueDistance(float a, float b) {
    float d = abs(a - b);
    return min(d, 1.0 - d);
}

float getHueWeight(float hue, float center, float overlap) {
    float baseWidth = 1.0 / 6.0;
    float feather = baseWidth * overlap;

    float d = hueDistance(hue, center);

    float inner = baseWidth * 0.5;
    float outer = inner + feather;

    return 1.0 - smoothstep(inner, outer, d);
}

float getModeWeight(float hue, int mode, float overlap) {
    if (mode == MODE_MASTER || mode == MODE_COLORIZE) return 1.0;

    if (mode == MODE_RED) {
        return max(
            getHueWeight(hue, 0.0, overlap),
            getHueWeight(hue, 1.0, overlap)
        );
    }

    float center = float(mode - 1) / 6.0;
    return getHueWeight(hue, center, overlap);
}

//=============================================================================
// Adjustment Functions
//=============================================================================

float adjustLightness(float l, float amount) {
    return amount > 0.0
        ? l + (1.0 - l) * amount
        : l + l * amount;
}

float adjustBrightness(float b, float amount) {
    return clamp(b + amount, 0.0, 1.0);
}

float adjustSaturation(float s, float amount) {
    return amount > 0.0
        ? s + (1.0 - s) * amount
        : s + s * amount;
}

vec3 colorize(vec3 rgb, float hue, float sat, float light) {
    float lum = dot(rgb, vec3(0.299, 0.587, 0.114));
    float l = adjustLightness(lum, light);

    vec3 hsl = vec3(fract(hue), clamp(sat, 0.0, 1.0), clamp(l, 0.0, 1.0));
    return hsl2rgb(hsl);
}

//=============================================================================
// Main
//=============================================================================

void main() {
    vec4 original = texture(u_image0, v_texCoord);

    float hueShift   = u_float0 / 360.0;   // -180..180 -> -0.5..0.5
    float satAmount  = u_float1 / 100.0;   // -100..100 -> -1..1
    float lightAmount= u_float2 / 100.0;   // -100..100 -> -1..1
    float overlap    = u_float3 / 100.0;   // 0..100 -> 0..1

    vec3 result;

    if (u_int0 == MODE_COLORIZE) {
        result = colorize(original.rgb, hueShift, satAmount, lightAmount);
        fragColor = vec4(result, original.a);
        return;
    }

    vec3 hsx = (u_int1 == COLORSPACE_HSL)
        ? rgb2hsl(original.rgb)
        : rgb2hsb(original.rgb);

    float weight = getModeWeight(hsx.x, u_int0, overlap);

    if (u_int0 != MODE_MASTER && hsx.y < EPSILON) {
        weight = 0.0;
    }

    if (weight > EPSILON) {
        float h = fract(hsx.x + hueShift * weight);
        float s = clamp(adjustSaturation(hsx.y, satAmount * weight), 0.0, 1.0);
        float v = (u_int1 == COLORSPACE_HSL)
            ? clamp(adjustLightness(hsx.z, lightAmount * weight), 0.0, 1.0)
            : clamp(adjustBrightness(hsx.z, lightAmount * weight), 0.0, 1.0);

        vec3 adjusted = vec3(h, s, v);
        result = (u_int1 == COLORSPACE_HSL)
            ? hsl2rgb(adjusted)
            : hsb2rgb(adjusted);
    } else {
        result = original.rgb;
    }

    fragColor = vec4(result, original.a);
}
