#version 300 es
precision highp float;

uniform sampler2D u_image0;
uniform float u_float0;   // Blur radius (0–20, default ~5)
uniform float u_float1;   // Edge threshold (0–100, default ~30)
uniform int u_int0;       // Step size (0/1 = every pixel, 2+ = skip pixels)

in vec2 v_texCoord;
out vec4 fragColor;

const int MAX_RADIUS = 20;
const float EPSILON = 0.0001;

// Perceptual luminance
float getLuminance(vec3 rgb) {
    return dot(rgb, vec3(0.299, 0.587, 0.114));
}

vec4 bilateralFilter(vec2 uv, vec2 texelSize, int radius,
                     float sigmaSpatial, float sigmaColor)
{
    vec4 center = texture(u_image0, uv);
    vec3 centerRGB = center.rgb;

    float invSpatial2 = -0.5 / (sigmaSpatial * sigmaSpatial);
    float invColor2   = -0.5 / (sigmaColor * sigmaColor + EPSILON);

    vec3 sumRGB = vec3(0.0);
    float sumWeight = 0.0;

    int step = max(u_int0, 1);
    float radius2 = float(radius * radius);

    for (int dy = -MAX_RADIUS; dy <= MAX_RADIUS; dy++) {
        if (dy < -radius || dy > radius) continue;
        if (abs(dy) % step != 0) continue;

        for (int dx = -MAX_RADIUS; dx <= MAX_RADIUS; dx++) {
            if (dx < -radius || dx > radius) continue;
            if (abs(dx) % step != 0) continue;

            vec2 offset = vec2(float(dx), float(dy));
            float dist2 = dot(offset, offset);
            if (dist2 > radius2) continue;

            vec3 sampleRGB = texture(u_image0, uv + offset * texelSize).rgb;

            // Spatial Gaussian
            float spatialWeight = exp(dist2 * invSpatial2);

            // Perceptual color distance (weighted RGB)
            vec3 diff = sampleRGB - centerRGB;
            float colorDist = dot(diff * diff, vec3(0.299, 0.587, 0.114));
            float colorWeight = exp(colorDist * invColor2);

            float w = spatialWeight * colorWeight;
            sumRGB += sampleRGB * w;
            sumWeight += w;
        }
    }

    vec3 resultRGB = sumRGB / max(sumWeight, EPSILON);
    return vec4(resultRGB, center.a); // preserve center alpha
}

void main() {
    vec2 texelSize = 1.0 / vec2(textureSize(u_image0, 0));

    float radiusF = clamp(u_float0, 0.0, float(MAX_RADIUS));
    int radius = int(radiusF + 0.5);

    if (radius == 0) {
        fragColor = texture(u_image0, v_texCoord);
        return;
    }

    // Edge threshold → color sigma
    // Squared curve for better low-end control
    float t = clamp(u_float1, 0.0, 100.0) / 100.0;
    t *= t;
    float sigmaColor = mix(0.01, 0.5, t);

    // Spatial sigma tied to radius
    float sigmaSpatial = max(radiusF * 0.75, 0.5);

    fragColor = bilateralFilter(
        v_texCoord,
        texelSize,
        radius,
        sigmaSpatial,
        sigmaColor
    );
}