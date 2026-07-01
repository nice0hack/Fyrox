// Shared functions for all shaders in the engine.
// Naga-compatible version: uses texture2D/texture3D/textureCube + sampler
// as separate parameters instead of combined sampler2D/sampler3D/samplerCube.

const float PI = 3.14159;

bool S_SolveQuadraticEq(float a, float b, float c, out float minT, out float maxT)
{
    float twoA = 2.0 * a;
    float det = b * b - 2.0 * twoA * c;
    if (det < 0.0) { minT = 0.0; maxT = 0.0; return false; }
    float sqrtDet = sqrt(det);
    float root1 = (-b - sqrtDet) / twoA;
    float root2 = (-b + sqrtDet) / twoA;
    minT = min(root1, root2);
    maxT = max(root1, root2);
    return true;
}

float S_LightDistanceAttenuation(float distance, float radius)
{
    return clamp(1.0 - distance * distance / (radius * radius), 0.0, 1.0);
}

vec3 S_Project(vec3 worldPosition, mat4 matrix)
{
    vec4 screenPos = matrix * vec4(worldPosition, 1);
    screenPos.xyz /= screenPos.w;
    return screenPos.xyz * 0.5 + 0.5;
}

vec3 S_UnProject(vec3 screenPos, mat4 matrix)
{
    vec4 clipSpacePos = vec4(screenPos * 2.0 - 1.0, 1.0);
    vec4 position = matrix * clipSpacePos;
    return position.xyz / position.w;
}

float S_DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float nom = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
    return nom / denom;
}

float S_GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;
    float nom = NdotV;
    float denom = NdotV * (1.0 - k) + k;
    return nom / denom;
}

float S_GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = S_GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = S_GeometrySchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
}

vec3 S_FresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(max(1.0 - cosTheta, 0.0), 5.0);
}

vec3 S_FresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness)
{
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(1.0 - cosTheta, 5.0);
}

struct TPBRContext {
    vec3 lightColor;
    vec3 viewVector;
    vec3 fragmentToLight;
    vec3 fragmentNormal;
    float metallic;
    float roughness;
    vec3 albedo;
};

vec3 S_PBR_CalculateLight(TPBRContext ctx) {
    vec3 F0 = mix(vec3(0.04), ctx.albedo, ctx.metallic);
    vec3 L = ctx.fragmentToLight;
    vec3 H = normalize(ctx.viewVector + L);
    float NDF = S_DistributionGGX(ctx.fragmentNormal, H, ctx.roughness);
    float G = S_GeometrySmith(ctx.fragmentNormal, ctx.viewVector, L, ctx.roughness);
    vec3 F = S_FresnelSchlick(max(dot(H, ctx.viewVector), 0.0), F0);
    vec3 numerator = NDF * G * F;
    float denominator = 4.0 * max(dot(ctx.fragmentNormal, ctx.viewVector), 0.0) * max(dot(ctx.fragmentNormal, L), 0.0) + 0.001;
    vec3 specular = numerator / denominator;
    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - ctx.metallic;
    float NdotL = max(dot(ctx.fragmentNormal, L), 0.0);
    return (kD * ctx.albedo / PI + specular) * ctx.lightColor * NdotL;
}

float S_InScatter(vec3 start, vec3 dir, vec3 lightPos, float d)
{
    vec3 q = start - lightPos;
    float b = dot(dir, q);
    float c = dot(q, q);
    float s = 1.0 / sqrt(c - b * b);
    float l = s * (atan((d + b) * s) - atan(b * s));
    return l;
}

vec3 S_RayleighScatter(vec3 start, vec3 dir, vec3 lightPos, float d)
{
    float scatter = S_InScatter(start, dir, lightPos, d);
    return vec3(0.55, 0.75, 1.0) * scatter;
}

bool S_RaySphereIntersection(vec3 origin, vec3 dir, vec3 center, float radius, out float minT, out float maxT)
{
    vec3 d = origin - center;
    float a = dot(dir, dir);
    float b = 2.0 * dot(dir, d);
    float c = dot(d, d) - radius * radius;
    return S_SolveQuadraticEq(a, b, c, minT, maxT);
}

float S_PointShadow(
    bool shadowsEnabled,
    bool softShadows,
    float fragmentDistance,
    float shadowBias,
    vec3 toLight,
    textureCube shadowMap_tex,
    sampler shadowMap_samp)
{
    if (shadowsEnabled)
    {
        float biasedFragmentDistance = fragmentDistance - shadowBias;

        if (softShadows)
        {
            const vec3 directions[20] = vec3[20](
            vec3(1, 1, 1), vec3(1, -1, 1), vec3(-1, -1, 1), vec3(-1, 1, 1),
            vec3(1, 1, -1), vec3(1, -1, -1), vec3(-1, -1, -1), vec3(-1, 1, -1),
            vec3(1, 1, 0), vec3(1, -1, 0), vec3(-1, -1, 0), vec3(-1, 1, 0),
            vec3(1, 0, 1), vec3(-1, 0, 1), vec3(1, 0, -1), vec3(-1, 0, -1),
            vec3(0, 1, 1), vec3(0, -1, 1), vec3(0, -1, -1), vec3(0, 1, -1)
            );

            const float diskRadius = 0.0025;

            float accumulator = 0.0;

            for (int i = 0; i < 20; ++i)
            {
                vec3 fetchDirection = -toLight + directions[i] * diskRadius;
                float shadowDistanceToLight = texture(samplerCube(shadowMap_tex, shadowMap_samp), fetchDirection).r;
                if (biasedFragmentDistance > shadowDistanceToLight)
                {
                    accumulator += 1.0;
                }
            }

            return clamp(1.0 - accumulator / 20.0, 0.0, 1.0);
        }
        else
        {
            float shadowDistanceToLight = texture(samplerCube(shadowMap_tex, shadowMap_samp), -toLight).r;
            return biasedFragmentDistance > shadowDistanceToLight ? 0.0 : 1.0;
        }
    } else {
        return 1.0;
    }
}

float S_SpotShadowFactor(
    bool shadowsEnabled,
    bool softShadows,
    float shadowBias,
    vec3 fragmentPosition,
    mat4 lightViewProjMatrix,
    float shadowMapInvSize,
    texture2D spotShadowTexture_tex,
    sampler spotShadowTexture_samp)
{
    if (shadowsEnabled)
    {
        vec3 lightSpacePosition = S_Project(fragmentPosition, lightViewProjMatrix);

        float biasedLightSpaceFragmentDepth = lightSpacePosition.z - shadowBias;

        if (softShadows)
        {
            float accumulator = 0.0;

            float step = 0.5;
            float kernelHalfSize = 2.0;
            float kernelSize = 2.0 * kernelHalfSize;
            float totalSamples = pow(kernelSize / step, 2.0);

            for (float y = -kernelHalfSize; y <= kernelHalfSize; y += step)
            {
                for (float x = -kernelHalfSize; x <= kernelHalfSize; x += step)
                {
                    vec2 fetchTexCoord = lightSpacePosition.xy + vec2(x, y) * shadowMapInvSize;
                    if (biasedLightSpaceFragmentDepth > texture(sampler2D(spotShadowTexture_tex, spotShadowTexture_samp), fetchTexCoord).r)
                    {
                        accumulator += 1.0;
                    }
                }
            }

            return clamp(1.0 - accumulator / totalSamples, 0.0, 1.0);
        }
        else
        {
            return biasedLightSpaceFragmentDepth > texture(sampler2D(spotShadowTexture_tex, spotShadowTexture_samp), lightSpacePosition.xy).r ? 0.0 : 1.0;
        }
    } else {
        return 1.0;
    }
}

float Internal_FetchHeight(texture2D heightTexture_tex, sampler heightTexture_samp, vec2 texCoords, float center) {
    return clamp(texture(sampler2D(heightTexture_tex, heightTexture_samp), texCoords).r - center, 0.0, 1.0);
}

vec2 S_ComputeParallaxTextureCoordinates(texture2D heightTexture_tex, sampler heightTexture_samp, vec3 eyeVec, vec2 texCoords, float center, float scale) {
    const float minLayers = 8.0;
    const float maxLayers = 15.0;
    const int maxIterations = 15;

    float t = max(0.0, abs(dot(vec3(0.0, 0.0, 1.0), eyeVec)));
    float numLayers = mix(maxLayers, minLayers, t);
    float layerDepth = 1.0 / numLayers;
    float currentLayerDepth = 0.0;

    vec2 deltaTexCoords = scale * eyeVec.xy / numLayers;

    vec2 currentTexCoords = texCoords;
    float currentDepthMapValue = Internal_FetchHeight(heightTexture_tex, heightTexture_samp, currentTexCoords, center);

    for (int i = 0; i < maxIterations; i++) {
        if (currentLayerDepth < currentDepthMapValue) {
            currentTexCoords -= deltaTexCoords;
            currentDepthMapValue = Internal_FetchHeight(heightTexture_tex, heightTexture_samp, currentTexCoords, center);
            currentLayerDepth += layerDepth;
        } else {
            break;
        }
    }

    vec2 prev = currentTexCoords + deltaTexCoords;
    float nextH = currentDepthMapValue - currentLayerDepth;
    float prevH = Internal_FetchHeight(heightTexture_tex, heightTexture_samp, prev, center) - currentLayerDepth + layerDepth;

    float weight = nextH / (nextH - prevH);

    return prev * weight + currentTexCoords * (1.0 - weight);
}

ivec2 S_LinearIndexToPosition(int index, int textureWidth) {
    int y = index / textureWidth;
    int x = index - textureWidth * y;
    return ivec2(x, y);
}

mat4 S_FetchMatrix(texture2D storage_tex, sampler storage_samp, int index) {
    int textureWidth = textureSize(storage_tex, 0).x;
    ivec2 pos = S_LinearIndexToPosition(4 * index, textureWidth);

    vec4 col1 = texelFetch(sampler2D(storage_tex, storage_samp), pos, 0);
    vec4 col2 = texelFetch(sampler2D(storage_tex, storage_samp), ivec2(pos.x + 1, pos.y), 0);
    vec4 col3 = texelFetch(sampler2D(storage_tex, storage_samp), ivec2(pos.x + 2, pos.y), 0);
    vec4 col4 = texelFetch(sampler2D(storage_tex, storage_samp), ivec2(pos.x + 3, pos.y), 0);

    return mat4(col1, col2, col3, col4);
}

struct TBlendShapeOffsets {
    vec3 position;
    vec3 normal;
    vec3 tangent;
};

TBlendShapeOffsets S_FetchBlendShapeOffsets(texture3D storage_tex, sampler storage_samp, int vertexIndex, int blendShapeIndex) {
    int textureWidth = textureSize(storage_tex, 0).x;
    ivec3 pos = ivec3(S_LinearIndexToPosition(3 * vertexIndex, textureWidth), blendShapeIndex);
    vec3 position = texelFetch(sampler3D(storage_tex, storage_samp), pos, 0).xyz;
    vec3 normal = texelFetch(sampler3D(storage_tex, storage_samp), ivec3(pos.x + 1, pos.y, pos.z), 0).xyz;
    vec3 tangent = texelFetch(sampler3D(storage_tex, storage_samp), ivec3(pos.x + 2, pos.y, pos.z), 0).xyz;
    return TBlendShapeOffsets(position, normal, tangent);
}

vec4 S_LinearToSRGB(vec4 color) {
    vec3 a = 12.92 * color.rgb;
    vec3 b = 1.055 * pow(color.rgb, vec3(1.0 / 2.4)) - 0.055;
    vec3 c = step(vec3(0.0031308), color.rgb);
    return vec4(mix(a, b, c), color.a);
}

vec4 S_SRGBToLinear(vec4 color) {
    vec3 a = color.rgb / 12.92;
    vec3 b = pow((color.rgb + 0.055) / 1.055, vec3(2.4));
    vec3 c = step(vec3(0.04045), color.rgb);
    return vec4(mix(a, b, c), color.a);
}

float S_Luminance(vec3 x) {
    return dot(x, vec3(0.2125, 0.7154, 0.0721));
}

vec2 S_RotateVec2(vec2 v, float angle)
{
    float c = cos(angle);
    float s = sin(angle);
    mat2 m = mat2(c, -s, s, c);
    return m * v;
}

vec3 S_ConvertRgbToXyz(vec3 rgb)
{
    vec3 xyz;
    xyz.x = dot(vec3(0.4124564, 0.3575761, 0.1804375), rgb);
    xyz.y = dot(vec3(0.2126729, 0.7151522, 0.0721750), rgb);
    xyz.z = dot(vec3(0.0193339, 0.1191920, 0.9503041), rgb);
    return xyz;
}

vec3 S_ConvertXyzToRgb(vec3 xyz)
{
    vec3 rgb;
    rgb.x = dot(vec3(3.2404542, -1.5371385, -0.4985314), xyz);
    rgb.y = dot(vec3(-0.9692660, 1.8760108, 0.0415560), xyz);
    rgb.z = dot(vec3(0.0556434, -0.2040259, 1.0572252), xyz);
    return rgb;
}

vec3 S_ConvertXyzToYxy(vec3 xyz)
{
    float inv = 1.0 / dot(xyz, vec3(1.0, 1.0, 1.0));
    return vec3(xyz.y, xyz.x * inv, xyz.y * inv);
}

vec3 S_ConvertYxyToXyz(vec3 Yxy)
{
    vec3 xyz;
    xyz.x = Yxy.x * Yxy.y / Yxy.z;
    xyz.y = Yxy.x;
    xyz.z = Yxy.x * (1.0 - Yxy.y - Yxy.z) / Yxy.z;
    return xyz;
}

vec3 S_ConvertRgbToYxy(vec3 rgb)
{
    return S_ConvertXyzToYxy(S_ConvertRgbToXyz(rgb));
}

vec3 S_ConvertYxyToRgb(vec3 Yxy)
{
    return S_ConvertXyzToRgb(S_ConvertYxyToXyz(Yxy));
}
