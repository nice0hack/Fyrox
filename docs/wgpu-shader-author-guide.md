# Fyrox wgpu Shader Author Guide

This guide covers writing shaders for the Fyrox wgpu backend.

## Color Space Convention

**Linear working space is mandatory.** All GPU compute and intermediate storage is linear. The OS compositor handles the final linear→sRGB encode.

### What NOT to do

- **Do not** call `S_SRGBToLinear()` or `S_LinearToSRGB()`. These helpers do not exist in the wgpu shader pipeline.
- **Do not** manually decode/encode in fragment shaders. The swapchain format is `Rgba8Unorm` (linear).

### Texture Sampling

| Texture Format | wgpu Format | Behavior |
|---------------|-------------|----------|
| sRGB (tagged at load) | `Rgba8UnormSrgb` | Sampler auto-decodes sRGB→linear on sample |
| Linear | `Rgba8Unorm` | Sampler returns raw linear values |

When you author a `.shader` file:
- If the texture contains sRGB color data (albedo, diffuse), tag it as `SRGBA8` or `SRGB8` at load time. The wgpu backend maps this to `Rgba8UnormSrgb`, and the sampler auto-decodes.
- If the texture contains linear data (normal maps, roughness, metallic), tag it as `RGBA8` or `RGB8`. The wgpu backend maps this to `Rgba8Unorm`, and samples return raw linear.

## Vertex Colors and Brush Colors

UI brush colors, particle vertex colors, and widget vertex colors are authored in sRGB (0..1 float range). The CPU-side conversion at upload time is:

```rust
if c <= 0.04045 { c / 12.92 } else { pow((c + 0.055) / 1.055, 2.4) }
```

Alternatively, use the inline `srgb_to_linear` helper in the vertex shader for per-vertex colors:

```wgsl
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        pow((c + 0.055) / 1.055, 2.4)
    }
}
```

## Pipeline Variants

Fyrox's rendering pipeline uses named render passes. When you declare a pass in a `.shader` file, the engine selects the appropriate variant based on the pass name (GBuffer, ForwardOpaque, ForwardTransparent, etc.).

Do not hard-code blend/depth state in the shader. The engine manages pipeline variants.

## WGPU Validation

Enable the wgpu validation layer during development:

```bash
WGPU_VALIDATION=1 cargo run --release -p editor
```

This catches:
- Invalid resource bindings (missing textures, wrong sampler types)
- Pipeline misconfigurations
- Out-of-bounds buffer accesses

## Common Mistakes

### Double Conversion
If you see washed-out colors, you may be double-converting. For example, sampling an sRGB texture and then calling `pow(sample, 2.2)` is a double-decode. The sampler already decoded the sRGB texture; just use the sample directly.

### Missing Texture Bindings
Meshes without material bindings will now render with `white_dummy` instead of black (0 binding). If you see unexpected white geometry, check that all texture slots are properly bound in the material.

### Mismatched Texture Formats
If a shader expects linear data but receives sRGB-tagged data (or vice versa), colors will appear wrong. Always tag textures correctly at load time based on their actual content type.
