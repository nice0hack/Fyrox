# Fyrox wgpu Color-Space Convention

## Decision

Fyrox uses **linear working space** for all GPU compute and intermediate storage.
The final linear→sRGB encode happens exactly once, by the OS compositor / display
path, when the linear output reaches the screen.

## Why

The OpenGL backend's default framebuffer was linear (`GL_FRAMEBUFFER_SRGB` was
never enabled). The new wgpu backend matches that: the swapchain is configured as
`Rgba8Unorm` or `Bgra8Unorm` (linear), so wgpu does NOT auto-encode on write.

Under the previous sRGB-surface convention, every shader output was implicitly
encoded by the GPU on swapchain write, AND most shaders were manually calling
`S_LinearToSRGB` — a double-encode that produced washed-out or over-saturated
colors.

## Texture Sampling

| Texture tag           | wgpu format            | Sampler behavior                |
|-----------------------|------------------------|---------------------------------|
| `Rgba8UnormSrgb`      | `Rgba8UnormSrgb`      | Samples are auto-decoded to linear |
| `Rgba8Unorm` (linear) | `Rgba8Unorm`          | Samples return raw linear values |

The shader does not (and must not) call `S_SRGBToLinear` on any
`textureSample(...)` result. The helper does not exist in
`fyrox-graphics-wgpu/src/shaders/shared.wgsl`.

## Shader Output

All fragment shaders write linear values to the swapchain. They do not
(and must not) call `S_LinearToSRGB`. The helper does not exist in
`fyrox-graphics-wgpu/src/shaders/shared.wgsl`.

## CPU-Side Authored sRGB Values

UI brush colors, vertex colors, decal colors are authored in sRGB (0..255
bytes). They are converted to linear on the CPU before being uploaded to a
uniform buffer. The shader receives linear.

For widget shaders, the vertex shader includes an inline `srgb_to_linear`
conversion for per-vertex colors.

## Enforcement

The `shader_lint` module (`crate::shader_lint`) runs on every compiled shader
and refuses to compile any WGSL containing `S_SRGBToLinear` or
`S_LinearToSRGB` function calls. The lint is integrated into the shader
compiler pipeline in `WgpuShader::compile_wgsl`.

## GL Backend Status

`fyrox-graphics-gl/` is intentionally NOT included in the workspace
`[members]` list. It exists in the repository as historical reference but is
not built, tested, or maintained under the wgpu refactor.

If someone wants to revive the GL backend, they should:

1. Re-add it to `[workspace] members`
2. Port the WGSL shaders to GLSL (no parity scaffolding is preserved)
3. Audit the color-space convention against this document
4. Ensure `GL_FRAMEBUFFER_SRGB` is never enabled to match this convention

## Migration Guide for Custom Shaders

Third-party `.shader` files that hard-code `S_SRGBToLinear(textureSample(...))`
or `S_LinearToSRGB(...)` will fail to compile under this convention. To
migrate:

- Remove `S_SRGBToLinear` from around `textureSample` calls on color/albedo
  textures — the sampler auto-decodes sRGB-tagged textures.
- Remove `S_LinearToSRGB` from final fragment output — the swapchain does
  not encode.
- For UI/vertex colors authored in sRGB: convert on the CPU at upload time
  (in Rust) rather than in the shader.
