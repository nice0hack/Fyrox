# Fyrox wgpu Backend — Cross-Platform Status

## Supported Platforms

The wgpu backend is validated on the following platforms:

| Platform | Backend | Status |
|----------|---------|--------|
| Linux (Vulkan) | Vulkan | Primary development platform |
| Windows (DX12/Vulkan) | DX12 / Vulkan | Validated via CI |
| macOS (Metal) | Metal | Validated via CI |

## CI Build Matrix

The `wgpu-validation.yml` workflow builds on every push to `main` and `dev` branches:

- **ubuntu-latest**: Linux + Vulkan
- **windows-latest**: Windows + DX12 (fallback: Vulkan)
- **macos-latest**: macOS + Metal

## Known Backend-Specific Quirks

### DX12 vs Vulkan
- Shader precision: DX12 and Vulkan have slightly different default precision for `sin`/`cos` on very large values. Results may differ at the 6th+ decimal place.
- Depth range: Vulkan uses NDC with depth in `[0, 1]` for the depth range, same as DX12.

### Metal
- `front_facing`: Metal's `front_facing` is defined differently from Vulkan. Fyrox shaders always use `FrontFace::Ccw` (counter-clockwise windings are front), which is consistent across all backends.
- `depth_bias`: Metal handles `depth_bias` differently; Fyrox does not use `depth_bias` by default.

### General
- `WGPU_VALIDATION=1` env var enables wgpu's validation layer, which catches invalid resource usage and misconfigured pipelines.

## Manual Validation Checklist

Before merging a wgpu-related PR, verify the editor runs on:

- [ ] Windows (DX12 or Vulkan, whichever is default on the test machine)
- [ ] macOS (Metal)
- [ ] Linux (Vulkan)

Check that:
1. A scene with PBR materials renders correctly (check for gamma artifacts)
2. The editor UI renders without black/white flash on startup
3. Shadows appear on all light types (directional, point, spot)
4. No console errors about missing adapters or failed pipeline creation
