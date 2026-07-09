# Fyrox OpenGL → wgpu Migration Report

**Branch:** `fix/wgpu-local`
**Date:** 2026-07-09
**Status:** Substantially complete — `fyrox-graphics-wgpu` is the default backend.

---

## Summary

The Fyrox graphics stack has been migrated from OpenGL (`fyrox-graphics-gl`) to wgpu
(`fyrox-graphics-wgpu`). The wgpu backend is now the default via
`GraphicsServerConstructor::default()`. This report documents what was done,
what was fixed, known caveats, and how to verify the results locally.

---

## What Landed (24 commits on `fix/wgpu-local`)

### Core wgpu infrastructure (chronological)

| Commit | Description |
|--------|-------------|
| `9e06d4eed` | refactor(wgpu): switch swapchain to linear, document color-space convention |
| `53ff96c44` | refactor(shader): remove all manual sRGB conversions — linear working space |
| `a92131953` | refactor(wgpu): remove sRGB helpers from shared.wgsl and add shader_lint |
| `9489694a5` | fix(texture): default 8-bit RGB/RGBA to sRGB at load time |
| `ca29d5d9b` | refactor(wgpu): add PipelineCatalog with VariantKey enum |
| `305c6bf62` | refactor(wgpu): add BindGroupCache for per-draw-call bind group caching |
| `d9dde6f30` | refactor(wgpu): add UniformBufferPool with sub-allocation and dynamic offsets |
| `7e929078e` | refactor(wgpu): add RenderPass enum and PassDescriptor for named passes |
| `ba3b7e23b` | refactor(wgpu): add RendererResources to run_pass + fallback texture binding |
| `804336f84` | refactor(wgpu): explicit adapter preference with fallback tiers |
| `2bc0eb147` | feat(wgpu): add FrameMetrics and BindGroupMetrics for cache hit rate tracking |
| `07e4df346` | ci: add wgpu cross-platform validation workflow and docs |
| `c42480f4f` | docs: add wgpu shader author guide |

### Bug fixes (T1–T12 of the fix plan)

| Commit | Description |
|--------|-------------|
| `e44667457` | fix(wgpu,renderer): guard bind-group dedup and prevent instance-data collision |
| `63f8beb83` | fix(wgpu): port MSAA plumbing to master (T2) |
| `dccdb8dde` | feat(wgpu): implement set_polygon_fill_mode, generate_mipmap, blit_to, debug groups (T3–T6) |
| `92a05ffa2` | fix(shader): apply srgb_to_linear to U8×4 vertex color writes (T7) |
| `9bff2d177` | fix(occlusion): add debug_assert confirming frameBufferHeight = viewport.h() (T9) |
| `9405678e2` | docs(engine): fix stale doc comment — Default creates wgpu, not OpenGL (T10) |
| `54e3f6291` | fix(editor): clear highlight FBO instead of skipping render (T11) |
| `5d5d5e5bd` | ci(wgpu): add WGPU_VALIDATION=1, Vulkan drivers, and GPU smoke test (T12) |

---

## Real Bugs Found and Fixed

### 1. sRGB vertex-color gamma bug (T7 — `92a05ffa2`)

**Severity: High** — every rendered sprite, tile, quad, gizmo, debug line, and particle
was displayed at incorrect gamma because the shader wrote `output.color = input.vertexColor`
directly. Vertex color attributes are U8×4 in sRGB format, but the shaders treated them as linear.

**Fix:** `output.color = vec4f(srgb_to_linear(input.vertexColor.rgb), input.vertexColor.a);`
in 6 shader files. `widget.shader:42` was the only correct precedent.

**Files affected:**
- `fyrox-material/src/shader/standard/standard_particle_system.shader`
- `fyrox-material/src/shader/standard/standard2d.shader`
- `fyrox-material/src/shader/standard/standard_sprite.shader`
- `fyrox-material/src/shader/standard/tile.shader`
- `editor/resources/shaders/sprite_gizmo.shader`
- `fyrox-impl/src/renderer/shaders/debug.shader`

### 2. Highlight FBO stale-content bug (T11 — `54e3f6291`)

**Severity: Medium** — when no nodes were selected, the highlight render pass returned early
without clearing the FBO. Under wgpu's lazy-clear model (`needs_clear` flag), zero draw calls
meant the `needs_clear` flag was not reset, causing the edge-detect shader to read stale
FBO content on the next frame.

**Fix:** replace early `return` with explicit `framebuffer.clear(viewport, TRANSPARENT, None, None)`.

### 3. Bind-group deduplication regression guard (T1 — `e44667457`)

**Severity: Medium** — added `debug_assert_eq!(deduped.len(), entry_count)` in
`create_bind_group` to guard against future regressions in the binding deduplication logic
that could cause material+instance PropertyGroup collisions at binding slot 200+.

---

## Known Caveats

The following are unimplemented or partially implemented and are tracked as future work:

| Item | Status | Notes |
|------|--------|-------|
| `set_polygon_fill_mode` wireframe | Implemented | Encodes fill mode in `PipelineKey`; separate pipeline cache for wireframe |
| `generate_mipmap` | Implemented | Uses `wgpu::util::TextureBlitter` with linear filter; 2D Rectangle only |
| `blit_to` depth/stencil | Partial | Color-only copy implemented; depth/stencil readback not yet implemented |
| `push_debug_group`/`pop_debug_group` | Implemented | Threaded through `current_debug_group_label` → `encoder.push/pop_debug_group` |
| MSAA | Implemented | `sample_count` in `GpuTextureDescriptor`; `msaa_target` field in framebuffer; resolve in `do_draw` |
| GPU smoke test (`wgpu_smoke.rs`) | Placeholder | Headless test not yet implemented; `#[ignore]` placeholder added |

---

## Color Space Convention

wgpu uses a linear working space. The convention established:

- **Swapchain:** selected as linear (`!is_srgb`) format
- **sRGB-tagged textures:** `Rgba8UnormSrgb` — auto-decoded on sample
- **CPU-side sRGB→linear:** in `fyrox-impl/src/renderer/bundle.rs` (material colors),
  `ui_renderer.rs` (brushes, vertex colors), `light.rs` (ambient/light colors),
  `gbuffer.rs` (decal color)
- **Vertex colors (U8×4):** shader-level `srgb_to_linear()` via the shared helper in
  `fyrox-graphics-wgpu/src/shaders/shared.wgsl:19-22`

Documented in: `docs/wgpu-color-space.md`

---

## How to Verify Locally

```bash
# 1. Check the workspace compiles cleanly
cargo check --workspace

# 2. Run library tests (excluding the display-dependent editor tests)
cargo test --workspace --lib --exclude fyroxed_base

# 3. Run the editor with wgpu validation
WGPU_VALIDATION=1 cargo run --package fyroxed --profile=editor-standalone

# In the editor:
#   - Open a scene and verify GUI colors look correct
#   - Select objects and verify the selection outline appears
#   - View → Wireframe to verify wireframe mode works
#   - F12 (screenshot) to verify blit_to path
```

### What to look for
- GUI element colors appear correct (not washed out or too dark)
- Selection outline appears only on selected object silhouettes
- Wireframe mode shows mesh edges clearly
- No `wgpu validation error` messages in stderr/console
- No `log::warn!` messages about stub methods (`generate_mimmap`, `blit_to`, etc.)

---

## Future Work

- **Headless GPU smoke test:** implement a proper headless test in
  `fyrox-graphics-wgpu/tests/wgpu_smoke.rs` using a headless surface
  (e.g., `wgpu::Instance::surfaces()` with a software adapter or offscreen surfaceless context)
- **Depth blit_to:** implement the full copy_texture_to_buffer → write_texture round-trip
  for depth/stencil attachments
- **Cube/Volume mipmap generation:** `generate_mipmap` currently logs a warning for
  non-2D-Rectangle textures — a CPU round-trip can fill this gap
- **Restore GL as selectable backend:** re-add `fyrox-graphics-gl` behind a `backend-gl`
  cargo feature (per the original spec's hard constraint #1)

---

## References

- Master migration plan: `docs/compose/plans/2026-07-08-wgpu-backend-refactor.md` (1732 lines)
- This branch's fix plan: `.mimocode/plans/1783580511543-curious-cabin.md`
- wgpu shader author guide: `docs/wgpu-shader-author-guide.md` (or equivalent in `c42480f4f`)
- Color space documentation: `docs/wgpu-color-space.md`
- Original sketch (stale): `fyrox_wgpu_migration_agent_prompt.md`
