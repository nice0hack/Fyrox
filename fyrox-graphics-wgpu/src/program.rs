// Copyright (c) 2019-present Dmitry Stepanov and Fyrox Engine contributors.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

use crate::server::WgpuGraphicsServer;
use fyrox_graphics::{
    core::log::{Log, MessageKind},
    error::FrameworkError,
    gpu_program::{GpuProgramTrait, GpuShaderTrait, SamplerKind, ShaderKind, ShaderPropertyKind, ShaderResourceDefinition, ShaderResourceKind},
};
use std::rc::Weak;

fn count_lines(src: &str) -> isize { src.bytes().filter(|b| *b == b'\n').count() as isize }

/// Returns the Vulkan GLSL 450 separate texture type name.
fn vulkan_texture_type(kind: SamplerKind) -> &'static str {
    match kind {
        SamplerKind::Sampler1D => "texture1D",
        SamplerKind::Sampler2D => "texture2D",
        SamplerKind::Sampler3D => "texture3D",
        SamplerKind::SamplerCube => "textureCube",
        SamplerKind::USampler1D => "utexture1D",
        SamplerKind::USampler2D => "utexture2D",
        SamplerKind::USampler3D => "utexture3D",
        SamplerKind::USamplerCube => "utextureCube",
    }
}

/// Returns the combined sampler constructor name (e.g. "sampler2D", "samplerCube").
fn sampler_constructor_name(kind: SamplerKind) -> &'static str {
    match kind {
        SamplerKind::Sampler1D => "sampler1D",
        SamplerKind::Sampler2D => "sampler2D",
        SamplerKind::Sampler3D => "sampler3D",
        SamplerKind::SamplerCube => "samplerCube",
        SamplerKind::USampler1D => "usampler1D",
        SamplerKind::USampler2D => "usampler2D",
        SamplerKind::USampler3D => "usampler3D",
        SamplerKind::USamplerCube => "usamplerCube",
    }
}

/// Maps a combined sampler type name to SamplerKind.
fn sampler_kind_from_type_name(type_name: &str) -> Option<SamplerKind> {
    match type_name {
        "sampler1D" => Some(SamplerKind::Sampler1D),
        "sampler2D" => Some(SamplerKind::Sampler2D),
        "sampler3D" => Some(SamplerKind::Sampler3D),
        "samplerCube" => Some(SamplerKind::SamplerCube),
        "usampler1D" => Some(SamplerKind::USampler1D),
        "usampler2D" => Some(SamplerKind::USampler2D),
        "usampler3D" => Some(SamplerKind::USampler3D),
        "usamplerCube" => Some(SamplerKind::USamplerCube),
        _ => None,
    }
}

/// Rewrites texture function calls, textureSize calls, and shared function argument
/// splitting for a given sampler name. Used for both global texture resources and
/// local function parameters.
fn rewrite_sampler_usages(source: &mut String, name: &str, kind: SamplerKind) {
    let constructor = sampler_constructor_name(kind);
    let sampler_expr = format!("{constructor}({name}_tex, {name}_samp)");

    // Transform texture functions: texture(name, ...) → texture(samplerXxx(name_tex, name_samp), ...)
    for func_name in &["texture", "textureLod", "textureGrad", "textureOffset", "textureLodOffset",
                       "texelFetch", "texelFetchOffset", "textureProj", "textureProjLod"] {
        for pattern in &[
            format!("{func_name}({name},"),
            format!("{func_name}({name} ,"),
            format!("{func_name}( {name},"),
        ] {
            *source = source.replace(pattern, &format!("{func_name}({sampler_expr},"));
        }
    }

    // textureSize takes the raw texture (not sampler)
    for pattern in &[
        format!("textureSize({name},"),
        format!("textureSize({name} ,"),
    ] {
        *source = source.replace(pattern, &format!("textureSize({name}_tex,"));
    }

    // Replace shared function calls that pass texture names as arguments.
    // These functions (in shared.glsl) take separate texture2D + sampler parameters.
    for func_name in &[
        "S_PointShadow", "S_SpotShadowFactor",
        "S_ComputeParallaxTextureCoordinates", "S_FetchMatrix", "S_FetchBlendShapeOffsets",
        "Internal_FetchHeight", "CsmGetShadow",
    ] {
        replace_texture_arg_in_calls(source, func_name, name);
    }
}

/// Rewrites local function definitions that have combined sampler parameters
/// (e.g. `in sampler2D name`) into Vulkan-compatible signatures with separate
/// `texture2D` + `sampler` parameters. Also rewrites the function body to use
/// the new parameter names.
fn rewrite_sampler_function_definitions(source: &mut String) {
    let sampler_type_keywords = [
        "sampler1D", "sampler2D", "sampler3D", "samplerCube",
        "usampler1D", "usampler2D", "usampler3D", "usamplerCube",
    ];

    // Phase 1: Find all function definitions with sampler params.
    // Store (paren_pos, body_close_brace_pos, vec<(param_name, SamplerKind)>).
    let mut regions: Vec<(usize, usize, Vec<(String, SamplerKind)>)> = Vec::new();

    let bytes = source.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'(' {
            // Check there's an identifier immediately before '(' (function name).
            let paren_pos = i;
            let mut j = paren_pos;
            while j > 0 && (bytes[j - 1].is_ascii_alphanumeric() || bytes[j - 1] == b'_') {
                j -= 1;
            }
            if j == paren_pos {
                // No identifier before '('
                i += 1;
                continue;
            }

            // Extract parameter list.
            let Some((args_str, after_paren)) = extract_call_args(source, paren_pos) else {
                i += 1;
                continue;
            };

            // Check if this looks like a function definition (followed by '{', possibly with whitespace).
            let rest = source[after_paren..].trim_start();
            if !rest.starts_with('{') {
                i += 1;
                continue;
            }

            // Parse parameters and find sampler types.
            let mut sampler_params: Vec<(String, SamplerKind)> = Vec::new();
            for param in args_str.split(',') {
                let param = param.trim();
                let tokens: Vec<&str> = param.split_whitespace().collect();
                if tokens.len() < 2 {
                    continue;
                }
                let mut type_idx = None;
                for (idx, tok) in tokens.iter().enumerate() {
                    if sampler_type_keywords.contains(tok) {
                        type_idx = Some(idx);
                        break;
                    }
                }
                let Some(ti) = type_idx else { continue };
                let kind = sampler_kind_from_type_name(tokens[ti]).unwrap();
                let param_name = tokens.last().unwrap().to_string();
                sampler_params.push((param_name, kind));
            }

            if !sampler_params.is_empty() {
                // Find the function body via brace matching.
                let brace_start = source[after_paren..].find('{').unwrap() + after_paren;
                let mut depth = 0i32;
                let mut k = brace_start;
                while k < bytes.len() {
                    match bytes[k] {
                        b'{' => depth += 1,
                        b'}' => {
                            depth -= 1;
                            if depth == 0 { break; }
                        }
                        _ => {}
                    }
                    k += 1;
                }
                if depth == 0 {
                    regions.push((paren_pos, k + 1, sampler_params));
                }
            }
        }
        i += 1;
    }

    // Phase 2: Apply rewrites from end to start to avoid position shifts.
    for (paren_pos, body_end, sampler_params) in regions.iter().rev() {
        // Extract parameter list text.
        let Some((args_str, after_paren)) = extract_call_args(source, *paren_pos) else { continue };

        // Rewrite parameter list: split each sampler param into _tex + _samp.
        let mut new_params: Vec<String> = Vec::new();
        for param in args_str.split(',') {
            let trimmed = param.trim();
            let tokens: Vec<&str> = trimmed.split_whitespace().collect();
            if tokens.len() < 2 {
                new_params.push(param.to_string());
                continue;
            }
            let mut type_idx = None;
            for (idx, tok) in tokens.iter().enumerate() {
                if sampler_type_keywords.contains(tok) {
                    type_idx = Some(idx);
                    break;
                }
            }
            if let Some(ti) = type_idx {
                let kind = sampler_kind_from_type_name(tokens[ti]).unwrap();
                let param_name = tokens.last().unwrap();
                // Collect qualifiers (everything before the type keyword).
                let qualifiers: Vec<&str> = tokens[..ti].to_vec();
                let qual_prefix = if qualifiers.is_empty() { String::new() } else { qualifiers.join(" ") + " " };
                let tex_type = vulkan_texture_type(kind);
                new_params.push(format!("{qual_prefix}{tex_type} {param_name}_tex"));
                new_params.push(format!("{qual_prefix}sampler {param_name}_samp"));
            } else {
                new_params.push(param.to_string());
            }
        }
        let new_args = new_params.join(", ");

        // Find the body text (from '{' to '}').
        let brace_start = source[after_paren..].find('{').unwrap() + after_paren;
        let body_text = &source[brace_start..*body_end].to_string();

        // Rewrite body for each sampler param.
        let mut new_body = body_text.clone();
        for (param_name, kind) in sampler_params {
            rewrite_sampler_usages(&mut new_body, param_name, *kind);
        }

        // Replace: from paren_pos to body_end.
        let before = &source[..*paren_pos].to_string();
        let after = &source[*body_end..].to_string();
        *source = format!("{before}({new_args}){new_body}{after}");
    }
}

/// Generates separate texture + sampler uniform declarations for wgpu bindings,
/// plus uniform block declarations with `layout(std140, binding=...)`.
fn generate_resource_declarations(resources: &[ShaderResourceDefinition], source: &mut String, line_offset: &mut isize) {
    let mut tex_decls = String::new();
    for res in resources {
        match res.kind {
            ShaderResourceKind::Texture { kind, .. } => {
                let tex_type = vulkan_texture_type(kind);
                tex_decls += &format!("layout(binding={}) uniform {} {}_tex;\n", res.binding, tex_type, res.name);
                tex_decls += &format!("layout(binding={}) uniform sampler {}_samp;\n", res.binding + 100, res.name);
            }
            ShaderResourceKind::PropertyGroup(ref fields) => {
                if fields.is_empty() { continue; }
                let mut block = format!("struct T{}{{\n", res.name);
                for f in fields {
                    let n = &f.name;
                    match f.kind {
                        ShaderPropertyKind::Float { .. } => block += &format!("\tfloat {n};\n"),
                        ShaderPropertyKind::FloatArray { max_len, .. } => block += &format!("\tfloat {n}[{max_len}];\n"),
                        ShaderPropertyKind::Int { .. } => block += &format!("\tint {n};\n"),
                        ShaderPropertyKind::IntArray { max_len, .. } => block += &format!("\tint {n}[{max_len}];\n"),
                        ShaderPropertyKind::UInt { .. } => block += &format!("\tuint {n};\n"),
                        ShaderPropertyKind::UIntArray { max_len, .. } => block += &format!("\tuint {n}[{max_len}];\n"),
                        ShaderPropertyKind::Bool { .. } => block += &format!("\tuint {n};\n"),
                        ShaderPropertyKind::Vector2 { .. } => block += &format!("\tvec2 {n};\n"),
                        ShaderPropertyKind::Vector2Array { max_len, .. } => block += &format!("\tvec2 {n}[{max_len}];\n"),
                        ShaderPropertyKind::Vector3 { .. } => block += &format!("\tvec3 {n};\n"),
                        ShaderPropertyKind::Vector3Array { max_len, .. } => block += &format!("\tvec3 {n}[{max_len}];\n"),
                        ShaderPropertyKind::Vector4 { .. } => block += &format!("\tvec4 {n};\n"),
                        ShaderPropertyKind::Vector4Array { max_len, .. } => block += &format!("\tvec4 {n}[{max_len}];\n"),
                        ShaderPropertyKind::Matrix2 { .. } => block += &format!("\tmat2 {n};\n"),
                        ShaderPropertyKind::Matrix2Array { max_len, .. } => block += &format!("\tmat2 {n}[{max_len}];\n"),
                        ShaderPropertyKind::Matrix3 { .. } => block += &format!("\tmat3 {n};\n"),
                        ShaderPropertyKind::Matrix3Array { max_len, .. } => block += &format!("\tmat3 {n}[{max_len}];\n"),
                        ShaderPropertyKind::Matrix4 { .. } => block += &format!("\tmat4 {n};\n"),
                        ShaderPropertyKind::Matrix4Array { max_len, .. } => block += &format!("\tmat4 {n}[{max_len}];\n"),
                        ShaderPropertyKind::Color { .. } => block += &format!("\tvec4 {n};\n"),
                    }
                }
                block += "};\n";
                block += &format!("layout(std140, binding={}) uniform U{} {{ T{} {}; }};\n", res.binding + 200, res.name, res.name, res.name);
                source.insert_str(0, &block);
                *line_offset -= count_lines(&block);
            }
        }
    }
    source.insert_str(0, &tex_decls);
    *line_offset -= count_lines(&tex_decls);
}

/// Preprocesses shader source for naga/wgpu compatibility.
///
/// Transformations:
/// 1. Local function definitions with sampler params → separate texture + sampler params
/// 2. `texture(name, uv)` → `texture(samplerXxx(name_tex, name_samp), uv)` for all texture functions
/// 3. `textureSize(name, ...)` → `textureSize(name_tex, ...)`
/// 4. Shared function calls with texture args: `S_PointShadow(..., name)` → `S_PointShadow(..., name_tex, name_samp)`
/// 5. `gl_InstanceID` → `gl_InstanceIndex`, `gl_VertexID` → `gl_VertexIndex`
/// 6. `properties.boolField` → `bool(properties.boolField)` (uint→bool cast)
fn preprocess_shader(source: &mut String, resources: &[ShaderResourceDefinition]) {
    // Rewrite local function definitions with sampler params first.
    rewrite_sampler_function_definitions(source);

    // Collect texture resources (longest first to avoid partial matches)
    let mut tex_resources: Vec<(&str, SamplerKind)> = resources
        .iter()
        .filter_map(|r| match r.kind {
            ShaderResourceKind::Texture { kind, .. } => Some((r.name.as_str(), kind)),
            _ => None,
        })
        .collect();
    tex_resources.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

    for (name, kind) in &tex_resources {
        rewrite_sampler_usages(source, name, *kind);
    }

    // Replace OpenGL built-in names with Vulkan equivalents.
    // Use `int(builtin + 0)` wrapper to work around a Naga bug where passing
    // gl_InstanceIndex/gl_VertexIndex directly as function arguments to functions
    // with texture2D/sampler parameters causes "Unknown function" errors.
    *source = source.replace("gl_InstanceID", "int(gl_InstanceIndex + 0)");
    *source = source.replace("gl_VertexID", "int(gl_VertexIndex + 0)");

    // Wrap bool uniform property accesses with bool() cast (uint→bool)
    for res in resources {
        let ShaderResourceKind::PropertyGroup(ref fields) = res.kind else { continue };
        let block_name = &*res.name;
        for f in fields {
            if !matches!(f.kind, ShaderPropertyKind::Bool { .. }) { continue; }
            let field_name = &*f.name;
            let search = format!("{block_name}.{field_name}");
            let replacement = format!("bool({block_name}.{field_name})");
            *source = source.replace(&search, &replacement);
        }
    }
}

/// Finds calls to `funcName(...)` in the source where `texture_name` appears as an argument,
/// and replaces it with `texture_name_tex, texture_name_samp`.
/// Handles multiple cases:
/// 1. Last argument: `funcName(..., name)` → `funcName(..., name_tex, name_samp)`
/// 2. Two consecutive args: `funcName(..., name, name, ...)` → `funcName(..., name_tex, name_samp, ...)`
/// 3. First/middle argument: `funcName(name, ...)` → `funcName(name_tex, name_samp, ...)`
fn replace_texture_arg_in_calls(source: &mut String, func_name: &str, texture_name: &str) {
    let search = format!("{func_name}(");
    // Process from end to start to avoid position shifts
    let mut positions: Vec<usize> = Vec::new();
    let mut pos = 0;
    while let Some(start) = source[pos..].find(&search) {
        positions.push(pos + start);
        pos = pos + start + search.len();
    }

    for &call_start in positions.iter().rev() {
        let paren_start = call_start + search.len() - 1;
        let Some((args_str, call_end)) = extract_call_args(source, paren_start) else { continue };

        // Case 1: last argument is the texture name
        let trimmed = args_str.trim_end();
        if trimmed.ends_with(texture_name) {
            let name_start = trimmed.len() - texture_name.len();
            let preceded_ok = name_start == 0 || {
                let prev = trimmed.as_bytes()[name_start - 1];
                matches!(prev, b',' | b' ' | b'\t' | b'\n')
            };
            if preceded_ok {
                let before = &source[..paren_start + 1 + name_start];
                let after = &source[call_end - 1..];
                *source = format!("{before}{texture_name}_tex, {texture_name}_samp{after}");
                continue;
            }
        }

        // Case 2: two consecutive args are the texture name (e.g., CsmGetShadow(name, name, ...))
        for sep in &[",", ", ", ",\n", ",\t"] {
            let double_pattern = format!("{texture_name}{sep}{texture_name}");
            if let Some(found) = args_str.find(&double_pattern) {
                let abs_in_args = found;
                let preceded_ok = abs_in_args == 0 || {
                    let prev = args_str.as_bytes()[abs_in_args - 1];
                    matches!(prev, b',' | b' ' | b'\t' | b'\n')
                };
                let after_end = abs_in_args + double_pattern.len();
                let followed_ok = after_end >= args_str.len() || {
                    let next = args_str.as_bytes()[after_end];
                    matches!(next, b',' | b' ' | b'\t' | b'\n')
                };
                if preceded_ok && followed_ok {
                    let args_start = paren_start + 1;
                    if let Some(found_in_source) = source[args_start..call_end - 1].find(&double_pattern) {
                        let abs_pos = args_start + found_in_source;
                        let before = &source[..abs_pos];
                        let after = &source[abs_pos + double_pattern.len()..];
                        *source = format!("{before}{texture_name}_tex,{texture_name}_samp{after}");
                    }
                    break;
                }
            }
        }

        // Case 3: texture name is the first/middle argument (not last, not doubled)
        // Find the texture name as a standalone identifier followed by a comma
        for sep in &[",", ", "] {
            let pattern = format!("{texture_name}{sep}");
            if let Some(found) = args_str.find(&pattern) {
                let abs_in_args = found;
                let preceded_ok = abs_in_args == 0 || {
                    let prev = args_str.as_bytes()[abs_in_args - 1];
                    matches!(prev, b',' | b' ' | b'\t' | b'\n')
                };
                // Make sure this isn't already handled by case 2 (double pattern)
                let after_sep = abs_in_args + pattern.len();
                let is_double = after_sep + texture_name.len() <= args_str.len()
                    && args_str[after_sep..].starts_with(texture_name);
                if preceded_ok && !is_double {
                    let args_start = paren_start + 1;
                    if let Some(found_in_source) = source[args_start..call_end - 1].find(&pattern) {
                        let abs_pos = args_start + found_in_source;
                        let before = &source[..abs_pos];
                        let after = &source[abs_pos + pattern.len()..];
                        *source = format!("{before}{texture_name}_tex,{texture_name}_samp{sep}{after}");
                    }
                    break;
                }
            }
        }
    }
}

/// Finds the arguments of a function call starting at the opening paren.
/// Returns (args_string, end_index) where end_index is the index AFTER the closing paren.
fn extract_call_args(source: &str, paren_start: usize) -> Option<(String, usize)> {
    let bytes = source.as_bytes();
    if bytes.get(paren_start) != Some(&b'(') { return None; }
    let mut depth = 1;
    let mut i = paren_start + 1;
    while i < bytes.len() && depth > 0 {
        match bytes[i] {
            b'(' => depth += 1,
            b')' => depth -= 1,
            _ => {}
        }
        i += 1;
    }
    if depth == 0 {
        Some((source[paren_start + 1..i - 1].to_string(), i))
    } else {
        None
    }
}

fn naga_stage(kind: &ShaderKind) -> naga::ShaderStage {
    match kind { ShaderKind::Vertex => naga::ShaderStage::Vertex, ShaderKind::Fragment => naga::ShaderStage::Fragment }
}

fn prepare_source(code: &str) -> String {
    let mut src = String::from("#version 450\n// include 'shared.glsl'\n");
    src += include_str!("shaders/shared.glsl");
    src += "\n// end of include\n";
    src += code;
    src
}

fn compile_glsl(device: &wgpu::Device, name: &str, kind: &ShaderKind, glsl: &str) -> Result<wgpu::ShaderModule, FrameworkError> {
    let mut frontend = naga::front::glsl::Frontend::default();
    let module = frontend.parse(&naga::front::glsl::Options { stage: naga_stage(kind), defines: Default::default() }, glsl).map_err(|errors| {
        let msg = format!("Failed to parse GLSL for {name}:\n{errors}");
        Log::writeln(MessageKind::Error, msg.clone());
        FrameworkError::ShaderCompilationFailed { shader_name: name.to_owned(), error_message: msg }
    })?;

    // Use relaxed validation — naga's GLSL frontend may produce types that don't
    // pass strict WGSL-oriented validation (e.g., bool in uniform blocks).
    let info = naga::valid::Validator::new(naga::valid::ValidationFlags::empty(), naga::valid::Capabilities::all())
        .validate(&module)
        .map_err(|e| {
            let msg = format!("Naga validation failed for {name}: {e}");
            Log::writeln(MessageKind::Error, msg.clone());
            FrameworkError::ShaderCompilationFailed { shader_name: name.to_owned(), error_message: msg }
        })?;

    // Convert naga module to SPIR-V for wgpu
    let spv = naga::back::spv::write_vec(&module, &info, &naga::back::spv::Options {
        flags: naga::back::spv::WriterFlags::empty(),
        ..Default::default()
    }, None).map_err(|e| {
        let msg = format!("SPIR-V generation failed for {name}: {e}");
        Log::writeln(MessageKind::Error, msg.clone());
        FrameworkError::ShaderCompilationFailed { shader_name: name.to_owned(), error_message: msg }
    })?;

    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(name),
        source: wgpu::ShaderSource::SpirV(std::borrow::Cow::Owned(spv)),
    });

    Log::writeln(MessageKind::Information, format!("Shader {name} compiled successfully!"));
    Ok(shader_module)
}

fn create_bind_group_layout(device: &wgpu::Device, resources: &[ShaderResourceDefinition]) -> wgpu::BindGroupLayout {
    let mut entries = Vec::new();
    for res in resources {
        match res.kind {
            ShaderResourceKind::Texture { kind, .. } => {
                let vd = match kind {
                    SamplerKind::Sampler1D | SamplerKind::USampler1D => wgpu::TextureViewDimension::D1,
                    SamplerKind::Sampler2D | SamplerKind::USampler2D => wgpu::TextureViewDimension::D2,
                    SamplerKind::Sampler3D | SamplerKind::USampler3D => wgpu::TextureViewDimension::D3,
                    SamplerKind::SamplerCube | SamplerKind::USamplerCube => wgpu::TextureViewDimension::Cube,
                };
                let st = match kind {
                    SamplerKind::USampler1D | SamplerKind::USampler2D | SamplerKind::USampler3D | SamplerKind::USamplerCube => wgpu::TextureSampleType::Uint,
                    _ => wgpu::TextureSampleType::Float { filterable: true },
                };
                entries.push(wgpu::BindGroupLayoutEntry { binding: res.binding as u32, visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { sample_type: st, view_dimension: vd, multisampled: false }, count: None });
                entries.push(wgpu::BindGroupLayoutEntry { binding: (res.binding + 100) as u32, visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None });
            }
            ShaderResourceKind::PropertyGroup { .. } => {
                entries.push(wgpu::BindGroupLayoutEntry { binding: (res.binding + 200) as u32, visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None });
            }
        }
    }
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("ShaderBindGroupLayout"), entries: &entries })
}

pub struct WgpuShader {
    _server: Weak<WgpuGraphicsServer>,
    module: wgpu::ShaderModule,
    _kind: ShaderKind,
}

impl GpuShaderTrait for WgpuShader {}

impl WgpuShader {
    pub fn new(server: &WgpuGraphicsServer, name: String, kind: ShaderKind, mut source: String, resources: &[ShaderResourceDefinition], mut line_offset: isize) -> Result<Self, FrameworkError> {
        for r in resources {
            for o in resources {
                if std::ptr::eq(r, o) { continue; }
                if std::mem::discriminant(&r.kind) == std::mem::discriminant(&o.kind) {
                    if r.binding == o.binding { return Err(FrameworkError::Custom(format!("Resource {} and {} same binding {}", r.name, o.name, r.binding))); }
                    if r.name == o.name { return Err(FrameworkError::Custom(format!("Duplicate resource name {}", r.name))); }
                }
            }
        }
        generate_resource_declarations(resources, &mut source, &mut line_offset);
        preprocess_shader(&mut source, resources);
        let full = prepare_source(&source);
        let module = compile_glsl(&server.state.device, &name, &kind, &full)?;
        Ok(Self { _server: server.weak_ref(), module, _kind: kind })
    }

    pub fn wgpu_module(&self) -> &wgpu::ShaderModule { &self.module }
}

pub struct WgpuProgram {
    _server: Weak<WgpuGraphicsServer>,
    name: String,
    vertex_module: wgpu::ShaderModule,
    fragment_module: wgpu::ShaderModule,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    resources: Vec<ShaderResourceDefinition>,
}

impl GpuProgramTrait for WgpuProgram {}

impl WgpuProgram {
    pub fn from_source(server: &WgpuGraphicsServer, name: &str, vs: String, vs_off: isize, fs: String, fs_off: isize, resources: &[ShaderResourceDefinition]) -> Result<Self, FrameworkError> {
        let vert = WgpuShader::new(server, format!("{name}_VS"), ShaderKind::Vertex, vs, resources, vs_off)?;
        let frag = WgpuShader::new(server, format!("{name}_FS"), ShaderKind::Fragment, fs, resources, fs_off)?;
        Self::from_modules(server, name, &vert, &frag, resources)
    }

    pub fn from_shaders(server: &WgpuGraphicsServer, name: &str, vs: &fyrox_graphics::gpu_program::GpuShader, fs: &fyrox_graphics::gpu_program::GpuShader, resources: &[ShaderResourceDefinition]) -> Result<Self, FrameworkError> {
        let vert = vs.as_any().downcast_ref::<WgpuShader>().ok_or_else(|| FrameworkError::Custom("Expected WgpuShader".into()))?;
        let frag = fs.as_any().downcast_ref::<WgpuShader>().ok_or_else(|| FrameworkError::Custom("Expected WgpuShader".into()))?;
        Self::from_modules(server, name, vert, frag, resources)
    }

    fn from_modules(server: &WgpuGraphicsServer, name: &str, vert: &WgpuShader, frag: &WgpuShader, resources: &[ShaderResourceDefinition]) -> Result<Self, FrameworkError> {
        let bgl = create_bind_group_layout(&server.state.device, resources);
        let pl = server.state.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{name}_PL")),
            bind_group_layouts: &[Some(&bgl)],
            ..Default::default()
        });
        Ok(Self {
            _server: server.weak_ref(), name: name.to_owned(),
            vertex_module: vert.module.clone(), fragment_module: frag.module.clone(),
            bind_group_layout: bgl, pipeline_layout: pl, resources: resources.to_vec(),
        })
    }

    pub fn vertex_module(&self) -> &wgpu::ShaderModule { &self.vertex_module }
    pub fn fragment_module(&self) -> &wgpu::ShaderModule { &self.fragment_module }
    pub fn pipeline_layout(&self) -> &wgpu::PipelineLayout { &self.pipeline_layout }
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout { &self.bind_group_layout }
    pub fn resources(&self) -> &[ShaderResourceDefinition] { &self.resources }
    pub fn name(&self) -> &str { &self.name }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rewrite_sampler_function_definitions() {
        let mut source = r#"float CsmGetShadow(in sampler2D shadowSampler, in vec3 fragmentPosition, in mat4 lightViewProjMatrix)
{
    float invSize = 1.0 / float(textureSize(shadowSampler, 0).x);
    return S_SpotShadowFactor(properties.shadowsEnabled, properties.softShadows,
        properties.shadowBias, fragmentPosition, lightViewProjMatrix, invSize, shadowSampler);
}"#.to_string();

        rewrite_sampler_function_definitions(&mut source);

        // Parameter list should be split
        assert!(source.contains("in texture2D shadowSampler_tex, in sampler shadowSampler_samp"));
        // textureSize should use _tex
        assert!(source.contains("textureSize(shadowSampler_tex, 0)"));
        // S_SpotShadowFactor call should have split args
        assert!(source.contains("shadowSampler_tex, shadowSampler_samp"));
        // Original combined sampler type should be gone
        assert!(!source.contains("sampler2D shadowSampler"));
    }

    #[test]
    fn test_rewrite_sampler_function_definitions_no_sampler() {
        let mut source = r#"float plainFunc(float a, float b) {
    return a + b;
}"#.to_string();

        let original = source.clone();
        rewrite_sampler_function_definitions(&mut source);
        assert_eq!(source, original);
    }

    #[test]
    fn test_rewrite_sampler_usages_global_resource() {
        let mut source = r#"vec3 c = texture(materialTexture, texCoord).rgb;
float d = textureSize(materialTexture, 0).x;"#.to_string();

        rewrite_sampler_usages(&mut source, "materialTexture", SamplerKind::Sampler2D);

        assert!(source.contains("texture(sampler2D(materialTexture_tex, materialTexture_samp),"));
        assert!(source.contains("textureSize(materialTexture_tex,"));
    }

    #[test]
    fn test_naga_shared_glsl_std140_with_gl_instance_index() {
        // Verify that the gl_InstanceIndex workaround (int(gl_InstanceIndex + 0))
        // works around the Naga bug where gl_InstanceIndex as a function argument
        // to functions with texture2D/sampler params causes "Unknown function" errors.
        let shared = include_str!("shaders/shared.glsl");
        let glsl = format!(r#"#version 450
{shared}

// end of include
layout(binding=0) uniform texture2D matrices_tex;
layout(binding=100) uniform sampler matrices_samp;

struct Tproperties{{
    mat4 viewProjection;
    int tileSize;
    float frameBufferHeight;
}};
layout(std140, binding=200) uniform Uproperties {{ Tproperties properties; }};

void main()
{{
    gl_Position = S_FetchMatrix(matrices_tex, matrices_samp, int(gl_InstanceIndex + 0)) * vec4(1.0);
}}
"#);

        let mut frontend = naga::front::glsl::Frontend::default();
        let result = frontend.parse(&naga::front::glsl::Options {
            stage: naga::ShaderStage::Vertex,
            defines: Default::default(),
        }, &glsl);

        match result {
            Ok(module) => {
                let info = naga::valid::Validator::new(
                    naga::valid::ValidationFlags::empty(),
                    naga::valid::Capabilities::all(),
                ).validate(&module);
                assert!(info.is_ok(), "Naga validation failed: {:?}", info.err());
            }
            Err(errors) => {
                panic!("Naga GLSL parse failed:\n{errors}");
            }
        }
    }
}
