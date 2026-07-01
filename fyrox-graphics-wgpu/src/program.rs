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
/// 1. `texture(name, uv)` → `texture(samplerXxx(name_tex, name_samp), uv)` for all texture functions
/// 2. `textureSize(name, ...)` → `textureSize(name_tex, ...)`
/// 3. `gl_InstanceID` → `gl_InstanceIndex`, `gl_VertexID` → `gl_VertexIndex`
/// 4. `properties.boolField` → `bool(properties.boolField)` (uint→bool cast)
/// 5. Shared function calls with texture args: `S_PointShadow(..., name)` → `S_PointShadow(..., name_tex, name_samp)`
fn preprocess_shader(source: &mut String, resources: &[ShaderResourceDefinition]) {
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
        let constructor = sampler_constructor_name(*kind);
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

        // Replace shared function calls that pass texture resource names as arguments.
        // These functions (in shared.glsl) take separate texture2D + sampler parameters.
        // The call site passes the combined name; we split it into _tex and _samp.
        for func_name in &[
            "S_PointShadow", "S_SpotShadowFactor",
            "S_ComputeParallaxTextureCoordinates", "S_FetchMatrix", "S_FetchBlendShapeOffsets",
            "Internal_FetchHeight", "CsmGetShadow",
        ] {
            replace_texture_arg_in_calls(source, func_name, name);
        }
    }

    // Replace OpenGL built-in names with Vulkan equivalents
    *source = source.replace("gl_InstanceID", "gl_InstanceIndex");
    *source = source.replace("gl_VertexID", "gl_VertexIndex");

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
