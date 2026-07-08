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
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

//! Shader source linting utilities.
//!
//! Enforces the linear working-space convention: sRGB encoding/decoding must NOT
//! appear in shader source — sRGB-tagged textures auto-decode on sample via wgpu,
//! and the swapchain is linear (OS compositor handles final encode).

use fyrox_graphics::core::log::{Log, MessageKind};

/// Checks `wgsl_source` for banned sRGB conversion helpers and logs warnings.
///
/// Banned patterns: `S_SRGBToLinear` and `S_LinearToSRGB`.
/// These helpers must not be used in any shader — the linear working-space convention
/// handles sRGB at the texture-format level (Rgba8UnormSrgb) and at the OS
/// compositor level respectively.
pub fn check_srgb_conversions(wgsl_source: &str, shader_name: &str) {
    let banned = ["S_SRGBToLinear", "S_LinearToSRGB"];

    for pattern in banned {
        if wgsl_source.contains(pattern) {
            Log::writeln(
                MessageKind::Warning,
                format!(
                    "shader_lint: shader `{}` contains banned sRGB helper `{}` — \
                    sRGB conversions must not appear in shader source. \
                    Use sRGB-tagged texture formats for auto-decode, or CPU-side \
                    conversion before upload.",
                    shader_name, pattern
                ),
            );
        }
    }
}
