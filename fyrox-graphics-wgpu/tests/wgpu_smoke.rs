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

//! GPU smoke tests for fyrox-graphics-wgpu.
//!
//! Run with: cargo test -p fyrox-graphics-wgpu --lib -- --ignored
//!
//! These tests create real wgpu devices and perform GPU operations.
//! They are gated behind #[ignore] since they require a physical GPU
//! and are too slow for normal CI. Enable them in CI via the
//! wgpu-validation workflow which sets WGPU_VALIDATION=1.

/// Smoke test: creates a wgpu device, builds a pipeline, and renders a triangle.
/// Fails if the device cannot be created or the pipeline fails to compile.
#[test]
#[cfg_attr(not(feature = "gpu_test"), ignore)]
fn test_wgpu_device_and_basic_pipeline() {
    // This test requires a display server and GPU, so it only runs when
    // explicitly invoked with --ignored flag on a machine with a GPU.
    // The actual implementation would create a wgpu device, compile shaders,
    // and draw — but that requires a winit Window and event loop.
    //
    // For now, this test is a placeholder that documents the expected behavior.
    // A full implementation would:
    // 1. Use raw-window-handle to create a surface from a headless buffer
    // 2. Request a wgpu Adapter with the appropriate backend
    // 3. Create a Device and Queue
    // 4. Compile a trivial vertex+fragment shader (WGLS inline)
    // 5. Create a render pipeline and submit a draw call
    // 6. Map the result buffer and verify no validation errors
    //
    // Until a proper headless GPU test harness is implemented (future work),
    // the primary GPU validation is done via WGPU_VALIDATION=1 in the editor.
    panic!("headless GPU smoke test not yet implemented — run with WGPU_VALIDATION=1 in editor");
}
