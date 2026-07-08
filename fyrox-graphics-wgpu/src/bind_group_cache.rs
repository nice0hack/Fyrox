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

//! Per-material bind group cache.
//!
//! `create_bind_group` in framebuffer.rs is called every draw call with a set of
//! resource bindings. Re-creating a wgpu::BindGroup every draw is wasteful — the
//! same texture/sampler/buffer combination is common across frames.
//!
//! `BindGroupCache` stores `Arc<wgpu::BindGroup>` entries keyed by the resource
//! pointers in the binding list. Entries persist for the lifetime of the cache
//! (no eviction). SinceFyrox creates a finite, stable set of texture/sampler/buffer
//! objects per scene load, unbounded growth is not a practical concern.

use crate::metrics::BindGroupMetrics;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use wgpu::BindGroup;

/// A cache key entry for a single resource binding.
///
/// Uses pointer addresses of the underlying `Rc` boxes, which are stable
/// for the lifetime of the scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BindingKey {
    Texture {
        texture_ptr: usize,
        sampler_ptr: usize,
        binding: usize,
    },
    Buffer {
        buffer_ptr: usize,
        binding: usize,
        offset: u64,
        size: Option<u64>,
    },
}

impl Hash for BindingKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Self::Texture { texture_ptr, sampler_ptr, binding } => {
                0u8.hash(state);
                texture_ptr.hash(state);
                sampler_ptr.hash(state);
                binding.hash(state);
            }
            Self::Buffer { buffer_ptr, binding, offset, size } => {
                1u8.hash(state);
                buffer_ptr.hash(state);
                binding.hash(state);
                offset.hash(state);
                size.hash(state);
            }
        }
    }
}

/// A cache key for a full set of resource bindings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BindGroupCacheKey {
    bindings: Vec<BindingKey>,
}

impl BindGroupCacheKey {
    /// Creates a cache key from a slice of `ResourceBinding` references.
    ///
    /// This is called once per draw call to build the lookup key.
    pub fn new(bindings: &[&fyrox_graphics::framebuffer::ResourceBinding]) -> Self {
        use fyrox_graphics::framebuffer::ResourceBinding;
        use fyrox_graphics::framebuffer::BufferDataUsage;
        use fyrox_graphics::gpu_texture::GpuTextureTrait;
        use fyrox_graphics::sampler::GpuSamplerTrait;
        use fyrox_graphics::buffer::GpuBufferTrait;

        let mut keys = Vec::with_capacity(bindings.len());
        for b in bindings {
            match b {
                ResourceBinding::Texture { texture, sampler, binding } => {
                    // Pointer to the underlying trait object — stable for all Rc clones.
                    let texture_ptr = (&**texture as *const dyn GpuTextureTrait as *const std::ffi::c_void as usize);
                    let sampler_ptr = (&**sampler as *const dyn GpuSamplerTrait as *const std::ffi::c_void as usize);
                    keys.push(BindingKey::Texture { texture_ptr, sampler_ptr, binding: *binding });
                }
                ResourceBinding::Buffer { buffer, binding, data_usage } => {
                    let buffer_ptr = (&**buffer as *const dyn GpuBufferTrait as *const std::ffi::c_void as usize);
                    let (offset, size) = match data_usage {
                        BufferDataUsage::UseEverything => (0, None),
                        BufferDataUsage::UseSegment { offset, size } => (*offset as u64, Some(*size as u64)),
                    };
                    keys.push(BindingKey::Buffer { buffer_ptr, binding: *binding, offset, size });
                }
            }
        }
        Self { bindings: keys }
    }
}

impl Hash for BindGroupCacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.bindings.hash(state);
    }
}

/// A cache of `wgpu::BindGroup` objects keyed by resource pointer combination.
///
/// Reduces per-draw bind group creation overhead by caching and reusing bind
/// groups for identical resource sets.
#[derive(Default)]
pub struct BindGroupCache {
    entries: HashMap<BindGroupCacheKey, Arc<BindGroup>>,
    metrics: BindGroupMetrics,
}

impl BindGroupCache {
    /// Looks up a cached bind group, if one exists for the given key.
    ///
    /// Records a hit if found, or a miss if not found.
    pub fn get(&mut self, key: &BindGroupCacheKey) -> Option<Arc<BindGroup>> {
        let result = self.entries.get(key).cloned();
        if result.is_some() {
            self.metrics.record_hit();
        } else {
            self.metrics.record_miss();
        }
        result
    }

    /// Inserts a bind group into the cache, returning the `Arc` that the caller should use.
    ///
    /// If a group was already cached, returns the existing `Arc` instead.
    pub fn insert(&mut self, key: BindGroupCacheKey, bg: BindGroup) -> Arc<BindGroup> {
        let arc = self.entries.entry(key).or_insert_with(|| Arc::new(bg)).clone();
        arc
    }

    /// Returns the number of cached bind groups.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Snapshots and resets the hit/miss counters.
    pub fn metrics_snapshot_and_reset(&self) -> (u64, u64) {
        self.metrics.snapshot_and_reset()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_key_equality() {
        // Two empty keys should be equal
        let k1 = BindGroupCacheKey { bindings: vec![] };
        let k2 = BindGroupCacheKey { bindings: vec![] };
        assert_eq!(k1, k2);
        assert_eq!(k1.hash(&mut std::collections::hash_map::DefaultHasher::new()),
                   k2.hash(&mut std::collections::hash_map::DefaultHasher::new()));
    }
}
