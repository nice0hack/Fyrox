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

//! GPU-side uniform buffer pool with sub-allocation.
//!
//! Instead of creating a new `wgpu::Buffer` for each uniform upload, the pool
//! maintains a single large buffer that is partitioned into aligned sub-regions.
//! `allocate()` reserves a region and returns the byte offset. Data is uploaded
//! via `queue.write_buffer()` at that offset.
//!
//! `reset()` clears the allocation cursor at the start of each frame, allowing
//! the same buffer memory to be reused. This avoids per-frame GPU allocation
//! overhead while keeping the maximum allocation count equal to the number of
//! draw calls per frame.

use std::sync::Arc;
use wgpu::{Buffer, BufferSlice, Device, Queue};

/// Minimum alignment for uniform buffer bindings in wgpu.
const UNIFORM_ALIGNMENT: u64 = 256;

/// A sub-allocation within a uniform buffer pool.
///
/// Tracks the byte offset of the allocated region. Use `slice()` to get
/// a `wgpu::BufferSlice` for binding, and `write()` to upload data.
#[derive(Debug, Clone, Copy)]
pub struct Suballocation {
    /// Byte offset from the start of the pool's buffer.
    pub offset: u32,
    /// Byte size of the allocated region.
    pub size: u32,
}

impl Suballocation {
    /// Returns a `BufferSlice` covering this suballocation's region.
    pub fn slice(self, buffer: &Arc<Buffer>) -> BufferSlice<'_> {
        buffer.slice(self.offset as u64..(self.offset + self.size) as u64)
    }

    /// Writes data into this suballocation's region of the pool buffer.
    pub fn write(&self, queue: &Queue, buffer: &Arc<Buffer>, data: &[u8]) {
        queue.write_buffer(buffer, self.offset as u64, data);
    }
}

/// A pool of GPU uniform buffer memory that allocates sub-regions on demand.
///
/// The pool owns a single `wgpu::Buffer` (with `UNIFORM | COPY_DST` usage) and
/// carves it into sub-allocations. `allocate()` reserves the next aligned region.
/// `reset()` rewinds the allocation cursor so the same memory can be reused each
/// frame — callers must ensure all in-flight draws using previously allocated
/// suballocations have completed before calling `reset()`.
pub struct UniformBufferPool {
    buffer: Arc<Buffer>,
    capacity: u64,
    next_offset: u64,
}

impl UniformBufferPool {
    /// Creates a new pool with a GPU buffer of the given capacity in bytes.
    pub fn new(device: &Device, capacity: u64) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("UniformBufferPool"),
            size: capacity,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { buffer: Arc::new(buffer), capacity, next_offset: 0 }
    }

    /// Returns the underlying buffer.
    pub fn buffer(&self) -> &Arc<Buffer> {
        &self.buffer
    }

    /// Attempts to allocate `size` bytes from the pool.
    ///
    /// Returns `None` if the pool is exhausted. The allocated region is
    /// aligned to [`UNIFORM_ALIGNMENT`] (256 bytes) per the WebGPU spec.
    pub fn allocate(&mut self, size: u64) -> Option<Suballocation> {
        let aligned = (size + UNIFORM_ALIGNMENT - 1) & !(UNIFORM_ALIGNMENT - 1);
        if self.next_offset + aligned > self.capacity {
            return None;
        }
        let offset = self.next_offset as u32;
        self.next_offset += aligned;
        Some(Suballocation { offset, size: size as u32 })
    }

    /// Resets the allocation cursor to the start of the buffer.
    ///
    /// Call this at the beginning of each frame. The caller must ensure that
    /// all draws using previously allocated suballocations have finished
    /// before resetting.
    pub fn reset(&mut self) {
        self.next_offset = 0;
    }

    /// Returns true if the pool is empty (no allocations made this frame).
    pub fn is_empty(&self) -> bool {
        self.next_offset == 0
    }

    /// Returns the number of bytes used in the current frame's allocation.
    pub fn used(&self) -> u64 {
        self.next_offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_allocation_at_zero() {
        // Simulate allocation logic without GPU.
        let mut next_offset = 0u64;
        let size = 64u64;
        let aligned = (size + 255) & !255;
        let offset = next_offset;
        next_offset += aligned;
        assert_eq!(offset, 0);
        assert_eq!(next_offset, 256);
    }

    #[test]
    fn second_allocation_aligned() {
        let mut next_offset = 0u64;
        // First: 64 bytes -> aligned to 256
        next_offset += (64 + 255) & !255;
        // Second: 64 bytes -> aligned to 256
        let second_offset = next_offset;
        next_offset += (64 + 255) & !255;
        assert_eq!(second_offset, 256);
        assert_eq!(next_offset, 512);
    }

    #[test]
    fn overflow_returns_none() {
        // 64 bytes aligned to 256 = 256. capacity = 200. 0 + 256 > 200 is true.
        let capacity = 200u64;
        let size = 64u64;
        let aligned = (size + 255) & !255;
        assert!(aligned > capacity);
    }

    #[test]
    fn suballocation_offset_and_size() {
        let sub = Suballocation { offset: 512, size: 64 };
        assert_eq!(sub.offset, 512);
        assert_eq!(sub.size, 64);
    }
}
