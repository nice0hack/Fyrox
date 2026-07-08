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

//! Frame-level performance metrics for the wgpu backend.

use std::sync::atomic::{AtomicU64, Ordering};

/// Metrics for a single frame, captured and reset each frame.
#[derive(Debug, Clone, Default)]
pub struct FrameMetrics {
    /// Number of bind group cache hits this frame.
    pub bind_group_cache_hits: u64,
    /// Number of bind group cache misses this frame.
    pub bind_group_cache_misses: u64,
    /// Number of draw calls this frame.
    pub draw_call_count: u64,
}

/// Accumulator for bind group cache metrics.
///
/// Uses atomic counters to allow concurrent access from multiple threads.
#[derive(Debug, Default)]
pub struct BindGroupMetrics {
    hits: AtomicU64,
    misses: AtomicU64,
}

impl BindGroupMetrics {
    /// Records a cache hit.
    pub fn record_hit(&self) {
        self.hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Records a cache miss.
    pub fn record_miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Snapshots the current counters and resets them.
    pub fn snapshot_and_reset(&self) -> (u64, u64) {
        let hits = self.hits.swap(0, Ordering::Relaxed);
        let misses = self.misses.swap(0, Ordering::Relaxed);
        (hits, misses)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bind_group_metrics_snapshot() {
        let metrics = BindGroupMetrics::default();

        metrics.record_hit();
        metrics.record_hit();
        metrics.record_miss();

        let (hits, misses) = metrics.snapshot_and_reset();
        assert_eq!(hits, 2);
        assert_eq!(misses, 1);

        // After reset, counters should be zero
        let (hits, misses) = metrics.snapshot_and_reset();
        assert_eq!(hits, 0);
        assert_eq!(misses, 0);
    }
}
