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
    core::{array_as_u8_slice, math::TriangleDefinition},
    error::FrameworkError,
    geometry_buffer::{AttributeKind, ElementsDescriptor, GpuGeometryBufferDescriptor, GpuGeometryBufferTrait},
    ElementKind,
};
use std::cell::Cell;
use std::rc::Weak;

fn attribute_format(kind: AttributeKind, cc: usize) -> wgpu::VertexFormat {
    match (kind, cc) {
        (AttributeKind::Float, 1) => wgpu::VertexFormat::Float32,
        (AttributeKind::Float, 2) => wgpu::VertexFormat::Float32x2,
        (AttributeKind::Float, 3) => wgpu::VertexFormat::Float32x3,
        (AttributeKind::Float, 4) => wgpu::VertexFormat::Float32x4,
        (AttributeKind::UnsignedByte, 4) => wgpu::VertexFormat::Uint8x4,
        (AttributeKind::UnsignedShort, 2) => wgpu::VertexFormat::Uint16x2,
        (AttributeKind::UnsignedShort, 4) => wgpu::VertexFormat::Uint16x4,
        (AttributeKind::UnsignedInt, 1) => wgpu::VertexFormat::Uint32,
        (AttributeKind::UnsignedInt, 2) => wgpu::VertexFormat::Uint32x2,
        (AttributeKind::UnsignedInt, 3) => wgpu::VertexFormat::Uint32x3,
        (AttributeKind::UnsignedInt, 4) => wgpu::VertexFormat::Uint32x4,
        _ => wgpu::VertexFormat::Float32x4,
    }
}

pub struct WgpuGeometryBuffer {
    _server: Weak<WgpuGraphicsServer>,
    vertex_buffers: Vec<wgpu::Buffer>,
    vertex_buffer_layouts: Vec<wgpu::VertexBufferLayout<'static>>,
    element_buffer: wgpu::Buffer,
    element_count: Cell<usize>,
    element_kind: ElementKind,
}

impl WgpuGeometryBuffer {
    pub fn new(server: &WgpuGraphicsServer, desc: GpuGeometryBufferDescriptor) -> Result<Self, FrameworkError> {
        let mut vertex_buffers = Vec::new();
        let mut vertex_buffer_layouts = Vec::new();

        for (i, buf) in desc.buffers.iter().enumerate() {
            let data_size = buf.data.bytes.map(|b| b.len()).unwrap_or(0);
            let label_str = format!("{}VB{i}", desc.name);
            let buffer = server.state.device.create_buffer(&wgpu::BufferDescriptor {
                label: if server.named_objects { Some(&label_str) } else { None },
                size: data_size.max(1) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            if let Some(data) = buf.data.bytes { if !data.is_empty() { server.state.queue.write_buffer(&buffer, 0, data); } }

            let mut attributes = Vec::new();
            let mut offset = 0u64;
            for attr in buf.attributes {
                attributes.push(wgpu::VertexAttribute { format: attribute_format(attr.kind, attr.component_count), offset, shader_location: attr.location });
                offset += (attr.kind.size() * attr.component_count) as u64;
            }

            let step_mode = if buf.attributes.iter().any(|a| a.divisor > 0) { wgpu::VertexStepMode::Instance } else { wgpu::VertexStepMode::Vertex };
            vertex_buffer_layouts.push(wgpu::VertexBufferLayout { array_stride: buf.data.element_size as u64, step_mode, attributes: attributes.leak() });
            vertex_buffers.push(buffer);
        }

        let (element_count, element_data) = match desc.elements {
            ElementsDescriptor::Triangles(t) => (t.len(), array_as_u8_slice(t)),
            ElementsDescriptor::Lines(l) => (l.len(), array_as_u8_slice(l)),
            ElementsDescriptor::Points(p) => (p.len(), array_as_u8_slice(p)),
        };

        let ib_label = format!("{}IB", desc.name);
        let element_buffer = server.state.device.create_buffer(&wgpu::BufferDescriptor {
            label: if server.named_objects { Some(&ib_label) } else { None },
            size: element_data.len().max(1) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !element_data.is_empty() { server.state.queue.write_buffer(&element_buffer, 0, element_data); }

        Ok(Self { _server: server.weak_ref(), vertex_buffers, vertex_buffer_layouts, element_buffer, element_count: Cell::new(element_count), element_kind: desc.elements.element_kind() })
    }

    pub fn element_count(&self) -> usize { self.element_count.get() }
    pub fn element_kind(&self) -> ElementKind { self.element_kind }
    pub fn vertex_buffers(&self) -> &[wgpu::Buffer] { &self.vertex_buffers }
    pub fn vertex_buffer_layouts(&self) -> &[wgpu::VertexBufferLayout<'static>] { &self.vertex_buffer_layouts }
    pub fn element_buffer(&self) -> &wgpu::Buffer { &self.element_buffer }
}

impl GpuGeometryBufferTrait for WgpuGeometryBuffer {
    fn set_buffer_data(&self, buffer: usize, data: &[u8]) {
        if let Some(server) = self._server.upgrade() {
            if let Some(buf) = self.vertex_buffers.get(buffer) { server.state.queue.write_buffer(buf, 0, data); }
        }
    }
    fn element_count(&self) -> usize { self.element_count.get() }
    fn set_triangles(&self, triangles: &[TriangleDefinition]) {
        if let Some(server) = self._server.upgrade() {
            self.element_count.set(triangles.len());
            server.state.queue.write_buffer(&self.element_buffer, 0, array_as_u8_slice(triangles));
        }
    }
    fn set_lines(&self, lines: &[[u32; 2]]) {
        if let Some(server) = self._server.upgrade() {
            self.element_count.set(lines.len());
            server.state.queue.write_buffer(&self.element_buffer, 0, array_as_u8_slice(lines));
        }
    }
    fn set_points(&self, points: &[u32]) {
        if let Some(server) = self._server.upgrade() {
            self.element_count.set(points.len());
            server.state.queue.write_buffer(&self.element_buffer, 0, array_as_u8_slice(points));
        }
    }
}
