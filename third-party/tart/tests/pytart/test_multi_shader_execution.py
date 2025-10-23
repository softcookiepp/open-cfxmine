import pytart
from shaders_for_test import *
import numpy as np


tart = pytart.Instance()

def test_multiple_shader_execution():
	dev = tart.create_device(0)
	shader = """
	#version 450

	layout (local_size_x = 1) in;

	layout(set = 0, binding = 0) buffer buf_in_a { float in_a[]; };
	layout(set = 0, binding = 1) buffer buf_in_b { float in_b[]; };
	layout(set = 0, binding = 2) buffer buf_out_a { uint out_a[]; };
	layout(set = 0, binding = 3) buffer buf_out_b { uint out_b[]; };

	layout(push_constant) uniform PushConstants {
		float val;
	} push_const;

	layout(constant_id = 0) const float const_one = 0;
	layout(constant_id = 1) const float const_two = 0;

	void main() {
		uint index = gl_GlobalInvocationID.x;
		out_a[index] += uint( in_a[index] * in_b[index] );
		out_b[index] += uint( const_two*(const_one * push_const.val ) );
	}
	"""
	a_np = np.ascontiguousarray(np.array([2., 2., 2.]).astype(np.float32))
	a = dev.allocate_buffer(a_np.nbytes)
	a.copy_in(a_np)
	
	b_np = np.ascontiguousarray(np.array([1., 2., 3.]).astype(np.float32))
	b = dev.allocate_buffer(b_np.nbytes)
	b.copy_in(b_np)

	a_out = dev.allocate_buffer(a_np.nbytes)
	b_out = dev.allocate_buffer(a_np.nbytes)
	
	spec_consts = np.ascontiguousarray(np.array([1.0, 2.0]).astype(np.float32))
	push_consts_a = np.ascontiguousarray(np.array([2.0], dtype = np.float32))
	push_consts_b = np.ascontiguousarray(np.array([3.0], dtype = np.float32))
	
	shader_module = dev.compile_glsl(shader)
	pipeline = dev.create_pipeline(shader_module, "main", spec_consts, push_consts_a)
	
	params = [a, b, a_out, b_out]
	sequence = dev.create_sequence()
	sequence.record_pipeline(pipeline, [3, 1, 1], params, push_consts_a)
	sequence.record_pipeline(pipeline, [3, 1, 1], params, push_consts_b)
	dev.submit_sequence(sequence)
	dev.sync()
	
	a_out_np = np.zeros(3, dtype = np.uint32)
	a_out.copy_out(a_out_np)
	
	b_out_np = np.zeros(3, dtype = np.uint32)
	b_out.copy_out(b_out_np)
	
	assert np.allclose(a_out_np - np.array([4., 8., 12.]), 0)
	assert np.allclose(b_out_np - np.array([10., 10., 10.]), 0)

