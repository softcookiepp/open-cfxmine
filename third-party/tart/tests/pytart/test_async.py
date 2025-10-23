import pytart
from shaders_for_test import *
import numpy as np

tart = pytart.Instance()

def test_allocation_destruction():
	dev = tart.create_device(0)
	buf = dev.allocate_buffer(1024)
	dev.deallocate_buffer(buf)
	assert buf.destroyed

def test_numpy_buffer_copy_in_out():
	dev = tart.create_device(0)
	a = np.arange(4)
	buf = dev.allocate_buffer(a.nbytes)
	buf.copy_in(a)
	b = np.zeros(4, dtype = a.dtype)
	buf.copy_out(b)
	dev.deallocate_buffer(buf)
	
	assert np.allclose(a - b, 0)

async_test_shader = """
#version 450

layout (local_size_x = 1) in;

layout(set = 0, binding = 0) buffer b { float pb[]; };

shared uint sharedTotal[1];

void main() {
	uint index = gl_GlobalInvocationID.x;

	sharedTotal[0] = 0;

	for (int i = 0; i < 1000; i++)
	{
		atomicAdd(sharedTotal[0], 1);
	}

	pb[index] = sharedTotal[0];
}
"""

def test_async_1():
	dev = tart.create_device(0)
	data = np.zeros(10, dtype = np.float32)
	result_async = np.zeros(10, dtype = np.float32) + 1000
	
	tensor_a = dev.allocate_buffer(data.nbytes)
	tensor_b = dev.allocate_buffer(data.nbytes)
	
	sq1 = dev.create_sequence()
	sq2 = dev.create_sequence()
	
	shader_module = dev.compile_glsl(async_test_shader)
	
	algo = dev.create_pipeline(shader_module, "main")
	
	sq1.record_pipeline(algo, [10], [tensor_a])
	sq2.record_pipeline(algo, [10], [tensor_b])
	
	# TODO: this crashes when a given buffer is recorded to 2 sequences at the same time, then submitted.
	# Find out why (see if it can be reproduced in C++ first)
	dev.submit_sequence(sq1)
	dev.submit_sequence(sq2)
	dev.sync()
	
	out_a = np.ascontiguousarray(np.zeros(10, dtype = np.float32))
	out_b = np.ascontiguousarray(np.zeros(10, dtype = np.float32))
	tensor_a.copy_out(out_a)
	tensor_b.copy_out(out_b)
	assert np.allclose(out_a, out_b)

def test_async_2():
	size = 10
	num_parallel = 2
