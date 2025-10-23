import pytart
from shaders_for_test import *
import numpy as np

tart = pytart.Instance()
dev = tart.create_device(0)

def test_cl():
	src = """
	__kernel
	void square(__global float* inp, __global float* out)
	{
		uint i = get_global_id(0);
		out[i] = inp[i] * inp[i];
	}
	"""
	a_in = np.arange(4).astype(np.float32)
	buf_a = dev.allocate_buffer(a_in.nbytes)
	buf_a.copy_in(a_in)
	buf_out = dev.allocate_buffer(a_in.nbytes)
	
	shader_module = dev.compile_cl(src)
	
	
	local_size = np.array([1, 1, 1]).astype(np.uint32)
	pipeline = dev.create_pipeline(shader_module, "square", local_size)
	
	dev.dispatch_pipeline(pipeline, [4, 1, 1], [buf_a, buf_out])
	dev.sync()
	
	out = np.zeros(4, dtype = np.float32)
	buf_out.copy_out(out)
	print(out)
	assert np.allclose(out, np.array([0., 1., 4., 9.]))
