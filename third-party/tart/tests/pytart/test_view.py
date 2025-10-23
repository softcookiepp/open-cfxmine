import pytart
from shaders_for_test import *
import numpy as np


tart = pytart.Instance()

def test_view():
	dev = tart.create_device(0)
	
	buf = dev.allocate_buffer(12*np.float32(0).nbytes)
	a = np.arange(4).astype(np.float32)
	b = np.arange(4).astype(np.float32) + 4
	c = np.arange(4).astype(np.float32) + 8
	buf.view(0).copy_in(a)
	buf.view(4*4).copy_in(b)
	buf.view(8*4).copy_in(c)
	
	expected = np.concatenate([a, b, c])
	result = np.zeros(12, dtype = np.float32)
	buf.copy_out(result)
	assert np.allclose(result, expected)
	
