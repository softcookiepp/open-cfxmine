import pytart
tart = pytart.Instance()
dev = tart.create_device(0)

import numpy as np

a = np.arange(4).astype(np.float32)
a_buf = dev.allocate_buffer(a.nbytes)
a_buf.copy_in(a)
b_buf = dev.allocate_buffer(a.nbytes)
with open("square-stripped.spv", "rb") as f:
	spv = f.read()
mod = dev.load_shader(spv)
pipeline = dev.create_pipeline(mod, "main", np.array([1, 1, 1], dtype = np.uint32))

pipeline.dispatch([4], [a_buf, b_buf])
dev.sync()
b = np.zeros(4, dtype = np.float32)
b_buf.copy_out(b)
assert np.allclose(b, np.array([0., 1., 4., 9.], dtype=np.float32) )
print("success!")
