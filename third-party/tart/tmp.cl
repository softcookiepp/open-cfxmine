
	__kernel
	void square(__global float* inp, __global float* out)
	{
		uint i = get_global_id(0);
		out[i] = inp[i] * inp[i];
	}
	