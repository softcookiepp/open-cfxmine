__kernel void Main(__global float* a, __global float* b)
{
	uint i = get_global_id(0);
	b[i] = a[i] * a[i];
}
