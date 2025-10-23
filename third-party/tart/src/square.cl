__kernel void other(__global int* a)
{
	uint i = get_global_id(0);
	a[i] = a[i] * a[i];
}

__kernel void Main(__global float* a, __global float* b)
{
	uint i = get_global_id(0);
	b[i] = a[i] * a[i];
}
