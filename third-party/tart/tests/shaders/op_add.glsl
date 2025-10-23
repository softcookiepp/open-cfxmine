#version 450

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer a_buf { float a[]; };
layout(set = 0, binding = 1) buffer b_buf { float b[]; };
layout(set = 0, binding = 2) buffer c_buf { float c[]; };


void main()
{
	uint i = gl_GlobalInvocationID.x;
	c[i] = b[i] + a[i];
}
