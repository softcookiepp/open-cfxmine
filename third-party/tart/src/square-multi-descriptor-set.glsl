#version 450

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer a_buf { float a[]; };
layout(set = 1, binding = 0) buffer b_buf { float b[]; };

layout(push_constant) uniform constants
{
	float k;
	float val;
} pc;

void main()
{
	uint i = gl_GlobalInvocationID.x;
	b[i] = (a[i] * a[i]) + pc.k*pc.val;
}
