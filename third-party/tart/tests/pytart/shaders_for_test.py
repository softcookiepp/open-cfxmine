op_add = """
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
"""

op_sub = """
#version 450

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer a_buf { float a[]; };
layout(set = 0, binding = 1) buffer b_buf { float b[]; };
layout(set = 0, binding = 2) buffer c_buf { float c[]; };


void main()
{
	uint i = gl_GlobalInvocationID.x;
	c[i] = b[i] - a[i];
}
"""

op_mul = """
#version 450

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer a_buf { float a[]; };
layout(set = 0, binding = 1) buffer b_buf { float b[]; };
layout(set = 0, binding = 2) buffer c_buf { float c[]; };


void main()
{
	uint i = gl_GlobalInvocationID.x;
	c[i] = b[i] * a[i];
}
"""

op_div = """
#version 450

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer a_buf { float a[]; };
layout(set = 0, binding = 1) buffer b_buf { float b[]; };
layout(set = 0, binding = 2) buffer c_buf { float c[]; };


void main()
{
	uint i = gl_GlobalInvocationID.x;
	c[i] = b[i] / a[i];
}
"""

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

cl_test = """
__kernel
void square(__global float* inp, __global float* out)
{
	uint i = get_global_id(0);
	out[i] = inp[i] * inp[i];
}
"""

multi_exec_test = """
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
