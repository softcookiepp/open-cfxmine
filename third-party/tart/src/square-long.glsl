#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer a_buf { int64_t a[]; };
layout(set = 1, binding = 0) buffer b_buf { int64_t b[]; };

void main()
{
	uint i = gl_GlobalInvocationID.x;
	b[i] = (a[i] * a[i]);
}
