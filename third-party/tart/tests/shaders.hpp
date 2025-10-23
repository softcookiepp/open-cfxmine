#ifndef TART_TEST_SHADERS
#define TART_TEST_SHADERS

#include <string>

namespace test_shaders
{

static const std::string op_add = R"(
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

)";

static const std::string op_sub = R"(
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

)";

static const std::string op_mul = R"(
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

)";

static const std::string op_div = R"(
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

)";

static const std::string asyncTestShader(R"(
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
)");

static const std::string cl_test = R"(

__kernel
void square(__global float* inp, __global float* out)
{
	uint i = get_global_id(0);
	out[i] = inp[i] * inp[i];
}

)";

} // namespace test_shaders

#endif
