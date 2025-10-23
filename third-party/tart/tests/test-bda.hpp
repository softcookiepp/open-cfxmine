#ifndef TART_TEST_BDA
#define TART_TEST_BDA
#include "tests-common.hpp"
#include "shaders.hpp"
#include "tart.hpp"

TEST_CASE("testing buffer device address (1/2)")
{
	tart::device_ptr device = getTestDevice();

	// initialize inputs
	std::vector<float> inAvalues({ 2., 2., 2. });
    auto bufA = device->allocateBuffer(inAvalues);

    std::vector<float> inBvalues({ 1., 2., 3. });
    auto bufB = device->allocateBuffer(inBvalues);
	
	std::string src = R"(
#version 450
#extension GL_EXT_buffer_reference : require

layout(buffer_reference, std430, buffer_reference_align = 4) buffer float_ptr { float data[]; };

layout(push_constant) uniform push
{
	float_ptr a;
	float_ptr b;
} args;

#define LOAD_BUF(buf, idx) args.buf.data[idx]

void main()
{
	uint idx = gl_WorkGroupID.x;
	args.a.data[idx] = 2*LOAD_BUF(b, idx);
}	
	)";
	
	tart::shader_module_ptr shaderModule = device->compileGLSL(src);
	
	struct {
		uint64_t a;
		uint64_t b;
	} push = {bufA->getAddress(), bufB->getAddress()};
	
	tart::pipeline_ptr pipeline = device->createPipeline(shaderModule, "main", {}, tart::packConstants(push) );
	
	device->dispatchPipeline(pipeline, {3}, {}, tart::packConstants(push));
	device->sync();
	std::vector<float> expected({2, 4, 6});
	CHECK(expected == bufA->copyOut<float>() );
}

TEST_CASE("testing buffer device address (1/2)")
{
	// here we benchmark both BDA and descriptor sets to determine which has better performance.
	tart::device_ptr device = getTestDevice();
	
	// initialize inputs
	std::vector<float> inAvalues = randn(4096*4096);
    auto bufA = device->allocateBuffer(inAvalues);

    std::vector<float> inBvalues = randn(4096*4096);
    auto bufB = device->allocateBuffer(inBvalues);
    
    auto bufC = device->allocateBuffer(4096*4096*sizeof(float));
    
std::string srcBDA = R"(
#version 450
#extension GL_EXT_buffer_reference : require

layout(buffer_reference, std430, buffer_reference_align = 4) buffer float_ptr { float data[]; };

layout(push_constant) uniform push
{
	float_ptr a;
	float_ptr b;
	float_ptr c;
} args;

#define INDEX_BUF(buf, idx) args.buf.data[idx]

void main()
{
	uint idx = gl_WorkGroupID.x;
	INDEX_BUF(c, idx) = sin(INDEX_BUF(a, idx)) + cos(INDEX_BUF(b, idx));
}	
	)";

#if 0
std::string srcDescriptors = R"(
#version 450

layout(binding = 0, std430) buffer a_buf { float a[]; };
layout(binding = 1, std430) buffer b_buf { float b[]; };
layout(binding = 2, std430) buffer c_buf { float c[]; };

#define INDEX_BUF(buf, idx) buf[idx]

void main()
{
	uint idx = gl_WorkGroupID.x;
	INDEX_BUF(c, idx) = sin(INDEX_BUF(a, idx)) + cos(INDEX_BUF(b, idx));
}	
	)";
#else
std::string srcDescriptors = R"(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types: require
#extension GL_EXT_shader_explicit_arithmetic_types_int64: require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
#define INFINITY uintBitsToFloat(0x7F800000)
#define NAN uintBitsToFloat(0x7FC00000)

#define PI 3.14159265358979323846

#define STORE_VEC2(buf, idx, value2) buf[idx] = value2.x; buf[idx + 1] = value2.y
#define STORE_VEC4(buf, idx, value4) buf[idx] = value4[0]; buf[idx + 1] = value4[1]; buf[idx + 2] = value4[2]; buf[idx + 3] = value4[3]
#define LOAD_VEC2(buf, idx, VEC_T) VEC_T(buf[idx], buf[idx + 1])
#define LOAD_VEC4(buf, idx, VEC_T) VEC_T(buf[idx], buf[idx + 1], buf[idx + 2], buf[idx + 3])

precise float64_t exp2(float64_t x) { precise float tmp =  float(x); tmp = pow(2.0, tmp); return float64_t(tmp); }

precise float _sin(float xlp) {
#if 0
	precise float64_t x = float64_t(mod(xlp + PI, 2.0 * PI) - PI);
	precise float64_t x2 = x * x;
	precise float64_t outp = x * (1.0 - x2 * (1.0/6.0 - x2 * (1.0/120.0 - x2 * (1.0/5040.0 - x2*(1.0/362880.0 - x2*(1.0/39916800.0 - x2*(1.0/6227020800.0)  - x2*(1.0/1307674368000.0) ) )  ))));
	return float(outp);
#else
    return sin(xlp);
#endif
}
precise float16_t sine_precise(float16_t t) { precise float s = float(t); s = _sin(s); return float16_t(s); }
precise float sine_precise(float t) { precise float s = t; s = _sin(s); return s; }
precise float64_t sine_precise(float64_t t) { precise float s = float(t); s = _sin(s); return float64_t(s); }

precise float16_t exp2_precise(float16_t t) { precise float16_t s = exp2(t); return s; }
precise float exp2_precise(float t) { precise float s = exp2(t); return s; }
precise float64_t exp2_precise(float64_t t) { precise float64_t s = exp2(t); return s; }

// looks like we really have to use domain hacks for log...
precise float16_t log2_precise(float16_t t)
{
	if (t == 0.0) return float16_t(-1.0*INFINITY);
	precise float16_t s = log2(t);
	return s;
}
precise float log2_precise(float t)
{
	if (t == 0.0) return -1.0*INFINITY;
	precise float s = log2(t);
	return s;
}
		
float nan() { uint bits = 0xffffffffu; return uintBitsToFloat(bits); }
layout(set = 0, binding = 0, std430) buffer data0_4096_buf { float data0_4096[]; };
layout(set = 0, binding = 1, std430) buffer data1_4096_buf { float data1_4096[]; };
void main()
{
  int gidx0 = int(gl_WorkGroupID.x); /* 32 */
  int lidx0 = int(gl_LocalInvocationID.x); /* 32 */
  int alu0 = ((gidx0<<(7))+(lidx0<<(2)));
  float val0 = data1_4096[alu0];
  int alu1 = (alu0+(1));
  float val1 = data1_4096[alu1];
  int alu2 = (alu0+(2));
  float val2 = data1_4096[alu2];
  int alu3 = (alu0+(3));
  float val3 = data1_4096[alu3];
  data0_4096[alu1] = (sin(val1)*float(1) / sin((1.5707963267948966f-val1)));
  data0_4096[alu2] = (sin(val2)*float(1) / sin((1.5707963267948966f-val2)));
  data0_4096[alu3] = (sin(val3)*float(1) / sin((1.5707963267948966f-val3)));
  data0_4096[alu0] = (sin(val0)*float(1) / sin((1.5707963267948966f-val0)));
}

)";
#endif
	tart::shader_module_ptr moduleBDA = device->compileGLSL(srcBDA);
	tart::shader_module_ptr moduleDescriptors = device->compileGLSL(srcDescriptors);

	struct {
		uint64_t a;
		uint64_t b;
		uint64_t c;
	} pushStruct = {bufA->getAddress(), bufB->getAddress(), bufC->getAddress() };
	std::vector<uint8_t> push = tart::packConstants(pushStruct);
	
	tart::pipeline_ptr pipelineBDA = device->createPipeline(moduleBDA, "main", {}, push);
	tart::pipeline_ptr pipelineDescriptors = device->createPipeline(moduleDescriptors, "main", {});
	
	auto startBDA = std::chrono::high_resolution_clock::now();
	//tart::command_sequence_ptr seq = device->createSequence();
	for (size_t i = 0; i < 1000; i += 1)
	{
		tart::command_sequence_ptr seq = device->createSequence();
		seq->recordPipeline(pipelineBDA, {4096}, {}, push);
		device->submitSequence(seq);
		device->sync();
	}
	auto endBDA = std::chrono::high_resolution_clock::now();
	
	auto startDescriptors = std::chrono::high_resolution_clock::now();
	
	device->deallocateBuffer(bufA);
	for (size_t i = 0; i < 1000; i += 1)
	{
		bufA = device->allocateBuffer(bufB->getSize());
		std::vector<tart::buffer_ptr> bufs({bufA, bufB});
		tart::command_sequence_ptr seq = device->createSequence();
		seq->recordPipeline(pipelineDescriptors, {131072}, bufs, push);
		device->submitSequence(seq);
		device->sync();
		device->deallocateBuffer(bufA);
	}
	auto endDescriptors = std::chrono::high_resolution_clock::now();
	uint64_t timeBDA = std::chrono::duration_cast<std::chrono::microseconds>(endBDA - startBDA).count()/1000;
	std::cout << "BDA benchmark: " << timeBDA << std::endl;
	uint64_t timeDescriptors = std::chrono::duration_cast<std::chrono::microseconds>(endDescriptors - startDescriptors).count()/1000;
	std::cout << "Descriptors benchmark: " << timeDescriptors << std::endl;
}

#endif
