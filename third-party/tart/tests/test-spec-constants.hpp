#ifndef TART_TEST_SPEC_CONSTANTS
#define TART_TEST_SPEC_CONSTANTS
#include "tests-common.hpp"
#include "shaders.hpp"
#include "tart.hpp"

TEST_CASE("testing spec constant reflection (1/1)")
{
		tart::device_ptr device = getTestDevice();

	// initialize inputs
	std::vector<float> inAvalues({ 2., 2., 2. });
    auto tensorInA = device->allocateBuffer(inAvalues);
    std::vector<float> inBvalues({ 1., 2., 3. });
    auto tensorInB = device->allocateBuffer(inBvalues);
    
    // initialize inputs
    std::vector<uint32_t> outInit({0, 0, 0});
    auto tensorOutA = device->allocateBuffer(outInit);
    auto tensorOutB = device->allocateBuffer(outInit);

    std::string shader = (R"(
        #version 450

        layout (local_size_x_id = 2) in;

        layout(set = 0, binding = 0) buffer buf_in_a { float in_a[]; };
        layout(set = 0, binding = 1) buffer buf_in_b { float in_b[]; };
        layout(set = 0, binding = 2) buffer buf_out_a { uint out_a[]; };
        layout(set = 0, binding = 3) buffer buf_out_b { uint out_b[]; };

        layout(push_constant) uniform PushConstants {
            float val;
        } push_const;

        layout(constant_id = 0) const float const_one = 0;
        layout(constant_id = 1) const float const_two = 0;
        layout(constant_id = 2) const uint local_size = 0;

        void main() {
            uint index = gl_GlobalInvocationID.x;
            out_a[index] += uint( in_a[index] * in_b[index] );
            out_b[index] += uint( const_two*(const_one * push_const.val ) );
        }
    )");

	std::vector<tart::buffer_ptr> params( {tensorInA, tensorInB, tensorOutA, tensorOutB} );

	struct { float a = 2; float b = 1; uint32_t localSize = 1; } specConsts;

	std::vector<float> pushConstsA({2.0});
	std::vector<float> pushConstsB({3.0});
	
	tart::shader_module_ptr shaderModule = device->compileGLSL(shader);
	tart::pipeline_ptr algorithm = device->createPipeline(shaderModule, "main",
		tart::packConstants(specConsts), tart::packConstants(pushConstsA) );
	
	tart::command_sequence_ptr sequence = device->createSequence();
	sequence->recordPipeline(algorithm, {3}, params, tart::packConstants(pushConstsA) );
	sequence->recordPipeline(algorithm, {3}, params, tart::packConstants(pushConstsB) );
	device->submitSequence(sequence);
	device->sync();

	CHECK(tensorOutA->copyOut<uint32_t>() == std::vector<uint32_t>({ 4, 8, 12 }) );
	CHECK(tensorOutB->copyOut<uint32_t>() == std::vector<uint32_t>({ 10, 10, 10 }) );
}

#endif
