#ifndef TART_TEST_MULTIPLE_SHADER_EXECUTIONS
#define TART_TEST_MULTIPLE_SHADER_EXECUTIONS
#include "tests-common.hpp"
#include "shaders.hpp"
#include "tart.hpp"




TEST_CASE("testing multiple shader execution (1/6)")
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
    )");

	std::vector<tart::buffer_ptr> params( {tensorInA, tensorInB, tensorOutA, tensorOutB} );

	struct { float a = 2; float b = 1; } specConsts;

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

TEST_CASE("testing multiple shader execution (2/6)")
{
	tart::device_ptr device = getTestDevice();

	std::vector<float> valuesA(3);
	tart::buffer_ptr tensorA = device->allocateBuffer(valuesA);
	
    std::string shader(R"(
      #version 450
      layout (local_size_x = 1) in;
      layout(set = 0, binding = 0) buffer a { float pa[]; };
      void main() {
          uint index = gl_GlobalInvocationID.x;
          pa[index] = pa[index] + 1;
      })");

	tart::shader_module_ptr shaderModule = device->compileGLSL(shader);
	tart::pipeline_ptr algo = device->createPipeline(shaderModule, "main");

	{
		tart::command_sequence_ptr sequence = device->createSequence();
		sequence->recordPipeline(algo, {tensorA->getSize()}, {tensorA});
		sequence->recordPipeline(algo, {tensorA->getSize()}, {tensorA});
		sequence->recordPipeline(algo, {tensorA->getSize()}, {tensorA});
		device->submitSequence(sequence);
		device->sync();
	}

	CHECK(tensorA->copyOut<float>() == std::vector<float>({ 3, 3, 3 }));
}

TEST_CASE("testing multiple shader execution (3/6)")
{
#if 1
	tart::device_ptr device = getTestDevice();
#else
	tart::Instance instance({"VK_LAYER_KHRONOS_validation"});
	std::vector<std::string> exts;
	tart::Device device = instance.createDevice(TART_TEST_DEVICE_INDEX, exts);
#endif
	std::vector<float> valuesA(3);
	tart::buffer_ptr tensorA = device->allocateBuffer(valuesA);

    std::string shader(R"(
      #version 450
      layout (local_size_x = 1) in;
      layout(set = 0, binding = 0) buffer a { float pa[]; };
      void main() {
          uint index = gl_GlobalInvocationID.x;
          pa[index] = pa[index] + 1;
      })");
    
	tart::shader_module_ptr shaderModule = device->compileGLSL(shader);
	tart::pipeline_ptr algorithm = device->createPipeline(shaderModule, "main");

	algorithm->execute({tensorA->getSize()}, {tensorA});
	algorithm->execute({tensorA->getSize()}, {tensorA});
	algorithm->execute({tensorA->getSize()}, {tensorA});
	
	CHECK(tensorA->copyOut<float>() == std::vector<float>({ 3, 3, 3 }));
}

TEST_CASE("testing multiple shader execution (4/6)")
{
#if 1
	tart::device_ptr device = getTestDevice();
#else
	tart::Instance instance({"VK_LAYER_KHRONOS_validation"});
	std::vector<std::string> exts;
	tart::Device device = instance.createDevice(TART_TEST_DEVICE_INDEX, exts);
#endif
	std::vector<float> valuesA(3);
	tart::buffer_ptr tensorA = device->allocateBuffer(valuesA);
	
    std::string shader(R"(
      #version 450
      layout (local_size_x = 1) in;
      layout(set = 0, binding = 0) buffer a { float pa[]; };
      void main() {
          uint index = gl_GlobalInvocationID.x;
          pa[index] = pa[index] + 1;
      })");

	tart::shader_module_ptr shaderModule = device->compileGLSL(shader);
	tart::pipeline_ptr algorithm = device->createPipeline(shaderModule, "main");

	tart::command_sequence_ptr sq = device->createSequence();
	for (size_t i = 0; i < 3; i += 1)
	{
		sq->recordPipeline(algorithm, {tensorA->getSize()}, {tensorA} );
		device->submitSequence(sq);
		device->sync();
	}
	CHECK(tensorA->copyOut<float>() == std::vector<float>({ 3, 3, 3 }));
}


/*
 * This test is for capability that I don't implement:
 * the repeated submission of a command buffer with exactly the same
 * parameters.
 */
TEST_CASE("testing multiple shader execution (5/6)")
{
#if 1
#else
    kp::Manager mgr;
#endif
#if 1
#else
    std::shared_ptr<kp::TensorT<float>> tensorA = mgr.tensor({ 0, 0, 0 });
#endif
    std::string shader(R"(
      #version 450
      layout (local_size_x = 1) in;
      layout(set = 0, binding = 0) buffer a { float pa[]; };
      void main() {
          uint index = gl_GlobalInvocationID.x;
          pa[index] = pa[index] + 1;
      })");

#if 1
#else
    std::vector<uint32_t> spirv = compileSource(shader);

    std::shared_ptr<kp::Algorithm> algorithm =
      mgr.algorithm({ tensorA }, spirv);
#endif
#if 1
#else
    std::shared_ptr<kp::Sequence> sq = mgr.sequence();

    sq->record<kp::OpSyncDevice>({ tensorA })->eval();

    sq->record<kp::OpAlgoDispatch>(algorithm)->eval()->eval()->eval();

    sq->record<kp::OpSyncLocal>({ tensorA })->eval();

    EXPECT_EQ(tensorA->vector(), std::vector<float>({ 3, 3, 3 }));
#endif
}

TEST_CASE("testing multiple shader execution (6/6)")
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

        layout (local_size_x = 1) in;

        // The input tensors bind index is relative to index in parameter passed
        layout(set = 0, binding = 0) buffer buf_in_a { float in_a[]; };
        layout(set = 0, binding = 1) buffer buf_in_b { float in_b[]; };
        layout(set = 0, binding = 2) buffer buf_out_a { uint out_a[]; };
        layout(set = 0, binding = 3) buffer buf_out_b { uint out_b[]; };

        // Kompute supports push constants updated on dispatch
        layout(push_constant) uniform PushConstants {
            float val;
        } push_const;

        // Kompute also supports spec constants on initalization
        layout(constant_id = 0) const float const_one = 0;

        void main() {
            uint index = gl_GlobalInvocationID.x;
            out_a[index] += uint( in_a[index] * in_b[index] );
            out_b[index] += uint( const_one * push_const.val );
        }
    )");

	std::vector<tart::buffer_ptr> params( {tensorInA, tensorInB, tensorOutA, tensorOutB} );
	// making these uint32_t so it is not as much of a pain to compare them
	struct { uint32_t a = 2; } specConsts;
	struct { uint32_t a = 2; } pushConstsA;
	struct { uint32_t a = 3; } pushConstsB;
	
	tart::shader_module_ptr shaderModule = device->compileGLSL(shader);
	std::vector<uint8_t> pushConstants = tart::packConstants(pushConstsA);
	std::vector<uint8_t> specConstants = tart::packConstants(specConsts);
	tart::pipeline_ptr algorithm = device->createPipeline(shaderModule, "main", specConstants, pushConstants);

	// TODO: change this to reflect new push constant API
	CHECK(algorithm->getSpecializationConstants()[0] == specConsts.a);
	//CHECK( ((uint32_t*)(algorithm->getLastPushConstants().data()))[0] == pushConstsB.a);
}

#endif
