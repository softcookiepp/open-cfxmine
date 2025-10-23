#ifndef TART_TEST_CLSPV
#define TART_TEST_CLSPV
#include "tests-common.hpp"
#include "shaders.hpp"
#include "tart.hpp"

TEST_CASE("testing clspv-compiled shader execution")
{
	std::vector<float> a = {0, 1, 2, 3};
	
	tart::device_ptr device = getTestDevice();
	
	tart::buffer_ptr aBuf = device->allocateBuffer(a);
	tart::buffer_ptr outBuf = device->allocateBuffer(a);
	
	tart::shader_module_ptr module = device->compileCL(test_shaders::cl_test);
	
	std::vector<uint32_t> localSize({1, 1, 1});
	
	tart::pipeline_ptr pipeline = device->createPipeline(module, "square", tart::packConstants(localSize) );
	
	pipeline->execute({4, 1, 1}, {aBuf, outBuf});
	
	const std::vector<float> out = outBuf->copyOut<float>();
	const std::vector<float> expected({0, 1, 4, 9});
	CHECK(out == expected);
}

TEST_CASE("testing CLProgram inteface")
{
	std::vector<float> a = {0, 1, 2, 3};
	
	std::vector<uint32_t> localSize({1, 1, 1});
	
	tart::device_ptr device = getTestDevice();
	
	tart::buffer_ptr aBuf = device->allocateBuffer(a);
	tart::buffer_ptr outBuf = device->allocateBuffer(a);
	
	tart::shader_module_ptr module = device->compileCL(test_shaders::cl_test);
	tart::cl_program_ptr clProgram = device->createCLProgram(module);
	
	tart::command_sequence_ptr sequence = clProgram->dispatch("square", {4, 1, 1}, localSize, {aBuf, outBuf});
	device->sync();
	
	const std::vector<float> out = outBuf->copyOut<float>();
	const std::vector<float> expected({0, 1, 4, 9});
	CHECK(out == expected);
}

#endif
