#ifndef TART_TEST_SEMAPHORES
#define TART_TEST_SEMAPHORES
#include "tests-common.hpp"
#include "shaders.hpp"
#include "tart.hpp"

std::vector<float> semaphoreTestReference(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c)
{
	// a, b, c must be same size
	std::vector<float> out(a.size());
	
	for (size_t i = 0; i < a.size(); i += 1)
	{
		out[i] = (a[i]*b[i]) + c[i];
	}
	return out;
}

std::vector<float> semaphoreTestTart(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c)
{
	tart::device_ptr device = getTestDevice();

	uint32_t buf_size = (uint32_t)(a.size()) * sizeof(float);
	
	tart::buffer_ptr bufA = device->allocateBuffer(buf_size);
	bufA->copyIn(a);
	tart::buffer_ptr bufB = device->allocateBuffer(buf_size);
	bufB->copyIn(b);
	tart::buffer_ptr bufC = device->allocateBuffer(buf_size);
	bufC->copyIn(c);
	
	tart::buffer_ptr bufOut = device->allocateBuffer(buf_size);

	tart::shader_module_ptr mulShader = device->compileGLSL(test_shaders::op_mul);
	tart::shader_module_ptr addShader = device->compileGLSL(test_shaders::op_add);
	
	tart::pipeline_ptr mulProgram = device->createPipeline(mulShader, "main");
	tart::pipeline_ptr addProgram = device->createPipeline(addShader, "main");
	
	tart::command_sequence_ptr sequence0 = device->createSequence();
	sequence0->recordPipeline(mulProgram, {a.size(), 1, 1}, {bufA, bufB, bufOut});
	
	tart::command_sequence_ptr sequence1 = device->createSequence();
	sequence1->recordPipeline(addProgram, {a.size(), 1, 1}, {bufOut, bufC, bufOut});
	
	// well it turns out that submission in reverse order causes the validation layers to screech at you.
	// I will have to modify my error handling to take account of that!
	device->submitSequence(sequence0);
	device->submitSequence(sequence1);
	
	// sync
	device->sync();
	
	// now copy bufOut to host
	return bufOut->copyOut<float>();
}

TEST_CASE("testing semaphore-based synchronization")
{
	std::vector<float> a = {0, 1, 2, 3};
	std::vector<float> b = {-1.4, -2.1, 3.9, 40.05};
	std::vector<float> c = {10, 5, 2.5, 12.25};
	
	std::vector<float> outReference = semaphoreTestReference(a, b, c);
	std::vector<float> outTart = semaphoreTestTart(a, b, c);
	
	CHECK(outReference == outTart);
}

#endif
