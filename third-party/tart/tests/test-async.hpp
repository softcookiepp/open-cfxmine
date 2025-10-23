#ifndef TART_TEST_ASYNC
#define TART_TEST_ASYNC
#include "shaders.hpp"
#include "tart.hpp"
#include "tests-common.hpp"

TEST_CASE("testing async operations (1/2)")
{
	uint32_t size = 10;

	std::vector<float> data(size, 0.0);
	std::vector<float> resultAsync(size, 1000);

	tart::device_ptr device = getTestDevice();
	
	tart::buffer_ptr tensorA = device->allocateBuffer(data);
	tart::buffer_ptr tensorB = device->allocateBuffer(data);

	tart::command_sequence_ptr sq1 = device->createSequence();
	tart::command_sequence_ptr sq2 = device->createSequence();
	
	tart::shader_module_ptr shaderModule = device->compileGLSL(test_shaders::asyncTestShader);
	
	// do we even need 2 of these?
	// I think not!
	tart::pipeline_ptr algo = device->createPipeline(shaderModule, "main");
	
	// begin here!
	auto startSync = std::chrono::high_resolution_clock::now();

	sq1->recordPipeline(algo, {tensorA->getSize(), 1, 1}, {tensorA} );
	sq2->recordPipeline(algo, {tensorB->getSize(), 1, 1}, {tensorB} );
	
	device->submitSequence(sq1);
	device->submitSequence(sq2);

	device->sync();

	auto endSync = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endSync - startSync).count();
	std::cout << "duration: " << duration << std::endl;

	/*
	 * The Kompute test I adapted this from made assumptions about hardware speed
	 * and GPU driver behavior.
	 * If running the test on software implementations like lavapipe or swiftshader for example,
	 * the test would fail because it was too slow.
	 * Thus I only print out the duration instead of using it in pass/fail evaluation.
	 */
	INFO(duration);
	CHECK(tensorA->copyOut<float>() == resultAsync);
	CHECK(tensorB->copyOut<float>() == resultAsync);
}

TEST_CASE("testing async operations (2/2)")
{
	uint32_t size = 10;

	uint32_t numParallel = 2;
	
	tart::device_ptr device = getTestDevice();

	tart::shader_module_ptr shaderModule = device->compileGLSL(test_shaders::asyncTestShader);
	tart::pipeline_ptr pipeline = device->createPipeline(shaderModule, "main");

	std::vector<float> data(size, 0.0);
	std::vector<float> resultSync(size, 1000);
	std::vector<float> resultAsync(size, 1000);

	std::vector<tart::buffer_ptr> inputsSyncB;

	for (uint32_t i = 0; i < numParallel; i++)
	{
		inputsSyncB.push_back(device->allocateBuffer(data));
	}

	auto startSync = std::chrono::high_resolution_clock::now();

	for (uint32_t i = 0; i < numParallel; i++)
	{
		// no need to wait for anyone else!
		device->dispatchPipeline(pipeline, {inputsSyncB[i]->getSize(), 1, 1}, {inputsSyncB[i]} );
	}

	auto endSync = std::chrono::high_resolution_clock::now();
	auto durationSync =
	  std::chrono::duration_cast<std::chrono::microseconds>(endSync - startSync)
		.count();
	device->sync();

	for (uint32_t i = 0; i < numParallel; i++)
	{
		CHECK(inputsSyncB[i]->copyOut<float>() == resultSync);
	}
}


#endif
