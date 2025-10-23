#ifndef TART_TEST_VIEW
#define TART_TEST_VIEW

#include "tests-common.hpp"
#include "shaders.hpp"
#include "tart.hpp"

// test_shaders::op_add
TEST_CASE("testing view creation and use (1/1)")
{
	tart::device_ptr dev = getTestDevice();
	
	std::vector<float> initValues({
		1.2, 2.4, 3.5,
		0.0, 0.0, 0.0,
		0.0, 0.0, 0.0
	});
	// this is what we expect if the views were sliced and diced correctly
	std::vector<float> expectedValues({
		1.2, 2.4, 3.5,
		3.1, 9.0,-7.8,
		4.3,11.4,-4.3
	});
	tart::buffer_ptr buf = dev->allocateBuffer(initValues);
	tart::shader_module_ptr module = dev->compileGLSL(test_shaders::op_add);
	tart::pipeline_ptr pipeline = dev->createPipeline(module, "main");
	
	// make views
	tart::buffer_ptr a = buf->view(0);
	tart::buffer_ptr b = buf->view(3*4);
	tart::buffer_ptr c = buf->view(6*4);
	
	// copy missing data to view b
	std::vector<float> bData({3.1, 9.0,-7.8});
	b->copyIn(bData);
	
	// verify result
	std::vector<float> bResult = b->copyOut<float>(3*4);
	CHECK(bData == bResult);
	
	pipeline->dispatch({3, 1, 1}, {a, b, c});
	dev->sync();
	
	std::vector<float> result = buf->copyOut<float>();
	CHECK(result == expectedValues);
}

#endif
