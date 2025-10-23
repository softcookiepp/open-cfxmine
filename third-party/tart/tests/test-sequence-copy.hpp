#ifndef TART_TEST_SEQUENCE_COPY
#define TART_TEST_SEQUENCE_COPY

#include "tests-common.hpp"
#include "shaders.hpp"
#include "tart.hpp"

// test_shaders::op_add
TEST_CASE("testing sequence-based buffer copying (1/1)")
{
	tart::device_ptr dev = getTestDevice();
	
	// values to copy
	std::vector<float> initValues({
		1.2, 2.4, 3.5,
		3.1, 9.0,-7.8,
		4.3,11.4,-4.3
	});
	
	tart::buffer_ptr src = dev->allocateBuffer(initValues);
	tart::buffer_ptr dst = dev->allocateBuffer(initValues.size()*sizeof(float));
	
	tart::command_sequence_ptr sequence = dev->createSequence();
	sequence->recordCopyBuffer(dst, src);
	dev->submitSequence(sequence);
	dev->sync();
	
	std::vector<float> result = dst->copyOut<float>();
	CHECK(result == initValues);
}

#endif
