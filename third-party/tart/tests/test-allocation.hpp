#ifndef TART_TEST_ALLOCATION
#define TART_TEST_ALLOCATION

#include "tests-common.hpp"
#include "tart.hpp"

TEST_CASE("testing allocation (1/1)")
{
	tart::device_ptr dev = getTestDevice();
	uint64_t size = 1024;
	size *= 1024;
	size *= 1024;
	size *= 8;
	bool result = true;
	try
	{
		tart::buffer_ptr buf = dev->allocateBuffer(size);
		dev->deallocateBuffer(buf);
	}
	catch (const std::exception& ex)
	{
		result = false;
	}
	REQUIRE_FALSE(result);
}

#endif
