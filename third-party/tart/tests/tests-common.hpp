#ifndef TART_TESTS_DOCTEST_INCLUDE
#define TART_TESTS_DOCTEST_INCLUDE
#include <chrono>
#include "../doctest/doctest/doctest.h"
#include "tart.hpp"
#include <random>

#ifndef TART_TEST_DEVICE_INDEX
/*
 * Device index used for testing.
 * Will add this as a configurable option to CMakeLists later
 */
#define TART_TEST_DEVICE_INDEX 0
#endif

tart::Instance gTartInstance;

tart::device_ptr getTestDevice()
{
	tart::device_ptr dev = gTartInstance.createDevice(TART_TEST_DEVICE_INDEX);
	return dev;
}

// random number generator for testing purposes
std::random_device rd;
std::mt19937 gGenerator(rd());
std::normal_distribution<> gDist(-1.0, 1.0);

float randn()
{
	return gDist(gGenerator);
}

std::vector<float> randn(uint32_t size)
{
	std::vector<float> v(size);
	for (size_t i = 0; i < v.size(); i += 1)
	{
		v[i] = randn();
	}
	return v;
}

#endif
