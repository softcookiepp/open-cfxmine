#ifndef TART_TEST_DESTROY
#define TART_TEST_DESTROY

#include "tests-common.hpp"
#include "shaders.hpp"
#include "tart.hpp"


TEST_CASE("testing destruction (1/3)")
{

	tart::buffer_ptr tensorA = nullptr;

    std::string shader(R"(
      #version 450
      layout (local_size_x = 1) in;
      layout(set = 0, binding = 0) buffer a { float pa[]; };
      void main() {
          uint index = gl_GlobalInvocationID.x;
          pa[index] = pa[index] + 1;
      })");

    {

		tart::command_sequence_ptr sq = nullptr;

        {
			tart::device_ptr device = getTestDevice();

            const std::vector<float> initialValues = { 0.0f, 0.0f, 0.0f };

			tensorA = device->allocateBuffer(initialValues);

			// create shader module and executor
			tart::shader_module_ptr shaderModule = device->compileGLSL(shader);
			tart::pipeline_ptr algo = device->createPipeline(shaderModule, "main");

			CHECK(tensorA->copyOut<float>() == initialValues);
			device->sync(); // lets seee....
			device->dispatchPipeline(algo, {tensorA->getSize(), 1, 1}, {tensorA});
			device->sync();


            const std::vector<float> expectedFinalValues = { 1.0f, 1.0f, 1.0f };

			CHECK(tensorA->copyOut<float>() == expectedFinalValues);

			device->deallocateBuffer(tensorA);
			CHECK( tensorA->isDestroyed() );

        }
		CHECK( tensorA->isDestroyed() );
    }
}

TEST_CASE("testing destruction (2/3)")
{

	tart::buffer_ptr tensorA = nullptr;
	tart::buffer_ptr tensorB = nullptr;

    std::string shader(R"(
      #version 450
      layout (local_size_x = 1) in;
      layout(set = 0, binding = 0) buffer a { float pa[]; };
      layout(set = 0, binding = 1) buffer b { float pb[]; };
      void main() {
          uint index = gl_GlobalInvocationID.x;
          pa[index] = pa[index] + 1;
          pb[index] = pb[index] + 2;
      })");

    {
		tart::command_sequence_ptr sq = nullptr;

        {

#if 1
			tart::device_ptr device = getTestDevice();
#else
			tart::Instance instance({"VK_LAYER_KHRONOS_validation"});
			std::vector<std::string> exts;
			tart::Device device = instance.createDevice(TART_TEST_DEVICE_INDEX, exts);
#endif

			std::vector<float> initialValues = { 1, 1, 1 };
			tensorA = device->allocateBuffer(initialValues);
			tensorB = device->allocateBuffer(initialValues);

			tart::shader_module_ptr shaderModule = device->compileGLSL(shader);
			tart::pipeline_ptr algo = device->createPipeline(shaderModule, "main");

			sq = device->dispatchPipeline(algo, {tensorA->getSize(), 1, 1}, {tensorA, tensorB});
			device->sync();

			CHECK(tensorA->copyOut<float>() == std::vector<float>({ 2, 2, 2 }));
			CHECK(tensorB->copyOut<float>() == std::vector<float>({ 3, 3, 3 }));

			device->deallocateBuffer(tensorA);
			device->deallocateBuffer(tensorB);
			CHECK(tensorA->isDestroyed());
			CHECK(tensorB->isDestroyed());
        }
    }
}

TEST_CASE("testing destruction (3/3)")
{
	tart::buffer_ptr tensorA = nullptr;

    std::string shader(R"(
      #version 450
      layout (local_size_x = 1) in;
      layout(set = 0, binding = 0) buffer a { float pa[]; };
      void main() {
          uint index = gl_GlobalInvocationID.x;
          pa[index] = pa[index] + 1;
      })");

    {
		tart::command_sequence_ptr sq = nullptr;
        {
#if 1
			tart::device_ptr device = getTestDevice();
#else
			tart::Instance instance({"VK_LAYER_KHRONOS_validation"});
			std::vector<std::string> exts;
			tart::Device device = instance.createDevice(TART_TEST_DEVICE_INDEX, exts);
#endif
			std::vector<float> initValues({ 0, 0, 0 });
			tensorA = device->allocateBuffer(initValues);


			tart::shader_module_ptr shaderModule = device->compileGLSL(shader);
			tart::pipeline_ptr executor = device->createPipeline(shaderModule, "main");
			device->dispatchPipeline(executor, {tensorA->getSize()}, {tensorA});
			device->sync();


			device->destroySequence(sq);
			CHECK(!sq );

			CHECK(tensorA->copyOut<float>() == std::vector<float>({ 1, 1, 1 }));
        }
    }
}

/*
 * there are sadly unresolved driver bugs related to repeated creation and destruction.
 * Therefore, this test shall be temporarily disabled until a workaround is implemented.
 */
#if 0
TEST_CASE("testing repeated destruction")
{
	for (size_t i = 0; i < 100; i += 1)
	{
		{
		#if 1
			tart::device_ptr device = getTestDevice();
		#else
			tart::Instance instance({"VK_LAYER_KHRONOS_validation"});
			std::vector<std::string> exts;
			tart::Device device = instance.createDevice(TART_TEST_DEVICE_INDEX, exts);
		#endif
			std::vector<float> initValues({ 0, 0, 0 });
			tart::buffer_ptr tensorA = device->allocateBuffer(initValues);
		}
	}
}
#endif

// definition
#endif
