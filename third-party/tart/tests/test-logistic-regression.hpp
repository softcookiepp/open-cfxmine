#ifndef TART_TEST_LOGISTIC_REGRESSION
#define TART_TEST_LOGISTIC_REGRESSION
#include "tests-common.hpp"
#include "shaders.hpp"
#include "tart.hpp"

/*
 * The original test either was not completely implemented, or is somehow
 * missing core components.
 * In any case, it is likely better to temporarily disable it until
 * I get around to refreshing my mind regarding how logistic regression works
 */
#if 0
TEST_CASE("testing logistic regression (1/2)")
{

    uint32_t ITERATIONS = 100;
    float learningRate = 0.1;

    {
#if 1
		tart::Instance instance({"VK_LAYER_KHRONOS_validation"});
		tart::Device device = instance.createDevice(TART_TEST_DEVICE_INDEX);
#else
        kp::Manager mgr;
#endif
#if 1
		std::vector<float> xIvector({ 0, 1, 1, 1, 1 });
		tart::buffer_ptr xI = device.allocateBuffer(xIvector);
		
		std::vector<float> xJvector({ 0, 0, 0, 1, 1 });
        tart::buffer_ptr xJ = device.allocateBuffer(xJvector);

        std::vector<float> yVector({ 0, 0, 0, 1, 1 });
        tart::buffer_ptr y = device.allocateBuffer(yVector);

        std::vector<float> wInVector({ 0.001, 0.001 });
        tart::buffer_ptr wIn = device.allocateBuffer(wInVector);
        
        std::vector<float> wOutIvector({ 0, 0, 0, 0, 0 });
        tart::buffer_ptr wOutI = device.allocateBuffer(wOutIvector);
        
        std::vector<float> wOutJvector({ 0, 0, 0, 0, 0 });
        tart::buffer_ptr wOutJ = device.allocateBuffer(wOutJvector);

        std::vector<float> bInVector({0});
        tart::buffer_ptr bIn = device.allocateBuffer(bInVector);
        
        std::vector<float> bOutVector({ 0, 0, 0, 0, 0 });
        tart::buffer_ptr bOut = device.allocateBuffer(bOutVector);

        std::vector<float> lOutVector({ 0, 0, 0, 0, 0 });
        tart::buffer_ptr lOut = device.allocateBuffer(lOutVector);

        std::vector<tart::buffer_ptr> params = { xI, xJ, y, wIn, wOutI, wOutJ,
                                                            bIn, bOut,  lOut };
#else
        std::shared_ptr<kp::TensorT<float>> xI = mgr.tensor({ 0, 1, 1, 1, 1 });
        std::shared_ptr<kp::TensorT<float>> xJ = mgr.tensor({ 0, 0, 0, 1, 1 });

        std::shared_ptr<kp::TensorT<float>> y = mgr.tensor({ 0, 0, 0, 1, 1 });

        std::shared_ptr<kp::TensorT<float>> wIn = mgr.tensor({ 0.001, 0.001 });
        std::shared_ptr<kp::TensorT<float>> wOutI =
          mgr.tensor({ 0, 0, 0, 0, 0 });
        std::shared_ptr<kp::TensorT<float>> wOutJ =
          mgr.tensor({ 0, 0, 0, 0, 0 });

        std::shared_ptr<kp::TensorT<float>> bIn = mgr.tensor({ 0 });
        std::shared_ptr<kp::TensorT<float>> bOut =
          mgr.tensor({ 0, 0, 0, 0, 0 });

        std::shared_ptr<kp::TensorT<float>> lOut =
          mgr.tensor({ 0, 0, 0, 0, 0 });

        std::vector<std::shared_ptr<kp::Memory>> params = { xI,  xJ,    y,
                                                            wIn, wOutI, wOutJ,
                                                            bIn, bOut,  lOut };
#endif
#if 0
        mgr.sequence()->eval<kp::OpSyncDevice>(params);
#endif
#if 1
#else
        std::vector<uint32_t> spirv(
          kp::TEST_LOGISTIC_REGRESSION_SHADER_COMP_SPV.begin(),
          kp::TEST_LOGISTIC_REGRESSION_SHADER_COMP_SPV.end());
#endif
        std::shared_ptr<kp::Algorithm> algorithm = mgr.algorithm(
          params, spirv, kp::Workgroup({ 5 }), std::vector<float>({ 5.0 }));

        std::shared_ptr<kp::Sequence> sq =
          mgr.sequence()
            ->record<kp::OpSyncDevice>({ wIn, bIn })
            ->record<kp::OpAlgoDispatch>(algorithm)
            ->record<kp::OpSyncLocal>({ wOutI, wOutJ, bOut, lOut });

        // Iterate across all expected iterations
        for (size_t i = 0; i < ITERATIONS; i++) {
            sq->eval();

            for (size_t j = 0; j < bOut->size(); j++) {
                wIn->data()[0] -= learningRate * wOutI->data()[j];
                wIn->data()[1] -= learningRate * wOutJ->data()[j];
                bIn->data()[0] -= learningRate * bOut->data()[j];
            }
        }

        // Based on the inputs the outputs should be at least:
        // * wi < 0.01
        // * wj > 1.0
        // * b < 0
        // TODO: Add EXPECT_DOUBLE_EQ instead
        EXPECT_LT(wIn->data()[0], 0.01);
        EXPECT_GT(wIn->data()[1], 1.0);
        EXPECT_LT(bIn->data()[0], 0.0);

        KP_LOG_WARN("Result wIn i: {}, wIn j: {}, bIn: {}",
                    wIn->data()[0],
                    wIn->data()[1],
                    bIn->data()[0]);
    }
}
#endif

// end
#endif
