// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
* \brief The entry point for the Inference Engine object_detection demo application
* \file object_detection_demo_ssd_async/main.cpp
* \example object_detection_demo_ssd_async/main.cpp
*/

#include <gflags/gflags.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>

#include <inference_engine.hpp>

#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>

#include "object_detection_demo_ssd_async.hpp"
#ifdef WITH_EXTENSIONS
#include <ext_list.hpp>
#endif

using namespace InferenceEngine;

bool ParseAndCheckCommandLine(int argc, char *argv[])
{
    // ---------------------------Parsing and validation of input args--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h)
    {
        showUsage();
        showAvailableDevices();
        return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_i.empty())
    {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty())
    {
        throw std::logic_error("Parameter -m is not set");
    }

    return true;
}

void frameToBlob(const cv::Mat &frame,
                 InferRequest::Ptr &inferRequest,
                 const std::string &inputName)
{
    if (FLAGS_auto_resize)
    {
        /* Just set input blob containing read image. Resize and layout conversion will be done automatically */
        inferRequest->SetBlob(inputName, wrapMat2Blob(frame));
    }
    else
    {
        /* Resize and copy data from the image to the input blob */
        Blob::Ptr frameBlob = inferRequest->GetBlob(inputName);
        matU8ToBlob<uint8_t>(frame, frameBlob);
    }
}

int main(int argc, char *argv[])
{
    std::vector<std::string> labels;
    labels.push_back("A");
    labels.push_back("B");
    labels.push_back("C");
    labels.push_back("D");
    labels.push_back("E");
    labels.push_back("F");
    labels.push_back("G");
    labels.push_back("H");
    labels.push_back("I");
    labels.push_back("J");
    labels.push_back("K");
    labels.push_back("L");
    labels.push_back("M");
    labels.push_back("N");
    labels.push_back("O");
    labels.push_back("P");
    labels.push_back("Q");
    labels.push_back("R");
    labels.push_back("S");
    labels.push_back("T");
    labels.push_back("U");
    labels.push_back("V");
    labels.push_back("W");
    labels.push_back("X");
    labels.push_back("Y");
    labels.push_back("Z");
    labels.push_back("del");
    labels.push_back("nothing");
    labels.push_back("space");

    try
    {
        /** This demo covers certain topology and cannot be generalized for any object detection **/
        std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv))
        {
            return 0;
        }

        slog::info << "Reading input" << slog::endl;
        cv::VideoCapture cap;
        if (!((FLAGS_i == "cam") ? cap.open(0) : cap.open(FLAGS_i.c_str())))
        {
            throw std::logic_error("Cannot open input file or camera: " + FLAGS_i);
        }
        // read input (video) frame
        cv::Mat curr_frame;
        cap >> curr_frame;
        cv::Mat next_frame;

        if (!cap.grab())
        {
            throw std::logic_error("This demo supports only video (or camera) inputs !!! "
                                   "Failed getting next frame from the " +
                                   FLAGS_i);
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 1. Load inference engine -------------------------------------
        slog::info << "Loading Inference Engine" << slog::endl;
        Core ie;

        slog::info << "Device info: " << slog::endl;
        std::cout << ie.GetVersions(FLAGS_d);

        /** Load extensions for the plugin **/

#ifdef WITH_EXTENSIONS
        /** Loading default extensions **/
        if (FLAGS_d.find("CPU") != std::string::npos)
        {
            /**
             * cpu_extensions library is compiled from "extension" folder containing
             * custom MKLDNNPlugin layer implementations. These layers are not supported
             * by mkldnn, but they can be useful for inferring custom topologies.
            **/
            ie.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>(), "CPU");
        }
#endif

        if (!FLAGS_l.empty())
        {
            // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
            IExtensionPtr extension_ptr = make_so_pointer<IExtension>(FLAGS_l.c_str());
            ie.AddExtension(extension_ptr, "CPU");
        }
        if (!FLAGS_c.empty())
        {
            // clDNN Extensions are loaded from an .xml description and OpenCL kernel files
            ie.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}}, "GPU");
        }

        /** Per layer metrics **/
        if (FLAGS_pc)
        {
            ie.SetConfig({{PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES}});
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 2. Read IR Generated by ModelOptimizer (.xml and .bin files) ------------
        slog::info << "Loading network files" << slog::endl;
        CNNNetReader netReader;
        /** Read network model **/
        netReader.ReadNetwork(FLAGS_m);
        /** Set batch size to 1 **/
        slog::info << "Batch size is forced to  1." << slog::endl;
        netReader.getNetwork().setBatchSize(1);
        /** Extract model name and load it's weights **/
        std::string binFileName = fileNameNoExt(FLAGS_m) + ".bin";
        netReader.ReadWeights(binFileName);
        // -----------------------------------------------------------------------------------------------------

        /** SSD-based network should have one input and one output **/
        // --------------------------- 3. Configure input & output ---------------------------------------------
        // --------------------------- Prepare input blobs -----------------------------------------------------
        slog::info << "Checking that the inputs are as the demo expects" << slog::endl;
        InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());

        std::string imageInputName, imageInfoInputName;
        size_t netInputHeight, netInputWidth;

        for (const auto &inputInfoItem : inputInfo)
        {
            if (inputInfoItem.second->getTensorDesc().getDims().size() == 4)
            { // first input contains images
                imageInputName = inputInfoItem.first;
                inputInfoItem.second->setPrecision(Precision::U8);
                if (FLAGS_auto_resize)
                {
                    inputInfoItem.second->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
                    inputInfoItem.second->getInputData()->setLayout(Layout::NHWC);
                }
                else
                {
                    inputInfoItem.second->getInputData()->setLayout(Layout::NCHW);
                }
                const TensorDesc &inputDesc = inputInfoItem.second->getTensorDesc();
                netInputHeight = getTensorHeight(inputDesc);
                netInputWidth = getTensorWidth(inputDesc);
            }
            else if (inputInfoItem.second->getTensorDesc().getDims().size() == 2)
            { // second input contains image info
                imageInfoInputName = inputInfoItem.first;
                inputInfoItem.second->setPrecision(Precision::FP32);
            }
            else
            {
                throw std::logic_error("Unsupported " +
                                       std::to_string(inputInfoItem.second->getTensorDesc().getDims().size()) + "D "
                                                                                                                "input layer '" +
                                       inputInfoItem.first + "'. "
                                                             "Only 2D and 4D input layers are supported");
            }
        }

        // --------------------------- Prepare output blobs -----------------------------------------------------
        slog::info << "Checking that the outputs are as the demo expects" << slog::endl;
        OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
        if (outputInfo.size() != 1)
        {
            throw std::logic_error("This demo accepts networks having only one output");
        }
        DataPtr &output = outputInfo.begin()->second;
        auto outputName = outputInfo.begin()->first;
        for (auto it = outputInfo.begin(); it != outputInfo.end(); ++it)
        {
            std::cout << it->second->getName() << std::endl;
        }
        //std::cout<<output->getDims();
        const static int num_classes = output->getTensorDesc().getDims()[1];
        const SizeVector outputDims = output->getTensorDesc().getDims();
        for (auto &element : outputDims)
        {
            std::cout << element << std::endl;
        }
        output->setPrecision(Precision::FP32);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 4. Loading model to the device ------------------------------------------
        slog::info << "Loading model to the device" << slog::endl;
        ExecutableNetwork network = ie.LoadNetwork(netReader.getNetwork(), FLAGS_d);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 5. Create infer request -------------------------------------------------
        InferRequest::Ptr async_infer_request_curr = network.CreateInferRequestPtr();
        InferRequest::Ptr async_infer_request_next = network.CreateInferRequestPtr();

        /* it's enough just to set image info input (if used in the model) only once */
        if (!imageInfoInputName.empty())
        {
            auto setImgInfoBlob = [&](const InferRequest::Ptr &inferReq) {
                auto blob = inferReq->GetBlob(imageInfoInputName);
                auto data = blob->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
                data[0] = static_cast<float>(netInputHeight); // height
                data[1] = static_cast<float>(netInputWidth);  // width
                data[2] = 1;
            };
            setImgInfoBlob(async_infer_request_curr);
            setImgInfoBlob(async_infer_request_next);
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 6. Do inference ---------------------------------------------------------
        slog::info << "Start inference " << slog::endl;

        bool isLastFrame = false;
        bool isAsyncMode = false;   // execution is always started using SYNC mode
        bool isModeChanged = false; // set to TRUE when execution mode is changed (SYNC<->ASYNC)

        typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
        auto total_t0 = std::chrono::high_resolution_clock::now();
        //auto wallclock = std::chrono::high_resolution_clock::now();
        //double ocv_decode_time = 0, ocv_render_time = 0;

        std::cout << "To close the application, press 'CTRL+C' here or switch to the output window and press ESC key" << std::endl;
        std::cout << "To switch between sync/async modes, press TAB key in the output window" << std::endl;
        while (true)
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            // Here is the first asynchronous point:
            // in the async mode we capture frame to populate the NEXT infer request
            // in the regular mode we capture frame to the CURRENT infer request
            if (!cap.read(next_frame))
            {
                if (next_frame.empty())
                {
                    isLastFrame = true; // end of video file
                }
                else
                {
                    throw std::logic_error("Failed to get frame from cv::VideoCapture");
                }
            }
            if (isAsyncMode)
            {
                if (isModeChanged)
                {
                    frameToBlob(curr_frame, async_infer_request_curr, imageInputName);
                }
                if (!isLastFrame)
                {
                    frameToBlob(next_frame, async_infer_request_next, imageInputName);
                }
            }
            else if (!isModeChanged)
            {
                frameToBlob(curr_frame, async_infer_request_curr, imageInputName);
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            //ocv_decode_time = std::chrono::duration_cast<ms>(t1 - t0).count();

            t0 = std::chrono::high_resolution_clock::now();
            // Main sync point:
            // in the truly Async mode we start the NEXT infer request, while waiting for the CURRENT to complete
            // in the regular mode we start the CURRENT request and immediately wait for it's completion
            if (isAsyncMode)
            {
                if (isModeChanged)
                {
                    async_infer_request_curr->StartAsync();
                }
                if (!isLastFrame)
                {
                    async_infer_request_next->StartAsync();
                }
            }
            else if (!isModeChanged)
            {
                async_infer_request_curr->StartAsync();
            }

            if (OK == async_infer_request_curr->Wait(IInferRequest::WaitMode::RESULT_READY))
            {
                // ---------------------------Process output blobs--------------------------------------------------
                // Processing results of the CURRENT request
                const float *detections = async_infer_request_curr->GetBlob(outputName)->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
                float max_element = detections[0];
                int max_element_index = 0;
                for (int i = 0; i < num_classes; i++)
                {
                    if (detections[i] > max_element)
                    {
                        max_element = detections[i];
                        max_element_index = i;
                    }
                }
                // now that you have the max confidence and index, find the label
                auto label = labels.at(max_element_index);
                if (!FLAGS_r)
                {
                    std::cout << "confidence: " << max_element << " "
                              << "label: " << label << std::endl;
                }
                cv::putText(curr_frame, label, cv::Point2f(0, 25), cv::FONT_HERSHEY_TRIPLEX, 0.6, cv::Scalar(0, 255, 0));
            }
            if (!FLAGS_no_show)
            {
                cv::imshow("Detection results", curr_frame);
            }

            t1 = std::chrono::high_resolution_clock::now();
            //ocv_render_time = std::chrono::duration_cast<ms>(t1 - t0).count();

            if (isLastFrame)
            {
                break;
            }

            if (isModeChanged)
            {
                isModeChanged = false;
            }

            // Final point:
            // in the truly Async mode we swap the NEXT and CURRENT requests for the next iteration
            curr_frame = next_frame;
            next_frame = cv::Mat();
            if (isAsyncMode)
            {
                async_infer_request_curr.swap(async_infer_request_next);
            }
            if (!FLAGS_no_show)
            {
                const int key = cv::waitKey(1);
                if (27 == key) // Esc
                    break;
                if (cv::waitKey(1) == 'p')
                    while (cv::waitKey(1) != 'p')
                        ;
                if (9 == key)
                { // Tab
                    isAsyncMode ^= true;
                    isModeChanged = true;
                }
            }
        }
        // -----------------------------------------------------------------------------------------------------
        auto total_t1 = std::chrono::high_resolution_clock::now();
        ms total = std::chrono::duration_cast<ms>(total_t1 - total_t0);
        std::cout << "Total Inference time: " << total.count() << std::endl;

        /** Show performace results **/
        if (FLAGS_pc)
        {
            printPerformanceCounts(*async_infer_request_curr, std::cout, getFullDeviceName(ie, FLAGS_d));
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << "[ ERROR ] " << error.what() << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << "[ ERROR ] Unknown/internal exception happened." << std::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;
    return 0;
}
