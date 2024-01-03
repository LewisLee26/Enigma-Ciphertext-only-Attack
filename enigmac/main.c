
#include <stdio.h>
// #include "enigma.h"

#include "onnxruntime_c_api.h"

int main() {
   // Initialize ONNX Runtime
   Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ModelRunner");

   // Initialize session options
   Ort::SessionOptions session_options;
   session_options.SetIntraOpNumThreads(1);

   // Load model
   const char* model_path = "path_to_your_model.onnx";
   Ort::Session session(env, model_path, session_options);

   // Prepare input data
   std::vector<float> input_tensor_values = {...}; // replace with your actual data
   std::array<int64_t, 4> input_shape = {1, 3, 224, 224}; // replace with your actual shape
   Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
   Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

   // Run inference
   auto output_tensors = session.Run(Ort::RunOptions{nullptr}, {"input_name"}, {&input_tensor}, 1, {"output_name"});

   return 0;
}