
#include "main_functions.h"
#include "model.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include <esp_heap_caps.h>
#include <esp_timer.h>
#include <esp_log.h>


#define IMG_W 96
#define IMG_H 96
#define IMG_C 3

namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;

  // An area of memory to use for input, output, and intermediate arrays.
  constexpr int kTensorArenaSize = 200 * 1024;
  static uint8_t *tensor_arena;
}  // namespace

void setup(void) 
{
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) 
  {
    MicroPrintf("Model provided is schema version %d not equal to supported "
                "version %d.", model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }
  if (tensor_arena == NULL) 
  {
    tensor_arena = (uint8_t *) malloc(kTensorArenaSize);
  }
  if (tensor_arena == NULL) {
    printf("Couldn't allocate memory of %d bytes\n", kTensorArenaSize);
    return;
  }

  //tflite::AllOpsResolver micro_op_resolver;
  static tflite::MicroMutableOpResolver<9> micro_op_resolver;
  micro_op_resolver.AddAveragePool2D();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddQuantize();
  micro_op_resolver.AddPad();
  micro_op_resolver.AddMean();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddDequantize();

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) 
  {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  input = interpreter->input(0);
}

void loop(uint8_t *buf, size_t len)
{
  int pix_count = 0;
  uint8_t hb, lb;
  pix_count = len / 2;
  for(int i=0; i<pix_count; i++) 
  {
    hb = buf[i];
    lb = buf[i+1];
    input->data.uint8[i] = (lb & 0x1F) << 3;
    input->data.uint8[i+1] = (hb & 0x07) << 5 | (lb & 0xE0) >> 3;
    input->data.uint8[i+2] = hb & 0xF8;
  }

  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != interpreter->Invoke()) {
    MicroPrintf("Invoke failed.");
  }

  TfLiteTensor* output = interpreter->output(0);

  // Process the inference results.
  uint8_t cat_score = output->data.uint8[0];
  uint8_t dog_score = output->data.uint8[1];

  //float cat_score_f = (cat_score - output->params.zero_point) * output->params.scale;
  //float dog_score_f = (dog_score - output->params.zero_point) * output->params.scale;

  if(cat_score > dog_score)
  {
    MicroPrintf(" %d %d CAT", cat_score, dog_score);
  }
  else
  {
    MicroPrintf(" %d %d DOG", cat_score, dog_score);
  }

  vTaskDelay(1); // to avoid watchdog trigger
}

