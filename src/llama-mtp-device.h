#pragma once
#include "ggml.h"
#include <cstddef>
ggml_tensor llama_mtp_device_rows(const ggml_tensor * tensor, int64_t first, int64_t count);
void llama_mtp_count_device_input(size_t bytes);
