#pragma once

#include <cstdint>

namespace mini_infer {

// IEEE-754 binary32 -> binary16 (host). Round-to-nearest-even.
uint16_t f32_to_f16_bits(float f);

}  // namespace mini_infer