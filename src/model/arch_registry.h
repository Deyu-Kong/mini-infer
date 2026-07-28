#pragma once

#include <string>

#include "model/model_config.h"

namespace mini_infer {

enum class NormType {
    RMSNorm = 0,
    LayerNorm,
};

enum class PosEncType {
    RoPE = 0,
    ALiBi,
    Learned,
};

struct ArchTraits {
    ModelArch   arch;
    NormType    norm_type       = NormType::RMSNorm;
    PosEncType  pos_enc_type    = PosEncType::RoPE;
    ActKind     default_act     = ActKind::Silu;
    bool        default_qkv_bias = false;
    bool        rmsnorm_add_one  = false;
    bool        embed_scale      = false;
    bool        double_norm_block = false;
};

class ArchRegistry {
public:
    static ArchTraits traits_for(ModelArch arch) {
        switch (arch) {
            case ModelArch::QwenLLaMA:
                return {ModelArch::QwenLLaMA, NormType::RMSNorm,
                        PosEncType::RoPE, ActKind::Silu,
                        false, false, false, false};
            case ModelArch::Gemma:
                return {ModelArch::Gemma, NormType::RMSNorm,
                        PosEncType::RoPE, ActKind::GeluTanh,
                        false, true, true, false};
            case ModelArch::GPT2:
                return {ModelArch::GPT2, NormType::LayerNorm,
                        PosEncType::Learned, ActKind::GeluTanh,
                        true, false, false, false};
            case ModelArch::Bloom:
                return {ModelArch::Bloom, NormType::LayerNorm,
                        PosEncType::ALiBi, ActKind::GeluTanh,
                        true, false, false, false};
        }
        return {ModelArch::QwenLLaMA};
    }

    static ArchTraits traits_for(const ModelConfig& cfg) {
        return traits_for(cfg.arch);
    }

    static bool is_supported(ModelArch arch) {
        return arch == ModelArch::QwenLLaMA || arch == ModelArch::Gemma;
    }
};

}  // namespace mini_infer
