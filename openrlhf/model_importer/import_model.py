def get_model_class(model_type):
    if model_type == 'qwen1':
        from .qwen.configuration_qwen import QWenConfig as ModelConfig
        from .qwen.modeling_qwen import QWenLMHeadModel as Model
        from .qwen.modeling_qwen import QWenBlock as DecoderLayer
    elif model_type == 'qwen2_sts':
        from .qwen2_sts.configuration_qwen2 import Qwen2Config as ModelConfig
        from .qwen2_sts.modeling_qwen2 import Qwen2ForCausalLM as Model
        from .qwen2_sts.modeling_qwen2 import Qwen2DecoderLayer as DecoderLayer
    elif model_type == 'qwen2':
        from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM as Model
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer as DecoderLayer
        from transformers.models.qwen2.configuration_qwen2 import Qwen2Config as ModelConfig
    elif model_type == 'llama':
        from transformers.models.llama.modeling_llama import LlamaForCausalLM as Model
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer as DecoderLayer
        from transformers.models.llama.configuration_llama import LlamaConfig as ModelConfig
    else:
        raise Exception("未知模型名称")
    return Model, DecoderLayer, ModelConfig