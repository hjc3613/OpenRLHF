from typing import Optional, Tuple, Union
import json
import os
import inspect

from torch import distributed as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
import transformers
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.deepspeed import HfDeepSpeedConfig

from .utils import log_probs_from_logits, freeze_transformer_layers_for_qwen_new, parse_freeze_strategy
from openrlhf.model_importer import get_model_class


def load_model_class(model_path, decoder_layer_name=None):
    # 1. 获取模型配置
    # config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    # 2. 从配置文件中读取 auto_map
    config_file = os.path.join(model_path, "config.json")
    with open(config_file, 'r') as f:
        config_dict = json.load(f)
    
    auto_map = config_dict.get("auto_map", {})
    architectures = config_dict.get("architectures", [])
    # 3. 获取 AutoModelForCausalLM 对应的类名
    model_class_path = auto_map.get("AutoModelForCausalLM")
    if model_class_path:
        # module_name, class_name = model_class_path.rsplit('.', 1)
        model_class = get_class_from_dynamic_module(
                    model_class_path, model_path,
                )
        model_class.register_for_auto_class()
        if decoder_layer_name is not None:
            decoder_class_path = model_class_path.rsplit('.', 1)[0]+'.'+decoder_layer_name
            decoder_class = get_class_from_dynamic_module(
                decoder_class_path, model_path
            )
        else:
            decoder_class = None
    else:
        model_class = getattr(transformers, architectures[0])
        if decoder_layer_name is not None:
            model_file = inspect.getfile(model_class).strip('.py').split('/')+[decoder_layer_name]
            transformers_idx = model_file.index('transformers')
            start_module = transformers
            for idx, submodel in enumerate(model_file[transformers_idx+1:]):
                start_module = getattr(start_module, submodel)
            decoder_class = start_module
        else:
            decoder_class = None
    return model_class, decoder_class

def load_huggingface_model(path, device_map, nf4_config, bf16, model_type, low_cpu=True):
    Model, DecoderLayer, ModelConfig = get_model_class(model_type)
    if 'qwen1' in model_type:
        flash_attn_args = {
                'use_flash_attn':True
            }
    else:
        flash_attn_args = {
                'attn_implementation':'flash_attention_2'
            }
    if low_cpu:
        if dist.get_rank() == 0:
            print(f'process: {torch.cuda.current_device()} load from disk to cpu')
            model = Model.from_pretrained(
                path,
                device_map=device_map,
                use_cache=False,
                trust_remote_code=True,
                # quantization_config=nf4_config,
                torch_dtype=torch.bfloat16 if bf16 else "auto",
                **flash_attn_args
            )
        else:
            config = ModelConfig.from_pretrained(
                path, 
                torch_dtype=torch.bfloat16 if bf16 else "auto", 
                trust_remote_code=True,
                use_cache=False, 
                # quantization_config=nf4_config,
                **flash_attn_args
            )
            print(f'process: {torch.cuda.current_device()} load from config to meta device')
            with torch.device('meta'):
                model = Model(config)
    else:
        print(f'each process load model from disk,process: {torch.cuda.current_device()}')
        model = Model.from_pretrained(
                path,
                trust_remote_code=True,
                quantization_config=nf4_config,
                torch_dtype=torch.bfloat16 if bf16 else "auto",
                device_map=device_map,
                **flash_attn_args
        )

    return model

class Actor(nn.Module):
    """
    Actor model base class.

    Args:
        model (nn.Module): Actor Model.
        lora_rank (int): LoRA rank.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(
        self,
        pretrain_or_model,
        use_flash_attention_2=False,
        bf16=True,
        load_in_4bit=False,
        lora_rank=0,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=None,
        ds_config=None,
        device_map=None,
        freeze_strategy=None,
        transformer_layers_path=None,
        low_cpu=False,
        model_type=None,
        is_ref=False,
        **kwargs,
    ) -> None:
        super().__init__()

        if isinstance(pretrain_or_model, str):
            attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

            # Note: dschf is defined in function scope to avoid global effects
            # https://huggingface.co/docs/transformers/deepspeed#non-trainer-deepspeed-integration
            if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
                dschf = HfDeepSpeedConfig(ds_config)
            else:
                dschf = None

            if load_in_4bit:
                assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                nf4_config = None

            # self.model = AutoModelForCausalLM.from_pretrained(
            #     pretrain_or_model,
            #     trust_remote_code=True,
            #     attn_implementation=attn_implementation,
            #     quantization_config=nf4_config,
            #     torch_dtype=torch.bfloat16 if bf16 else "auto",
            #     device_map=device_map,
            # )
            
            self.model = load_huggingface_model(
                pretrain_or_model, 
                device_map=device_map, 
                nf4_config=nf4_config, 
                bf16=bf16, 
                low_cpu=low_cpu,
                model_type=model_type
            )
            if freeze_strategy and not is_ref:
                assert lora_rank <=0, "冻结模式与LORA不能同时开启"
                assert transformer_layers_path is not None, "开启冻结模式，需提供transformer_layers_path"
                activation, layers = parse_freeze_strategy(freeze_strategy)
                freeze_transformer_layers_for_qwen_new(self.model, layers=layers, layer_path=transformer_layers_path, action=activation)

            # LoRA
            if lora_rank > 0:
                assert freeze_strategy is not None,"冻结模式与LORA不能同时开启"
                # https://github.com/huggingface/peft/issues/137
                self.model.enable_input_require_grads()
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                )
                self.model = get_peft_model(self.model, lora_config)

                if load_in_4bit:
                    for name, module in self.model.named_modules():
                        if isinstance(module, LoraLayer):
                            module = module.to(torch.bfloat16)
                        if "norm" in name:
                            module = module.to(torch.float32)
                        if "lm_head" in name or "embed_tokens" in name:
                            if hasattr(module, "weight"):
                                module = module.to(torch.bfloat16)

            # MoE - balancing loss
            model_config = self.model.config.to_dict()
            if "output_router_logits" in model_config:
                print("[MoE] set output_router_logits as True")
                self.model.config.output_router_logits = True

            # https://github.com/huggingface/transformers/issues/26877
            # Use `model.generate(use_cache=True)` instead.`
            self.model.config.use_cache = False
        else:
            self.model = pretrain_or_model

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, **kwargs) -> Union[
        Tuple[torch.LongTensor, torch.LongTensor],
        Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor],
    ]:
        generate_args = {
            "input_ids": input_ids,
            "top_k": kwargs.get("top_k", None),
            "top_p": kwargs.get("top_p", None),
            "do_sample": kwargs.get("do_sample", True),
            "early_stopping": True,
            "temperature": kwargs.get("temperature", 1),
            "use_cache": True,
            "num_beams": kwargs.get("num_beams", 1),
            "attention_mask": kwargs.get("attention_mask"),
            "eos_token_id": kwargs.get("eos_token_id"),
            "pad_token_id": kwargs.get("pad_token_id"),
            "min_new_tokens": kwargs.get("min_new_tokens", 1),
        }

        if kwargs.get("max_new_tokens", None):
            generate_args["max_new_tokens"] = kwargs.get("max_new_tokens")
        if kwargs.get("max_length", None):
            generate_args["max_length"] = kwargs.get("max_length")

        # Call generate
        sequences = self.model.generate(**generate_args)

        # Prepare mask tensor
        eos_token_id = generate_args["eos_token_id"]
        pad_token_id = generate_args["pad_token_id"]

        return self.process_sequences(sequences, input_ids.size(1), eos_token_id, pad_token_id)

    def process_sequences(self, sequences: torch.Tensor, input_len, eos_token_id, pad_token_id):
        attention_mask = (sequences.ne(eos_token_id) & sequences.ne(pad_token_id)).to(dtype=torch.long)
        seq_length = attention_mask.size(1)

        # The following code is equivalent to:
        #
        # for i in range(attention_mask.size(0)):
        #     for t in reversed(range(seq_length)):
        #         if attention_mask[i][t] > 0.5:
        #             attention_mask[i][min(t + 1, seq_length - 1)] = True
        #             sequences[i][min(t + 1, seq_length - 1)] = eos_token_id
        #             break
        #
        eos_indices = seq_length - attention_mask.long().fliplr().argmax(dim=1, keepdim=True).clamp(min=1)
        sequences.scatter_(dim=1, index=eos_indices, value=eos_token_id)

        # For Llama3 and Qwen2 models, there are some eos_tokens in the middle of the prompt.
        first_token_indices = attention_mask.long().argmax(dim=1, keepdim=True)
        mask = torch.arange(seq_length).unsqueeze(0).expand(sequences.size(0), -1).to(device=sequences.device)
        attention_mask = (mask >= first_token_indices) & (mask <= eos_indices).to(dtype=torch.long)

        # in RL, state_i (current token) + action_i (next token) -> state_i+1 (next token)
        state_seq = sequences[:, input_len - 1 : -1]
        action_mask = state_seq.ne(eos_token_id) & state_seq.ne(pad_token_id)
        action_mask[:, 0] = 1

        return sequences, attention_mask, action_mask

    def forward(
        self,
        sequences: torch.LongTensor,
        num_actions: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
        packing_samples=False,
    ) -> torch.Tensor:
        """Returns action log probs"""
        if not packing_samples:
            # https://github.com/OpenLLMAI/OpenRLHF/issues/217
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        else:
            position_ids = None

        output = self.model(sequences, attention_mask=attention_mask, position_ids=position_ids)
        log_probs = log_probs_from_logits(output["logits"][:, :-1, :], sequences[:, 1:])

        if return_output:
            return output if num_actions is None else (log_probs[:, -num_actions:], output)
        else:
            return log_probs[:, -num_actions:]

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={"use_reentrant": False}):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()
