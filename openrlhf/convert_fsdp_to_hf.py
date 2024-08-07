import sys
sys.path.append('.')
from torch.distributed._shard.checkpoint import FileSystemReader
import torch.distributed._shard.checkpoint as dist_cp
from openrlhf.model_importer import get_model_class
from transformers import AutoTokenizer

def load_sharded_model_single_gpu(model,model_path):
    
    reader = FileSystemReader(model_path)
    
    state_dict = {
        "model": model.state_dict()
    }
    
    dist_cp.load_state_dict(
                state_dict=state_dict,
                storage_reader= FileSystemReader(model_path),
                no_dist=True,
            )
    
    model.load_state_dict(state_dict["model"])
    
    print(f"Sharded state checkpoint loaded from {model_path}")
    return model
def convert_fsdp_to_hf(args):
    Model, _, _ =  get_model_class(args.model_type)
    model = Model.from_pretrained(args.pretrain_path)
    print("model is loaded from pretrain")
    load_sharded_model_single_gpu(model, args.fsdp_path)
    print("model is loaded from FSDP checkpoints")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_path,trust_remote_code=True)
    tokenizer.save_pretrained(args.consolidated_model_path)
    model.save_pretrained(args.consolidated_model_path)
    print(f"HuggingFace model checkpoints has been saved in {args.consolidated_model_path}")
    ...

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--model_type', required=True, help='qwen1, qwen2, llama, mixtral...')
    parser.add_argument('--pretrain_path', required=True, help='base pretrained path')
    parser.add_argument('--fsdp_path', required=True, help='fsdp ckpt path')
    parser.add_argument('--consolidated_model_path', required=True, help='consolidated model path')
    args = parser.parse_args()
    convert_fsdp_to_hf(args)