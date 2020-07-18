import os

import torch

from pytorch_pretrained_bert.optimization import BertAdam
from transformers import XLNetTokenizer
# updated with relevant files
# from pytorch_pretrained_bert.tokenization_xlnet import (
#     PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES, VOCAB_FILES_NAMES,
# )
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "xlnet-base-cased": None,
    "xlnet-large-cased": None,
}
VOCAB_FILES_NAMES = {"vocab_file": "spiece.model"}

TF_PYTORCH_XLNET_NAME_MAP = {
    "xlnet-large-cased": "xlnet-large-cased",
    "xlnet-base-cased": "xlnet-base-cased",
}
XLNET_ALL_DIR = './'
# change variable name


def get_xlnet_config_path(xlnet_model_name):
    # change to handle xlnet all dir path 

    return os.path.join(XLNET_ALL_DIR, TF_PYTORCH_XLNET_NAME_MAP[xlnet_model_name])


def load_overall_state(xlnet_load_path, relaxed=True):
    if xlnet_load_path is None:
        if relaxed:
            return None
        else:
            raise RuntimeError("Need 'xlnet_load_path'")
    else:
        return torch.load(xlnet_load_path)


def create_tokenizer(xlnet_model_name, xlnet_load_mode, do_lower_case, xlnet_vocab_path=None):
    if xlnet_load_mode == "from_pretrained":
        assert xlnet_vocab_path is None
        tokenizer = XLNetTokenizer.from_pretrained(
            xlnet_model_name, do_lower_case=do_lower_case)
    elif xlnet_load_mode in ["model_only", "state_model_only", "state_all", "state_full_model",
                             "full_model_only",
                             "state_adapter"]:
        print("xlnet vocab-path: ",xlnet_vocab_path), 
        tokenizer = load_tokenizer(
            xlnet_model_name=xlnet_model_name,
            do_lower_case=do_lower_case,
            xlnet_vocab_path=xlnet_vocab_path,
        )
    else:
        raise KeyError(xlnet_load_mode)
    return tokenizer


def load_tokenizer(xlnet_model_name, do_lower_case, xlnet_vocab_path=None):
    if xlnet_vocab_path is None:
        print("checking again: ", xlnet_model_name, VOCAB_FILES_NAMES)
        xlnet_vocab_path = os.path.join(
            get_xlnet_config_path(xlnet_model_name), VOCAB_FILES_NAMES["vocab_file"])
    # max_len = min(
    #     PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES[xlnet_model_name], int(1e12))
    # get rid of max length, not sure what happen
    tokenizer = XLNetTokenizer(
        vocab_file=xlnet_vocab_path,
        do_lower_case=do_lower_case,
        # max_len=max_len,
    )
    return tokenizer


def get_opt_train_steps(num_train_examples, args):
    num_train_steps = int(
        num_train_examples
        / args.train_batch_size
        / args.gradient_accumulation_steps
        * args.num_train_epochs,
    )
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    return t_total


def create_optimizer(model, learning_rate, t_total, loss_scale, fp16, warmup_proportion, state_dict):
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = [
        'bias', 'LayerNorm.bias', 'LayerNorm.weight',
        'adapter.down_project.weight', 'adapter.up_project.weight',
    ]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex "
                              "to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=learning_rate,
                             warmup=warmup_proportion,
                             t_total=t_total)

    if state_dict is not None:
        optimizer.load_state_dict(state_dict)
    return optimizer


def stage_model(model, fp16, device, local_rank, n_gpu):
    if fp16:
        model.half()
    model.to(device)
    if local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex "
                              "to use distributed and fp16 training.")
        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
    return model


def get_tunable_state_dict(model, verbose=True):
    # Drop non-trainable params
    # Sort of a hack, because it's not really clear when we want/don't want state params,
    #   But for now, layer norm works in our favor. But this will be annoying.
    model_state_dict = model.state_dict()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            if verbose:
                print("    Skip {}".format(name))
            del model_state_dict[name]
    return model_state_dict
