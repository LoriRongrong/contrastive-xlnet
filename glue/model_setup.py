import os
import torch
# just a class for model initiation
# from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from transformers import XLNetModel, XLNetForSequenceClassification
# TODO need some further clarification
# have big trouble understanding what it is
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

import pytorch_pretrained_bert.utils as utils
# TODO check if it is bert specific
from shared.model_setup import stage_model, get_xlnet_config_path, get_tunable_state_dict
from glue.tasks import TaskType


def create_model(task_type, xlnet_model_name, xlnet_load_mode, xlnet_load_args,
                 all_state,
                 num_labels, device, n_gpu, fp16, local_rank,
                 xlnet_config_json_path=None):
    if xlnet_load_mode == "from_pretrained":
        assert xlnet_load_args is None
        assert all_state is None
        assert xlnet_config_json_path is None
        cache_dir = PYTORCH_PRETRAINED_BERT_CACHE / \
            'distributed_{}'.format(local_rank)
        model = create_from_pretrained(
            task_type=task_type,
            xlnet_model_name=xlnet_model_name,
            cache_dir=cache_dir,
            num_labels=num_labels,
        )
    elif xlnet_load_mode in ["model_only", "state_model_only", "state_all", "state_full_model",
                             "full_model_only"]:
        assert xlnet_load_args is None
        model = load_xlnet(
            task_type=task_type,
            xlnet_model_name=xlnet_model_name,
            xlnet_load_mode=xlnet_load_mode,
            all_state=all_state,
            num_labels=num_labels,
            xlnet_config_json_path=xlnet_config_json_path,
        )
    # delete "state_adapter" because I have no idea what it does

    else:
        raise KeyError(xlnet_load_mode)
    model = stage_model(model, fp16=fp16, device=device,
                        local_rank=local_rank, n_gpu=n_gpu)
    return model

# https://mccormickml.com/2019/09/19/XLNet-fine-tuning/


def create_from_pretrained(task_type, xlnet_model_name, cache_dir, num_labels):
    if task_type == TaskType.CLASSIFICATION:
        model = XLNetForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=xlnet_model_name,
            cache_dir=cache_dir,
            num_labels=num_labels
        )
    # delete the regression task because sentiment analysis doesn't have regression
    else:
        raise KeyError(task_type)
    return model


def load_xlnet(task_type, xlnet_model_name, xlnet_load_mode, all_state, num_labels,
               xlnet_config_json_path=None):
    if xlnet_config_json_path is None:
        xlnet_config_json_path = os.path.join(
            get_xlnet_config_path(xlnet_model_name), "xlnet_config.json")
    if xlnet_load_mode in ("model_only", "full_model_only"):
        state_dict = all_state
    elif xlnet_load_mode in ["state_model_only", "state_all", "state_full_model"]:
        state_dict = all_state["model"]
    else:
        raise KeyError(xlnet_load_mode)

    if task_type == TaskType.CLASSIFICATION:
        if xlnet_load_mode in ("state_full_model", "full_model_only"):
            model = XLNetForSequenceClassification.from_state_dict_full(
                config_file=xlnet_config_json_path,  # need to figure out what the config file is
                state_dict=state_dict,
                num_labels=num_labels,
            )
        else:
            model = XLNetForSequenceClassification.from_state_dict(
                config_file=xlnet_config_json_path,
                state_dict=state_dict,
                num_labels=num_labels,
            )
    else:
        raise KeyError(task_type)
    return model


def save_xlnet(model, optimizer, args, save_path, save_mode="all", verbose=True):
    assert save_mode in [
        "all", "tunable", "model_all", "model_tunable",
    ]
    # need to figure out what the individual parts are
    save_dict = dict()

    # Save args
    save_dict["args"] = vars(args)

    # Save model
    model_to_save = model.module if hasattr(
        model, 'module') else model  # Only save the model itself
    if save_mode in ["all", "model_all"]:
        model_state_dict = model_to_save.state_dict()
    elif save_mode in ["tunable", "model_tunable"]:
        model_state_dict = get_tunable_state_dict(model_to_save)
    else:
        raise KeyError(save_mode)
    if verbose:
        print("Saving {} model elems:".format(len(model_state_dict)))
    save_dict["model"] = utils.to_cpu(model_state_dict)

    # Save optimizer
    if save_mode in ["all", "tunable"]:
        optimizer_state_dict = utils.to_cpu(
            optimizer.state_dict()) if optimizer is not None else None
        if verbose:
            print("Saving {} optimizer elems:".format(
                len(optimizer_state_dict)))

    torch.save(save_dict, save_path)
