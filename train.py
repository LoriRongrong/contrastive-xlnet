import argparse
import json
import os
import pandas as pd

import logging

# transformer data processing code
# liscence added
from glue.tasks import get_task     # already adjusted to sentiment analysis
from glue.runners import GlueTaskRunner, RunnerParameters   # given, no need to modify

from glue import model_setup as glue_model_setup
from shared import model_setup as shared_model_setup
from pytorch_pretrained_bert.utils import at_most_one_of, random_sample
import shared.initialization as initialization
import shared.log_info as log_info


def get_args(*in_args):
    parser = argparse.ArgumentParser()

    # === Required parameters === #
    parser.add_argument("--data_dir",
                        default='./data/SST-2',
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    # TODO: change the help method: change it into xlnet
    parser.add_argument("--xlnet_model", default='xlnet-base-cased', type=str, required=True,
                        help="XLNet pre-trained model selected in the list: xlnet-large-cased, xlnet-base-cased")
    parser.add_argument("--task_name",
                        default='sst', # having some problem with passing in using command line
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # === Model parameters === #
    parser.add_argument("--xlnet_load_path", default=None, type=str)
    # if the model is vanilla or pre-trained
    parser.add_argument("--xlnet_load_mode", default="from_pretrained", type=str,
                        help="from_pretrained, model_only, state_model_only, state_all")
    parser.add_argument("--xlnet_load_args", default=None, type=str)
    # should be pre-trained saved parameters
    parser.add_argument("--xlnet_config_json_path", default=None, type=str)
    parser.add_argument("--xlnet_vocab_path", default=None, type=str)
    parser.add_argument("--xlnet_save_mode", default="all", type=str)

    # === Other parameters === #
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_save", action="store_true")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_val",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_val_history",
                        action='store_true',
                        help="")
    parser.add_argument("--train_examples_number", type=int, default=None)
    parser.add_argument("--train_save_every", type=int, default=None)
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=10,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=-1,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. "
                             "Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--print-trainable-params', action="store_true")
    parser.add_argument('--not-verbose', action="store_true")
    parser.add_argument('--force-overwrite', action="store_true")
    args = parser.parse_args(*in_args)
    # print ("checking: ", args)
    return args


def main():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    args = get_args()
    log_info.print_args(args)

    device, n_gpu = initialization.init_cuda_from_args(args, logger=logger)
    initialization.init_seed(args, n_gpu=n_gpu, logger=logger)
    initialization.init_train_batch_size(args)
    initialization.init_output_dir(args)
    # could cause problem because imdb is not part of the tasks defined
    task = get_task(args.task_name, args.data_dir)

    # create tokenizer using given model input
    # I think xlnet also use the same tokenizer
    tokenizer = shared_model_setup.create_tokenizer(
        xlnet_model_name=args.xlnet_model,  # need to change
        xlnet_load_mode=args.xlnet_load_mode,  # need to change
        do_lower_case=args.do_lower_case,
        xlnet_vocab_path=args.xlnet_vocab_path,  # not sure how to modify
    )
    all_state = shared_model_setup.load_overall_state(
        args.xlnet_load_path, relaxed=True)  # probably will be the pre-trained one

    model = glue_model_setup.create_model(
        task_type=task.processor.TASK_TYPE,
        xlnet_model_name=args.xlnet_model,
        xlnet_load_mode=args.xlnet_load_mode,
        xlnet_load_args=args.xlnet_load_args,
        all_state=all_state,
        num_labels=len(task.processor.get_labels()),
        device=device,
        n_gpu=n_gpu,
        fp16=args.fp16,
        local_rank=args.local_rank,
        xlnet_config_json_path=args.xlnet_config_json_path,
    )
    if args.do_train:
        if args.print_trainable_params:
            log_info.print_trainable_params(model)
        train_examples = task.get_train_examples()
        if args.train_examples_number is not None:
            train_examples = random_sample(
                train_examples, args.train_examples_number)
        t_total = shared_model_setup.get_opt_train_steps(
            num_train_examples=len(train_examples),
            args=args,
        )

        optimizer = shared_model_setup.create_optimizer(
            model=model,
            learning_rate=args.learning_rate,
            t_total=t_total,
            loss_scale=args.loss_scale,
            fp16=args.fp16,
            warmup_proportion=args.warmup_proportion,
            state_dict=all_state["optimizer"] if args.xlnet_load_mode == "state_all" else None,
        )
    else:
        train_examples = None
        t_total = 0
        optimizer = None
    # TODO: what does xlnet runner do???\
    # initial answer: probably do
    runner = GlueTaskRunner(
        model=model,
        optimizer=optimizer,
        tokenizer=tokenizer,
        label_list=task.get_labels(),
        device=device,
        rparams=RunnerParameters(
            max_seq_length=args.max_seq_length,
            local_rank=args.local_rank, n_gpu=n_gpu, fp16=args.fp16,
            learning_rate=args.learning_rate, gradient_accumulation_steps=args.gradient_accumulation_steps,
            t_total=t_total, warmup_proportion=args.warmup_proportion,
            num_train_epochs=args.num_train_epochs,
            train_batch_size=args.train_batch_size, eval_batch_size=args.eval_batch_size,
        )
    )

    if args.do_train:
        assert at_most_one_of([args.do_val_history, args.train_save_every])
        if args.do_val_history:
            val_examples = task.get_dev_examples()
            results = runner.run_train_val(
                train_examples=train_examples,
                val_examples=val_examples,
                task_name=task.name,
            )
            metrics_str = json.dumps(results, indent=2)
            with open(os.path.join(args.output_dir, "val_metrics_history.json"), "w") as f:
                f.write(metrics_str)
        elif args.train_save_every:
            train_dataloader = runner.get_train_dataloader(
                train_examples, verbose=not args.not_verbose)
            for epoch in range(int(args.num_train_epochs)):
                for step, _, _ in runner.run_train_epoch_context(train_dataloader):
                    if step % args.train_save_every == args.train_save_every - 1 \
                            or step == len(train_dataloader) - 1:
                        glue_model_setup.save_xlnet(
                            model=model, optimizer=optimizer, args=args,
                            save_path=os.path.join(
                                args.output_dir, f"all_state___epoch{epoch:04d}___batch{step:06d}.p"),
                            save_mode=args.xlnet_save_mode,
                            verbose=not args.not_verbose,
                        )
        else:
            runner.run_train(train_examples)

    if args.do_save:
        # Save a trained model
        glue_model_setup.save_xlnet(
            model=model, optimizer=optimizer, args=args,
            save_path=os.path.join(args.output_dir, "all_state.p"),
            save_mode=args.xlnet_save_mode,
        )
    # remove the hack part for MultiNLI Mismatched dataset
    if args.do_val:
        val_examples = task.get_dev_examples()
        results = runner.run_val(
            val_examples, task_name=task.name, verbose=not args.not_verbose)
        df = pd.DataFrame(results["logits"])
        df.to_csv(os.path.join(args.output_dir, "val_preds.csv"),
                  header=False, index=False)
        metrics_str = json.dumps(
            {"loss": results["loss"], "metrics": results["metrics"]}, indent=2)
        print(metrics_str)
        with open(os.path.join(args.output_dir, "val_metrics.json"), "w") as f:
            f.write(metrics_str)

    if args.do_test:
        test_examples = task.get_test_examples()
        logits = runner.run_test(test_examples, verbose=not args.not_verbose)
        df = pd.DataFrame(logits)
        df.to_csv(os.path.join(args.output_dir, "test_preds.csv"),
                  header=False, index=False)


if __name__ == "__main__":
    main()
