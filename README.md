# contrastive-xlnet
- First prepare augmented.csv

- Then download pretrained xlnet-base-cased to local using:\\
```
wget https://storage.googleapis.com/xlnet/released_models/cased_L-12_H-768_A-12.zip

unzip cased_L-12_H-768_A-12.zip
```
- and then rename it as xlnet-base-cased

- then
```
mkdir moco_model
```
- Run 
```
Python MOCO.py
```
- Run 
```
Python trans.py
```
- Then go to [**huggingface**](https://github.com/huggingface/transformers) github to clone some dependencies of files (I cloned the whole directory of transformers). 
but the **run_glue.py** is customed (hard-coded some args). 

- Run 
```
python ./examples/text-classification/run_glue.py \
    --model_name_or_path ../moco_model/pytorch_model.bin \
    --task_name SST-2 \
    --do_train \
    --do_eval \
    --cache_dir ../moco_model/ \
    --config_name ../moco_model/config.json \
    --data_dir ../data/SST-2/ \
    --max_seq_length 128 \
    --per_device_eval_batch_size=8   \
    --per_device_train_batch_size=8   \
    --learning_rate 2e-5 \
    --num_train_epochs 1.0 \
    --output_dir ../sst_output
 ```   
 - The result is in eval_results_sst-2.txt

