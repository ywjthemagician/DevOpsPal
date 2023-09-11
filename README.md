<div align="center">
<h1>
 DevOpsPal
</h1>
</div>

<p align="center">
🤗 <a href="https://huggingface.co" target="_blank">DevOpsPal-7B-Base</a> 
  • 
🤗 <a href="https://huggingface.co" target="_blank">DevOpsPal-7B-Chat</a> 
</p>

<div align="center">
<h4 align="center">
    <p>
        <b>中文</b> |
        <a href="https://github.com/luban-agi/DevOpsPal/main/README.md">English</a>
    <p>
</h4>
</div>

# 最新
- [2023.9.30] ......

<br>
<br>
<br>

# 介绍
DevOpsPal 是一系列基于 DevOps 知识的开源大语言模型，旨在为了帮助工程师在 DevOps 整个生命周期做到有问题，问 DevOpsPal。

DevOps 将整个项目生命周期划分为了七个阶段，分别为：计划，编码，构建，测试，部署，运维，观察。如下图所示，整个周期属于一个无限的循环。
<br>
<div  align="center">
 <img src="https://github.com/luban-agi/DevOpsPal/blob/main/image/devops_flow.png" width = "700" height = "350" alt="devops flow" align=center />
</div>

在这个循环中，每一步在实施的时候，都会产生各种的问题需要去搜寻答案，比如在构建过程中，需要了解某个包的函数。以往会主要从网络上搜索相关的答案，这一步会比较耗时，而且可能还找不到想要的结果。

所以我们基于自己收集的 DevOps 的相关数据，产出了首个以帮助工程师在整个 DevOps 生命周期中可能遇到的问题为目的的大模型，取名为 DevOpsPal。我们分别开源了经过 DevOps 语料加训过的 Base 模型和经过 DevOps QA 数据对齐过后的 Chat 模型。在 DevOps 的评测榜单上，我们的模型取得了同参数量级别 SOTA 的水平。相关训练数据集和评测数据集也已经开源。

<br>
<br>
<br>


# 模型

| **模型名** | **参数量** | **训练数据** | **基础模型** | **地址** |
| :--- | :---- |:----| :---- | :----| 
| DevOpsPal-7B | 7B | DevOps Corpus|Qwen-7B | Coming soon|
| DevOpsPal-7B-Chat | 7B | DevOps Corpus + DevOps QA| Qwen-7B | Coming soon|
| DevOpsPal-13B | 13B | DevOps Corpus| Baichuan-13B | Coming soon|
| DevOpsPal-13B-Chat | 13B | DevOps Corpus + DevOps QA| Baichuan-13B | Coming soon|

<br>
<br>
<br>



# 评测
Coming soon

<br>
<br>
<br>

# 使用
## 推理
Coming soon

<br>
<br>
<br>

## 微调
整体训练框架是基于 LLaMA-Efficient-Tuning 开源库修改而来，感谢作者的付出。更加详细的文档可以移步  [LLaMA-Efficient-Tuning](https://github.com/hiyouga/LLaMA-Efficient-Tuning/tree/main) 。

### 数据准备
代码内部通过调用 datasets.load_dataset 读取数据，支持 load_dataset 所支持的数据读取方式，比如 json，csv，自定义读取脚本等方式（但推荐数据准备为 jsonl 格式的文件）。然后还需要更新 `data/dataset_info.json` 文件，具体可以参考 `data/README.md`。


### Pretrain
如果收集了一批文档之类的语料（比如公司内部产品的文档），可以执行 `scripts/devopspal-pt.sh` 来发起一次加训来让模型学习到这批文档的知识，代码如下:

```bash
set -v 

torchrun --nproc_per_node=4 --nnodes=$WORLD_SIZE --master_port=$MASTER_PORT --master_addr=$MASTER_ADDR --node_rank=$RANK src/train_bash.py \
    --deepspeed conf/deepspeed_config.json \    # deepspeed 配置地址
	--stage pt \    # 代表执行 pretrain
    --model_name_or_path path_to_model \    # huggingface下载的模型地址
    --do_train \
    --report_to 'tensorboard' \
    --dataset your_corpus \    # 数据集名字，要和在 dataset_info.json 中定义的一致
    --template default \    # template，pretrain 就是 default
    --finetuning_type full \  # 全量或者 lora
    --output_dir path_to_output_checkpoint_path \    # 模型 checkpoint 保存的路径
    --overwrite_cache \
    --per_device_train_batch_size 8 \    
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --evaluation_strategy steps \
    --logging_steps 10 \
    --max_steps 1000 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --learning_rate 5e-6 \
    --plot_loss \
    --max_source_length=2048 \
    --dataloader_num_workers 8 \
    --val_size 0.01 \
    --bf16 \
    --overwrite_output_dir
```

使用者可以在这个基础上调整来发起自己的训练，更加详细的可配置项建议通过 `python src/train_bash.py -h` 来获取完整的参数列表。


### SFT


<br>
<br>
<br>

## 量化
Coming soon

<br>
<br>
<br>

# 数据
## DevOps Corpus
Coming soon

<br>
<br>
<br>

## DevOps QA
Coming soon

<br>
<br>
<br>


# 免责声明


<br>
<br>
<br>

# Star History
[![Star History Chart](https://api.star-history.com/svg?repos=luban-agi/DevOpsPal&type=Date)](https://star-history.com/#luban-agi/DevOpsPal&Date)
