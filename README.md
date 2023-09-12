<div align="center">
<h1>
 DevOpsPal
</h1>
</div>

<p align="center">
🤗 <a href="https://huggingface.co" target="_blank">Hugging Face</a> • 
🤖 <a href="https://modelscope.cn/" target="_blank">ModelScope</a> • 
💬 <a href="https://github.com/" target="_blank">WeChat</a>
</p

<div align="center">
<h4 align="center">
    <p>
        <b>中文</b> |
        <a href="https://github.com/luban-agi/DevOpsPal/main/README.md">English</a>
    <p>
</h4>
</div>

DevOpsPal是国内首个开源的**开发运维大模型**，主要致力于在 DevOps 领域发挥实际价值。目前，DevOpsPal 能够帮助工程师在 DevOps 生命周期做到有问题，问 DevOpsPal。

我们开源了经过高质量 DevOps 语料训练的 Base 模型和经过 DevOps QA 数据对齐后的 Chat 模型。
在开发运维领域评测基准 [DevOpsEval](https://github.com/luban-agi/DevOps-Eval) 上，DevOpsPal 取得同规模**最佳**的效果。
欢迎阅读我们的[技术报告](https://arxiv.org)获取更多信息。

<!--
DevOps 将整个项目生命周期划分为了七个阶段，分别为：计划，编码，构建，测试，部署，运维，观察。如下图所示，整个周期属于一个无限的循环。
<br>
<div  align="center">
 <img src="https://github.com/luban-agi/DevOpsPal/blob/main/image/devops_flow.png" width = "700" height = "350" alt="devops flow" align=center />
</div>

在这个循环中，每一步在实施的时候，都会产生各种的问题需要去搜寻答案，比如在构建过程中，需要了解某个包的函数。以往会主要从网络上搜索相关的答案，这一步会比较耗时，而且可能还找不到想要的结果。

所以我们基于自己收集的 DevOps 的相关数据，产出了首个以帮助工程师在整个 DevOps 生命周期中可能遇到的问题为目的的大模型，取名为 DevOpsPal。我们分别开源了经过 DevOps 语料加训过的 Base 模型和经过 DevOps QA 数据对齐过后的 Chat 模型。在 DevOps 的评测榜单上，我们的模型取得了同参数量级别 SOTA 的水平。相关训练数据集和评测数据集也已经开源。
-->

开源模型和下载链接见下表：
|         | 基座模型  | 对齐模型 | 对齐模型 Int4 量化 |
|:-------:|:-------:|:-------:|:-----------------:|
| 7B      | 🤗 [DevOpsPal-7B-Base](https://huggingface.co) | 🤗 [DevOpsPal-7B-Chat](https://huggingface.co) | 🤗 [DevOpsPal-7B-Chat-4bits](https://huggingface.co) |
| 13B     | 🤗 [DevOpsPal-13B-Base](https://huggingface.co) | 🤗 [DevOpsPal-13B-Chat](https://huggingface.co) | 🤗 [DevOpsPal-13B-Chat-4bits](https://huggingface.co) |


# 最新消息
- [2023.9.30] 开源 DevOpsPal-7B-Base 和 DevOpsPal-7B-Chat，以及 Chat 版本的 Int4 量化模型。

# 模型评测
Coming soon

# 快速使用
我们提供简单的示例来说明如何利用 🤗 Transformers 快速使用 DevopsPal-7B 和 DevopsPal-7B-Chat。

## 安装依赖

```bash
pip install -r requirements.txt
```
## Chat 模型推理示例

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

# 请注意：分词器默认行为已更改为默认关闭特殊token攻击防护。
tokenizer = AutoTokenizer.from_pretrained("path_to_DevOpsPal-7B-Chat", trust_remote_code=True)

# 默认使用自动模式，根据设备自动选择精度
model = AutoModelForCausalLM.from_pretrained("path_to_DevOpsPal-7B-Chat", device_map="auto", trust_remote_code=True).eval()

# 可指定不同的生成长度、top_p等相关超参
model.generation_config = GenerationConfig.from_pretrained("path_to_DevOpsPal-7B-Chat", trust_remote_code=True)

# 第一轮对话
response, history = model.chat(tokenizer, "你好", history=None)
print(response)

# 第二轮对话
response, history = model.chat(tokenizer, "。。。", history=history)
print(response)

# 第三轮对话
response, history = model.chat(tokenizer, "。。。", history=history)
print(response)
```
## Base 模型推理示例

# 模型训练

## 数据准备
代码内部通过调用 datasets.load_dataset 读取数据，支持 load_dataset 所支持的数据读取方式，比如 json，csv，自定义读取脚本等方式（但推荐数据准备为 jsonl 格式的文件）。然后还需要更新 `data/dataset_info.json` 文件，具体可以参考 `data/README.md`。

## 预训练
如果收集了一批文档之类的语料（比如公司内部产品的文档）想要在 devopspal 模型上加训，可以执行 `scripts/devopspal-pt.sh` 来发起一次加训来让模型学习到这批文档的知识，具体代码如下:

```bash
set -v 

torchrun --nproc_per_node=4 --nnodes=$WORLD_SIZE --master_port=$MASTER_PORT --master_addr=$MASTER_ADDR --node_rank=$RANK src/train_bash.py \
    --deepspeed conf/deepspeed_config.json \    # deepspeed 配置地址
	--stage pt \    # 代表执行 pretrain
    --model_name_or_path path_to_model \    # huggingface下载的 devopspal 模型地址
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

## 指令微调
如果收集了一批 QA 数据想要针对 devopspal 再进行对齐的话，可以执行 `scripts/devopspal-sft.sh` 来发起一次加训来让模型在收集到的模型上进行对齐，具体代码如下:
```bash
set -v 

torchrun --nproc_per_node=2 --nnodes=$WORLD_SIZE --master_port=$MASTER_PORT --master_addr=$MASTER_ADDR --node_rank=$RANK src/train_bash.py \
    --deepspeed conf/deepspeed_config.json \    # deepspeed 配置地址
	--stage pt \    # 代表执行 pretrain
    --model_name_or_path path_to_model \    # huggingface下载的模型地址
    --do_train \
    --report_to 'tensorboard' \
    --dataset your_corpus \    # 数据集名字，要和在 dataset_info.json 中定义的一致
    --template chatml \    # template qwen 模型固定写 chatml
    --finetuning_type full \    # 全量或者 lora
    --output_dir /mnt/llm/devopspal/model/trained \     # 模型 checkpoint 保存的路径
    --overwrite_cache \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --evaluation_strategy steps \
    --logging_steps 10 \
    --max_steps 1000 \
    --save_steps 100 \
    --eval_steps 100 \
    --learning_rate 5e-5 \
    --plot_loss \
    --max_source_length=2048 \
    --dataloader_num_workers 8 \
    --val_size 0.01 \
    --bf16 \
    --overwrite_output_dir
```

使用者可以在这个基础上调整来发起自己的 SFT 训练，更加详细的可配置项建议通过 `python src/train_bash.py -h` 来获取完整的参数列表。

## 量化
我们提供了 DevOpsPal-Chat 系列的量化模型，当然也可以通过以下代码来量化自己加训过的模型

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.gptq import GPTQQuantizer, load_quantized_model
import torch

# 加载模型
model_name = "path_of_your_model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# 加载数据
# todo

# 开始量化
quantizer = GPTQQuantizer(bits=4, dataset="c4", block_name_to_quantize = "model.decoder.layers", model_seqlen = 2048)
quantized_model = quantizer.quantize_model(model, tokenizer)

# 保存量化后的模型
out_dir = 'save_path_of_your_quantized_model'
quantized_model.save_quantized(out_dir)
```

# 数据

## 预训练数据
DevOps Corpus Coming soon

## 指令微调数据
DevOps QA Coming soon

# 免责声明
由于语言模型的特性，模型生成的内容可能包含幻觉或者歧视性言论。请谨慎使用 DevOpsPal 系列模型生成的内容。
如果要公开使用或商用该模型服务，请注意服务方需承担由此产生的不良影响或有害言论的责任，本项目开发者不承担任何由使用本项目（包括但不限于数据、模型、代码等）导致的危害或损失。

# 引用
如果使用本项目的代码、数据或模型，请引用本项目论文：

链接：[DevOpsPal](https://arxiv.org)

```
@article{devopspal2023,
  title={},
  author={},
  journal={arXiv preprint arXiv},
  year={2023}
}
```

# 致谢
本项目参考了以下开源项目，在此对相关项目和研究开发人员表示感谢。
- [LLaMA-Efficient-Tuning](https://github.com/hiyouga/LLaMA-Efficient-Tuning)

# 点赞历史
[![Star History Chart](https://api.star-history.com/svg?repos=luban-agi/DevOpsPal&type=Date)](https://star-history.com/#luban-agi/DevOpsPal&Date)
