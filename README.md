# Language Model Evaluation Harness

![](https://github.com/EleutherAI/lm-evaluation-harness/workflows/Build/badge.svg)
[![codecov](https://codecov.io/gh/EleutherAI/lm-evaluation-harness/branch/master/graph/badge.svg?token=JSG3O2427J)](https://codecov.io/gh/EleutherAI/lm-evaluation-harness)


## 已添加任务


- 毒害指标评测

|注册名|任务文件|说明|评价指标|
|:----:|:----|:----:|:-----:|
| coldzh      | coldzh.py  |中文毒害数据| |

- 偏见指标评测

|注册名|任务文件|说明|评价指标|
|:----:|:----|:----:|:-----:|
| stereosetzh      | stereosetzh.py  || |

- 分类任务评测

|注册名|任务文件|说明|评价指标|
|:----:|:----|:----:|:-----:|
| pclue_dev_classify | pclue_dev_classify.py  |中文评测集Pclue的dev数据中classify任务| |

- NLI任务评测

|注册名|任务文件|说明|评价指标|
|:----:|:----|:----:|:-----:|
|pclue_dev_nli|pclue_dev_nli.py|中文评测集Pclue里的dev数据集nli任务||

- 其他任务

|注册名|任务文件|说明|评价指标|
|:----:|:----|:----:|:-----:|
|chnsenticorp|chnsenticorp.py|情感分析任务ChenSentiCorp|acc|

- 生成任务
   - 中文评测集Pclue里的dev数据集generate任务：注册为pclue_dev_generate，bleu和chrf结果越高越好，chrf仅供参考，可以只看bleu

|注册名|任务文件|说明|评价指标|
|:----:|:----|:----:|:-----:|
|pclue_dev_generate|pclue_dev_generate.py|中文评测集Pclue里的dev数据集generate任务|**bleu**, chrf|

## Overview

This project provides a unified framework to test generative language models on a large number of different evaluation tasks.

Features:

- 200+ tasks implemented. See the [task-table](./docs/task_table.md) for a complete list.
- Support for the Hugging Face `transformers` library, GPT-NeoX, Megatron-DeepSpeed, and the OpenAI API, with flexible tokenization-agnostic interface.
- Support for evaluation on adapters (e.g. LoRa) supported in [HuggingFace's PEFT library](https://github.com/huggingface/peft).
- Task versioning to ensure reproducibility.

## Install

To install `lm-eval` from the github repository main branch, run:

```bash
git clone https://github.com/RekkimiARG/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e .
```

## 其他依赖

运行一些模型的时候可能会报错缺少部分库，比如sentencepiece，安装一下一般就解决了。

## llama tokenizer

llama huggingface的config中对tokenizer的命名和transformers库中的不一致，导致无法使用AutoTokenizer加载，但是本库的运行默认使用AutoTokenizer。虽然该问题由来已久，但是目前仍没有修复，所以我们要手动改一下transformers的源码。

有两种方法，
1.安装好transformers库之后，去对应环境site-packages下找到transformers库，做如下修改：

```
replace LlamaTokenizer to LLaMATokenizer
1. convert_slow_tokenizer.py
2. models/auto/tokenization_auto.py
3. models/llama/__init__.py
4. models/llama/convert_llama_weights_to_hf.py
5. models/llama/tokenization_llama_fast.py
6. models/llama/tokenization_llama.py
```

2. 从这里拉源码安装

```
git clone https://github.com/vxfla/transformers.git
cd transformers/
pip install -e .
```

## Basic Usage

> **Note**: When reporting results from eval harness, please include the task versions (shown in `results["versions"]`) for reproducibility. This allows bug fixes to tasks while also ensuring that previously reported scores are reproducible. See the [Task Versioning](#task-versioning) section for more info.

To evaluate a model hosted on the [HuggingFace Hub](https://huggingface.co/models) (e.g. GPT-J-6B) you can use the following command:


```bash
python main.py \
    --model hf-causal \
    --model_args pretrained=EleutherAI/gpt-j-6B \
    --tasks lambada_openai,hellaswag \
    --device cuda:0
```

Additional arguments can be provided to the model constructor using the `--model_args` flag. Most notably, this supports the common practice of using the `revisions` feature on the Hub to store partialy trained checkpoints:

```bash
python main.py \
    --model hf-causal \
    --model_args pretrained=EleutherAI/pythia-160m,revision=step100000 \
    --tasks lambada_openai,hellaswag \
    --device cuda:0
```

To evaluate models that are called via `AutoSeq2SeqLM`, you instead use `hf-seq2seq`.

> **Warning**: Choosing the wrong model may result in erroneous outputs despite not erroring.

To use with [PEFT](https://github.com/huggingface/peft), take the call you would run to evaluate the base model and add `,peft=PATH` to the `model_args` argument as shown below:
```bash
python main.py \
    --model hf-causal \
    --model_args pretrained=EleutherAI/gpt-j-6b,peft=nomic-ai/gpt4all-j-lora \
    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \ 
    --device cuda:0 
```

Our library also supports the OpenAI API:

```bash
export OPENAI_API_SECRET_KEY=YOUR_KEY_HERE
python main.py \
    --model gpt3 \
    --model_args engine=davinci \
    --tasks lambada_openai,hellaswag
```

While this functionality is only officially mantained for the official OpenAI API, it tends to also work for other hosting services that use the same API such as [goose.ai](goose.ai) with minor modification. We also have an implementation for the [TextSynth](https://textsynth.com/index.html) API, using `--model textsynth`.

To verify the data integrity of the tasks you're performing in addition to running the tasks themselves, you can use the `--check_integrity` flag:

```bash
python main.py \
    --model gpt3 \
    --model_args engine=davinci \
    --tasks lambada_openai,hellaswag \
    --check_integrity
```

To evaluate mesh-transformer-jax models that are not available on HF, please invoke eval harness through [this script](https://github.com/kingoflolz/mesh-transformer-jax/blob/master/eval_harness.py).

💡 **Tip**: You can inspect what the LM inputs look like by running the following command:

```bash
python write_out.py \
    --tasks all_tasks \
    --num_fewshot 5 \
    --num_examples 10 \
    --output_base_path /path/to/output/folder
```

This will write out one text file for each task.

## Implementing new tasks

To implement a new task in the eval harness, see [this guide](./docs/task_guide.md).

## Task Versioning

To help improve reproducibility, all tasks have a `VERSION` field. When run from the command line, this is reported in a column in the table, or in the "version" field in the evaluator return dict. The purpose of the version is so that if the task definition changes (i.e to fix a bug), then we can know exactly which metrics were computed using the old buggy implementation to avoid unfair comparisons. To enforce this, there are unit tests that make sure the behavior of all tests remains the same as when they were first implemented. Task versions start at 0, and each time a breaking change is made, the version is incremented by one.

When reporting eval harness results, please also report the version of each task. This can be done either with a separate column in the table, or by reporting the task name with the version appended as such: taskname-v0.

## Test Set Decontamination

To address concerns about train / test contamination, we provide utilities for comparing results on a benchmark using only the data points nto found in the model trainign set. Unfortunately, outside of models trained on the Pile ans C4, its very rare that people who train models disclose the contents of the training data. However this utility can be useful to evaluate models you have trained on private data, provided you are willing to pre-compute the necessary indices. We provide computed indices for 13-gram exact match deduplication against the Pile, and plan to add additional precomputed dataset indices in the future (including C4 and min-hash LSH deduplication).

For details on text decontamination, see the [decontamination guide](./docs/decontamination.md).

Note that the directory provided to the `--decontamination_ngrams_path` argument should contain the ngram files and info.json. See the above guide for ngram generation for the pile, this could be adapted for other training sets.

```bash
python main.py \
    --model gpt2 \
    --tasks sciq \
    --decontamination_ngrams_path path/containing/training/set/ngrams \
    --device cuda:0
```

## Cite as

```
@software{eval-harness,
  author       = {Gao, Leo and
                  Tow, Jonathan and
                  Biderman, Stella and
                  Black, Sid and
                  DiPofi, Anthony and
                  Foster, Charles and
                  Golding, Laurence and
                  Hsu, Jeffrey and
                  McDonell, Kyle and
                  Muennighoff, Niklas and
                  Phang, Jason and
                  Reynolds, Laria and
                  Tang, Eric and
                  Thite, Anish and
                  Wang, Ben and
                  Wang, Kevin and
                  Zou, Andy},
  title        = {A framework for few-shot language model evaluation},
  month        = sep,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v0.0.1},
  doi          = {10.5281/zenodo.5371628},
  url          = {https://doi.org/10.5281/zenodo.5371628}
}
```
