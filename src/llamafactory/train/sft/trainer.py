# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
from transformers.trainer_pt_utils import LabelSmoother

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler()

    @override
    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        """
        Override the compute_loss method to give equal weight to the thinking and answer parts.
        """
        self.label_smoother = LabelSmoother(epsilon=self.args.label_smoothing_factor)
        labels=inputs.get("labels")
        # Get the original outputs from the model
        outputs = model(**inputs)

        # 保存past state（如果存在）
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        
        # 获取logits
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
        
        # Get the special token ID for <thinking_end>
        thinking_end_token_id = self.processing_class.convert_tokens_to_ids('<thinking_end>')
        
        batch_size, seq_len, vocab_size = logits.shape
        total_loss = 0
        thinking_loss = 0.0
        strategy_loss = 0.0
        thinking_logits=[]
        strategy_logits=[]
        thinking_labels=[]
        strategy_labels=[]
        for i in range(batch_size):
            # Find the position of the special token
            thinking_end_positions = (labels[i] == thinking_end_token_id).nonzero(as_tuple=True)[0]
            print('thinking_end_positions',thinking_end_positions)
            if len(thinking_end_positions) == 0:
                # print('!!!! not find thinking part')
                return super().compute_loss(model, inputs, *args, **kwargs)
            else:
                # 找到第一个<thinking_end>位置
                split_pos = thinking_end_positions[0]
                
                # 分割标签和logits为thinking部分和strategy部分
                thinking_label= labels[i][:split_pos+1]  # 包含<thinking_end>标记
                strategy_label = labels[i][split_pos:]    # 从<thinking_end>开始
                
                thinking_logit = logits[i][:split_pos+1]
                strategy_logit = logits[i][split_pos:]

                thinking_labels.append(thinking_label)
                strategy_labels.append(strategy_label)
                thinking_logits.append(thinking_logit)
                strategy_logits.append(strategy_logit)

        # 处理序列长度不一致的问题
        max_thinking_len = max(len(label) for label in thinking_labels)
        max_strategy_len = max(len(label) for label in strategy_labels)

         # 使用padding处理不同长度的序列
        padded_thinking_labels = []
        padded_thinking_logits = []
        padded_strategy_labels = []
        padded_strategy_logits = []
        
        for i in range(len(thinking_labels)):
            t_label = thinking_labels[i]
            t_logit = thinking_logits[i]
            s_label = strategy_labels[i]
            s_logit = strategy_logits[i]
            
            # 填充thinking部分
            if len(t_label) < max_thinking_len:
                pad_length = max_thinking_len - len(t_label)
                padding = torch.full((pad_length,), -100, dtype=t_label.dtype, device=t_label.device)
                padded_thinking_labels.append(torch.cat([t_label, padding], dim=0))
                
                logit_padding = torch.zeros((pad_length, vocab_size), dtype=t_logit.dtype, device=t_logit.device)
                padded_thinking_logits.append(torch.cat([t_logit, logit_padding], dim=0))
            else:
                padded_thinking_labels.append(t_label)
                padded_thinking_logits.append(t_logit)
            
            # 填充strategy部分
            if len(s_label) < max_strategy_len:
                pad_length = max_strategy_len - len(s_label)
                padding = torch.full((pad_length,), -100, dtype=s_label.dtype, device=s_label.device)
                padded_strategy_labels.append(torch.cat([s_label, padding], dim=0))
                
                logit_padding = torch.zeros((pad_length, vocab_size), dtype=s_logit.dtype, device=s_logit.device)
                padded_strategy_logits.append(torch.cat([s_logit, logit_padding], dim=0))
            else:
                padded_strategy_labels.append(s_label)
                padded_strategy_logits.append(s_logit)

        # 将列表转换为张量
        padded_thinking_labels = torch.stack(padded_thinking_labels)
        padded_thinking_logits = torch.stack(padded_thinking_logits)
        padded_strategy_labels = torch.stack(padded_strategy_labels)
        padded_strategy_logits = torch.stack(padded_strategy_logits)

        thinking_outputs={"logits": padded_thinking_logits, "labels": padded_thinking_labels}
        strategy_outputs={"logits": padded_strategy_logits, "labels": padded_strategy_labels}        
        

        thinking_loss = self.label_smoother(thinking_outputs, thinking_outputs['labels'], shift_labels=True)
        strategy_loss = self.label_smoother(strategy_outputs, strategy_outputs['labels'], shift_labels=True)
        # thinking_loss = super().compute_loss(model, inputs, outputs=False, *args, **kwargs)      
        # 两部分权重相同
        loss = thinking_loss + strategy_loss
        
        print(loss.item(),thinking_loss.item(), strategy_loss.item())
        return (loss, outputs) if return_outputs else loss

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
