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
        # Get the original outputs from the model
        outputs = model(**inputs)
        
        # If there's no labels, just return the original loss
        if "labels" not in inputs:
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss
        
        # Get the logits and labels
        logits = outputs.logits
        labels = inputs["labels"]
        
        # Get the special token ID for <thinking_end>
        thinking_end_token_id = self.processing_class.convert_tokens_to_ids('<thinking_end>')
        
        batch_size = labels.shape[0]
        total_loss = 0
        
        for i in range(batch_size):
            # Find the position of the special token
            delimiter_pos = -1
            for j in range(labels.shape[1]):
                if labels[i, j] == thinking_end_token_id:
                    delimiter_pos = j
                    break
            
            # If the delimiter is not found, use the original loss calculation
            if delimiter_pos == -1:
                sample_logits = logits[i].unsqueeze(0)
                sample_labels = labels[i].unsqueeze(0)
                sample_loss = F.cross_entropy(
                    sample_logits.view(-1, sample_logits.size(-1)),
                    sample_labels.view(-1),
                    ignore_index=IGNORE_INDEX,
                    reduction='mean'
                )
                total_loss += sample_loss
            else:
                # Calculate loss for thinking part
                thinking_logits = logits[i, :delimiter_pos].unsqueeze(0)
                thinking_labels = labels[i, :delimiter_pos].unsqueeze(0)
                thinking_loss = F.cross_entropy(
                    thinking_logits.view(-1, thinking_logits.size(-1)),
                    thinking_labels.view(-1),
                    ignore_index=IGNORE_INDEX,
                    reduction='sum'
                )
                
                # Calculate loss for answer part
                answer_logits = logits[i, delimiter_pos+1:].unsqueeze(0)  # +1 to skip the delimiter token
                answer_labels = labels[i, delimiter_pos+1:].unsqueeze(0)
                answer_loss = F.cross_entropy(
                    answer_logits.view(-1, answer_logits.size(-1)),
                    answer_labels.view(-1),
                    ignore_index=IGNORE_INDEX,
                    reduction='sum'
                )
                
                # Count valid tokens (non-padding) in each part
                thinking_valid = (thinking_labels != IGNORE_INDEX).sum().item()
                answer_valid = (answer_labels != IGNORE_INDEX).sum().item()
                
                # Normalize losses by the number of valid tokens
                if thinking_valid > 0:
                    thinking_loss = thinking_loss / thinking_valid
                if answer_valid > 0:
                    answer_loss = answer_loss / answer_valid
                
                # Equal weighting of both parts
                sample_loss = (thinking_loss + answer_loss) / 2
                total_loss += sample_loss
        
        # Average the loss over the batch
        loss = total_loss / batch_size
        
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
