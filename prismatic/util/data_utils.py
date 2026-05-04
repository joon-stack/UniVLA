"""
data_utils.py

General utilities and classes for facilitating data loading and collation.
"""
import re
import string
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Sequence, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


def tree_map(fn: Callable, tree: dict) -> dict:
    """Maps a function over a nested dictionary."""
    return {k: tree_map(fn, v) if isinstance(v, dict) else fn(v) for k, v in tree.items()}


def tree_map_with_key(fn: Callable, tree: dict, keys: Sequence = ()) -> dict:
    """Maps a function over a nested dictionary."""
    return {
        k: tree_map_with_key(fn, v, (*keys, k)) if isinstance(v, dict) else fn((*keys, k), v) for k, v in tree.items()
    }


@dataclass
class PaddedCollatorForLanguageModeling:
    model_max_length: int
    pad_token_id: int
    default_image_resolution: Tuple[int, int, int]
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        self.dummy_pixel_values = torch.zeros(self.default_image_resolution, dtype=self.pixel_values_dtype)

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]

        # For now, we only support Tokenizers with `padding_side = "right"` during Training (but plan to extend!)
        #   => Handle padding via RNN Utils => `pad_sequence`
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # === Handle "unimodal" (language-only) vs. "multimodal" ===

        # Some examples are "language-only" --> build a Tensor of `multimodal_indices` that we can slice into easily
        multimodal_indices = torch.tensor(
            [idx for idx in range(len(pixel_values)) if pixel_values[idx] is not None], dtype=torch.long
        )

        # Stack all `pixel_values` --> depending on type (torch.Tensor, or Dict[str, torch.Tensor]) & presence of None
        if len(multimodal_indices) == 0:
            pixel_values = torch.stack([self.dummy_pixel_values for _ in range(len(input_ids))])
        elif isinstance(pv_example := pixel_values[multimodal_indices[0]], torch.Tensor):
            pixel_values = torch.stack(
                [
                    pixel_values[idx] if idx in multimodal_indices else self.dummy_pixel_values
                    for idx in range(len(input_ids))
                ]
            )
        elif isinstance(pv_example, dict):
            pixel_values = {
                k: torch.stack(
                    [
                        pixel_values[idx][k] if idx in multimodal_indices else self.dummy_pixel_values
                        for idx in range(len(input_ids))
                    ]
                )
                for k in pv_example
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            multimodal_indices=multimodal_indices,
        )


@dataclass
class PaddedCollatorForActionPrediction:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

        # For now, we only support Tokenizers with `padding_side = "right"` during training
        #   => Handle padding via RNN Utils => `pad_sequence`
        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # [Contract] For VLA Training =>> No "Unimodal" Data!
        assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"

        # Stack all `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]
        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values = torch.stack(pixel_values)
        elif isinstance(pixel_values[0], dict):
            pixel_values = {
                k: torch.stack([pixel_values[idx][k] for idx in range(len(input_ids))]) for k in pixel_values[0]
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        output = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output


@dataclass
class PaddedCollatorForLatentActionPrediction:
    model_max_length: int
    pad_token_id: int
    tokenizer: Any
    latent_action_model: torch.nn.Module
    prompt_builder_fn: Callable
    padding_side: str = "right"
    predict_stop_token: bool = True

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        profile_steps = int(os.environ.get("UNIVLA_PROFILE_STEPS", "0"))
        profile_start = time.perf_counter() if profile_steps > 0 else None
        pixel_values = [instance["pixel_values"] for instance in instances]
        dataset_names = [instance["dataset_name"] for instance in instances]

        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values = torch.stack(pixel_values)
        elif isinstance(pixel_values[0], dict):
            pixel_values = {
                k: torch.stack([pixel_values[idx][k] for idx in range(len(instances))]) for k in pixel_values[0]
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")
        profile_pixel_done = time.perf_counter() if profile_steps > 0 else None

        if os.environ.get("UNIVLA_DUMMY_LATENT_ACTIONS", "0") == "1":
            latent_action_idx = torch.zeros((len(instances), 1), dtype=torch.long)
        elif "initial_lam_pixel_values" in instances[0]:
            initial = torch.stack([instance["initial_lam_pixel_values"] for instance in instances])
            target = torch.stack([instance["target_lam_pixel_values"] for instance in instances])
            device = next(self.latent_action_model.parameters()).device
            video = torch.stack([initial, target], dim=1).to(device, non_blocking=True)
            with torch.inference_mode():
                latent_action_idx = self.latent_action_model.vq_encode(video)["indices"].view(len(instances), -1)
        else:
            latent_action_idx = torch.zeros((len(instances), 1), dtype=torch.long)
        profile_lam_done = time.perf_counter() if profile_steps > 0 else None

        input_ids, labels = [], []
        for instance, action_indices in zip(instances, latent_action_idx.cpu()):
            action_tokens = "".join(f"<ACT_{idx}>" for idx in action_indices.tolist())
            prompt_builder = self.prompt_builder_fn("openvla")
            conversation = [
                {"from": "human", "value": f"What action should the robot take to {instance['lang']}?"},
                {"from": "gpt", "value": action_tokens},
            ]
            for turn in conversation:
                prompt_builder.add_turn(turn["from"], turn["value"])

            cur_input_ids = self.tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
            cur_labels = list(cur_input_ids)
            cur_input_ids, cur_labels = torch.tensor(cur_input_ids), torch.tensor(cur_labels)
            cur_labels[: -(len(action_indices) + 1)] = IGNORE_INDEX
            if not self.predict_stop_token:
                cur_labels[-1] = IGNORE_INDEX
            input_ids.append(cur_input_ids)
            labels.append(cur_labels)
        profile_token_done = time.perf_counter() if profile_steps > 0 else None

        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]
        attention_mask = input_ids.ne(self.pad_token_id)

        output = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            dataset_names=dataset_names,
        )
        if profile_steps > 0:
            profile_end = time.perf_counter()
            output["profile"] = {
                "collator_total": profile_end - profile_start,
                "collator_pixel_stack": profile_pixel_done - profile_start,
                "collator_lam": profile_lam_done - profile_pixel_done,
                "collator_tokenize": profile_token_done - profile_lam_done,
                "collator_pad": profile_end - profile_token_done,
            }
        return output


@dataclass
class PaddedCollatorForActionPrediction_LIBERO:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

        # For now, we only support Tokenizers with `padding_side = "right"` during training
        #   => Handle padding via RNN Utils => `pad_sequence`
        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # For low-level policy training
        actions = [torch.from_numpy(instance["actions"]) for instance in instances]
        actions = torch.stack(actions, dim=0)

        # Get latent action indexes
        latent_action_idx = [instance["latent_action_idx"] for instance in instances]
        latent_action_idx = torch.stack(latent_action_idx, dim=0)

        # [Contract] For VLA Training =>> No "Unimodal" Data!
        assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"

        # Stack all `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]
        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values = torch.stack(pixel_values)
        elif isinstance(pixel_values[0], dict):
            pixel_values = {
                k: torch.stack([pixel_values[idx][k] for idx in range(len(input_ids))]) for k in pixel_values[0]
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        output = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            actions=actions,
            latent_action_idx=latent_action_idx,
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output

@dataclass
class PaddedCollatorForActionPrediction_R2R:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        
        initial_pixel_values = [instance["initial_pixel_values"] for instance in instances]
        target_pixel_values = [instance["target_pixel_values"] for instance in instances]
        
        
        initial_pixel_values_hist, target_pixel_values_hist = [], []
        with_hist = []
        for instance in instances:
            if instance["initial_pixel_values_hist"] is not None:
                initial_pixel_values_hist.append(torch.stack(instance["initial_pixel_values_hist"]))
                target_pixel_values_hist.append(torch.stack(instance["target_pixel_values_hist"]))
                with_hist.append(torch.tensor(True))
            else:
                with_hist.append(torch.tensor(False))   

        
        pixel_values = [instance["pixel_values"] for instance in instances]
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

        # For low-level policy training
        actions = [instance["actions"] for instance in instances]
        actions = torch.stack(actions, dim=0)

        instructions = [instance["lang"] for instance in instances]


        # [Contract] For VLA Training =>> No "Unimodal" Data!
        assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"

        # Stack all `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]

        pixel_values = torch.stack(pixel_values)
        initial_pixel_values = torch.stack(initial_pixel_values)
        target_pixel_values = torch.stack(target_pixel_values)
        initial_pixel_values_hist = torch.stack(initial_pixel_values_hist) if len(initial_pixel_values_hist) > 0 else []
        target_pixel_values_hist = torch.stack(target_pixel_values_hist) if len(target_pixel_values_hist) > 0 else []
        with_hist = torch.stack(with_hist)

        output = dict(
            pixel_values=pixel_values,
            initial_pixel_values=initial_pixel_values,
            target_pixel_values=target_pixel_values,
            initial_pixel_values_hist=initial_pixel_values_hist,
            target_pixel_values_hist=target_pixel_values_hist,
            instructions=instructions,
            with_hist=with_hist,
            # input_ids=input_ids,
            # attention_mask=attention_mask,
            # labels=labels,
            actions=actions,
            # proprio=proprio
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output
        
@dataclass
class PaddedCollatorForActionPrediction_CALVIN:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        
        initial_pixel_values = [instance["initial_pixel_values"] for instance in instances]
        target_pixel_values = [instance["target_pixel_values"] for instance in instances]

        initial_pixel_values_hist, target_pixel_values_hist = [], []
        with_hist = []
        for instance in instances:
            if instance["initial_pixel_values_hist"] is not None:
                initial_pixel_values_hist.append(instance["initial_pixel_values_hist"])
                target_pixel_values_hist.append(instance["target_pixel_values_hist"])
                with_hist.append(torch.tensor(True))
            else:
                with_hist.append(torch.tensor(False))     



        pixel_values = [instance["pixel_values"] for instance in instances]
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None


        # For low-level policy training
        actions = [instance["actions"] for instance in instances]
        actions = torch.stack(actions, dim=0)

        proprio = [instance["proprio"] for instance in instances]
        proprio = torch.stack(proprio, dim=0)

        instructions = [instance["lang"] for instance in instances]


        # [Contract] For VLA Training =>> No "Unimodal" Data!
        assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"

        # Stack all `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]
        pixel_values = torch.stack(pixel_values)
        initial_pixel_values = torch.stack(initial_pixel_values)
        target_pixel_values = torch.stack(target_pixel_values)
        initial_pixel_values_hist = torch.stack(initial_pixel_values_hist) if len(initial_pixel_values_hist) > 0 else []
        target_pixel_values_hist = torch.stack(target_pixel_values_hist) if len(target_pixel_values_hist) > 0 else []
        with_hist = torch.stack(with_hist)

        output = dict(
            pixel_values=pixel_values,
            initial_pixel_values=initial_pixel_values,
            target_pixel_values=target_pixel_values,
            initial_pixel_values_hist=initial_pixel_values_hist,
            target_pixel_values_hist=target_pixel_values_hist,
            instructions=instructions,
            with_hist=with_hist,
            # input_ids=input_ids,
            # attention_mask=attention_mask,
            # labels=labels,
            actions=actions,
            proprio=proprio
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output


@dataclass
class CollatorForLatentAction:
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

        initial_pixel_values = [instance["initial_pixel_values"] for instance in instances]
        initial_pixel_values = torch.stack(initial_pixel_values)
        
        target_pixel_values = [instance["target_pixel_values"] for instance in instances]
        target_pixel_values = torch.stack(target_pixel_values)
        if "middle_pixel_values" in instances[0]:
            middle_pixel_values = [instance["middle_pixel_values"] for instance in instances]
            middle_pixel_values = torch.stack(middle_pixel_values)
            pixel_values = torch.stack([initial_pixel_values, target_pixel_values], dim=1)
            radprog_mid_offsets = torch.tensor(
                [float(instance["radprog_h1_offset"]) for instance in instances],
                dtype=torch.float32,
            )
            radprog_future_offsets = torch.tensor(
                [float(instance["radprog_h2_offset"]) for instance in instances],
                dtype=torch.float32,
            )
            radprog_valid = torch.tensor(
                [float(instance["radprog_valid"]) for instance in instances],
                dtype=torch.float32,
            )
        else:
            pixel_values = torch.stack([initial_pixel_values, target_pixel_values], dim=1)


        action = [torch.from_numpy(instance["action"]) for instance in instances]
        action = torch.stack(action)

        # removing all punctuation in task instruction
        task_instruction = [re.sub('[{}]'.format(string.punctuation),"",instance["task_instruction"]) for instance in instances]


        output = dict(
            videos=pixel_values,
            task_instruction=task_instruction,
            action=action,
        )
        if "middle_pixel_values" in instances[0]:
            output["radprog_mid_pixel_values"] = middle_pixel_values
            output["radprog_mid_offsets"] = radprog_mid_offsets
            output["radprog_future_offsets"] = radprog_future_offsets
            output["radprog_valid"] = radprog_valid
        if dataset_names is not None:
            output["dataset_names"] = dataset_names

        return output


@dataclass
class CollatorForMultiViewVideo:
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

        initial_pixel_values = [instance["initial_pixel_values"] for instance in instances]
        initial_pixel_values = torch.stack(initial_pixel_values)
        
        target_pixel_values = [instance["target_pixel_values"] for instance in instances]
        target_pixel_values = torch.stack(target_pixel_values)
        pixel_values = torch.stack([initial_pixel_values, target_pixel_values], dim=1)


        initial_pixel_values_view2 = [instance["initial_pixel_values_view2"] for instance in instances]
        initial_pixel_values_view2 = torch.stack(initial_pixel_values_view2)
        
        target_pixel_values_view2 = [instance["target_pixel_values_view2"] for instance in instances]
        target_pixel_values_view2 = torch.stack(target_pixel_values_view2)
        pixel_values_view2 = torch.stack([initial_pixel_values_view2, target_pixel_values_view2], dim=1)
        


        action = [torch.from_numpy(instance["action"]) for instance in instances]
        action = torch.stack(action)

        # removing all punctuation in task instruction
        task_instruction = [re.sub('[{}]'.format(string.punctuation),"",instance["task_instruction"]) for instance in instances]


        output = dict(
            videos=pixel_values,
            videos_view2=pixel_values_view2,
            task_instruction=task_instruction,
            action=action,
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names

        return output
