from typing import Any, Optional, Union
import warnings
from contextlib import patch
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
import torch
from trl import GRPOTrainer, GRPOConfig
from transformers import PreTrainedModel, PreTrainedTokenizerBase, GenerationConfig
from torch.utils.data import Dataset, IterableDataset
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import is_peft_available
from transformers.utils import is_vllm_available
from transformers.trainer_pt_utils import pad

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation



if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_vllm_available():
    from vllm import LLM, SamplingParams

class GRPOVerifierTrainer(GRPOTrainer):
    
    _tag_names = ["trl", "grpo"]
    
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        env: Optional[Any] = None,
        use_vllm: bool = False,
    ):
    
        self.env = env
        self.use_vllm = use_vllm
        
        super().__init__(
            model,
            reward_funcs,
            args,
            train_dataset,
            eval_dataset,
            processing_class,
            reward_processing_classes,
            callbacks,
            optimizers,
            peft_config,
        )
        
        if self.use_vllm:
                if not is_vllm_available():
                    raise ImportError(
                        "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                        "`pip install vllm` to use it."
                    )

                if self.accelerator.is_main_process:
                    vllm_device = self.args.vllm_device
                    if vllm_device == "auto":
                        if torch.cuda.device_count() == 1:
                            vllm_device = "cuda:0"  # particular case when training with onyl 1 GPU: share it
                        else:
                            vllm_device = f"cuda:{self.accelerator.num_processes}"  # take the next GPU idx
                    # Check that the requested device is available
                    if vllm_device.split(":")[0] == "cuda" and int(vllm_device.split(":")[1]) >= torch.cuda.device_count():
                        raise ValueError(
                            f"The requested device for vllm ({vllm_device}) is not available. You are likely using vLLM "
                            "without restricting the number of GPUs for training. Set the `--num_processes` argument to a "
                            "value lower than the number of GPUs available on your machine—typically, reducing it by one "
                            f"is sufficient. In your case: `--num_processes {torch.cuda.device_count() - 1}`."
                        )
                    # Check that the requested device is not also used for training
                    if vllm_device in {f"cuda:{idx}" for idx in range(self.accelerator.num_processes)}:
                        warnings.warn(
                            f"The requested device {vllm_device} is also being used for training. For higher throughput "
                            "and to avoid out-of-memory errors, it is recommended to use a dedicated device for vLLM. "
                            "If this is intentional, you may ignore this warning but should adjust "
                            "`vllm_gpu_memory_utilization` accordingly."
                        )
                    # vLLM is not compatible with accelerate. So we need to patch it to make sure we can (1) place the vLLM
                    # model on the desired device (world_size_patch) and (2) avoid a test that is not designed for our
                    # setting (profiling_patch).
                    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
                     profiling_patch = patch(
                        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None
                    )
                    with world_size_patch, profiling_patch:
                        self.llm = LLM(
                            model=model.name_or_path,
                            device=vllm_device,
                            gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
                            dtype=self.args.vllm_dtype,
                            enable_prefix_caching=True,
                            max_model_len=self.args.vllm_max_model_len,
                        )
                    self.sampling_params = SamplingParams(
                        temperature=args.temperature,
                        max_tokens=self.max_completion_length,
                    )

                self._last_loaded_step = 0  # tag to avoid useless loading during grad accumulation

                # When using vLLM, the main process is responsible for loading the model weights. This can cause process
                # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
                # synchronize all processes after vLLM has been fully initialized.
                self.accelerator.wait_for_everyone()
            else:
                self.generation_config = GenerationConfig(
                    max_new_tokens=self.max_completion_length,
                    do_sample=True,
                    temperature=args.temperature,
                    pad_token_id=processing_class.pad_token_id,
                )
            
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)
                
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs
    
    def _sample_completions(self, 
                            model, 
                            prompts, 
                            prompts_inputs,
                            prompt_texts,
                            device):
        if self.args.use_vllm:
            if self.state.global_step != self._last_loaded_step:
                with unwrap_model_for_generation(model, self.accelerator) as model:
                    state_dict = unwrapped_model.state_dict()
                if self.accelerator.is_main_process:
                    llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                    llm_model.load_weights(state_dict.items())
                self._last_loaded_step = self.state.global_step
            
            all_prompts_text = gather_object(prompts_text)
            all_prompts = gather_object(prompts)
            if self.accelerator.is_main_process:
                if self.env is not None:
                    outputs = self.env.generate(
                        prompts=all_prompts,
                        llm=self.llm,
                        sampling_params=self.sampling_params
                    )
                    completion_ids = [out.token_ids for completions in outputs for out in completions.outputs]
                else:
                    outputs = self.llm.generate(all_prompts_text, sampling_params=self.sampling_params, use_tqdm=False)
                    completion_ids = [out.token_ids for completions in outputs for out in completions.outputs]
            else:
                completion_ids = [None] * len(all_prompts_text) * self.num_generations

            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts_text) * self.num_generations,
                (self.accelerator.process_index + 1) * len(prompts_text) * self.num_generations,
            )
            completion_ids = completion_ids[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_inputs_repeated = torch.repeat_interleave(prompt_inputs["input_ids"], self.num_generations, dim=0)
            prompt_completion_ids = torch.cat([prompt_inputs_repeated, completion_ids], dim=1)
        else:
            # Regular generation path
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(
                    **prompt_inputs, generation_config=self.generation_config
                )
                
        return prompt_completion_ids
    
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        device = self.accelerator.device

        # text prompts
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]

        # tokenize prompts
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        # truncate prompts if necessary
        if self.max_prompt_length is not None:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length :]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length :]

        # Generate completions using either vLLM or regular generation (returns prompt_completion_ids)
        prompt_length = prompt_inputs["input_ids"].size(1)
        prompt_completion_ids = self._sample_completions(model, prompts, prompt_inputs, prompts_text, device)
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        prompt_mask_repeated = prompt_inputs["attention_mask"].repeat_interleave(self.num_generations, dim=0)
        attention_mask = torch.cat([prompt_mask_repeated, completion_mask], dim=1)  # (B*G, P+C)

        # Get the per-token log probabilities for the completions for the model and the reference model
        def get_per_token_logps(model, input_ids, attention_mask, logits_to_keep):
            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            logits = model(
                input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1
            ).logits  # (B, L, V)
            logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

            # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
            per_token_logps = []
            for logits_row, input_ids_row in zip(logits, input_ids[:, -logits_to_keep:]):
                log_probs = logits_row.log_softmax(dim=-1)
                token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
                per_token_logps.append(token_log_prob)
            return torch.stack(per_token_logps)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        per_token_logps = get_per_token_logps(model, prompt_completion_ids, attention_mask, logits_to_keep)

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = get_per_token_logps(
                        model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        # Compute the KL divergence between the model and the reference model
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        # Compute the rewards
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        # Repeat each value in the column for `num_generations` times
                        reward_kwargs[key].extend([example[key]] * self.num_generations)
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Sum the rewards from all reward functions
        rewards = rewards_per_func.sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # x - x.detach() allows for preserving gradients from x
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())

        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        return loss
                
    
