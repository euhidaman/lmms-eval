import torch

torch.backends.cuda.matmul.allow_tf32 = True

import copy
import warnings
from datetime import timedelta
from typing import List, Optional, Tuple, Union

from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from packaging import version
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.utils import stop_sequences_criteria

warnings.filterwarnings("ignore")

from loguru import logger as eval_logger

try:
    from tinyllava.data import ImagePreprocess, TextPreprocess
    from tinyllava.model import load_pretrained_model
    from tinyllava.utils.constants import DEFAULT_IMAGE_TOKEN
    from tinyllava.utils.message import Message
except Exception as e:
    eval_logger.debug("TinyLLaVA_Factory is not installed. Please install TinyLLaVA_Factory to use this model.\nError: %s" % e)

# inference implementation for attention, can be "sdpa", "eager", "flash_attention_2". Seems FA2 is not effective during inference: https://discuss.huggingface.co/t/flash-attention-has-no-effect-on-inference/73453/5
# if is_flash_attn_2_available:
#     best_fit_attn_implementation = "flash_attention_2" # flash_attn has a bug that says: ERROR Error query and key must have the same dtype in generating

if version.parse(torch.__version__) >= version.parse("2.1.2"):
    best_fit_attn_implementation = "sdpa"
else:
    best_fit_attn_implementation = "eager"


@register_model("tinyllava")
class TinyLlava(lmms):
    """
    TinyLlava Model
    """

    def __init__(
        self,
        pretrained: str = "tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B",
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        device_map="cuda:0",
        conv_mode="phi",  # TODO
        use_cache=True,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        self._model, self._tokenizer, self._image_processor, self._max_length = load_pretrained_model(pretrained, device_map=self.device_map)
        data_args = self._model.config
        self._image_processor = ImagePreprocess(self._image_processor, data_args)
        assert self._tokenizer.padding_side == "right", "Not sure but seems like `right` is a natural choice for padding?"
        self._text_processor = TextPreprocess(self._tokenizer, conv_mode)

        self._config = self._model.config
        self.model.eval()
        self.model.tie_weights()
        # self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        # self.conv_template = conv_template
        self.use_cache = use_cache
        # self.truncate_context = truncate_context

        # assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")

            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._world_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        try:
            return self.tokenizer.decode(tokens)
        except:
            return self.tokenizer.decode([tokens])

    def flatten(self, input):
        if not input or any(i is None for i in input):
            return []
        new_list = []
        for i in input:
            if i:
                for j in i:
                    new_list.append(j)
        return new_list

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # encode, pad, and truncate contexts for this batch
            if type(doc_to_target) == str:
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            image_sizes = [[visual.size[0], visual.size[1]] for visual in visuals]
            if visuals:
                # https://github.com/zjysteven/TinyLLaVA_Factory/blob/main/tinyllava/data/image_preprocess.py
                # tinyllava's image processor seems to take each individual image as input
                image = [self._image_processor(v) for v in visuals]
                if type(image) is list:
                    image = [_image.to(dtype=torch.float16, device=self.device) for _image in image]
                    # as of 2024/06, tinyllava only accepts `images` input to be a tensor
                    image = torch.stack(image)
                else:
                    image = image.to(dtype=torch.float16, device=self.device)
            else:
                image = None

            prompts_input = contexts[0] if isinstance(contexts, list) else contexts

            if image is not None and len(image) != 0 and DEFAULT_IMAGE_TOKEN not in prompts_input:
                """
                Three senarios:
                1. No image, and there for, no image token should be added.
                2. image token is already specified in the context, so we don't need to add it.
                3. image token is not specified in the context and there is image inputs, so we need to add it. In this case, we add the image token at the beginning of the context and add a new line.
                """
                image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visuals)
                image_tokens = " ".join(image_tokens)
                prompts_input = image_tokens + "\n" + (contexts[0] if isinstance(contexts, list) else contexts)

            msg = Message()
            msg.add_message(prompts_input)

            # Process text input and get input_ids
            contxt_id = self._text_processor(msg.messages, mode="eval")["input_ids"]

            # Set the continuation as the second role's response
            msg._messages[1]["value"] = continuation
            input_ids = self._text_processor(msg.messages, mode="eval")["input_ids"]

            # Prepare labels and ensure the correct shape
            labels = input_ids.clone()
            if labels.dim() == 1:
                labels = labels.unsqueeze(0)  # Convert to (1, seq_len) if needed

            if len(contxt_id.shape) == 1:
                contxt_id = contxt_id.unsqueeze(0)  # Convert to (1, context_len)

            # Mask the context part to ignore it in loss computation
            labels[:, : contxt_id.shape[1]] = -100

            # Move tensors to the correct device
            device = self.device
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            if len(input_ids.shape) == 1:
                input_ids = input_ids.unsqueeze(0)  # Ensure it is (batch_size, seq_len)

            # Handle image input if available
            if image is None:
                image_sizes = []
                with torch.inference_mode():
                    outputs = self.model(input_ids=input_ids, labels=labels, use_cache=True)
            else:
                with torch.inference_mode():
                    outputs = self.model(input_ids=input_ids, labels=labels, images=image, use_cache=True, image_sizes=image_sizes)

            loss = outputs["loss"]
            # loss = torch.exp(loss)
            logits = outputs["logits"]
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = input_ids[:, contxt_id.shape[1] :]  # [1, seq]
            greedy_tokens = greedy_tokens[:, contxt_id.shape[1] : input_ids.shape[1]]  # [1, seq]
            max_equal = (greedy_tokens == cont_toks).all()
            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)
        pbar.close()
        return res

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            batched_visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]  # [B, N]
            flattened_visuals = self.flatten(batched_visuals)  # [B*N]
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            until = [self.tok_decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")

            if "image_aspect_ratio" in gen_kwargs.keys() and "image_aspect_ratio" not in self._config.__dict__:
                # here we should pop it out of gen_kwargs so that it doesn't get passed to the model for next step of generation
                self._config.image_aspect_ratio = gen_kwargs.pop("image_aspect_ratio")
                eval_logger.info(f"Setting image aspect ratio: {self._config.image_aspect_ratio}")
            # encode, pad, and truncate contexts for this batch
            if flattened_visuals:
                image_tensor = [self._image_processor(v) for v in flattened_visuals]
                if type(image_tensor) is list:
                    image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                    # as of 2024/06, tinyllava only accepts `images` input to be a tensor
                    image_tensor = torch.stack(image_tensor)
                else:
                    image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)
            else:
                image_tensor = None

            # prompts_input = contexts[0]

            question_input = []

            for visual, context in zip(batched_visuals, contexts):
                if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in context:
                    """
                    Three senarios:
                    1. No image, and there for, no image token should be added.
                    2. image token is already specified in the context, so we don't need to add it.
                    3. image token is not specified in the context and there is image inputs, so we need to add it. In this case, we add the image token at the beginning of the context and add a new line.
                    """
                    image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visual) if isinstance(visual, list) else [DEFAULT_IMAGE_TOKEN]
                    image_tokens = " ".join(image_tokens)
                    question = image_tokens + "\n" + context
                else:
                    question = context

                msg = Message()
                msg.add_message(question)
                prompt_question = self._text_processor(msg.messages, mode="eval")["prompt"]
                question_input.append(prompt_question)

            # The above for loop has bugs. When there is no visuals, e.g. pure text,
            # there will be no for loop execute resulting in an empty question_input (because no visuals)
            # Scenario 1 won't even be execute
            if len(flattened_visuals) == 0:
                for context in contexts:
                    question = context
                    msg = Message()
                    msg.add_message(question)
                    prompt_question = self._text_processor(msg.messages, mode="eval")["prompt"]
                    question_input.append(prompt_question)

            # input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            # preconfigure gen_kwargs with defaults
            gen_kwargs["image_sizes"] = [flattened_visuals[idx].size for idx in range(len(flattened_visuals))]
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            # input_ids_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in question_input]
            input_ids_list = [self._text_processor.template.tokenizer_image_token(prompt, self.tokenizer, return_tensors="pt") for prompt in question_input]
            pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device)
            attention_masks = input_ids.ne(pad_token_ids).to(self.device)
            # These steps are not in LLaVA's original code, but are necessary for generation to work
            # TODO: attention to this major generation step...
            try:
                cont = self.model.generate(
                    input_ids,
                    attention_mask=attention_masks,
                    pad_token_id=pad_token_ids,
                    images=image_tensor,
                    image_sizes=gen_kwargs["image_sizes"],
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                )
                text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
            except Exception as e:
                raise e
                eval_logger.error(f"Error {e} in generating")
                cont = ""
                text_outputs = [""]

            # cont_toks_list = cont.tolist()
            # for cont_toks, context in zip(cont_toks_list, contexts):
            # discard context + left-padding toks if using causal decoder-only LMM
            # if self.truncate_context:
            #     cont_toks = cont_toks[input_ids.shape[1] :]
            # use secondary stop seqs to cut off should-have-been-stopped content post-hoc
            # if self.truncate_context:
            #     for term in until:
            #         if len(term) > 0:
            #             # ignore '' separator,
            #             # for seq2seq case where self.tok_decode(self.eot_token_id) = ''
            #             text_outputs = text_outputs.split(term)[0]
            res.extend(text_outputs)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")


@register_model("embervlm")
class EmberVLM(lmms):
    """
    EmberVLM Model - Tiny VLM for Robot Fleet Selection
    """

    def __init__(
        self,
        pretrained: str = "outputs/mobilevit_xs_smollm_135m/final",
        device: str = "cuda",
        batch_size: int = 1,
        max_length: int = 2048,
        **kwargs,
    ) -> None:
        super().__init__()
        self.batch_size_per_gpu = batch_size
        self._max_length = max_length

        # Get checkpoint path from environment or parameter
        import os
        from pathlib import Path
        import torch
        checkpoint_path = os.environ.get('EMBERVLM_CHECKPOINT', pretrained)
        checkpoint_path = Path(checkpoint_path)

        eval_logger.info(f"Loading EmberVLM from {checkpoint_path}")

        # Load EmberVLM model
        try:
            from embervlm.models import EmberVLM as EmberVLMModel
            from transformers import AutoTokenizer

            # Load model from checkpoint directory
            self.model = EmberVLMModel.from_pretrained(str(checkpoint_path))

            # Move to device first to enable embedding inspection
            self.model = self.model.to(device).eval()
            self._device = torch.device(device)
            self._config = getattr(self.model, 'config', None)
            self.image_preprocessor = getattr(self.model, 'image_preprocessor', None)

            # Get the actual embedding vocab size from the loaded model FIRST
            self._actual_vocab_size = self._get_embedding_vocab_size()
            if self._actual_vocab_size is None:
                raise RuntimeError("❌ Could not determine model embedding size! Cannot safely run evaluation.")

            eval_logger.info(f"📊 Model embedding matrix size: {self._actual_vocab_size}")

            # Resolve tokenizer path: prefer sibling 'tokenizer' directory
            tokenizer_path = checkpoint_path / 'tokenizer'
            if not tokenizer_path.exists():
                # checkpoint_path = .../stage2/checkpoint-epoch-X
                # tokenizer lives at .../stage2/../tokenizer
                tokenizer_path = checkpoint_path.parent.parent / 'tokenizer'

            if tokenizer_path.exists():
                self._tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
                eval_logger.info(f"✓ Loaded tokenizer from {tokenizer_path}")
            else:
                self._tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
                eval_logger.warning(f"⚠️  Tokenizer not found at {tokenizer_path}, using default SmolLM-135M")

            tokenizer_vocab_size = len(self._tokenizer)
            eval_logger.info(f"📊 Tokenizer vocabulary size: {tokenizer_vocab_size}")

            # CRITICAL FIX: Resize embeddings if there's a mismatch
            if tokenizer_vocab_size != self._actual_vocab_size:
                eval_logger.warning(
                    f"⚠️  MISMATCH: Tokenizer({tokenizer_vocab_size}) ≠ Model({self._actual_vocab_size})"
                )

                if tokenizer_vocab_size > self._actual_vocab_size:
                    eval_logger.warning(f"   → Resizing model embeddings from {self._actual_vocab_size} to {tokenizer_vocab_size}...")

                    # Resize the embeddings
                    try:
                        if hasattr(self.model, 'language_model'):
                            if hasattr(self.model.language_model, 'resize_token_embeddings'):
                                self.model.language_model.resize_token_embeddings(tokenizer_vocab_size)
                                eval_logger.info(f"   ✓ Resized via language_model.resize_token_embeddings")
                            elif hasattr(self.model.language_model, 'model') and hasattr(self.model.language_model.model, 'resize_token_embeddings'):
                                self.model.language_model.model.resize_token_embeddings(tokenizer_vocab_size)
                                eval_logger.info(f"   ✓ Resized via language_model.model.resize_token_embeddings")
                            else:
                                raise RuntimeError("No resize_token_embeddings method found")

                            # Update config vocab_size
                            if hasattr(self.model.language_model, 'config'):
                                self.model.language_model.config.vocab_size = tokenizer_vocab_size
                            if hasattr(self.model.language_model, 'model') and hasattr(self.model.language_model.model, 'config'):
                                self.model.language_model.model.config.vocab_size = tokenizer_vocab_size

                            self._actual_vocab_size = tokenizer_vocab_size
                            eval_logger.info(f"✓ Model embeddings resized to {self._actual_vocab_size}")
                        else:
                            raise RuntimeError("language_model not found in model")
                    except Exception as e:
                        eval_logger.error(f"❌ Failed to resize embeddings: {e}")
                        raise
                else:
                    eval_logger.warning(f"   → Tokenizer is smaller - this is unusual but proceeding with caution")

            # Ensure pad token is set
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
                eval_logger.info(f"✓ Set pad_token = eos_token ({self._tokenizer.eos_token_id})")

            # Fix all special token IDs to be within valid range
            self._fix_all_token_ids(self._actual_vocab_size)

            # Final validation
            final_vocab = self._get_embedding_vocab_size()
            eval_logger.info(f"")
            eval_logger.info(f"{'='*60}")
            eval_logger.info(f"FINAL TOKENIZER/MODEL ALIGNMENT:")
            eval_logger.info(f"  Model embedding size: {final_vocab}")
            eval_logger.info(f"  Tokenizer vocab size: {len(self._tokenizer)}")
            eval_logger.info(f"  EOS token ID: {self._tokenizer.eos_token_id}")
            eval_logger.info(f"  PAD token ID: {self._tokenizer.pad_token_id}")
            eval_logger.info(f"  Status: {'✓ ALIGNED' if final_vocab == len(self._tokenizer) else '❌ MISALIGNED'}")
            eval_logger.info(f"{'='*60}")
            eval_logger.info(f"")

            eval_logger.info(f"✓ EmberVLM loaded successfully on {device}")

        except Exception as e:
            eval_logger.error(f"Failed to load EmberVLM: {e}")
            raise

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model_instance(self):
        return self.model

    @property
    def eot_token_id(self):
        if self.tokenizer:
            return self.tokenizer.eos_token_id
        return None

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    def _get_embedding_vocab_size(self) -> Optional[int]:
        """Get the actual vocab size from model embeddings."""
        try:
            # Try multiple paths to find the embedding layer
            if hasattr(self.model, 'language_model'):
                lm = self.model.language_model
                # SmolLMBackbone or PretrainedTinyLLMBackbone wrap the actual HF model in .model
                if hasattr(lm, 'model'):
                    inner_model = lm.model
                    if hasattr(inner_model, 'get_input_embeddings'):
                        emb = inner_model.get_input_embeddings()
                        if emb is not None:
                            return emb.weight.shape[0]
                # Direct access
                if hasattr(lm, 'get_input_embeddings'):
                    emb = lm.get_input_embeddings()
                    if emb is not None:
                        return emb.weight.shape[0]
            # Fallback: check model directly
            if hasattr(self.model, 'get_input_embeddings'):
                emb = self.model.get_input_embeddings()
                if emb is not None:
                    return emb.weight.shape[0]
            # Fallback to config vocab size if embeddings aren't accessible
            if hasattr(self.model, 'config') and getattr(self.model.config, 'vocab_size', None):
                return int(self.model.config.vocab_size)
        except Exception as e:
            eval_logger.warning(f"Could not determine embedding vocab size: {e}")
        return None

    def _fix_all_token_ids(self, vocab_size: int):
        """Fix all token IDs in tokenizer and model configs to be within vocab_size."""
        safe_eos_id = (vocab_size - 1) if vocab_size > 0 else 0

        # 1. Fix tokenizer special token IDs
        if self._tokenizer is not None:
            if self._tokenizer.eos_token_id is None or self._tokenizer.eos_token_id >= vocab_size:
                eval_logger.warning(f"Fixing tokenizer eos_token_id from {self._tokenizer.eos_token_id} to {safe_eos_id}")
                self._tokenizer.eos_token_id = safe_eos_id
            if self._tokenizer.pad_token_id is None or self._tokenizer.pad_token_id >= vocab_size:
                eval_logger.warning(f"Fixing tokenizer pad_token_id from {self._tokenizer.pad_token_id} to {self._tokenizer.eos_token_id}")
                self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        # 2. Fix the underlying HuggingFace model config (used by generate())
        try:
            if hasattr(self.model, 'language_model'):
                lm = self.model.language_model
                # Fix the wrapper's config
                if hasattr(lm, 'config'):
                    if hasattr(lm.config, 'eos_token_id') and (lm.config.eos_token_id is None or lm.config.eos_token_id >= vocab_size):
                        lm.config.eos_token_id = safe_eos_id
                    if hasattr(lm.config, 'pad_token_id') and (lm.config.pad_token_id is None or lm.config.pad_token_id >= vocab_size):
                        lm.config.pad_token_id = safe_eos_id
                    if hasattr(lm.config, 'bos_token_id') and lm.config.bos_token_id is not None and lm.config.bos_token_id >= vocab_size:
                        lm.config.bos_token_id = safe_eos_id

                # Fix the inner HF model's config (this is what generate() actually uses!)
                if hasattr(lm, 'model') and hasattr(lm.model, 'config'):
                    inner_config = lm.model.config
                    if hasattr(inner_config, 'eos_token_id') and (inner_config.eos_token_id is None or inner_config.eos_token_id >= vocab_size):
                        eval_logger.warning(f"Fixing inner model eos_token_id from {inner_config.eos_token_id} to {safe_eos_id}")
                        inner_config.eos_token_id = safe_eos_id
                    if hasattr(inner_config, 'pad_token_id') and (inner_config.pad_token_id is None or inner_config.pad_token_id >= vocab_size):
                        eval_logger.warning(f"Fixing inner model pad_token_id from {inner_config.pad_token_id} to {safe_eos_id}")
                        inner_config.pad_token_id = safe_eos_id
                    if hasattr(inner_config, 'bos_token_id') and inner_config.bos_token_id is not None and inner_config.bos_token_id >= vocab_size:
                        inner_config.bos_token_id = safe_eos_id

                # Also fix generation_config if it exists
                if hasattr(lm, 'model') and hasattr(lm.model, 'generation_config'):
                    gen_config = lm.model.generation_config
                    if hasattr(gen_config, 'eos_token_id') and gen_config.eos_token_id is not None:
                        if isinstance(gen_config.eos_token_id, int) and gen_config.eos_token_id >= vocab_size:
                            gen_config.eos_token_id = safe_eos_id
                        elif isinstance(gen_config.eos_token_id, list):
                            gen_config.eos_token_id = [min(x, vocab_size - 1) for x in gen_config.eos_token_id]
                    if hasattr(gen_config, 'pad_token_id') and gen_config.pad_token_id is not None and gen_config.pad_token_id >= vocab_size:
                        gen_config.pad_token_id = safe_eos_id
                    if hasattr(gen_config, 'bos_token_id') and gen_config.bos_token_id is not None and gen_config.bos_token_id >= vocab_size:
                        gen_config.bos_token_id = safe_eos_id
        except Exception as e:
            eval_logger.warning(f"Failed to fix model config token IDs: {e}")

    def _ensure_generation_token_ids(self) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """Ensure pad/eos/bos token IDs are valid for generation and return them."""
        vocab_size = self._get_embedding_vocab_size()
        if vocab_size is None:
            vocab_size = getattr(self, '_actual_vocab_size', None)
        if vocab_size is None and self._tokenizer is not None:
            vocab_size = len(self._tokenizer)

        if vocab_size is None:
            return None, None, None

        safe_id = (vocab_size - 1) if vocab_size > 0 else 0

        eos_id = self._tokenizer.eos_token_id if self._tokenizer and self._tokenizer.eos_token_id is not None else safe_id
        if eos_id >= vocab_size:
            eos_id = safe_id

        pad_id = self._tokenizer.pad_token_id if self._tokenizer and self._tokenizer.pad_token_id is not None else eos_id
        if pad_id >= vocab_size:
            pad_id = eos_id

        bos_id = None
        if self._tokenizer and getattr(self._tokenizer, 'bos_token_id', None) is not None:
            bos_id = self._tokenizer.bos_token_id
            if bos_id >= vocab_size:
                bos_id = safe_id

        # Update tokenizer ids
        if self._tokenizer is not None:
            self._tokenizer.eos_token_id = eos_id
            self._tokenizer.pad_token_id = pad_id
            if bos_id is not None:
                self._tokenizer.bos_token_id = bos_id

        # Update model configs used by generate()
        try:
            if hasattr(self.model, 'language_model'):
                lm = self.model.language_model
                if hasattr(lm, 'config'):
                    lm.config.eos_token_id = eos_id
                    lm.config.pad_token_id = pad_id
                    if bos_id is not None:
                        lm.config.bos_token_id = bos_id
                if hasattr(lm, 'model') and hasattr(lm.model, 'config'):
                    lm.model.config.eos_token_id = eos_id
                    lm.model.config.pad_token_id = pad_id
                    if bos_id is not None:
                        lm.model.config.bos_token_id = bos_id
                if hasattr(lm, 'model') and hasattr(lm.model, 'generation_config'):
                    gen_config = lm.model.generation_config
                    gen_config.eos_token_id = eos_id
                    gen_config.pad_token_id = pad_id
                    if bos_id is not None:
                        gen_config.bos_token_id = bos_id
        except Exception as e:
            eval_logger.warning(f"Failed to enforce generation token ids: {e}")

        return pad_id, eos_id, bos_id

    def flatten(self, input):
        if not input:
            return []
        new_list = []
        for i in input:
            if isinstance(i, (list, tuple)):
                new_list.extend(i)
            else:
                new_list.append(i)
        return new_list

    def _load_image(self, visual):
        from PIL import Image
        if isinstance(visual, Image.Image):
            return visual.convert('RGB')
        if isinstance(visual, str):
            return Image.open(visual).convert('RGB')
        return None

    def _pil_to_tensor(self, image):
        import numpy as np
        import torch
        image_np = np.array(image).astype('float32') / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
        return image_tensor

    def tok_encode(self, string: str, **kwargs):
        if self.tokenizer:
            return self.tokenizer.encode(string, add_special_tokens=False)
        return []

    def tok_decode(self, tokens, **kwargs):
        if self.tokenizer:
            return self.tokenizer.decode(tokens)
        return ""

    def clean(self):
        """Override clean method with safer CUDA cleanup."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception as e:
            eval_logger.warning(f"CUDA cleanup warning (non-critical): {e}")

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("EmberVLM does not support loglikelihood evaluation")

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """
        Generate responses for EmberVLM evaluation.
        
        Args:
            requests: List of Instance objects containing:
                - doc: Dictionary with 'image' and 'question' fields
                - arguments: Generation parameters
        
        Returns:
            List of generated text responses
        """
        import torch

        results = []
        for request in tqdm(requests, desc="EmberVLM inference"):
            try:
                contexts, gen_kwargs, doc_to_visual, doc_id, task, split = request.arguments
                gen_kwargs = gen_kwargs or {}

                max_new_tokens = gen_kwargs.get("max_new_tokens", 128)
                temperature = gen_kwargs.get("temperature", 0.0)
                top_p = gen_kwargs.get("top_p", 0.9)
                top_k = gen_kwargs.get("top_k", 50)
                do_sample = gen_kwargs.get("do_sample", temperature > 0)

                visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
                visuals = self.flatten(visuals)
                image = None
                if visuals:
                    image = self._load_image(visuals[0])

                prompt = contexts or ""
                prompt = prompt.replace("<|image|>", "").replace("<image>", "").strip()

                if self.tokenizer:
                    # Avoid padding for multimodal generation to prevent invalid pad token usage
                    # Also cap text length to leave room for visual tokens within model position limits
                    model_max_positions = None
                    if self.model is not None and hasattr(self.model, 'language_model'):
                        lm_config = getattr(self.model.language_model, 'config', None)
                        if lm_config is not None:
                            model_max_positions = getattr(lm_config, 'max_position_embeddings', None)
                            if model_max_positions is None:
                                model_max_positions = getattr(lm_config, 'n_positions', None)

                    num_visual_tokens = getattr(getattr(self.model, 'config', None), 'num_visual_tokens', 0) or 0
                    max_text_len = 1024
                    if model_max_positions is not None:
                        max_text_len = max(1, min(max_text_len, model_max_positions - num_visual_tokens))

                    inputs = self.tokenizer(
                        prompt,
                        return_tensors='pt',
                        padding=False,
                        truncation=True,
                        max_length=max_text_len,
                    )
                    input_ids = inputs['input_ids'].to(self.device)
                    # Clamp/replace token IDs to model vocab size to avoid CUDA index errors
                    vocab_size = getattr(self, '_actual_vocab_size', None)
                    if vocab_size is not None:
                        input_ids = input_ids.clone()
                        oov_mask = (input_ids >= vocab_size) | (input_ids < 0)
                        if oov_mask.any():
                            replacement_id = min(self.tokenizer.eos_token_id or 0, vocab_size - 1)
                            eval_logger.debug(f"Replacing {oov_mask.sum().item()} OOV tokens with {replacement_id}")
                            input_ids[oov_mask] = replacement_id
                    attention_mask = None
                else:
                    input_ids = None
                    attention_mask = None

                pixel_values = None
                if image is not None and self.image_preprocessor is not None:
                    image_tensor = self._pil_to_tensor(image).to(self.device, dtype=torch.float32)
                    pixel_values = self.image_preprocessor(image_tensor)
                    pixel_values = pixel_values.to(self.device)

                image_positions = None
                if pixel_values is not None and input_ids is not None:
                    image_positions = torch.zeros(input_ids.size(0), dtype=torch.long, device=self.device)

                # FINAL SAFETY CHECK: Validate input_ids are all within bounds
                if input_ids is not None:
                    vocab_size = self._actual_vocab_size
                    max_id = input_ids.max().item()
                    min_id = input_ids.min().item()
                    if max_id >= vocab_size or min_id < 0:
                        eval_logger.error(
                            f"❌ CRITICAL: Token IDs still out of bounds before generation! "
                            f"Range: [{min_id}, {max_id}], Valid: [0, {vocab_size-1}]"
                        )
                        # Emergency clamp on CPU
                        input_ids = torch.clamp(input_ids.cpu(), 0, vocab_size - 1).to(self.device)
                        eval_logger.warning(f"   → Emergency clamped to valid range")

                self._ensure_generation_token_ids()
                with torch.no_grad():
                    try:
                        outputs = self.model.generate(
                            input_ids=input_ids,
                            pixel_values=pixel_values,
                            attention_mask=attention_mask,
                            image_positions=image_positions,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                            do_sample=do_sample,
                        )

                        # CRITICAL: Validate generated output tokens
                        if isinstance(outputs, torch.Tensor):
                            vocab_size = self._actual_vocab_size
                            max_output_id = outputs.max().item()
                            min_output_id = outputs.min().item()

                            if max_output_id >= vocab_size or min_output_id < 0:
                                eval_logger.warning(
                                    f"⚠️ Generated tokens out of bounds: "
                                    f"Range [{min_output_id}, {max_output_id}], Valid [0, {vocab_size-1}]. "
                                    f"Clamping outputs."
                                )
                                outputs = torch.clamp(outputs, 0, vocab_size - 1)

                        if isinstance(outputs, torch.Tensor) and self.tokenizer is not None:
                            prompt_len = input_ids.size(1) if input_ids is not None else 0
                            decoded = self.tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
                        else:
                            decoded = str(outputs)

                        results.append(decoded.strip())
                    except RuntimeError as e:
                        if 'CUDA' in str(e) or 'assert' in str(e).lower():
                            eval_logger.error(f"CUDA error in generation (skipping): {e}")
                            results.append("")
                            # Clear CUDA error state
                            try:
                                torch.cuda.synchronize()
                            except:
                                pass
                        else:
                            raise
            except Exception as e:
                eval_logger.error(f"Error in EmberVLM generation: {e}")
                results.append("")

        return results

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("EmberVLM does not support multi-round generation")
