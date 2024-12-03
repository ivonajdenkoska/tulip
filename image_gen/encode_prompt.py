import sys
sys.path.append('..')
from diffusers import DiffusionPipeline
import torch
import types
from open_clip import create_model_and_transforms, get_tokenizer, get_model_config
from open_clip.transformer_rope import TextTransformerRoPE
from open_clip.transformer_cope import TextTransformerCoPE
from open_clip.transformer_cope import _expand_token
from einops import rearrange
import torch.nn.functional as F

import pdb


import torch.nn as nn
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    is_invisible_watermark_available,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)

from diffusers.loaders import (
    FromSingleFileMixin,
    IPAdapterMixin,
    StableDiffusionXLLoraLoaderMixin,
    TextualInversionLoaderMixin,
)


from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import retrieve_timesteps


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.
    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.
    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.
    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

        

    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def encode_text_full(self, text, normalize: bool = False, output_hidden_states=False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        #x, _ = text_global_pool(x, text, self.text_pool_type)
        #if self.text_projection is not None:
        #    if isinstance(self.text_projection, nn.Linear):
        #        x = self.text_projection(x)
        #    else:
        #        x = x @ self.text_projection

        #return F.normalize(x, dim=-1) if normalize else x
        return x

def modified_encode_text(self, text):
    
    cast_dtype = self.transformer.get_cast_dtype()
    seq_len = text.shape[1]
    x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
    attn_mask = self.attn_mask
    if self.cls_emb is not None:
        seq_len += 1
        x = torch.cat([x, _expand_token(self.cls_emb, x.shape[0])], dim=1)
        cls_mask = self.build_cls_mask(text, cast_dtype)
        if attn_mask is not None:
            attn_mask = attn_mask[None, :seq_len, :seq_len] + cls_mask[:, :seq_len, :seq_len]

    x = x.permute(1, 0, 2)  # NLD -> LND
    x= self.transformer(x, attn_mask=attn_mask)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = self.ln_final(x)
    return x

def kps(model):
    positional_embedding_pre = model.positional_embedding       
    length, dim = positional_embedding_pre.shape
    keep_len = 20
    posisitonal_embedding_new = torch.zeros([4*length-3*keep_len, dim])
    for i in range(keep_len):
        posisitonal_embedding_new[i] = positional_embedding_pre[i]
    for i in range(length-1-keep_len):
        posisitonal_embedding_new[4*i + keep_len] = positional_embedding_pre[i + keep_len]
        posisitonal_embedding_new[4*i + 1 + keep_len] = 3*positional_embedding_pre[i + keep_len]/4 + 1*positional_embedding_pre[i+1+keep_len]/4
        posisitonal_embedding_new[4*i + 2+keep_len] = 2*positional_embedding_pre[i+keep_len]/4 + 2*positional_embedding_pre[i+1+keep_len]/4
        posisitonal_embedding_new[4*i + 3+keep_len] = 1*positional_embedding_pre[i+keep_len]/4 + 3*positional_embedding_pre[i+1+keep_len]/4

    posisitonal_embedding_new[4*length -3*keep_len - 4] = positional_embedding_pre[length-1] + 0*(positional_embedding_pre[length-1] - positional_embedding_pre[length-2])/4
    posisitonal_embedding_new[4*length -3*keep_len - 3] = positional_embedding_pre[length-1] + 1*(positional_embedding_pre[length-1] - positional_embedding_pre[length-2])/4
    posisitonal_embedding_new[4*length -3*keep_len - 2] = positional_embedding_pre[length-1] + 2*(positional_embedding_pre[length-1] - positional_embedding_pre[length-2])/4
    posisitonal_embedding_new[4*length -3*keep_len - 1] = positional_embedding_pre[length-1] + 3*(positional_embedding_pre[length-1] - positional_embedding_pre[length-2])/4
            
    positional_embedding_res = posisitonal_embedding_new.clone()
            
    model.positional_embedding = nn.Parameter(posisitonal_embedding_new)

    return model

def build_causal_mask(context_length):
    # lazily create causal attention mask, with full attention between the tokens
    # pytorch uses additive attention mask; fill with -inf
    mask = torch.empty(context_length, context_length)
    mask.fill_(float("-inf"))
    mask.triu_(1)  # zero out the lower diagonal
    return mask

context_length = 248
bigG_model, _, bigG_preprocess = create_model_and_transforms('ViT-bigG-14', 
        pretrained='/home/mderakh/Long-CLIP/SDXL/checkpoints/ViT-bigG-14/open_clip_pytorch_model.bin')
bigG_model = kps(bigG_model)
bigG_model.eval().cuda()
bigG_model.encode_text_full = types.MethodType(encode_text_full, bigG_model)
bigG_model.attn_mask = build_causal_mask(context_length).cuda()
bigG_encoder = bigG_model.encode_text_full
openclip_tokenizer = get_tokenizer('ViT-bigG-14', context_length=context_length)

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = 'ViT-L-14'
model_path = "/nvmestore/mderakh/logs/sharegpt4v/rope/2024_09_12-14_32_44-dynamic_ntk_model_ViT-L-14-lr_1e-05-b_40-j_8-p_amp_bf16-BASE_CTX_77-EXT_CTX_248-scale_f_8.0-loss_clip_loss/checkpoints/epoch_5.pt"
tokenizer = get_tokenizer(model_name, context_length=context_length)  
teacher_cfg = get_model_config(model_name)
vitl_model = TextTransformerRoPE(context_length=context_length,  
            vocab_size=teacher_cfg["text_cfg"]["vocab_size"],
            width=teacher_cfg["text_cfg"]["width"],
            heads=teacher_cfg["text_cfg"]["heads"],
            layers=teacher_cfg["text_cfg"]["layers"],
            output_dim=teacher_cfg["text_cfg"]["width"])
checkpoint = torch.load(model_path)
checkpoint['state_dict'] = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
vitl_model.load_state_dict(checkpoint['state_dict'], strict=False)
vitl_model = vitl_model.to(device)
vitl_model.encode_text_full = types.MethodType(modified_encode_text, vitl_model)
vitl_model.eval()
vitL_encoder = vitl_model.encode_text_full

def tokenize(texts: Union[str, List[str]]) -> torch.LongTensor:
    return tokenizer(texts)

with torch.no_grad():
    def encode_prompt(
        pipe,
        prompt: str,
        prompt_2: Optional[str] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[str] = None,
        negative_prompt_2: Optional[str] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        device = device or pipe._execution_device
        if pipe.text_encoder is not None:
            old_text_encoder = pipe.text_encoder
            old_tokenizer = pipe.tokenizer
            pipe.text_encoder = vitL_encoder
            pipe.tokenizer = tokenize

        if pipe.text_encoder_2 is not None:
            old_text_encoder_2 = pipe.text_encoder_2
            old_tokenizer_2 = pipe.tokenizer_2
            pipe.text_encoder_2 = bigG_encoder
            pipe.tokenizer_2 = openclip_tokenizer

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(pipe, StableDiffusionXLLoraLoaderMixin):
            pipe._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if pipe.text_encoder is not None:
                if not USE_PEFT_BACKEND:
                    adjust_lora_scale_text_encoder(pipe.text_encoder, lora_scale)
                else:
                    scale_lora_layers(pipe.text_encoder, lora_scale)

            if pipe.text_encoder_2 is not None:
                if not USE_PEFT_BACKEND:
                    adjust_lora_scale_text_encoder(pipe.text_encoder_2, lora_scale)
                else:
                    scale_lora_layers(pipe.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        

        # Define tokenizers and text encoders
        tokenizers = [pipe.tokenizer, pipe.tokenizer_2] if pipe.tokenizer is not None else [pipe.tokenizer_2]
        text_encoders = (
            [pipe.text_encoder, pipe.text_encoder_2] if pipe.text_encoder is not None else [bigG_encoder]
        )
        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # textual inversion: procecss multi-vector tokens if necessary
            prompt_embeds_list = []
            prompts = [prompt, prompt_2]
            for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):

                text_inputs = tokenizer(
                    prompt
                )

                text_input_ids = text_inputs
                
                untruncated_ids = tokenizer(prompt)

                if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
                ):
                    removed_text = tokenizer.decoder(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                    logger.warning(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {tokenizer.model_max_length} tokens: {removed_text}"
                    )
            
                prompt_embeds = text_encoder(text_input_ids.to(device))

                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = bigG_model.encode_text(text_input_ids.to(device))
                if clip_skip is None:
                    prompt_embeds = prompt_embeds
                else:
                    # "2" because SDXL always indexes from the penultimate layer.
                    prompt_embeds = prompt_embeds

                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        # get unconditional embeddings for classifier free guidance
        zero_out_negative_prompt = negative_prompt is None and pipe.config.force_zeros_for_empty_prompt
        if do_classifier_free_guidance and negative_prompt_embeds is None and zero_out_negative_prompt:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        elif do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt

            # normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_2 = (
                batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
            )

            uncond_tokens: List[str]
            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = [negative_prompt, negative_prompt_2]

            negative_prompt_embeds_list = []
            for negative_prompt, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, text_encoders):
                max_length = prompt_embeds.shape[1]
                uncond_input = tokenizer(
                    negative_prompt
                )

                negative_prompt_embeds = text_encoder(
                    uncond_input.to(device),
                )
                # We are only ALWAYS interested in the pooled output of the final text encoder
                negative_pooled_prompt_embeds = bigG_model.encode_text(uncond_input.to(device))
                negative_prompt_embeds = negative_prompt_embeds

                negative_prompt_embeds_list.append(negative_prompt_embeds)

            negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

        if pipe.text_encoder_2 is not None:
            prompt_embeds = prompt_embeds.to(dtype=pipe.unet.dtype, device=device)
        else:
            prompt_embeds = prompt_embeds.to(dtype=pipe.unet.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            if pipe.text_encoder_2 is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=pipe.unet.dtype, device=device)
            else:
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=pipe.unet.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )
        if do_classifier_free_guidance:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
                bs_embed * num_images_per_prompt, -1
            )

        if pipe.text_encoder is not None:
            if isinstance(pipe, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(pipe.text_encoder, lora_scale)

        if pipe.text_encoder_2 is not None:
            if isinstance(pipe, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(pipe.text_encoder_2, lora_scale)

        if pipe.text_encoder is not None:
            pipe.text_encoder = old_text_encoder
            pipe.tokenizer = old_tokenizer

        if pipe.text_encoder_2 is not None:
            pipe.text_encoder_2 = old_text_encoder_2
            pipe.tokenizer_2 = old_tokenizer_2

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds
