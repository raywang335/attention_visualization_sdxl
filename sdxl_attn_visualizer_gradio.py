import gradio as gr
from diffusers import StableDiffusionXLPipeline
from diffusers.models import attention_processor
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from einops import rearrange
import torch
from types import MethodType
from pynvml import *
import os
import numpy as np
from PIL import Image
import gc


@torch.no_grad()
def get_attn(emb, res, layer):
    def hook(self, sd_in, sd_out):
        if "attn2" in layer:
            key = self.to_k(emb)
        else:
            key = self.to_k(sd_in[0])
        query = self.to_q(sd_in[0])
        heads = self.heads
        query, key = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=heads), (query, key))
        attn = torch.einsum("b i d, b j d -> b i j", query, key)
        attn = attn * self.scale
        attn = attn.softmax(dim=-1)
        res[layer] = attn
    return hook

@torch.no_grad()
def attn_vis(
        self, 
        prompt=None,
        prompt_2=None,
        height=None,
        width=None,
        num_inference_steps=50,
        denoising_end=None,
        guidance_scale = 5.0,
        negative_prompt=None,
        negative_prompt_2=None,
        num_images_per_prompt = 1,
        eta = 0.0,
        generator = None,
        latents = None,
        prompt_embeds = None,
        negative_prompt_embeds = None,
        pooled_prompt_embeds = None,
        negative_pooled_prompt_embeds = None,
        output_type = "pil",
        return_dict = True,
        callback = None,
        callback_steps = 1,
        cross_attention_kwargs = None,
        guidance_rescale: float = 0.0,
        original_size = None,
        crops_coords_top_left = (0, 0),
        target_size = None,
        ):

    with torch.no_grad():
        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. basic check
        self.check_inputs(
                prompt,
                prompt_2,
                height,
                width,
                callback_steps,
                negative_prompt,
                negative_prompt_2,
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)
        # get text embeddings
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        self.scheduler.set_timesteps(num_inference_steps, device=device)

        timesteps = self.scheduler.timesteps

        # Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype
        )

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # 7.1 Apply denoising_end
        if denoising_end is not None and type(denoising_end) == float and denoising_end > 0 and denoising_end < 1:
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]
        
        cross_hidden_layers = {}
        self_hidden_layers = {}
        for m,n in self.unet.named_modules():
            if isinstance(n, attention_processor.Attention):
                query_dim = n.to_q.weight.shape
                key_dim = n.to_k.weight.shape
                if query_dim != key_dim:
                    cross_hidden_layers[m] = n
                else:
                    self_hidden_layers[m] = n

        cross_attn_map_res = {}
        self_attn_map_res = {}
        for layer in cross_hidden_layers.keys():
            cross_hidden_layers[layer].register_forward_hook(get_attn(prompt_embeds, cross_attn_map_res, layer))
        for layer in self_hidden_layers.keys():
            self_hidden_layers[layer].register_forward_hook(get_attn(prompt_embeds, self_attn_map_res, layer))

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # make sure the VAE is in float32 mode, as it overflows in float16
        if self.vae.dtype == torch.float16 and self.vae.config.force_upcast:
            self.upcast_vae()
            latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = latents
        # apply watermark if available
        if self.watermark is not None:
            image = self.watermark.apply_watermark(image)

        image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image,)
        cross_attn_map_res = change_tensor_to_np(cross_attn_map_res)
        self_attn_map_res = change_tensor_to_np(self_attn_map_res)

        return image, cross_attn_map_res, self_attn_map_res

def change_tensor_to_np(dict_data):
    for key in dict_data.keys():
        dict_data[key] = dict_data[key].cpu().numpy()
    return dict_data

@torch.no_grad()
def self_tokenize(self, prompt):
    tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
    text_inputs = self.tokenizer(
        prompt,
        padding="max_length",
        max_length=self.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids
    text_tokens = self.tokenizer.convert_ids_to_tokens(text_inputs[0])
    text_dict = {}
    for i, token in enumerate(text_tokens):
        text_dict[i] = token
        if "endoftext" in token:
            break
    return text_dict

# check the usage of gpu
# def use_gpu(used_percentage=0.2):
#     nvmlInit()
#     gpu_num = nvmlDeviceGetCount()
#     out = ""
#     for i in range(gpu_num):
#         handle = nvmlDeviceGetHandleByIndex(i)
#         info = nvmlDeviceGetMemoryInfo(handle)
#         used_percentage_real = info.used / info.total
#         if out == "":
#             if used_percentage_real < used_percentage:
#                 out += str(i)
#         else:
#             if used_percentage_real < used_percentage:
#                 out += "," + str(i)
#     nvmlShutdown()
#     return out


def gen_img(prompt, model_path, seed):
    ## prepare sdxl attnvis pipeline
    global cross_attn_map_list
    global self_attn_map_list
    global pipe
    device = 'cuda'
    if pipe is None:
        print("init pipeline")
        pipe = StableDiffusionXLPipeline.from_single_file(
            model_path, torch_dtype=torch.float16, variant="fp16", use_safetensors=True, local_files_only=True, 
            original_config_file="./sd_xl_base.yaml"
        )
        pipe.enable_xformers_memory_efficient_attention()
        pipe.to(device)
        pipe.attnvis = MethodType(attn_vis,pipe)
        pipe.self_tokenize = MethodType(self_tokenize, pipe)
    print("start inference")
    generator = torch.Generator(device=device)
    generator = generator.manual_seed(int(seed))
    img, cross_attn_map_list, self_attn_map_list = pipe.attnvis(prompt, generator=generator)
    token_idx = pipe.self_tokenize(prompt)
    return img[0], token_idx

def to_norm(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def return_cross_attn_map(layer, token_idx):
    global cross_attn_map_list
    attn_map_list = cross_attn_map_list
    attn_layer = attn_map_list[layer]
    attn_mean_layer = np.mean(attn_layer, axis=0)
    size = np.sqrt(attn_mean_layer.shape[0]).astype(int)
    attn_map_single = attn_mean_layer.reshape(size, size, 77)
    np_img = attn_map_single[:, :, token_idx]
    attn_img = Image.fromarray((to_norm(np_img) * 255).astype(np.uint8))
    return attn_img

def return_group_cross_attn_map(layer, token_idx):
    group_name = ".".join(layer.split(".")[:2])
    global cross_attn_map_list
    attn_map_list = cross_attn_map_list
    np_img = None
    for key in cross_attn_map_list.keys():
        if group_name in key:
            attn_layer = attn_map_list[key]
            attn_mean_layer = np.mean(attn_layer, axis=0)
            size = np.sqrt(attn_mean_layer.shape[0]).astype(int)
            attn_map_single = attn_mean_layer.reshape(size, size, 77)
            np_img = to_norm(attn_map_single[:, :, token_idx]) if np_img is None else np.concatenate((np_img, to_norm(attn_map_single[:, :, token_idx])), axis=1)
    height, length = np_img.shape[0], np_img.shape[1]
    while length > height * 3:
        np_img = np.concatenate((np_img[:,:int(length//2)], np_img[:,int(length//2):]),axis=0)
        height, length = np_img.shape[0], np_img.shape[1]
    attn_img = Image.fromarray((np_img * 255).astype(np.uint8))
    return attn_img

def return_self_attn_map(layer):
    global self_attn_map_list
    attn_map_list = self_attn_map_list
    attn_layer = attn_map_list[layer]
    attn_mean_layer = np.mean(attn_layer, axis=0)
    size = np.sqrt(attn_mean_layer.shape[0]).astype(int)
    attn_map_single = attn_mean_layer.reshape(-1, size, size)
    np_img = np.mean(attn_map_single, axis=0)
    attn_img = Image.fromarray((to_norm(np_img) * 255).astype(np.uint8))
    return attn_img
    
def return_group_self_attn_map(layer):
    group_name = ".".join(layer.split(".")[:2])
    global self_attn_map_list
    attn_map_list = self_attn_map_list
    np_img = None
    for key in self_attn_map_list.keys():
        if group_name in key:
            attn_layer = attn_map_list[key]
            attn_mean_layer = np.mean(attn_layer, axis=0)
            size = np.sqrt(attn_mean_layer.shape[0]).astype(int)
            attn_map_single = attn_mean_layer.reshape(-1, size, size)
            np_img = to_norm(np.mean(attn_map_single, axis=0)) if np_img is None else np.concatenate((np_img, to_norm(np.mean(attn_map_single, axis=0))), axis=1)
    height, length = np_img.shape[0], np_img.shape[1]
    if 'up_blocks.0' in layer:
        np_img = np.concatenate((np_img[:,:int(length//5)], np_img[:,int(length//5):2*int(length//5)],  np_img[:,int(length//5)*2:3*int(length//5)],  np_img[:, int(length//5)*3:4*int(length//5)],  np_img[:,int(length//5)*4:]),axis=0)
    else:
        while length > height * 3:
            np_img = np.concatenate((np_img[:,:int(length//2)], np_img[:,int(length//2):]),axis=0)
            height, length = np_img.shape[0], np_img.shape[1]
    attn_img = Image.fromarray((np_img * 255).astype(np.uint8))
    return attn_img

def update_cross_layers_name():
    global cross_attn_map_list
    return gr.Dropdown.update(choices=list(cross_attn_map_list.keys()))

def update_self_layers_name():
    global self_attn_map_list
    return gr.Dropdown.update(choices=list(self_attn_map_list.keys()))

# to save memory
def clear():
    global cross_attn_map_list
    cross_attn_map_list = None
    global self_attn_map_list
    self_attn_map_list = None
    global pipe
    pipe = None
    torch.cuda.empty_cache()
    gc.collect()



if __name__ ==  "__main__":
    global cross_attn_map_list
    global self_attn_map_list
    global pipe
    cross_attn_map_list, self_attn_map_list, pipe = None, None, None
    with gr.Blocks() as visualizer:
        gr.Markdown("Attntion map visualization in sdxl")
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Textbox(placeholder="Input prompts", label='Prompts',)
                seed = gr.Textbox(placeholder="Seed", label='Seed', value="-1")
                model_path = gr.Textbox(label='Model path', placeholder="Model path", value="/path/to/sdxl_model/sd_xl_base_1.0.safetensors")
                out_token_idx = gr.Textbox(placeholder="Token index 0", label='Token&IDs Mapping',)
            with gr.Column(scale=1):
                out_img = gr.Image(shape=(256, 256))
            with gr.Column(scale=1):
                out2 = gr.Image(shape=(256, 256))
        with gr.Row():
            with gr.Column(scale=1):
                cross_layers_name = gr.Dropdown(
                    label='Cross layers',
                    choices=[],
                    elem_id='choose the target layer',
                )
                self_layers_name = gr.Dropdown(
                    label='Self layers',
                    choices=[],
                    elem_id='choose the target layer',
                )
                out_img.change(fn=update_cross_layers_name, outputs=[cross_layers_name])
                out_img.change(fn=update_self_layers_name, outputs=[self_layers_name])
                token_idx = gr.Slider(minimum=0, maximum=76, step=1, default=0, label="Token index")
            with gr.Column(scale=2):
                out_group = gr.Image()
                # framework = gr.Image(value="./sdxl_framework.png", shape=(256, 256))


        btn = gr.Button("Run")
        with gr.Row():
            cross_attn_btn_s = gr.Button("Get single cross attn map")
            cross_attn_btn_g = gr.Button("Get group cross attn maps")
        with gr.Row():
            self_attn_btn_s = gr.Button("Get single self attn map")
            self_attn_btn_g = gr.Button("Get group self attn maps")
        clear_btn = gr.Button("Clear")
        

        btn.click(fn=gen_img, inputs=[inp, model_path, seed], outputs=[out_img, out_token_idx])
        cross_attn_btn_s.click(fn=return_cross_attn_map, inputs=[cross_layers_name, token_idx], outputs=out2)
        cross_attn_btn_g.click(fn=return_group_cross_attn_map, inputs=[cross_layers_name, token_idx], outputs=out_group)
        self_attn_btn_s.click(fn=return_self_attn_map, inputs=[self_layers_name], outputs=out2)
        self_attn_btn_g.click(fn=return_group_self_attn_map, inputs=[self_layers_name], outputs=out_group)
        clear_btn.click(fn=clear)

    visualizer.launch()