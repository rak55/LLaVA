import argparse
from transformers import AutoTokenizer, AutoConfig
from transformers import CLIPImageProcessor, CLIPVisionModel
from transformers import StoppingCriteria, BitsAndBytesConfig

from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init

import os
from PIL import Image
from io import BytesIO
import re
import requests
import json
from tqdm import tqdm
import torch
from llava import LlavaLlamaForCausalLM

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

# new stopping implementation
class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(
        self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            outputs = self.tokenizer.batch_decode(
                output_ids[:, self.start_len :], skip_special_tokens=True
            )[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False


#added old funcs.
def read_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                ex = json.loads(line)
                yield ex

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def patch_config(config):
    patch_dict = {
        "use_mm_proj": True,
        "mm_vision_tower": "openai/clip-vit-large-patch14",
        "mm_hidden_size": 1024,
    }

    cfg = AutoConfig.from_pretrained(config)
    if not hasattr(cfg, "mm_vision_tower"):
        print(
            f"`mm_vision_tower` not found in `{config}`, applying patch and save to disk."
        )
        for k, v in patch_dict.items():
            setattr(cfg, k, v)
        cfg.save_pretrained(config)
        
def load_pretrained_model(
    model_name,
    load_8bit=False,
    load_4bit=False,
    load_bf16=False,
    device_map="auto",
    device="cuda",
):
    kwargs = {"device_map": device_map}

    if device != "cuda":
        kwargs["device_map"] = {"": device}

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
              kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if load_bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    if load_bf16:
        kwargs["torch_dtype"] = torch.bfloat16
    else:
        kwargs["torch_dtype"] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_name, **kwargs
    )                                                                            #took out low_mem_usga=False.

            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKE
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    vision_tower = model.model.vision_tower[0]
    if vision_tower.device.type == "meta":
        vision_tower = CLIPVisionModel.from_pretrained(
            vision_tower.config._name_or_path,
            torch_dtype=torch.bfloat16 if load_bf16 else torch.float16,
            low_cpu_mem_usage=False,
        ).cuda()
        model.model.vision_tower[0] = vision_tower
    else:
        vision_tower.to(
            device="cuda", dtype=torch.bfloat16 if load_bf16 else torch.float16
        )

    vision_config = vision_tower.config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids(
        [DEFAULT_IMAGE_PATCH_TOKEN]
    )[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    if mm_use_im_start_end:
        (
            vision_config.im_start_token,
            vision_config.im_end_token,
        ) = tokenizer.convert_tokens_to_ids(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
    image_processor = CLIPImageProcessor.from_pretrained(
        model.config.mm_vision_tower,
        torch_dtype=torch.bfloat16 if load_bf16 else torch.float16,
    )
    image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2
    return model, tokenizer, image_processor, image_token_len

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, use_cache=True
    ).cuda()
    image_processor = CLIPImageProcessor.from_pretrained(
        model.config.mm_vision_tower, torch_dtype=torch.float16
    )
    vision_tower = model.model.vision_tower[0]
    vision_tower.to(device="cuda", dtype=torch.float16)

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    # import pdb; pdb.set_trace()
    vision_config = vision_tower.config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids(
        [DEFAULT_IMAGE_PATCH_TOKEN]
    )[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    if mm_use_im_start_end:
        (
            vision_config.im_start_token,
            vision_config.im_end_token,
        ) = tokenizer.convert_tokens_to_ids(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN]
        )
    image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2
    return model, tokenizer, image_processor, image_token_len

       

def eval_model(args):
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name,load_4bit=args.load_4bit)

    dataset = list(read_jsonl(args.dataset))
    demos = list(read_jsonl(args.demos))
    demos = demos[:args.num_demos]
    ex_demos = []
    for d_item in demos:
        ex_demos.append(
            {"image": load_image(os.path.join(args.images_path, d_item["file_name"])),"frame":d_item["frame"],"rationale":d_item["rationale"],"problems":d_item["problems"]})
    answers_file = os.path.expanduser(args.output)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    seen_ids = set()
    if os.path.exists(answers_file):
        with open(answers_file, "r") as f:
            for line in f:
                line = json.loads(line)
                seen_ids.add(line["file_name"])
                
    r_prompt = "Reason about whether the posting contains a frame (or more frames), or just states something factual or an experience."
    a_prompt = "If the posting contains a frame, articulate that frame succinctly."
    
    def add_r_turn(conv, question: str, rationale: str | None = None):
        qs = f"{question} {r_prompt}"
        if getattr(model.config, "mm_use_im_start_end", False):
            qs = (
                qs
                + "\n"
                + DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN)
        else:
            qs = qs + "\n" + DEFAULT_IMAGE_TOKEN
        conv.append_message(conv.roles[0], qs)
        if rationale is not None:
            #rationale = format_rationale(rationale)
            conv.append_message(conv.roles[1], rationale + "\n")
        else:
            conv.append_message(conv.roles[1],None)
        
    def add_a_turn(conv, answer: str | None = None):
        qs = a_prompt
        conv.append_message(conv.roles[0], qs)
        if answer is not None:
            conv.append_message(conv.roles[1],answer + "\n")
        else:
            conv.append_message(conv.roles[1],None)
    
    def run(conv,images):
        prompt = conv.get_prompt()
        input_ids = (tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda())
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        
        with torch.inference_mode():
            output_ids = model.generate(
            input_ids,
            images=images,
            do_sample=True,
            temperature=0.2,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            stopping_criteria=[stopping_criteria],
            )
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
          print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
          outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        return outputs
       
    for idx in tqdm(range(len(dataset))):
        if dataset[idx]["file_name"] in seen_ids:
            continue
        ex = dataset[idx]
        image_path = ex["file_name"]
        image = load_image(os.path.join(args.images_path, image_path))
        image_tensor = image_processor.preprocess([d["image"] for d in ex_demos] + [image], return_tensors="pt")["pixel_values"]
        images = image_tensor.half().cuda()
        question = "You will be tasked with identifying and articulating misogyny framings on the social media postings. Each social media posting provided may or may not contain one or more frames of communication."
        "List all the frames and the corresponding reasoning."
        conv = conv_templates[args.conv_mode].copy()
        for d in ex_demos:
            add_r_turn(
                conv,
                question=question,
                rationale=d["rationale"],            #changed near demos. 
            )
            add_a_turn(
                conv,
                answer=d["frame"],                   #change category wrt output.
            )

        final_conv = conv.copy()

        add_r_turn(
            conv,
            question=question,
            
        )
        print(conv.get_prompt())
        rationale = run(conv, images)

        add_r_turn(
            final_conv,
            question=question,
            rationale=rationale,
                                       
        )
        full_conv = final_conv.copy()
        add_a_turn(final_conv)
        pred = run(final_conv, images)
            
        add_a_turn(full_conv, answer=pred)
        
        with open(answers_file, "a") as f:
            f.write(
                json.dumps(
                    {
                        "id": ex["file_name"],
                        "rationale": rationale,
                        "pred": pred,
                    }
                )
                + "\n"
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--demos", type=str, required=True)
    parser.add_argument("--images_path", type=str, required=True)
    parser.add_argument("--num_demos", type=int, default=3)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.7)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--load-4bit", action="store_true")
    args = parser.parse_args()
    
    eval_model(args)
