from llava.model.builder import load_pretrained_model
from langchain_community.llms import HuggingFaceHub
from langchain.schema import (
    HumanMessage, SystemMessage, AIMessage
)
from langchain_community.chat_models.huggingface import ChatHuggingFace
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path, KeywordsStoppingCriteria
)
import os
from PIL import Image
from io import BytesIO
import re
import requests
import json
from tqdm import tqdm
import argparse
import torch
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
def image_parser(args):
    out = args.image_file.split(args.sep)
    return out

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


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(os.path.join(args.images_path, image_file))
        out.append(image)
    return out

def eval_model(args):
  model_name = get_model_name_from_path(args.model_path)
  tokenizer, model, image_processor, context_len = load_pretrained_model(
          args.model_path, args.model_base, model_name,load_4bit=args.load_4bit
      )
  """
  prompt= SystemMessage(content = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides and assist the user with a variety of tasks using natural language.")
  new_prompt=(prompt+HumanMessage(content="Hi! What is a Frame of Communication?")+AIMessage(content="Frames of communication select particular aspects of an issue and make them salient in communicating a message. Social science stipulates that discourse almost inescapably involves framing – a strategy of highlighting certain issues to promote a certain interpretation or attitude. It has been argued that to frame is to select some aspects of a perceived reality and make them more salient in a communicating text, in such a way as to promote problem definition, causal interpretation, moral evaluation, and/or treatment recommendation.\n"))
  qs = "List all the frames present in this image."
  
  image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
  
  if model.config.mm_use_im_start_end:
    qs = image_token_se + "\n" + qs
  else:
    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    
  final_p=new_prompt + HumanMessage(content = qs)
  """
  image_files = image_parser(args)
  images = load_images(image_files)
  image_sizes = [x.size for x in images]
  images_tensor = process_images(
      images,
      image_processor,
      model.config
  ).to(model.device, dtype=torch.float16)
  
  """input_ids = (
        tokenizer_image_token(final_p, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )"""
  
  llm = HuggingFaceHub(
      repo_id="liuhaotian/llava-v1.6-vicuna-7b",
      task="text-generation",
      model_kwargs={
          "max_new_tokens": 512,
          "top_p": 0.5,
          "temperature": 0.2,
          "do_sample"=True,
          "use_cache"=True,
          "image_sizes"=image_sizes,
          "images"=images_tensor,
          "input_ids"=input_ids
      },
  )
  messages = [
    SystemMessage(content="You're a helpful assistant"),
    HumanMessage(
        content="Describe the image in detail"
    ),
  ]

  chat_model = ChatHuggingFace(llm=llm)
  res = chat_model.invoke(messages)
  print(res.content)
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--sep", type=str, default=",")
    eval_model(args)
