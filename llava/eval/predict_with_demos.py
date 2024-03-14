import argparse
from transformers import AutoTokenizer, AutoConfig
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)

from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image
from io import BytesIO
import re
import requests
import json

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
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    dataset = list(read_jsonl(args.dataset))
    demos = list(read_jsonl(args.demos))
    demos = demos[: args.num_demos]
    ex_demos = []
    for d_item in demos:
        ex_demos.append(
            {"image":d_item["file_name"],"frame":d_item["frames"],"stance":d_item["stance"],"rationale":d_item["rationale"]})
    answers_file = os.path.expanduser(args.output)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    r_prompt = "Reason about whether the posting contains a frame (or more frames), or just states something factual or an experience."
    a_prompt = "If the posting contains a frame, articulate that frame succinctly."
    
    def add_r_turn(conv, question: str, rationale: str | None = None):
        #s= "The stance of the image for the corresponding frames is:"
        #h= " ".join(stance)
        #sf=s+" "+h
        #f=" ".join(frames)
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
        images = load_images(image_files)
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
              images,
              image_processor,
              model.config
        ).to(model.device, dtype=torch.float16)
        
        input_ids = (tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
              .unsqueeze(0)
              .cuda()
        )
        
        with torch.inference_mode():
            output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            )
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
    for idx in tqdm(range(len(dataset))):
        if dataset[idx]["file_name"] in seen_ids:
            continue
        ex = dataset[idx]
        image_path = ex["file_name"]                 #i think we have to load the actual image.
        question = "You will be tasked with identifying and articulating misogyny framings on the social media postings. Each social media posting provided may or may not contain one or more frames of communication."
        "List all the frames and the corresponding reasoning."
        img_list=[d["image"] for d in ex_demos]]
        img_list.append(image_path)
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

        rationale = run(conv, img_list)

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
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    
    args = parser.parse_args()
    
    eval_model(args)
