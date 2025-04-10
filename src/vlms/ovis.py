import torch
from transformers import AutoModelForCausalLM
from PIL import Image
from tqdm import tqdm

from .benchmark_evaluator import VLMEvaluator


class OvisEvaluator(VLMEvaluator):

    def load_vlm(self, *args, **kwargs):
        self.model = AutoModelForCausalLM.from_pretrained(self.vlm_name,
                                             torch_dtype=torch.bfloat16,
                                             multimodal_max_length=32768,
                                             trust_remote_code=True).cuda()
        self.text_tokenizer = self.model.get_text_tokenizer()
        self.visual_tokenizer = self.model.get_visual_tokenizer()


    def evaluate_dataset(self, data_dicts, *args, **kwargs):
        responses = {}
        for data_dict in tqdm(data_dicts):
            img_path = data_dict['image_path']
            images = [Image.open(img_path).convert('RGB')]
            q_id = data_dict['question_id']

            max_partition = 9

            prompt = data_dict['prompt']
            query = f'<image>\n{prompt}'

            # format conversation
            prompt, input_ids, pixel_values = self.model.preprocess_inputs(query, images, max_partition=max_partition)
            attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
            input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
            attention_mask = attention_mask.unsqueeze(0).to(device=self.model.device)
            if pixel_values is not None:
                pixel_values = pixel_values.to(dtype=self.visual_tokenizer.dtype, device=self.visual_tokenizer.device)
            pixel_values = [pixel_values]

            # generate output
            with torch.inference_mode():
                gen_kwargs = dict(
                    max_new_tokens=1024,
                    do_sample=False,
                    top_p=None,
                    top_k=None,
                    temperature=None,
                    repetition_penalty=None,
                    eos_token_id=self.model.generation_config.eos_token_id,
                    pad_token_id=self.text_tokenizer.pad_token_id,
                    use_cache=True
                )
                output_ids = self.model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
                response = self.text_tokenizer.decode(output_ids, skip_special_tokens=True)
                responses[q_id] = response
        
        return responses