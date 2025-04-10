from tqdm import tqdm

from .benchmark_evaluator import VLMEvaluator

import base64
from .openai_key import key as API_KEY
import openai


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    

class OpenAIEvaluator(VLMEvaluator):

    def load_vlm(self, *args, **kwargs):
        self.client = openai.OpenAI(api_key=API_KEY)


    def evaluate_dataset(self, data_dicts, *args, **kwargs):
        responses = {}
        for data_dict in tqdm(data_dicts):
            img_path = data_dict['image_path']
            base64_image = encode_image(img_path)

            prompt = data_dict['prompt']
            q_id = data_dict['question_id']

            response = self.client.chat.completions.create(
                model=self.vlm_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"},
                            },
                        ],
                    }
                ],
            )

            resp_text = response.choices[0].message.content
            responses[q_id] = resp_text

            return responses