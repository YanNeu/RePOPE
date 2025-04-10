import torch
from PIL import Image
from tqdm import tqdm 
from torch.utils.data import DataLoader

from .dash_utils.vlm_utils import load_vlm_model, get_config_from_name
from .dash_utils.finetuning_utils import make_vlm_datasets_dataset, load_ft_model
from .benchmark_evaluator import VLMEvaluator


class DASHEvaluator(VLMEvaluator):

    def load_vlm(self, *args, checkpoint=None, gpu=0, **kwargs):

        device = torch.device(f'cuda:{gpu}')
        self.vlm_config = get_config_from_name(self.vlm_name)

        if checkpoint is None:
            self.vlm_model = load_vlm_model(self.vlm_name, device)
            self.vlm_name = self.vlm_model.name
        else:
            self.vlm_model = load_ft_model(self.vlm_name, checkpoint, device)
            self.vlm_name = f"{self.vlm_model.name}_{checkpoint.split('/')[-2]}"
        
        print(f"Loaded {self.vlm_name}")

        self.vlm_processor_function = self.vlm_model.get_processor_function()

    def evaluate_dataset(self, data_dicts, *args, batchsize=32, num_workers=1, **kwargs):
        
        dataset = make_vlm_datasets_dataset(data_dicts)

        def collate_fn(examples):
            prompts = [example['prompt'] for example in examples]

            images = []
            for example in examples:
                try:
                    img = Image.open(example['image_path'])
                    img = img.convert('RGB')
                except Exception as e:
                    print(f"Could not load {example['image_path']}: {e}")
                    img = Image.new('RGB', (512, 512))
                images.append(img)

            batch = {'image':images, 'prompt':prompts}
            inputs = self.vlm_processor_function(batch)
            additional_infos = {}
            for key in examples[0].keys():
                if not key in ["prompt", "image_path"]:
                    additional_infos[key] = [example[key] for example in examples]

            return inputs, additional_infos
        
        data_loader = DataLoader(
            dataset,
            collate_fn=collate_fn,
            shuffle=False,
            batch_size=batchsize,
            num_workers=num_workers,
        )

        all_responses = {}
        for inputs, additional_infos in tqdm(data_loader):
            inputs = inputs.to(self.vlm_model.device, self.vlm_model.dtype)
            outputs = self.vlm_model(inputs, generation_kwargs=None)
            outputs = self.vlm_model.decode(inputs, outputs)

            responses = outputs['response']
            for i in range(len(responses)):
                all_responses[additional_infos['question_id'][i]] = responses[i]
            #all_responses.extend(responses)
        return all_responses