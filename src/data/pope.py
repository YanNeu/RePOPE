import os
import json

COCO_ROOT_DIR = '/mnt/datasets/coco/val2014'
JSON_RANDOM = 'annotations/coco_pope_random.json'
JSON_POPULAR = 'annotations/coco_pope_popular.json'
JSON_ADVERSARIAL = 'annotations/coco_pope_adversarial.json'


def load_pope_json(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_pope_dictionaries(coco_root=COCO_ROOT_DIR):
    data_info_random = load_pope_json(JSON_RANDOM)
    data_info_popular = load_pope_json(JSON_POPULAR)
    data_info_adversarial = load_pope_json(JSON_ADVERSARIAL)
    data_info_pope = {
        'random': data_info_random,
        'popular': data_info_popular,
        'adversarial': data_info_adversarial
    }
    data_dicts = {}
    
    for variant in data_info_pope:
        data_dicts[variant] = []
        for data_info in data_info_pope[variant]:
            data_dicts[variant].append({
                'image_path': os.path.join(coco_root, data_info['image']),
                'prompt': data_info['text'],
                'dataset': variant,
                'label': data_info['label'],
                'question_id': str(data_info['question_id']),
            })

        data_dicts[variant] = sorted(data_dicts[variant], key=lambda x: int(x['question_id']))
    return data_dicts
