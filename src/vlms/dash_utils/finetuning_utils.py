import os
import torch
import numpy as np
from typing import List, Dict, Optional
from tqdm import tqdm
from datasets import Dataset
import json

from peft import PeftModel

from .vlm_utils import get_standard_spurious_prompt, load_vlm_model
from .utils import load_results, get_cluster_info_filepath, get_full_image_folder_path
from .common_elements_utils import load_object_image_folders

DEFAULT_TRAIN_CONFIG = {
    'output_dir': ".",
    'eval_strategy': 'steps',
    'max_grad_norm': 1.0,
    'learning_rate': 1e-5,
    'dataloader_num_workers':1,
    'dataloader_persistent_workers':True,
    'lr_scheduler_type': "linear",
    'warmup_steps': 0,
    'eval_accumulation_steps': 1, 
    'eval_on_start': True,
    'remove_unused_columns': False,
    'dataloader_persistent_workers':False,
    'skip_memory_metrics': True,
    'eval_do_concat_batches': True,
    'save_strategy': 'steps',
    'save_steps': 500,
}


def make_vlm_datasets_dataset(data_dicts: List[Dict]):
    assert len(data_dicts) > 0

    #transform from List[dict] to dict[List]
    dict_first_data = {k: [] for k in data_dicts[0].keys()}
    for data_dict in data_dicts:
        for k, v in data_dict.items():
            dict_first_data[k].append(v)

    dataset = Dataset.from_dict(dict_first_data)
    dataset = dataset.to_iterable_dataset()
    dataset = dataset.with_format('torch')
    return dataset


def create_simple_dataset(results_dict, num_pos_per_obj, num_neg_per_obj, num_eval_per_obj, seed=123, prompt_id=0):
    np.random.seed(seed)
    # load data
    nn_image_paths = {}
    all_img_paths = []
    print('load image paths')
    for object_name in tqdm(results_dict):
        img_paths = results_dict[object_name]['img_paths']
        
        nn_image_paths[object_name] = [img_paths[idx] for idx in img_paths]
        
        all_img_paths.extend(nn_image_paths[object_name])
    all_img_paths = list(set(all_img_paths))
    num_unique_images_all = len(all_img_paths)

    pos_data_dicts = []
    neg_data_dicts = []
    
    pos_data_dicts_eval = []
    neg_data_dicts_eval = []
        
    print('collect positive and negative examples')

    pos_target = "yes" #get_target_str(args.model_id, 'yes')
    neg_target = "no" #get_target_str(args.model_id, 'no')
    for object_name in tqdm(nn_image_paths):
        prompt = get_standard_spurious_prompt(object_name, prompt_id)
        
        num_total_pos = num_pos_per_obj + num_eval_per_obj
        num_total_neg = num_neg_per_obj + num_eval_per_obj
        
        if len(nn_image_paths[object_name]) < num_total_pos: continue
        
        pos_paths_random = np.random.choice(nn_image_paths[object_name], size=num_total_pos, replace=False)
        pos_data_dicts.extend([{'image_path':img_paths, 'prompt':prompt, 'target':pos_target} for img_paths in pos_paths_random[:num_pos_per_obj]])
        pos_data_dicts_eval.extend([{'image_path':img_paths, 'prompt':prompt, 'target':pos_target} for img_paths in pos_paths_random[num_pos_per_obj:]])

        neg_paths_random = []

        num_neg = 0
        idx = 0
        random_indices = np.random.permutation(range(num_unique_images_all))
        while num_neg < num_total_neg and idx < num_unique_images_all:
            rand_idx = random_indices[idx]
            if not all_img_paths[rand_idx] in nn_image_paths[object_name]:
                if num_neg < num_total_neg:
                    neg_paths_random.append(all_img_paths[rand_idx])
                    num_neg = len(neg_paths_random)
            idx += 1
        neg_data_dicts.extend([{'image_path':img_paths, 'prompt':prompt, 'target':neg_target} for img_paths in neg_paths_random[:num_neg_per_obj]])
        neg_data_dicts_eval.extend([{'image_path':img_paths, 'prompt':prompt, 'target':neg_target} for img_paths in neg_paths_random[num_neg_per_obj:]])
    
    train_data_dicts = pos_data_dicts
    train_data_dicts.extend(neg_data_dicts)
    
    val_data_dicts = pos_data_dicts_eval
    val_data_dicts.extend(neg_data_dicts_eval)
    return train_data_dicts, val_data_dicts


def nn_ids_by_cluster(source_folder, results_dict, linkage, distance_threshold, object2dataset, dataset2source_dir, seed=123):
    np.random.seed(seed)
    nn_image_paths = {}

    variant = source_folder.split('/')[-1]
    image_paths_json = f"data/{variant}_all_nn_ids_by_cluster.json"
    if os.path.exists(image_paths_json):
        with open(image_paths_json, 'r') as f:
            nn_image_paths = json.load(f)
        return nn_image_paths

    for obj_name in results_dict:
        nn_image_paths[obj_name] = {}
        source_dir = dataset2source_dir[object2dataset[obj_name]]
        obj_subdir = os.path.join(source_dir, obj_name)
        if not os.path.exists(obj_subdir): 
            print(f"Skipping {obj_name} (no nn folder)")
            continue

        cluster_file_path = get_cluster_info_filepath(obj_subdir, linkage, distance_threshold)
        cluster_info = torch.load(cluster_file_path)
        
        for cluster_id, cluster_nn_ids in enumerate(cluster_info['clusters']):
            nn_image_paths[obj_name][cluster_id] = [os.path.join(obj_subdir, f"{nn_id}.png") for nn_id in cluster_nn_ids]

    with open(image_paths_json, 'w', encoding='utf-8') as f:
        json.dump(nn_image_paths, f, ensure_ascii=False, indent=4)

    return nn_image_paths


def load_reference_train_paths(dataset, object_label, num_samples=1000):
    if "objects_365" in dataset.lower():
        object_img_path_json = f"data/objects_365_pos/{object_label}.json"
        with open(object_img_path_json, 'r') as f:
            obj_img_paths = json.load(f)
        obj_img_paths = obj_img_paths[object_label][:num_samples]
    else:
        raise NotImplementedError()

    return obj_img_paths


def create_ft_train_set(num_neg_per_obj, pos_samples_results_dict, ratio_pos_samples=1.0, use_obj365_train=False, target_strings={'pos':'yes', 'neg': 'no'}, prompt_id=4):
    neg_image_paths_json = "data/train_ids_neg.json"
    assert os.path.exists(neg_image_paths_json)
    neg_image_paths = json.load(open(neg_image_paths_json))
    neg_image_paths_merged = {}

    if use_obj365_train:
        with open('data/objects_365_100.txt', 'r') as f:
            obj365_labels = f.readlines()
        obj365_labels = [obj.strip() for obj in obj365_labels]
        obj365_labels.remove('Billiards')
        num_pos_per_obj = int(num_neg_per_obj * ratio_pos_samples)

    # merge images from text based/prompt based
    for source in neg_image_paths:
        for obj_name in neg_image_paths[source]:
            if not obj_name in neg_image_paths_merged: neg_image_paths_merged[obj_name] = []
            neg_image_paths_merged[obj_name].extend(neg_image_paths[source][obj_name])
    
    # load positive sample paths
    pos_image_paths = {}
    print('load image paths')
    for object_name in tqdm(pos_samples_results_dict):

        if use_obj365_train and object_name in obj365_labels:
            img_paths = load_reference_train_paths("objects_365", object_label=object_name, num_samples=num_pos_per_obj)
            pos_image_paths[object_name] = img_paths
        else:
            img_paths = pos_samples_results_dict[object_name]['img_paths']
            pos_image_paths[object_name] = [img_paths[idx] for idx in img_paths] #[img_paths[idx] for idx in pos_samples_results_dict['success_ids']]

    print('gather positive and negative samples')
    data_dicts = []    
    for object_name in tqdm(neg_image_paths_merged):
        prompt = get_standard_spurious_prompt(object_name, prompt_id)

        neg_data_dicts = []
        # gather negative samples
        neg_paths = neg_image_paths_merged[object_name]

        if len(neg_paths) < num_neg_per_obj:
            continue
            #n_samples = len(neg_paths)
            #data_dicts.extend([
            #    {'image_path': neg_path, 'prompt': prompt, 'target': target_strings['neg']} 
            #    for neg_path in neg_paths
            #])
        else:
            n_samples = num_neg_per_obj
            random_indices = np.random.choice(range(len(neg_paths)), replace=False, size=num_neg_per_obj)
            neg_data_dicts = [
                {'image_path': neg_paths[rnd_idx], 'prompt': prompt, 'target': target_strings['neg']}
                for rnd_idx in random_indices
            ]
        
        num_pos_samples = int(n_samples * ratio_pos_samples)
        if len(pos_image_paths[object_name]) < num_pos_samples: continue

        # gather positive samples
        assert num_pos_samples <= len(pos_image_paths[object_name])

        random_indices = np.random.choice(range(len(pos_image_paths[object_name])), replace=False, size=num_pos_samples)
        data_dicts.extend([
            {'image_path': pos_image_paths[object_name][rnd_idx], 'prompt':prompt, 'target': target_strings['pos']}
            for rnd_idx in random_indices
        ])
        data_dicts.extend(neg_data_dicts)
    
    return data_dicts


def create_ft_test_set(img_paths, target_strings={'pos':'yes', 'neg': 'no'}, prompt_id=4):
    data_dicts = []
    
    for source in img_paths:
        for obj_name in img_paths[source]:
            for image_path in img_paths[source][obj_name]:
                data_dict = {
                    'image_path': image_path,
                    'prompt': get_standard_spurious_prompt(obj_name, prompt_id),
                    'target': target_strings['neg']
                }
                data_dicts.append(data_dict)
    return data_dicts


def create_ft_test_sets():
    mixed_json = "data/test_ids_mixed_neg.json"
    only_json = "data/test_ids_only_neg.json"
    assert os.path.exists(mixed_json) and os.path.exists(only_json)

    with open(mixed_json, 'r') as f:
        img_paths_mixed = json.load(f)

    test_dicts_mixed = create_ft_test_set(img_paths_mixed)

    with open(only_json, 'r') as f:
        img_paths_only = json.load(f)

    test_dicts_only = create_ft_test_set(img_paths_only)
    
    test_dicts = {
        'mixed': test_dicts_mixed,
        'only': test_dicts_only
    }
    return test_dicts


def compute_number_of_steps(num_train_points, num_epochs, gradient_accumulation_steps, train_batchsize):
    train_points_per_step = gradient_accumulation_steps * train_batchsize
    
    num_steps_per_epoch = num_train_points // train_points_per_step
    if num_train_points % train_points_per_step > 0:
        print(f"Total number of training points is not divisible by gradient_accumulation_steps * train_batchsize")
    
    num_max_steps = num_steps_per_epoch * num_epochs

    print(f'Number of steps per epoch: {num_steps_per_epoch}')
    print(f'Number of epochs: {num_epochs} -> max steps {num_max_steps}')
    return num_max_steps, num_steps_per_epoch


def load_all_results(source_root_dir, source_datasets, detector_name, vlm_name, detection_threshold, prompt_id=0, load_path=False, vlm_config=None, detection_config=None, prompt_type=None, agnostic=False):

    all_results_dict = {}
    object_to_dataset = {}
    dataset_to_source_dir = {}
    for dataset in source_datasets:
        if load_path:
            source_dir = get_full_image_folder_path(source_root_dir, dataset, prompt_type, prompt_id, vlm_config, detection_config, agnostic=agnostic)
        else:
            source_dir = os.path.join(source_root_dir, dataset)
        
        if not os.path.exists(source_dir):
            continue
        
        dataset_to_source_dir[dataset] = source_dir

        
        object_imgs_dirs = load_object_image_folders(source_dir)
        results_dict = load_results(object_imgs_dirs, detector_name, vlm_name, detection_threshold, prompt_id)

        for obj in results_dict:
            if obj in all_results_dict:
                print(f"Found duplicate object {obj} (keeping object from {object_to_dataset[obj]}, ignoring object in {dataset})")
                print(dataset)
            #assert not obj in all_results_dict
            all_results_dict[obj] = results_dict[obj]
            object_to_dataset[obj] = dataset
    
    if load_path:
        return all_results_dict, object_to_dataset, dataset_to_source_dir
    return all_results_dict, object_to_dataset


def load_ft_model(model_id, checkpoint_dir, device):
    vlm_model = load_vlm_model(model_id, device)
    vlm_model.model = PeftModel.from_pretrained(vlm_model.model, checkpoint_dir)
    return vlm_model
