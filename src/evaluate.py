import os
import argparse

from eval_utils import compute_results, save_answers
from data import load_pope_dictionaries, load_repope_dictionaries
from vlms import get_evaluator, VLMEvaluator


class CustomEvaluator(VLMEvaluator):
    def load_vlm(self, *args, **kwargs):
        """
        Loads the model and processors/tokenizers required for inference.

        """
        # Implement loading of model and processors/tokenizers here
        raise NotImplementedError()
    
    def evaluate_dataset(self, data_dicts, *args, **kwargs):
        """
        Args:
            data_dicts: list of dictionaries, each containing the following keys:
                - image_path: path to the image
                - prompt: text query 
                
        Returns a list of model response strings for each image-query in the dictionaries.
        """
        # Implement model evaluation here
        raise NotImplementedError()


def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate a VLM on the DASH-B benchmark.')
    parser.add_argument('--vlm_name', type=str, default='AIDC-AI/Ovis2-1B', 
                        help='Name of the vision language model to evaluate')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save results')
    parser.add_argument('--bs', type=int, default=32,
                        help='batchsize')

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_arguments()
    vlm_name = args.vlm_name    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving results to {output_dir}")
    if args.vlm_name == 'custom':
        vlm_evaluator = CustomEvaluator(vlm_name)
    else:
        vlm_evaluator = get_evaluator(vlm_name)

    pope_data = load_pope_dictionaries()
    repope_data = load_repope_dictionaries()

    responses = {}
    
    for subset in pope_data:
        responses[subset] = vlm_evaluator.evaluate_dataset(pope_data[subset], batchsize=args.bs)
    
    answers_pope = save_answers(pope_data, responses, 'POPE', output_dir, vlm_evaluator.vlm_name)
    answers_repope = save_answers(repope_data, responses, 'RePOPE', output_dir, vlm_evaluator.vlm_name)

    results_pope = compute_results(answers_pope, output_dir, vlm_evaluator, variant='POPE')
    results_repope = compute_results(answers_repope, output_dir, vlm_evaluator, variant='RePOPE')


    print(vlm_evaluator.vlm_name)
    print(results_pope)
    print(results_repope)
    