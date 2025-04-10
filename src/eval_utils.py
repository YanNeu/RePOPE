import os 
import json


def compute_answer(response):
    answer = None
    if "yes" in response.lower():
        answer = "yes"
    elif "no" in response.lower():
        answer = "no"  
    elif "not" in response.lower():
        answer = "no"        
    return answer


def save_answers(benchmark_data, responses, variant, output_dir, vlm_name):
    answers = {}
    for subset in benchmark_data:
        answers[subset] = {}
        for data_dict in benchmark_data[subset]:
            q_id = data_dict['question_id']
            response = responses[subset][q_id]
            answers[subset][q_id] = {
                'question_id': data_dict['question_id'],
                'text': data_dict['prompt'],
                'label': data_dict['label'],
                'dataset': data_dict['dataset'],
                'response': response,
                'answer': compute_answer(response)
            }
    answers_json = os.path.join(output_dir, f"{vlm_name.replace('/', '_')}_{variant}_answers.json")
    with open(answers_json, 'w') as f:
        f.write(json.dumps(answers, indent=4))
    
    return answers

def compute_results(answers, output_dir, vlm_evaluator, variant='POPE'):

    all_results = {}
    for subset in answers:
        pos = "yes"
        neg = "no"
        TP, TN, FP, FN = 0, 0, 0, 0
        
        for q_id in answers[subset]:
            pred = answers[subset][q_id]['answer']    
            label = answers[subset][q_id]['label']
        
            if pred == pos and label == pos:
                TP += 1
            elif pred == pos and label == neg:
                FP += 1
            elif pred == neg and label == neg:
                TN += 1
            elif pred == neg and label == pos:
                FN += 1

        print('TP\tFP\tTN\tFN\t')
        print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))


        precision = float(TP) / float(TP + FP)
        recall = float(TP) / float(TP + FN)
        f1 = 2*precision*recall / (precision + recall)
        acc = (TP + TN) / (TP + TN + FP + FN)
        yes_ratio = (TP + FP) / (TP + TN + FP + FN)

        results = {
            'TP': TP,
            'FP': FP,
            'TN': TN,
            'FN': FN,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'ACC': acc,
            'Yes Ratio': yes_ratio
        }
        all_results[subset] = results
    results_json = os.path.join(output_dir, f"{vlm_evaluator.vlm_name.replace('/', '_')}_{variant}_results.json")
    with open(results_json, 'w') as f:
        f.write(json.dumps(all_results, indent=4))
    return all_results
