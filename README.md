# RePOPE: Impact of Annotation Errors on the POPE Benchmark

**Yannic Neuhaus**, **Matthias Hein**

**Tübingen AI Center - University of Tübingen**

**[RePOPE Annotation Files](#annotations)** | **[Requirements](#setup)** | **[arXiv](https://arxiv.org/abs/2503.23573)** | **[Citation](#citation)**

<a id="annotations"></a>
## RePOPE Annotation Files
We provide the corrected annotation files in the same format as the original POPE files:

    annotations/coco_repope_random.json
    annotations/coco_repope_popular.json
    annotations/coco_repope_adversarial.json

<a id="setup"></a>
## Requirements
Install the conda environment as follows to reproduce our results:

    conda create --name repope python=3.12
    conda activate dash
    conda install nvidia/label/cuda-12.1.0::cuda-nvcc
    pip install -r requirements_pip.txt
    pip install flash-attn

## Model Evaluation
You can run the following command to evaluate a model on POPE and RePOPE. Supported models can be found [here](.assets/available_models.md).
   
    CUDA_VISIBLE_DEVICES=<GPU index> python src/evaluate.py --vlm_name <VLM name> --bs <batchsize> &
