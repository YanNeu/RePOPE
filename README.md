# RePOPE: Impact of Annotation Errors on the POPE Benchmark

**Yannic Neuhaus**, **Matthias Hein**

**Tübingen AI Center - University of Tübingen**

**[RePOPE Annotation Files](#annotations)** | **[Requirements](#setup)** | **[arXiv](https://arxiv.org/abs/2504.15707)** | **[Citation](#citation)**

This repository contains a relabeling of the POPE. We also created [DASH-B](https://github.com/YanNeu/DASH-B) a harder and less saturated object hallucination benchmark for VLMs.
## RePOPE
We introduce RePOPE, a relabeling of the commonly used object hallucination benchmark COCO POPE. We correct wrong annotations and remove ambiguous ones. The imbalance between incorrect "Yes" and incorrect "No" labels (9.3% vs 1.7%) has a significant effect on the F1 scores.

<p align="center">
  <img width="1000" src="./assets/pope_vs_repope.png", alt="Relabeling results">
</p>

## (Re-)Annotation Examples

<p align="center">
  <img width="1000" src="./assets/annotation_examples.png", alt="Annotation error examples">
</p>


<a id="annotations"></a>
## RePOPE Annotation Files
We provide the corrected annotation files in the same format as the original POPE files:

    annotations/coco_repope_random.json
    annotations/coco_repope_popular.json
    annotations/coco_repope_adversarial.json

<a id="setup"></a>
## Requirements
Install the conda environment as follows to reproduce the results using our code:

    conda create --name repope python=3.12
    conda activate repope
    conda install nvidia/label/cuda-12.1.0::cuda-nvcc
    pip install -r requirements_pip.txt
    pip install flash-attn

## Model Evaluation
You can run the following command to evaluate a model on POPE and RePOPE. Supported models can be found [here](./assets/available_models.md).
   
    CUDA_VISIBLE_DEVICES=<GPU index> python src/evaluate.py --vlm_name <VLM name> --bs <batchsize> &


<a id="citation"></a>
## Citation

    @article{neuhaus2025repope,
        title={RePOPE: Impact of Annotation Errors on the POPE Benchmark},
        author={Neuhaus, Yannic and Hein, Matthias},
        journal={arXiv preprint arXiv:2504.15707},
        year={2025}
    }

