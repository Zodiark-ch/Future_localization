# knowledge-neurons
Code for the ACL-2022 paper "Knowledge Neurons in Pretrained Transformers"

# Introduction

This project helps you to reproduce all the results presented in our work about knowledge neurons, including calculating the knowledge attribution scores, identifying knowledge neurons, computing all the statistics, and plotting all the figures.

# Execution Steps

This section provides step-by-step instructions for executing the CSAT project using Edge Pruning methodology.

## Step 1: Dataset Creation
Execute the dataset creation script using Python:
```bash
python Edge-Pruning/get_wmdp.py
```

Additional datasets can be obtained from the [Edge-Pruning repository](https://github.com/princeton-nlp/Edge-Pruning). 

**Note**: If you encounter insufficient GPU memory, you can modify the `max_token` parameter to control the token length of the generated dataset. However, this may lead to unexpected deviations in the final results.

## Step 2: Circuit Generation
Execute the pruning scripts to generate Noising and Denoising circuits:
```bash
python Edge-Pruning/src/prune/fmistral_cyber.py
python Edge-Pruning/src/prune/fmistral_cyber_or.py
```

Please execute the corresponding Python files based on your specific dataset requirements.

## Step 3: Edge Refinement
Generate refined edges.json files using the visualization script:
```bash
python Edge-Pruning/src/modeling/vis_mistral.py
```

**Note**: Noising and denoising processes will produce two different JSON files.

## Step 4: Logical Circuit Generation
Extract circuits containing conjunction and disjunction operations:
```bash
python Edge-Pruning/src/solver/get_logical_edge.py
```

## Step 5: Conflict Analysis
Identify forget nodes and conflict nodes:
```bash
python Edge-Pruning/src/solver/conflict.py
```

## Step 6: Mask Generation
Generate masks for the corresponding nodes:
```bash
python Edge-Pruning/src/solver/mask_generation.py
```

## Step 7: Model Unlearning
Perform the unlearning process:
```bash
python src/exec/unlearn_model_conlict.py
```

# Code Usage

First please change the working directory to `src/`.

### Calculate the Attribution Scores
Run `bash 1_run_mlm.sh param1`, where param1 is the relation name to analyze, such as "P101". You can write a script to run this command for each of the 34 relations. This command will calculate the attribution scores for all the facts.

### Identify Knowledge Neurons
Run `bash 2_run_kn.sh`. This command will identify and refine knowledge neurons for each fact, and give their statistics along with a figure about the knowledge neuron distribution.

### Modify Knowledge Neurons
Run `3_run_modify_activation.sh`. This command will modify the activation values of knowledge neurons and record the corresponding results.

### Check Knowledge Neuron Activation for Prompts
Run `4_run_distant.sh`. This command will check the activation values of knowledge neurons for different types of prompts crawled from web pages.

### Produce Activating Prompts
Run `5_run_trigger_examples.sh`. This command will produce activating prompts.

### Update Facts
Run `6_run_edit.sh param1 param2`, where param1 and param2 are two hyper-parameters. In our paper, they are set to 1 and 8, respectively. This command will edit sampled facts.

### Erase Relations
Run `7_run_erase.sh param1`, where param1 is the relation name to erase. This command will erase a relation. In our paper, we try to erase P19, P27, P106, and P937, which can be regarded as privacy information. Of course, you can erase any relation as you like.

### Plot Figures
Run `8_run_plot.sh`. This command will plot two figures that visualize the results from `3_run_modify_activation.sh` and `4_run_distant.sh`.

