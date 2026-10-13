# Multi-Task BERT with Task Vector Composition

Jupyter notebooks that fine-tune BERT on two tasks, sentiment analysis and named entity recognition, then combine them into one model by averaging the weight changes from each task.

Final project for ECE 6524 (Deep Learning), Virginia Tech. The project report will be added to this repo once it is finalized.

## Why it exists

Running one model per task means storing and serving several copies of BERT. The idea tested here is called task vector composition: fine-tune a copy of BERT on each task, subtract the original weights to get a "task vector" for that task, then add the vectors together to get one model that does both. If it works, you keep one set of encoder weights instead of two.

## How it works

1. **Fine-tune per task.** Start from `bert-base-uncased` and fine-tune one copy on each task:
   - Sentiment analysis on SST-2 (from the GLUE benchmark), 2 labels.
   - Named entity recognition on CoNLL-2003, 9 labels.

   Both use the same settings: learning rate 2e-5, batch size 16, 3 epochs, weight decay 0.01.
2. **Extract task vectors.** For each task, subtract the original BERT weights from the fine-tuned weights. The result is that task's vector.
3. **Compose.** Average the two vectors: `(A + B) / 2`. The notebook also has a weighted option (for example 0.7 and 0.3), commented out.
4. **Build the multi-task model.** One BERT encoder with the merged vector applied, plus a separate output layer for each task.
5. **Evaluate.** Run the merged model on the SST-2 and CoNLL-2003 validation sets.

Built with PyTorch and the Hugging Face `transformers`, `datasets` and `evaluate` libraries.

## How to run it

The notebooks depend on each other's output files, so run them in this order:

| Step | Notebook | What it produces |
|---|---|---|
| 1 | `TaskA_Fine_Tuning.ipynb` | `base_model/` and `model_taskA_sst2/` |
| 2 | `TaskB_Fine_Tuning.ipynb` | `model_taskB_ner/` |
| 3 | `TaskA Task Vector Extraction and Merging.ipynb` | `delta_taskA_sst2.pt` |
| 4 | `TaskB Task Vector Extraction and Merging.ipynb` | `delta_taskB_ner.pt` and `delta_merged.pt` |
| 5 | `Applying the Merged Vector to the Model.ipynb` | `delta_merged_encoder_only.pt` and `clean_multi_task_model.pt` |
| 6 | `Evaluation of Multi-Task Model.ipynb` | metrics and a bar chart |

The saved outputs were produced with Python 3.11, torch 2.6, transformers 4.51 and datasets 3.5. The first cell of each notebook installs the packages. The NER fine-tune took about 31 minutes in the saved run (1,869 seconds). A GPU is recommended.

**Model files are not included.** The saved model checkpoints are about 200 GB in total and live only on my computer. They are gitignored, so running the notebooks creates them again.

## Results

Results will be in the project report, which will be added to this repo once it is finalized. It covers the base model, both fine-tuned models, and the merged model with simple and weighted averaging.

The output saved in `Evaluation of Multi-Task Model.ipynb` is from an early run. Use the report for the final numbers.

## Tech stack

Python, PyTorch, Hugging Face Transformers, Datasets and Evaluate, seqeval, scikit-learn, matplotlib, seaborn, Jupyter. Model: `bert-base-uncased` (110M parameters). Datasets: GLUE SST-2 and CoNLL-2003.
