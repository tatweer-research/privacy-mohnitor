## Privacy Policy Analysis: Leveraging Bidirectional Language Model Pretraining and Instruction Fine-Tuning

This repository contains the research work done in the master's thesis of Mohammad Al Zoubi with the title "Privacy Policy Analysis: Leveraging Bidirectional Language Model Pretraining and Instruction Fine-Tuning". The thesis was supervised by Prof. Dr. Matthias Grabmair and advised by Santosh Tokala at TUM Chair for LegalTech.

## Research Objective

This research aims to explore the effectiveness of generative language models for privacy policy-related tasks. We investigate the benefits of pre-training on the PrivaSeer corpus and compare the performance of PrivaT5 variants with the widely-used T5 model on a range of privacy tasks. Additionally, we explore single-task and multi-task learning approaches and evaluate the effectiveness of instruction-finetuned FlanT5 models in a zero-shot setting.

## Repository Structure

This repository is organized into the following subdirectories:

* **pretraining**: Contains scripts and code for pre-training T5 models on the PrivaSeer corpus.
* **instruction_finetuning**: Includes scripts and code for instruction-finetuning T5 and FlanT5 models on privacy policy-related tasks.
* **alignment**: Contains scripts for aligning pre-trained and fine-tuned models.
* **utils**: Provides utility functions and scripts used across the project.

## Getting Started

1. Clone the repository:

```
git clone https://github.com/tatweer-research/privacy-mohnitor.git
```

2. Install the required dependencies:

```
pip install -r instruction_finetuning/requirements.txt
```

3. Follow the instructions in each subdirectory for specific tasks and experiments.

## Contributing

Contributions are welcome!

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.