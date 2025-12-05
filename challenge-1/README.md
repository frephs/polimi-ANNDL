# polimi-an2dl

Authors: Alessandro Del Fatti, Matteo Garzone, Francesco Genovese
Date: November 2025

Please find attached the: 
- [The main notebook](pirates_pain_classification.ipynb)
- [Final report](report.pdf)
- [Data profiling report](data_profiling.ipynb)

- Additional logs from the grid search:
    - [Log grid search 1](logs/log_grid_search_1.txt)
    - [Log grid search 2](logs/log_grid_search_2.txt)

The project is structured in modules, imported in the main notebook, that serves as a pipeline to execute the different steps of training, validation and submission creation.


The main notebook also have some disabled parts, that were only used during the validation process.

## Folder Structure
Please find below the folder structure of the project:

```
challenge-1/
├── logs/
│   ├── log_grid_search_1.txt
│   └── log_grid_search_2.txt
├── src/
│   ├── data_loading/
│   ├── evaluation/
│   ├── models/
│   ├── preprocessing/
│   ├── training/
│   └── utils/
├── configs/
│   └── default_config.yaml
├── pirates_pain_classification.ipynb
├── data_profiling_report.html
└── README.md
```