# CS224N default final project (2022 IID SQuAD track)

Starter code provided by course staff. See set up instructions at the end of this document.

## My implementations
Additional models and layers implemented by me as part of the final project are included in `models.py` and `layers.py`.
Some examples include:
1. Character level embedding
2. Self attention
3. Answer Pointer output


## Training
Run `python train.py` to start training. Set the `--model` flag to specify which model to train. For details, see the `initialize_model` method in `models.py`.

## Testing
Run `python test.py` to test a trained model. Set the `--model` flag to specify which model to test. Set the `--load_path` flag to load trained weights.

## Other arguments
See  `args.py` for which additional command line arguments are supported.

## Setup

1. Make sure you have [Miniconda](https://conda.io/docs/user-guide/install/index.html#regular-installation) installed
    1. Conda is a package manager that sandboxes your project’s dependencies in a virtual environment
    2. Miniconda contains Conda and its dependencies with no extra packages by default (as opposed to Anaconda, which installs some extra packages)

2. cd into src, run `conda env create -f environment.yml`
    1. This creates a Conda environment called `squad`

3. Run `conda activate squad`
    1. This activates the `squad` environment
    2. Do this each time you want to write/test your code

4. Run `python setup.py`
    1. This downloads SQuAD 2.0 training and dev sets, as well as the GloVe 300-dimensional word vectors (840B)
    2. This also pre-processes the dataset for efficient data loading
    3. For a MacBook Pro on the Stanford network, `setup.py` takes around 30 minutes total  

5. Browse the code in `train.py`
    1. The `train.py` script is the entry point for training a model. It reads command-line arguments, loads the SQuAD dataset, and trains a model.
    2. You may find it helpful to browse the arguments provided by the starter code. Either look directly at the `parser.add_argument` lines in the source code, or run `python train.py -h`.
