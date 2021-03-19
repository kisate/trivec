# trivec_internship
Temporary repository of Trivec for summer internship applications

## Train logging
We use [neptune](https://neptune.ai/) to track metrics and losses during training.

## Usage
To train model only on drug-drug data (without proteins) use command
```
$ python run_trivec.py --metrics_separately --gpu --epoch 25
```
* ```--use_proteins``` for train with proteins.
* ```--log``` and ```--neptune_project [PROJECT_NAME]``` to tracking metrics and loses with neptune.
* ```--experiment_name '<EXP_NAME>'``` to add custom experiment name for neptune and model saving path.
