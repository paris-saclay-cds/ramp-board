import pandas as pd
import numpy as np
import click
from generic import train_models

@click.command()
@click.option('--alias', default='kegl/rf300')
def train(alias):
	models = pd.read_csv("output/submissions.csv")
	model = models[models['alias'] == alias]
	train_models(model)

if __name__ == '__main__':
    train()