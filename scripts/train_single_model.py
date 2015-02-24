import click
import numpy as np
import pandas as pd

from databoard.generic import train_models


@click.command()
@click.option('--alias', default='kegl/rf300')
def train(alias):
    models = pd.read_csv("output/submissions.csv")
    model = models[models['alias'] == alias]
    train_models(model)

if __name__ == '__main__':
    train()