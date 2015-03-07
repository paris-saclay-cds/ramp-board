import click
import numpy as np
import pandas as pd

from databoard.generic import train_models
from databoard.model import shelve_database

@click.command()
@click.option('--alias', default='kegl/rf300')
def train(path):

    with shelve_database() as db:
        models = db['models']
    model = models[models['path'] == path]
    train_models(model)

if __name__ == '__main__':
    train()