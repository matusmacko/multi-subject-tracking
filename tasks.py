import glob
import os

from invoke import task
from src.formatters import read, write
from src import processing
from src import settings


@task()
def process(ctx):
    """Process all .txt files in the given source path."""
    for input_file in glob.glob("{}/*.txt".format(settings.SOURCE_PATH)):
        data = read(input_file)
        data, trajectories = processing.process(data)
        write(
            settings.RESULTS_PATH + "/" + os.path.basename(input_file),
            data,
            trajectories,
        )
        print("processed file {}".format(input_file))
