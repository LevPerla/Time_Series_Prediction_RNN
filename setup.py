from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
source_root = Path(".")
with (source_root / "README.md").open(encoding="utf-8") as f:
    long_description = f.read()

# Read the requirements
with (source_root / "requirements.txt").open(encoding="utf8") as f:
    requirements = f.readlines()

setup(
    name='ts_rnn',
    version='0.1',
    author="Lev Perla",
    author_email="levperla@mail.ru",
    description='Package to forecast time series with recurrent neural network',
    packages=find_packages(),
    url='http://https://github.com/LevPerla/Time_Series_Prediction_RNN',
    license="MIT",
    python_requires=">=3.7",
    install_requires=requirements,
    keywords="pandas data-science data-analysis python jupyter ipython",
    long_description=long_description,
    long_description_content_type="text/markdown",
)