import os
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='hsmm_mvpy',
    version='0.0.2',
    license='BSD-3-Clause',
    author="Gabriel Weindel, Leendert van Maanen, Jelmer Borst",
    author_email='gabriel.weindel@gmail.com',
    packages=find_packages('hsmm_mvpy'),
    package_dir={'': 'hsmm_mvpy'},
    url='https://github.com/GWeindel/hsmm_mvpy',
    keywords='neuroscience EEG stage state bump brain Semi-Markov',
    install_requires=required,
)
