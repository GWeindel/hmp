from setuptools import setup, find_packages


setup(
    name='hsmm_mvpy',
    version='0.0.1-alpha',
    license='BSD-3-Clause',
    author="Gabriel Weindel, Leendert van Maanen, Jelmer Borst",
    author_email='gabriel.weindel@gmail.com',
    packages=find_packages('hsmm_mvpy'),
    package_dir={'': 'hsmm_mvpy'},
    url='https://github.com/GWeindel/hsmm_mvpy',
    keywords='neuroscience EEG stage brain Hidden Semi Markov Model',
    install_requires=[
          'mne','xarray'
      ],
)
