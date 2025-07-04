HMP
==========


![](plots/general_illustration.png)


HMP is an open-source Python package to analyze neural time-series (e.g. EEG) to estimate Hidden Multivariate Patterns.  HMP is described in Weindel, van Maanen & Borst (2024, [paper](https://direct.mit.edu/imag/article/doi/10.1162/imag_a_00400/125469/Trial-by-trial-detection-of-cognitive-events-in)
) and is a generalized and simplified version of the HsMM-MVPA method developed by Anderson, Zhang, Borst, & Walsh  ([2016](https://psycnet.apa.org/doi/10.1037/rev0000030)).

As a summary of the method, an HMP model parses the reaction time into a number of successive events determined based on patterns in a neural time-serie (e.g. EEG, MEG). Hence any reaction time (or any other relevant behavioral duration) can then be described by a number of cognitive events and the duration between them estimated using HMP. The important aspect of HMP is that it is a whole-brain analysis (or whole scalp analysis) that estimates the peak of trial-recurrent multivariate events on a single-trial basis. These by-trial estimates allow you then to further dig into any aspect you are interested in a signal:
- Describing an experiment or a clinical sample in terms of events detected in the EEG signal
- Describing experimental effects based on the time onset of a particular event
- Estimating the effect of trial-wise manipulations on the identified event presence and time occurrence (e.g. the by-trial variation of stimulus strength or the effect of time-on-task)
- Time-lock EEG signal to the onset of a given event and perform classical ERPs or time-frequency analysis based on the onset of a new event
- And many more (e.g. evidence accumulation models, classification based on the number of events in the signal,...)

# Documentation

The documentation for the latest version is available on  [readthedocs: https://hmp.readthedocs.io/en/latest/welcome.html](https://hmp.readthedocs.io/en/latest/welcome.html)

# To get started
To get started with the code you can run the different tutorials in `docs/source/notebooks` after having installed HMP (see documentation)
- [General aspects on HMP (tutorial 1)](docs/source/notebooks/1-How_HMP_works.ipynb)
- [The different estimation methods (tutorial 2)](docs/source/notebooks/2-The_different_model_classes.ipynb)
- [Applying HMP to real data (tutorial 3)](docs/source/notebooks/3-Applying_HMP_to_real_data.ipynb)
- [Load your own EEG data ](docs/source/notebooks/Data_loading.ipynb)

# Citation: 
To cite the HMP method you can use the following paper: 
```
Weindel, G., van Maanen, L., & Borst, J. P. (2024). Trial-by-trial detection of cognitive events in neural time-series. Imaging Neuroscience, 2, 1-28.
```