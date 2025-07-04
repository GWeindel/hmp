Installation Guide
==================

This document will guide you through the process of installing the ``hmp`` package.

It is recommended to use a virtual environment for installing dependencies. See the `official Python venv guide <https://docs.python.org/3/tutorial/venv.html>`_ for instructions on creating and managing Python environments, or use `Anaconda <https://www.anaconda.com/products/distribution>`_ to create a virtual environment using conda.

Step 1: Python Installation
---------------------------

Install `Python <https://www.python.org/>`_.

Step 2: Install PIP
-------------------

If you haven't installed pip, refer to the `Pip Installation Guide <https://pip.pypa.io/en/stable/installation/>`_ for instructions.

Step 3: Install hmp
-------------------

.. code-block:: bash

    pip install hmp==1.0.0-b1

For the cutting edge version, you can clone the repository using *git* (if git is already installed):

Open a terminal and type:

.. code-block:: bash

    git clone https://github.com/gweindel/hmp.git
    git switch devel  # Optional, bleeding edge version

Then move to the cloned repository and run:

.. code-block:: bash

    pip install -e .

Step 4: Verifying Installation
-----------------------------

To ensure ``hmp`` has been successfully installed, run the following command in a Python console:

.. code-block:: python

    import hmp
