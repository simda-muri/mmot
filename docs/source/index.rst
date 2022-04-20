Welcome to MMOT2D's documentation!
==================================

This repository contains an implementation of the Multi-Marginal Optimal Transport (MMOT) algorithm described in

    B. Zhou and M. Parno (2022) `Efficient and Exact Multimarginal Optimal Transport with Pairwise Costs <https://arxiv.org/>`_

It is built on top of some of the core functionality (c-transforms and measure transformatiosn) provided by Flavien Leger in the `BFM package <https://github.com/Math-Jacobs/bfm>`_ .

Installation 
--------------

First install the `cairo` and `jupyterlab` packages using conda:

.. code:: bash

    conda install -c conda-forge cairo jupyterlab


Then you can install mmot by cloning this repository and install the package with pip:

.. code:: bash

    git clone git@github.com:simda-muri/mmot.git
    pip install -e mmot


If you plan on building the sphinx documentation, you will also need to install sphinx and the pydata-sphinx-theme:

.. code:: bash

    conda install -c conda-forge sphinx pydata-sphinx-theme nbsphinx pandoc ipython ipykernel

Then you can build the documentation using 

.. code:: bash 

    sphinx-build -b html docs/source/ docs/build/html


Typical Usage
--------------


Contents 
--------------

.. toctree::
   :maxdepth: 1
   :caption: Examples 

   examples/Introduction.ipynb
   examples/Barycenter.ipynb
   examples/MNISTBarycenter.ipynb
   
.. toctree::
   :maxdepth: 2
   :caption: API Documentation:

   solver
   bfm 
   graph



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
