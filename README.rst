Multimarginal Optimal Transport (MMOT)
========================================

Code to accompany paper 

    B. Zhou and M. Parno (2022) "Efficient and Exact Multimarginal Optimal Transport with Pairwise Costs"

Installation 
--------------

First install the `cairo` and `jupyterlab` packages using conda:

.. code:: 

    conda install -c conda-forge cairo jupyterlab

If you plan on building the sphinx documentation, you will also need to install sphinx and the pydata-sphinx-theme:

.. code::
    conda install -c conda-forge sphinx pydata-sphinx-theme


Then you can install mmot by cloning this repository and install the package with pip:

.. code::

    git clone git@github.com:simda-muri/mmot.git
    pip install -e mmot
    
On Colab, you can install mmot by:
.. code::
    
    !git clone https://github.com:simda-muri/mmot.git
    !pip install -e mmot
   


Example Usage
--------------
