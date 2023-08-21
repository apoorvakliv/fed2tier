.. _installation:

Installation 
============

Install the package
-------------------

Follow this procedure to prepare the environment and install **Fed2Tier**:

1. Install a Python 3.8 (>=3.6, <=3.9) virtual environment using venv.
   
 See the `Venv installation guide <https://docs.python.org/3/library/venv.html>`_ for details.

2. Create a new environment for the project.
    A. Using Virtual Environment
        .. code-block:: console

            $ python3 -m venv env

    B. Using conda
        .. code-block:: console

            $ conda create -n env python=3.9

3. Activate the virtual environment.

    A. Virtual Environment
        .. code-block:: console

            $ source env/bin/activate

    B. Conda Environment
        .. code-block:: console

            $ conda activate env

4. Install the **Fed2Tier** package.

    1. Clone the **Fed2Tier** repository:
    
        .. code-block:: console
        
            $ git clone https://github.com/apoorvakliv/fed2tier
            $ cd Fed2Tier

    2. Install dependencies:
    
        .. code-block:: console
        
            $ pip install -r requirements.txt

