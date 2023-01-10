Contributing guide
~~~~~~~~~~~~~~~~~~

Contributing to BLIMP
---------------------
Clone BLIMP from source as::

    git clone https://github.com/berrygroup/blimp
    cd blimp

Install the test and development mode::

    pip install -e'.[dev,test]'

Optionally install pre-commit. This will ensure that
the pushed code passes the linting steps::

    pre-commit install

Although the last step is not necessary, it is highly recommended,
since it will help you to pass the linting step
(see `Code style guide`_). If you did install ``pre-commit``
but are unable in deciphering some flags, you can
still commit using the ``--no-verify``.

To build documentation for ipython notebooks, you will also have to install
`pandoc <https://pandoc.org/installing.html>`_. This is possible for example using conda::

    conda install -c conda-forge pandoc


Code style guide
----------------
We rely on ``black`` and ``isort`` to do the most of the formatting
- both of them are integrated as pre-commit hooks.
We use ``flake8`` and ``mypy`` to further analyze the code.
Use ``# noqa: <error1>,<error2>`` to ignore certain ``flake8`` errors and
``# type: ignore[error1,error2]`` to ignore specific ``mypy`` errors.

You can use ``tox`` to check the changes::

    tox -e lint


Testing
-------
We use ``tox`` to automate our testing, as well as linting and
documentation creation.
To run the tests, run::

    tox -e py38

If needed, recreate the ``tox`` environment::

    tox -e py38 --recreate

Writing documentation
---------------------
We use ``numpy``-style docstrings for the documentation.

In order to build the documentation, run::

    tox -e docs

To validate the links inside the documentation, run::

    tox -e check-docs

If you need to clean the artifacts from previous documentation builds, run::

    tox -e clean-docs
