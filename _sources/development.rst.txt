###################
 Development guide
###################

**************
 Installation
**************

You need a Python environment with the following dependencies:

.. code:: bash

   - python 3.13.11
   - pytorch 2.8.0
   - pyyaml 6.0.3
   - h5py 3.15.1
   - pydantic 2.12.4
   - trueskill 0.4.5
   - make 4.4.1
   - graphviz 2.42.4
   - pylint 4.0.4
   - mypy 1.71.1
   - pyright 1.1.408
   - pytest 9.0.2
   - pytest-asyncio 1.3.0
   - sphinx 9.1.0
   - sphinx_rtd_theme 3.1.0
   - rstfmt 0.0.14
   - gprof2dot 2025.4.14
   - types-PyYAML 6.0.12.20250915

You can use the Dockerfile to create a Docker image.

.. code:: bash

   docker compose build mrl_dev      # Create the image
   docker compose run --rm mrl_dev   # Run a container from the image

If you want to use the GUI-based features, you must also configure the
connection between the Docker container and your system display. The
required settings are already included in the ``docker-compose.yaml``
file, but they may need adjustment depending on your system.

On macOS systems you will need to install and use `XQuartz
<https://www.xquartz.org/>`_.

Once the container is running, you can verify that the library works by
running:

.. code:: bash

   make quick_tests

***************
 Running tests
***************

You can use the Makefile commands to run the test suites. The tests are
organized into the following categories:

-  ``quick_tests``: fast unit tests that execute in a short time.

-  ``slow_tests``: integration tests that combine multiple components
   and take longer to run.

-  ``performance_tests``: tests that measure execution time for certain
   operations. These tests may fail depending on the performance of your
   machine.

-  ``manual_tests``: tests for the terminal interface of the built-in
   games.

-  ``gui_tests``: tests for the GUI interface of the built-in games.

-  ``debug_tests``: tests intended for debugging purposes. This set is
   initially empty.

To run a specific test repeatedly during debugging, you can annotate it
with the ``debug`` marker so that it is included in the ``debug_tests``
suite. You can then run it using:

.. code:: bash

   make debug_tests

*****************
 Running linters
*****************

You can run static code checks using the Makefile commands (for example
``make pylint``).

Three static analysis tools are used:

-  ``pylint`` for style, syntax, and general code quality checks;
-  ``mypy`` and ``pyright`` for static type checking.

***********
 Profiling
***********

If you want to investigate performance issues, you can use the profiling
command:

.. code:: bash

   make profile_debug_tests

This command runs the ``debug_tests`` suite and generates a graphical
report in PNG format called ``profile_record.png``.

Note that consecutive runs overwrite this file without warning. If you
want to keep previous results, move the file to another location before
running the command again.

***************
 Documentation
***************

The documentation source files are located in ``doc/source``.

You can format the edited source files using:

.. code:: bash

   make format_doc

You can build the documentation using:

.. code:: bash

   make doc

The generated documentation will be available in HTML format in the
``doc/build`` directory.
