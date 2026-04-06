FROM continuumio/miniconda3:v25.11.1-1 AS mrl_base_image
WORKDIR /mrl
COPY pyproject.toml .
COPY source/ source/
ENTRYPOINT ["/bin/bash"]
RUN conda install pytorch -y -c pytorch && \
    conda install -y pyyaml h5py && \
    python3 -m pip install trueskill


FROM mrl_base_image as mrl_production_image
RUN python3 -m pip install . && \
    rm -rf *


FROM mrl_base_image as mrl_development_image
RUN conda install -y \
        pylint \
        mypy \
        types-PyYAML \
        pytest \
        pytest-asyncio \
        sphinx \
        sphinx_rtd_theme && \
    apt-get update && \
    apt-get install -y make graphviz && \
    rm -rf /var/lib/apt/lists/* && \
    python3 -m pip install pyright rstfmt gprof2dot && \
    python3 -m pip install -e .[dev] && \
    rm -rf *
