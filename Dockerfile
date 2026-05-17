FROM mambaorg/micromamba:2.3.2 AS mrl_base_image
WORKDIR /mrl
COPY --chown=$MAMBA_USER:$MAMBA_USER environment_prod.yaml .
RUN micromamba install -y -n base -f environment_prod.yaml && \
    micromamba clean --all --yes
SHELL ["micromamba", "run", "-n", "base", "/bin/bash", "-c"]


FROM mrl_base_image AS mrl_production_image
COPY --chown=$MAMBA_USER:$MAMBA_USER pyproject.toml .
COPY --chown=$MAMBA_USER:$MAMBA_USER source/ source/
RUN pip install --no-cache-dir . && \
    rm -rf *
ENTRYPOINT ["/bin/bash"]


FROM mrl_base_image AS mrl_development_image
COPY --chown=$MAMBA_USER:$MAMBA_USER environment_dev.yaml .
RUN micromamba install -y -n base -f environment_dev.yaml && \
    micromamba clean --all --yes
COPY --chown=$MAMBA_USER:$MAMBA_USER pyproject.toml .
COPY --chown=$MAMBA_USER:$MAMBA_USER source/ source/
RUN pip install --no-cache-dir -e .[dev]
ENTRYPOINT ["/bin/bash"]
