# Synthetic Data Generation and Evaluation Pipelines

A collection of reusable, Docker-containerized components for privacy-preserving data processing, synthetic data generation, and evaluation. Components can run independently or be combined into pipelines using Prefect or other orchestration tools.

The reference workflow compares **Gaussian Copula**, **CTGAN**, and **CopulaGAN**, assessing statistical fidelity, machine learning utility, and privacy-related disclosure risks.

## Documentation

| Guide | What you will find |
|---|---|
| **[Reproducibility Guide](Docs/reproducibility-guide.md)** | Set up the environment and reproduce the reference experiments. |
| **[Components Guide](Docs/components-guide.md)** | Explore the available Docker components, their functionalities, configuration parameters, and standalone execution instructions. |

**To reproduce the experiments:** Start with the [Reproducibility Guide](Docs/reproducibility-guide.md).

## Repository Resources

- [Reference Prefect Workflow](Workflows/artificial_data_pipeline.py)
- [Input Dataset](Datasets/data.csv)

## Citation

If you use this software in your research, please refer to [CITATION.cff](CITATION.cff) for citation information. Details of the associated publication will be added when available.

## License

This repository is distributed under the [Apache License 2.0](LICENSE). Third-party dependencies and Docker images may be subject to separate license terms.

## Acknowledgements

Developed in the context of the **INSAFEDARE** project.
