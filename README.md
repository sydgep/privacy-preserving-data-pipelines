# Synthetic Data Generation and Evaluation Pipelines

A collection of reusable, Docker-containerized components for privacy-preserving data processing, synthetic data generation, and evaluation. Components can run independently or be combined into pipelines using Prefect or other orchestration tools.

The reference workflow compares **Gaussian Copula**, **CTGAN**, and **CopulaGAN**, assessing statistical fidelity, machine learning utility, and privacy-related disclosure risks.

## Documentation

| Guide | What you will find |
|---|---|
| **[Reproducibility Guide](Docs/reproducibility-guide.md)** | Set up your environment and reproduce the reference experiments |
| **[Components Guide](Docs/components-guide.md)** | Understand each Docker component, its parameters, and how to run it independently |
| **[Workflow Orchestration Guide](Docs/workflow-orchestration.md)** | Connect components and build custom pipelines, with or without Prefect |

**Co-authors reproducing the experiments:** Start with the [Reproducibility Guide](Docs/reproducibility-guide.md).

## Repository resources

- [Reference Prefect workflow](Workflows/artificial_data_pipeline.py)
- [Input dataset](Datasets/data.csv)

## Citation

For citation information, see [CITATION.cff](CITATION.cff). Details of the associated publication will be updated when available.

## License

See [LICENSE](LICENSE). Third-party dependencies and container images may have their own license terms.

## Acknowledgements

Developed in the context of the **INSAFEDARE** project.
