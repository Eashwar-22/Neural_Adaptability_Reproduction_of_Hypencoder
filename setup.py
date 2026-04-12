import setuptools

setuptools.setup(
    name="hypencoder_cb",
    version="0.0.1",
    author="Hypencoder Team",
    description="A Dual-Encoder framework with Q-Net adaptivity.",
    packages=setuptools.find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy",
        "matplotlib",
        "seaborn",
        "tqdm",
        "ir_datasets",
        "scikit-learn",
        "scipy"
    ],
)
