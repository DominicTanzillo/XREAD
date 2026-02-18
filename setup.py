from setuptools import setup, find_packages

setup(
    name="xread",
    version="0.1.0",
    description="Radiology Report Pathology Classification with LLM-Based Uncertainty Resolution",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.1.0",
        "pyarrow>=14.0.0",
        "numpy>=1.26.0",
        "tqdm>=4.66.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "pyyaml>=6.0.0",
        "requests>=2.31.0",
        "gradio>=4.0.0",
        "joblib>=1.3.0",
    ],
    entry_points={
        "console_scripts": [
            "xread=main:main",
        ],
    },
)
