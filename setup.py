"""Setup configuration for Predictive Maintenance MLOps package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="predictive-maintenance-mlops",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="End-to-end MLOps pipeline for predictive maintenance using TensorFlow and Azure",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/predictive-maintenance-mlops",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=3.0",
            "black>=23.0",
            "flake8>=6.0",
            "isort>=5.0",
            "mypy>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "train-model=train_model:main",
            "run-inference=inference_service:main",
            "detect-drift=drift_detection:main",
        ],
    },
)
