from setuptools import setup, find_packages

setup(
    name="mini-infer-sdk",
    version="0.1.0",
    description="Python SDK for mini-infer — a lightweight C++/CUDA LLM inference engine",
    author="mini-infer team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "tokenizers>=0.15.0",
    ],
    extras_require={
        "dev": ["pytest", "pytest-benchmark"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)