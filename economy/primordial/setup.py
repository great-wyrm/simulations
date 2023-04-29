from setuptools import find_packages, setup

long_description = ""
with open("README.md") as ifp:
    long_description = ifp.read()

setup(
    name="primordial",
    version="0.0.1",
    packages=find_packages(),
    install_requires=["torch", "torchvision", "torchaudio"],
    extras_require={
        "dev": ["black", "isort"],
    },
    description="primordial: Pre-launch simulations of Great Wyrm game economy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="zomglings",
    author_email="neeraj@moonstream.to",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "primordial=primordial.cli:main",
        ]
    },
    include_package_data=True,
)
