from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["numpy", "torch"]

setup(
    name="tqdnn",
    version="0.0.1",
    author="TuringQ-ToolChain-Team",
    author_email="xieqikai@turingq.com",
    description="TuringQ DNN library",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/HybridQuanta/QZone-Dnn",
    packages=find_packages(),
    install_requires=requirements,
)
