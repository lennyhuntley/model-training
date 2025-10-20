from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = ["wandb==0.15.11", "python-json-logger==2.0.7"]

setup(
    name="know-now-app-trainer-lh",
    version="0.0.1",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    description="Know Now App Trainer Application",
)
