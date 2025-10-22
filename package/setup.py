import subprocess
import sys
from setuptools import find_packages, setup
from setuptools.command.install import install


class PreInstallCommand(install):
    """Custom pre-installation command to handle dependency conflicts."""
    def run(self):
        # Force install a compatible protobuf version before other packages
        # This is to resolve conflicts with pre-installed libraries in Vertex AI
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'protobuf==3.20.3'])
        # Ensure python-json-logger is installed
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'python-json-logger==2.0.7'])
        install.run(self)



REQUIRED_PACKAGES = [
    "wandb==0.15.11",
    "deeplake<4",
    "google-cloud-storage",
    # Note: python-json-logger is handled in the pre-install command
]

setup(
    name="know-now-app-trainer-lh",
    version="0.0.1",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    description="Know Now App Trainer Application",
    cmdclass={
        'install': PreInstallCommand,
    },
)
