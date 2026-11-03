from setuptools import find_packages, setup

# install_requires stays empty: anomalib/torch/openvino come from the hipsterbrown/anomalib-train container,
# and listing them here would make pip reinstall over the container's CUDA builds.
setup(name="anomalib-train", version="0.3.0", packages=find_packages(), install_requires=[])
