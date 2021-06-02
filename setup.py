from setuptools import setup, find_packages

setup(
    name='UncertaintyLearning',
    version='1.0.0',
    description='Library implementing methods used in DEUP: Direct Epistemic Uncertainty Estimation',
    packages=find_packages(),
    install_requires=[
        'matplotlib>=3.3.1',
        'numpy>=1.19.1',
        'torch>=1.7.0',
        'scikit-learn',
        'botorch',
        'gpytorch',
        'bsuite',
        'torchvision',
        'requests',
        'tqdm'
    ],
)
