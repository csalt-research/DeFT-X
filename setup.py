from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

install_requires = [
    'datasets>=2.20.0',
    'evaluate==0.4.0',
    'huggingface-hub>=0.0.17',
    'transformers==4.43.3',
    'scikit-learn',
    'accelerate>=0.21.0'
]

setup(
    name='DeFT-X',
    version='0.0.1',
    description='Tools for training, composing and using sparse fine-tunined DeFT-X vectors for cross-lingual transfer in PyTorch',
    license="Apache",
    long_description=long_description,
    long_description_context_type='text/markdown',
    url='',
    author='',
    author_email='',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=install_requires,
    python_requires='>=3.9',
)
