from setuptools import setup, find_packages

setup(
    name='dimensionality_reduction',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',    # List all dependencies here
        'scipy',
        'scikit-learn',
    ],
    author='Your Name',
    author_email='your.email@example.com',
    description='A package for different dimensionality reduction methods',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/dimensionality_reduction',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)