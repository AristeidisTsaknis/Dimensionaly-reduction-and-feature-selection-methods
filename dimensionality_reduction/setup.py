from setuptools import setup, find_packages

setup(
    name='dimensionality_reduction',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',    
        'scipy',
        'scikit-learn',
        'Kneed',
        'Matplotlib',
        'statsmodels'
    ],
    author='Aristeidis Tsaknis',
    author_email='ece01744@uowm.gr',
    description='A package for different dimensionality reduction methods',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/dimensionality_reduction',
    classifiers=[
        'Programming Language :: Python :: 3.10',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)