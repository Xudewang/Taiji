# -*- coding: utf-8 -*-
'''This sets up the package.
Stolen from http://python-packaging.readthedocs.io/en/latest/everything.html and indirectly from Huang Song
and modified by me.
'''
__version__ = '0.1.0'

from setuptools import setup, find_packages


def readme():
    """Load the README file."""
    with open('README.md') as f:
        return f.read()


# let's be lazy and put requirements in one place
# what could possibly go wrong?
# with open('requirements.txt') as infd:
#     INSTALL_REQUIRES = [x.strip('\n') for x in infd.readlines()]

# Running setup
setup(
    name='Taiji',
    version=__version__,
    description=('My tools to do basic galaxy photometry.'),
    long_description=readme(),
    long_description_content_type="text/markdown",
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    keywords='astronomy, photometry',
    author='Dewang Xu',
    author_email='dwxu.astro@gmail.com',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3.6',
)
