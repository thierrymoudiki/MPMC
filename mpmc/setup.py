import platform
from setuptools import setup, find_packages
from codecs import open
from os import path

__version__ = '0.1.0'

# get the dependencies and installs
here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [
    x.strip() for x in all_reqs if "git+" not in x
]

dependency_links = [
    x.strip().replace("git+", "")
    for x in all_reqs
    if x.startswith("git+")
]

setup(
    name='mpmc',
    version=__version__,
    description='Message-passing Monte Carlo',
    long_description='Message-passing Monte Carlo for quasi-random numbers generation',
    url='',
    #alias='mpmc',
    download_url='',
    license='MIT',
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'Programming Language :: Python :: 3',
    ],
    keywords="",
    packages=find_packages(exclude=["docs", "tests*"]),
    include_package_data=True,
    author='T. Moudiki',
    install_requires=install_requires,
    dependency_links=dependency_links,
    author_email='thierry.moudiki@gmail.com'
)
