from setuptools import setup, find_packages

setup(
  name="EmbeddingMachine",
  version="0.0.1",
  author="WSDM@ICT",
  author_email="xiaoyanict@foxmail.com",
  description=("EmbeddingMachine is a toolkit for text and image embedding"),
  license="Apache-2.0",
  keywords="embedding models",
  url="https://github.com/shallyan/EmbeddingMachine",
  packages=find_packages(),
  classifiers=[
      # How mature is this project? Common values are
      "Development Status :: 3 - Alpha",
      'Environment :: Console',
      'Operating System :: POSIX :: Linux',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      "License :: OSI Approved :: Apache License",
      'Programming Language :: Python :: 2.7',
      'Programming Language :: Python :: 3',
      'Programming Language :: Python :: 3.4',
      'Programming Language :: Python :: 3.5',
      'Programming Language :: Python :: 3.6'
  ],
  install_requires=[
      'keras >= 2.0.5',
      'tensorflow >= 1.1.0',
      'numpy >= 1.12.1',
      'tqdm >= 4.19.4',
  ]
)
