from setuptools import setup, find_packages

setup(name='parriemann',
      version='0.1',
      description='parallel Riemannian Geometry for python',
      url='',
      author='Gabriel Wagner vom Berg',
      author_email='gabriel@bccn-berlin.de',
      license='BSD (3-clause)',
      packages=find_packages(),
      install_requires=['numpy', 'scipy', 'scikit-learn',  'joblib', 'pandas', 'numba'],
      zip_safe=False)
