import os

from setuptools import setup


def find_version(*file_paths):
    def read(*parts):
        here = os.path.abspath(os.path.dirname(__file__))
        with open(os.path.join(here, *parts)) as fp:
            return fp.read().strip()

    import re
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def readme():
    with open('README.md', 'r') as f:
        return f.read()


version = find_version("leaspy", "__init__.py")

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

EXTRAS_REQUIRE = {
    'docs': [
        # TODO: read docs/requirements.txt instead?
        'sphinx==3.3.1',
        'sphinx-gallery',
        'numpydoc'
    ]
}

setup(name="leaspy",
      version=version,

      description='Leaspy is a software package for the statistical analysis of longitudinal data.',
      long_description=readme(),
      long_description_content_type='text/markdown',
      license='GNU GPL v3',  # TODO

      url='https://gitlab.com/icm-institute/aramislab/leaspy',
      project_urls={
          'Bug Reports': 'https://gitlab.com/icm-institute/aramislab/leaspy/issues',
          'Source': 'https://gitlab.com/icm-institute/aramislab/leaspy',
      },

      author='Igor Koval, Raphael Couronne, Arnaud Valladier, Etienne Maheux, Benoit Martin, Pierre-Emmanuel Poulet, Cecile Di Folco, Juliette Ortholand, Mkrtich Vatinyan, Benoit Sauty De Chalon, Stanley Durrleman',  # TODO
      author_email='igor.koval@icm-institute.org',

      python_requires='>=3.6',

      keywords='leaspy longitudinal',

      packages=['leaspy',

                'leaspy.algo',
                #'leaspy.algo.data',
                'leaspy.algo.fit',
                'leaspy.algo.personalize',
                'leaspy.algo.samplers',
                'leaspy.algo.simulate',
                'leaspy.algo.others',

                'leaspy.datasets',
                #'leaspy.datasets.data',

                'leaspy.io',
                'leaspy.io.data',
                'leaspy.io.settings',
                'leaspy.io.realizations',
                'leaspy.io.outputs',
                'leaspy.io.logs',
                'leaspy.io.logs.visualization',

                'leaspy.models',
                #'leaspy.models.data',
                'leaspy.models.utils',
                'leaspy.models.utils.attributes',
                'leaspy.models.utils.initialization',

                'leaspy.utils',
                'leaspy.utils.parallel',
                'leaspy.utils.posterior_analysis',
                'leaspy.utils.resampling',
                ],

      install_requires=requirements,
      include_package_data=True,
      data_files=[('', ['requirements.txt'])],

      # tests_require=["unittest"],
      test_suite='test',

      classifiers=[
          "Intended Audience :: Science/Research",
          "Intended Audience :: Developers",
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ],
      extras_require=EXTRAS_REQUIRE
      )
