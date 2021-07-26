from setuptools import _install_setup_requires, setup, version
setup(name='Task1',
      version='0.0.1',
      install_requires=['gym', 'pygame'],
      package_data={'':['README.md', 'LICENSE'],
                    'assets':['*.bmp']},
      zip_safe=False

)