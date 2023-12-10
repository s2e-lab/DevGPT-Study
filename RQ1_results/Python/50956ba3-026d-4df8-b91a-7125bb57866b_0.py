from setuptools import setup, find_packages

setup(
    name='axolotl_plugin_mysuperplugin',
    version='0.1',
    packages=find_packages(),  # This automatically finds all packages and subpackages
    namespace_packages=['axolotl.plugins'],  # Declare the namespace
    install_requires=[
        # Any dependencies your plugin has can go here
        # 'somepackage>=1.0.0',
    ],
    classifiers=[
        # Some useful classifiers you might want to use
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    author='Your Name',
    author_email='your.email@example.com',
    description='A super plugin for axolotl',
    keywords='axolotl plugin',
    url='https://github.com/yourusername/axolotl_plugin_mysuperplugin',
)
