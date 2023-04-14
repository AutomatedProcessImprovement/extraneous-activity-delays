from setuptools import setup

setup(
    name='extraneous_activity_delays',
    version='2.0.0',
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        'pandas',
        'setuptools',
        'lxml',
        'numpy',
        'scipy',
        'hyperopt'
    ]
)
