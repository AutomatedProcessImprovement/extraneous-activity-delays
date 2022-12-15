from setuptools import setup

setup(
    name='extraneous_activity_delays',
    version='1.3.2',
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        'pandas',
        'setuptools',
        'lxml',
        'numpy',
        'scipy',
        'hyperopt',
        'log_similarity_metrics',
        'estimate_start_times'
    ]
)
