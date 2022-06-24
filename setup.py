from setuptools import setup

setup(
    name='extraneous_activity_delays',
    version='1.0.0',
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        'pandas'
    ],
    entry_points={
        'console_scripts': [
            'extraneous_activity_delays = main:main',
        ]
    }
)
