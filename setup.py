from setuptools import setup, find_packages

setup(
    name="driven_quantum_cluster",
    version="0.1.0",
    description="Simulation of quantum networks in acoustically driven solid-state systems",
    author="Dirk Englund Group",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20.0',
        'matplotlib>=3.5.0',
        'networkx>=2.6.0',
        'scipy>=1.7.0',
    ],
    python_requires='>=3.8',
) 