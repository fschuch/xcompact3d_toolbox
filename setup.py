import setuptools
import versioneer

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="xcompact3d_toolbox",
    version=versioneer.get_version(),
    author="Felipe N. Schuch",
    author_email="felipe.schuch@edu.pucrs.br",
    description="A set of tools for pre and postprocessing prepared for the high-order Navier-Stokes solver Xcompact3d",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fschuch/xcompact3d_toolbox",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.6",
    install_requires=[
        "notebook>=6.1",
        "numpy>=1.19",
        "scipy>=1.5"
        "traitlets>=4.3"
        "ipywidgets>=7.5"
        "matplotlib>=3.2"
        "pandas>=1.1"
        "xarray>=0.16"
        "dask>=2.22"
        "numba>=0.50",
        "plotly>=4.8",
        "tqdm",
    ],
)
