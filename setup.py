import itertools

import setuptools

import versioneer

with open("README.md", "r") as fh:
    long_description = fh.read()

extras_require = dict(
    visu=[
        "matplotlib>=3.2",
        "bokeh>=2.3",
        "datashader>=0.13",
        "hvplot>=0.7",
        "panel>=0.12",
        "holoviews>=1.14",
    ],
    docs=["sphinx>=1.4", "nbsphinx", "sphinx-autobuild", "sphinx-rtd-theme"],
    dev=["versioneer", "black", "jupyterlab>=3.1", "pooch"],
    test=["pytest>=3.8", "hypothesis>=4.53"],
)

# Add all extra requirements
extras_require["all"] = list(set(itertools.chain(*extras_require.values())))

setuptools.setup(
    name="xcompact3d_toolbox",
    version=versioneer.get_version(),
    author="Felipe N. Schuch",
    author_email="felipe.schuch@edu.pucrs.br",
    description="A set of tools for pre and postprocessing prepared for the high-order Navier-Stokes solver XCompact3d",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fschuch/xcompact3d_toolbox",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy==1.20", # because of numba
        "scipy>=1.5",
        "traitlets>=4.3",
        "ipywidgets>=7.5",
        "pandas>=1.1",
        "xarray>=0.16",
        "netcdf4",
        "dask[complete]>=2.22",
        "numba>=0.50",
        "tqdm>=4.62",
        "numpy-stl>=2.16.3",
    ],
    extras_require=extras_require,
    tests_require=["pytest"],
)
