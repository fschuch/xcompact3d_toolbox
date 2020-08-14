import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="xcompact3d_toolbox",
    version="0.0.1",
    author="Felipe N. Schuch",
    author_email="felipe.schuch@edu.pucrs.br",
    description="A set of tools for pre and postprocessing prepared for the high-order Navier-Stokes solver Xcompact3d",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fschuch/xcompact3d_toolbox",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL-3.0 License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
