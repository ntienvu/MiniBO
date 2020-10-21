from setuptools import setup, find_packages

setup(
    name='mini_bo',
    version='1',
    packages = ["bayes_opt",
    "mini_bo.functions",
    "mini_bo.gp",
	"mini_bo.utility",
	"mini_bo.visualization"],
    include_package_data = True,
    description='Mini Bayesian Optimization',
    install_requires=[
        "numpy >= 1.9.0",
        "scipy >= 0.14.0",
        "scikit-learn >= 0.16.1",
        "tabulate">=0.8.7",
    ],
)
