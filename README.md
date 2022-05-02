# MiniBayesOpt
Mini Bayesian Optimization package at ACML2020 Tutorial on Bayesian Optimization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-rAr71cNaeu-E-Y75fXdrT_QwgS6gf7r)

# Website: http://vu-nguyen.org/BOTutorial_ACML20.html

# Python environment
To create the working environment, please use
```
conda env create -f environment.yml
```

# Demo and Visualization in 1d and 2d
```
demo_1dimension_BO.ipynb
demo_2dimension_BO.ipynb
```

# Demo and Visualization for batch BO
```
demo_batch_BO.ipynb
```

# Customize your own black-box function
```
demo_customize_your_own_function.ipynb
```

# Dependencies
* numpy=1.9.0
* scipy=1.14.0
* scikit-learn=0.16.1
* tabulate=0.8.7

# Error with scipy=1.15
```
ValueError: `f0` passed has more than 1 dimension.
```
If this is the case, please downgrade to scipy=1.14.1

# Slides and Presentation
```
Visit http://vu-nguyen.org/BO_Part_1.pdf and http://vu-nguyen.org/BO_Part_2.pdf
```

# Video
```
http://videolectures.net/acml2020_Nguyen20c/
```

# Reference
```
Vu Nguyen.  "Tutorial on Recent Advances in Bayesian Optimization" Asian Conference on Machine Learning (ACML), 2020.
```
