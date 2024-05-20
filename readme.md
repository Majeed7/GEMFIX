# GEM-FIX
## Unlocking the Game: Estimating Essential Games in Mobius Representation for Explanation and High-order Interaction Detection

### Introduction
This package inlcudes the implementation of GEM-FIX: Game Estimation in Mobius representation for Feature Interaction detection eXplanation

### Features
- **Model Agnostic:** GEM-FIX can explain any model
- **Shapley Value:** GEM-FIX is a additive feature attribution method based on the Shapley value
- **Interaction Detection** GEM-FIX is able to detect high-order significant interactions from exponentially many possibilities


### Credits
GEM-FIX is used the SHAP (See [SHAP](https://github.com/shap/shap)) infrastructure for the implementaiton

### Requirements
See requirements.txt for all the requirements.



### Usage
Below is an example illustrating how to use GEM-FIX:

```python
import transformers
from explainer.gemfix import GEMFIX
 

# load a transformers pipeline model
model = transformers.pipeline('sentiment-analysis', return_all_scores=True)

# explain the model on a sample
explainer = gemfix.Explainer(model)
gemfix_values = explainer.shap_values(["What a great movie! ...if you have no taste."])

```
