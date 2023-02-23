# Stochastic Frank Wolfe library for TensorFlow and PyTorch
This repository contains the Stochastic Frank Wolfe (SFW) implementation in TensorFlow and Pytorch that was developed alongside the two following publications:

### Deep Neural Network Training with Frank-Wolfe ([arXiv:2010.07243](https://arxiv.org/abs/2010.07243))
*Authors: [Sebastian Pokutta](http://www.pokutta.com/), [Christoph Spiegel](http://christophspiegel.berlin/), [Max Zimmer](https://maxzimmer.org/)*

Colab Notebooks to reproduce the exact experiments of the paper:
* [Colab Notebook for visualization of constraints (TensorFlow)](https://colab.research.google.com/drive/1t-AbwNQSjNSCoOE0_snF9t-TkFTPClge)
* [Colab Notebook for sparseness during training (TensorFlow)](https://colab.research.google.com/drive/1qDKhGVjN6eH2vGKNHC1lBp-ryMzYC38t)
* [Colab Notebook for comparing stochastic Frank–Wolfe methods (TensorFlow)](https://colab.research.google.com/drive/1BBoEZ5PZfNjIB1iLanPu08YKFdREtM84)
* [Colab Notebook for large network training (PyTorch)](https://colab.research.google.com/drive/1F6EMYDSx29A_T9DEMuOhAtVW5rbxt05W)

In case you find the paper or the implementation useful for your own research, please consider citing:

```
@article{pokutta2020deep,
  title={Deep neural network training with frank-wolfe},
  author={Pokutta, Sebastian and Spiegel, Christoph and Zimmer, Max},
  journal={arXiv preprint arXiv:2010.07243},
  year={2020}
}
```


### Projection-Free Adaptive Gradients for Large-Scale Optimization ([arXiv:2009.14114](https://arxiv.org/abs/2009.14114))
*Authors: [Cyrille W. Combettes](https://cyrillewcombettes.github.io/), [Christoph Spiegel](http://christophspiegel.berlin/), [Sebastian Pokutta](http://www.pokutta.com/)*

Colab Notebooks to reproduce the exact experiments of the paper:
* [Colab Notebook for convex objectives](https://colab.research.google.com/drive/1hwa0bzMcMtpVESIiQfCw8il9jKC-pbfa) (not using this repository)
* [Colab Notebook for non-convex objectives (TensorFlow)](https://colab.research.google.com/drive/13su1LqZEWajgEucJHpCdniQ5W91q3NNW)

In case you find the paper or the implementation useful for your own research, please consider citing:

```
@article{combettes2020projection,
  title={Projection-free adaptive gradients for large-scale optimization},
  author={Combettes, Cyrille W and Spiegel, Christoph and Pokutta, Sebastian},
  journal={arXiv preprint arXiv:2009.14114},
  year={2020}
}
```
