# Keras implementation of Domain-Adversarial Training of Neural Networks (DANN)


Code used for these publications: 

* Domain Adaptation for Handwritten Symbol Recognition: A Case of Study in Old Music Manuscripts 
  (https://link.springer.com/chapter/10.1007/978-3-030-31321-0_12)
  
* Incremental Unsupervised Domain-Adversarial Training of Neural Networks 
  (https://ieeexplore.ieee.org/document/9216604)
    * In this case it was used as the base implementation of the DANN algorithm integrated in the incremental loop. 

* Unsupervised neural domain adaptation for document image binarization (https://www.sciencedirect.com/science/article/pii/S0031320321002867)
    * Used as the basis of the implementation. The repository for this paper can be found at: https://github.com/ajgallego/SAE-DANN


If you use this code, consider citing one or more of the following papers: 


``` 
@article{gallego2020incremental,
  author={Gallego, Antonio-Javier and Calvo-Zaragoza, Jorge and Fisher, Robert B.},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Incremental Unsupervised Domain-Adversarial Training of Neural Networks}, 
  year={2020},
  volume={},
  number={},
  pages={1-15},
  doi={10.1109/TNNLS.2020.3025954}
}

@article{Castellanos2021saedann,
  title = {Unsupervised neural domain adaptation for document image binarization},
  journal = {Pattern Recognition},
  volume = {119},
  pages = {108099},
  year = {2021},
  issn = {0031-3203},
  doi = {https://doi.org/10.1016/j.patcog.2021.108099},
  author = {Francisco J. Castellanos and Antonio-Javier Gallego and Jorge Calvo-Zaragoza}
}

@InProceedings{10.1007/978-3-030-31321-0_12,
    author="Mateiu, Tudor N. and Gallego, Antonio-Javier and Calvo-Zaragoza, Jorge",
    editor="Morales, Aythami and Fierrez, Julian and S{\'a}nchez, Jos{\'e} Salvador and Ribeiro, Bernardete",
    title="Domain Adaptation for Handwritten Symbol Recognition: A Case of Study in Old Music Manuscripts",
    booktitle="Pattern Recognition and Image Analysis",
    year="2019",
    publisher="Springer International Publishing",
    address="Cham",
    pages="135--146",
    isbn="978-3-030-31321-0"
}
```


## Usage

Run the following script to download the datasets:

```
bash download_datasets.sh 
```

And "`run_experiment.sh`" to run the experiments: 

```
bash run_experiment.sh
```



