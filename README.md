# Keras implementation of Domain-Adversarial Training of Neural Networks (DANN)


Code used for these publications: 

* Incremental Unsupervised Domain-Adversarial Training of Neural Networks 
  (https://arxiv.org/abs/2001.04129)

* Domain Adaptation for Handwritten Symbol Recognition: A Case of Study in Old Music Manuscripts 
  (https://link.springer.com/chapter/10.1007/978-3-030-31321-0_12)


Use the following BibTex to cite these papers: 

``` 
@misc{gallego2020incremental,
    title={Incremental Unsupervised Domain-Adversarial Training of Neural Networks},
    author={Antonio-Javier Gallego and Jorge Calvo-Zaragoza and Robert B. Fisher},
    year={2020},
    eprint={2001.04129},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
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



