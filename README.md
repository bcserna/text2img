This is an implementation of the [AttnGAN](https://arxiv.org/abs/1711.10485) in PyTorch, with some experimental additions and changes.

### Dataset

* Download the [Caltech-UCSD Birds-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) dataset and extract it to the root folder of the project.
* Download [metadata](https://drive.google.com/open?id=1O_LtUP9sch09QH3s_EBAgLEctBQ5JBSJ) (includes captions) and copy its contents to the dataset folder.

### Experimenting

* To train a DAMSM model, use the `python -m src.main train-damsm <EPOCHS> <NAME> [OPTIONS]` command. `EPOCHS` sets the number of training epochs, `NAME` is the name the model is going to be saved with and further referenced by. Options include:
  * Set patience for early stopping: `--patience=20`
  * Set device: `--device=cuda:0`

* To train the GAN, use `python -m src.main train-gan <EPOCHS> <NAME> <DAMSM> [OPTIONS]`. `EPOCHS` and `NAME` are the number of training epochs and the name of the model respectively. `DAMSM` is the name of the DAMSM model to be used for text-encoding and auxiliary DAMSM-loss. Options include:
  * Continue training of a saved model: `--gan=ExampleModelName`
  * Set device: `--device=cuda:1`

* To generate an image for each sample in the test set, use `python -m src.main validate-gan GAN DAMSM SAVEDIR [OPTIONS]`. `GAN` and `DAMSM` are the names of the models to be used. `SAVEDIR` is the output directory. Options include:
  * Set device: `--device=cuda:2`

For different hyperparameters, change values in `config.py`.
