# Command cheatsheet

- Create dataset

```bash
docker-compose run --rm base python data_creation/create_molecules.py --atom-size 20 && \
docker-compose run --rm base python data_creation/create_molecules.py --atom-size 25 && \
docker-compose run --rm base python data_creation/create_molecules.py --atom-size 30
```

## DiGress

<https://arxiv.org/abs/2209.14734>

```bash
tmux new-session -d -s dd1 'docker-compose run --rm base /bin/bash -c "cd /app/DiGress && python dgd/main.py model=discrete general.gpus=0 max_num_atoms=8"'
tmux new-session -d -s dd2 'docker-compose run --rm base /bin/bash -c "cd /app/DiGress && python dgd/main.py model=continuous general.gpus=0 max_num_atoms=8"'
```

## Molar

<https://pypi.org/project/molecule-generation/>

```bash
cp data/molecules/size_12/train.smi data/molecules/size_12/train.smiles
cp data/molecules/size_12/train.smi data/molecules/size_12/valid.smiles
cp data/molecules/size_12/train.smi data/molecules/size_12/test.smiles
```

```bash
dc run --rm base molecule_generation preprocess data/molecules/size_12/ data/molecules/size_12_moler/ data/molecules/size_12_moler_tracke/
```
