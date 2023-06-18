# Command cheatsheet

- Create dataset

```bash
docker-compose run --rm base python data_creation/create_molecules.py --atom-size 8 && \
docker-compose run --rm base python data_creation/create_molecules.py --atom-size 20 && \
docker-compose run --rm base python data_creation/create_molecules.py --atom-size 25 && \
docker-compose run --rm base python data_creation/create_molecules.py --atom-size 30
```

## DiGress

<https://arxiv.org/abs/2209.14734>

```bash
tmux new-session -d -s dd20 'docker-compose run --rm base /bin/bash -c "cd /app/DiGress && python dgd/main.py model=discrete general.gpus=1 max_num_atoms=20"'
tmux new-session -d -s dd20 'docker-compose run --rm base /bin/bash -c "cd /app/DiGress && python dgd/main.py model=continuous general.gpus=1 max_num_atoms=20"'
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

```bash
dc run --rm base molecule_generation train MoLeR data/molecules/size_12_moler_tracke/ --save-dir output_moler_12
dc run --rm base molecule_generation train CGVAE data/molecules/size_12_moler_tracke/ --save-dir output_cgvae_12
```

```bash
dc run --rm base molecule_generation sample output_cgvae_12 1000
```

## DEG

```bash
python main.py --training_data="../data/molecules/size_8/train.smiles" --output output_deg_8
```
