# Command cheatsheet

- Create dataset

```bash
docker-compose run --rm base python data_creation/create_molecules.py --atom-size 8 && \
docker-compose run --rm base python data_creation/create_molecules.py --atom-size 12 && \
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

## Moler

<https://pypi.org/project/molecule-generation/>

```bash
cp data/molecules/size_20/train.smi data/molecules/size_20/train.smiles
cp data/molecules/size_20/train.smi data/molecules/size_20/valid.smiles
cp data/molecules/size_20/train.smi data/molecules/size_20/test.smiles
```

```bash
dc run --rm base molecule_generation preprocess data/molecules/size_20/ data/molecules/size_20_moler/ data/molecules/size_20_moler_tracke/
```

```bash
dc run --rm base molecule_generation train MoLeR data/molecules/size_20_moler_tracke/ --save-dir output_moler_20
```

```bash
dc run --rm base molecule_generation sample output_moler_20 10
```

## DEG

```bash
python main.py --training_data="../data/molecules/size_8/train.smiles" --output output_deg_8
```

```bash
python retro_star_listener.py --filenames "output_deg_8/generated_samples.txt" "output_deg_12/generated_samples.txt" --output_filenames "output_deg_8/output_syn.txt" "output_deg_12/output_syn.txt"
```

## RNN

```bash
dc run --rm base python Molecule-RNN/vocab/chembl_selfies_vocab.py /app/data/molecules/size_8/train.smi /app/data/molecules/size_8/selfies.rnn.vocab
dc run --rm base python Molecule-RNN/vocab/chembl_regex_vocab.py /app/data/molecules/size_8/train.smi /app/data/molecules/size_8/regex.rnn.vocab
dc run --rm base python Molecule-RNN/vocab/chembl_char_vocab.py /app/data/molecules/size_8/train.smi /app/data/molecules/size_8/char.rnn.vocab
```

```bash
dc run --rm base python Molecule-RNN/train.py /app/data/molecules/size_8/train.smi selfies /app/data/molecules/size_8/selfies.rnn.vocab /app/output_rnn_8_selfies/
dc run --rm base python Molecule-RNN/train.py /app/data/molecules/size_8/train.smi regex /app/data/molecules/size_8/regex.rnn.vocab /app/output_rnn_8_regex/
dc run --rm base python Molecule-RNN/train.py /app/data/molecules/size_8/train.smi char /app/data/molecules/size_8/char.rnn.vocab /app/output_rnn_8_char/
```

```bash
dc run --rm base python Molecule-RNN/sample.py /app/output_rnn_8_selfies/ 1000
dc run --rm base python Molecule-RNN/sample.py /app/output_rnn_8_regex/ 1000
dc run --rm base python Molecule-RNN/sample.py /app/output_rnn_8_char/ 1000
```
