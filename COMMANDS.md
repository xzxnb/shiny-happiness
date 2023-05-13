# Command cheatsheet

- Create dataset

```bash
docker-compose run --rm base python data_creation/create_molecules.py --atom-size 8
```

- Start DiGress training

```bash
tmux new-session -d -s dd1 'docker-compose run --rm base /bin/bash -c "cd /app/DiGress && python dgd/main.py model=discrete general.gpus=0 max_num_atoms=8"'
tmux new-session -d -s dd2 'docker-compose run --rm base /bin/bash -c "cd /app/DiGress && python dgd/main.py model=continuous general.gpus=0 max_num_atoms=8"'
```
