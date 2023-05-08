# Command cheatsheet

- Start DiGress training

```bash
tmux new-session -d -s dd1 'docker-compose run --rm base /bin/bash -c "cd /app/DiGress && PYTHONPATH=. python dgd/main.py model=discrete general.gpus=0 max_num_atoms=7"'
tmux new-session -d -s dd2 'docker-compose run --rm base /bin/bash -c "cd /app/DiGress && PYTHONPATH=. python dgd/main.py model=continuous general.gpus=0 max_num_atoms=7"'
```
