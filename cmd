# ESOL (reg)
python -m src.train --dataset ESOL --arch gcn
python -m src.train --dataset ESOL --arch gat
python -m src.train --dataset ESOL --arch mpnn

# HIV (binary)
python -m src.train --dataset HIV --arch gcn
python -m src.train --dataset HIV --arch gat
python -m src.train --dataset HIV --arch mpnn

# TOX21 (multi-label)
python -m src.train --dataset TOX21 --arch gcn
python -m src.train --dataset TOX21 --arch gat
python -m src.train --dataset TOX21 --arch mpnn