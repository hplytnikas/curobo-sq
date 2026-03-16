mkdir -p checkpoints/shapenet
mkdir -p checkpoints/normalized

gdown 1Nsgtm_nCyp6qbRgnenoJVqL88eS1GXmC -O checkpoints/shapenet/config.yaml
gdown 1ypCViehSOzkCFL6dcCDfdPRzuj_MIayz -O checkpoints/shapenet/ckpt.pt

gdown 1l0wpNssH7f3V61SUA4VcjrVy-ganmIp_ -O checkpoints/normalized/config.yaml
gdown 1r1ydYXkMf7q6U99ze78-zkLiKnO3ICGk -O checkpoints/normalized/ckpt.pt
