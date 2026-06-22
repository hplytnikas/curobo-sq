# Dataset splits

These `train.lst` / `val.lst` / `test.lst` files pin the **exact** instance split
used for the tabletop fine-tune, so results are reproducible. Each `.lst` is a
plain text file with one model/instance id per line.

## Layout
```
splits/<category>/{train,val,test}.lst
```
where `<category>` is a ShapeNet synset id or `gso`:

| id         | category |
|------------|----------|
| `02876657` | bottle   |
| `02880940` | bowl     |
| `03624134` | knife    |
| `03642806` | laptop   |
| `03797390` | mug      |
| `gso`      | GSO      |

## How the pipeline consumes them
Scripts read the lists from the data tree at `data/ShapeNet/<category>/<split>.lst`.
After placing the data, mirror these splits in (copy or symlink):

```bash
for c in 02876657 02880940 03624134 03642806 03797390 gso; do
  cp splits/$c/*.lst data/ShapeNet/$c/   # or: ln -s
done
```
