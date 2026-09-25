# Test datasets (copied from https://github.com/mrvr/NMI)

| File | Role |
|------|------|
| `data.txt` … `data5.txt` | Small / synthetic examples (`?` = missing; decision often `dec`) |
| `data*_train.txt` / `data*_test.txt` | Train/test splits |
| `data*_master.txt` | Complete ground-truth companions |
| `dengue.csv` | Dengue clinical sample for system tests |

These files are vendored so DengueCAD can run tests locally and on GitHub Actions
without depending on a live checkout of NMI for *data* (the NMI *library* is still
loaded via `NMI_ROOT` / sibling `../NMI`).
