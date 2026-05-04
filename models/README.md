# Models directory

This folder holds trained artifacts:
- `sql_qa_checkpoint.pth` (periodic training checkpoint)
- `sql_qa_model.pth` (final model)

These files are **ignored by git** by default because GitHub blocks large files (>100MB).

## Option A (default): keep weights untracked
Nothing to do — `.gitignore` already ignores `models/*.pth`.

## Option B: track weights with Git LFS
If you want to upload `.pth` weights to GitHub, use Git LFS:

```bash
git lfs install
git lfs track "*.pth"
```

This creates/updates `.gitattributes`. Then add and commit:
```bash
git add .gitattributes models/*.pth
git commit -m "Track model weights with Git LFS"
```

Notes:
- Your GitHub account/repo must allow LFS storage/bandwidth.
- Alternatively, publish weights as a GitHub Release asset or external download link.
