# Three Retained Versions

This workspace has been cleaned to keep only the three requested methods:

1. `v14` method  
   - Code: `C:\Users\13411\MyPython\project\v14_conditional_wgan_gp.py`  
   - Output: `C:\Users\13411\MyPython\project\gan_background_outputs_v14_best`

2. `guo` method  
   - Code: `C:\Users\13411\MyPython\project\guo_3dconv_stattn_conditional_wgan_gp.py`  
   - Output: `C:\Users\13411\MyPython\project\gan_background_outputs_guo_kmfix_opt2_stdfix`

3. `kmeans` method  
   - Code: `C:\Users\13411\MyPython\project\kmeans_markov_monthly_rebalance.py`  
   - Output: `C:\Users\13411\MyPython\project\background_8760_outputs`

## Notes

- The previous old script names were renamed to algorithm-based names.
- Non-target version output folders were removed.
- Temporary tuning files (`_tmp_*`) were removed.

## Quick Run Commands

```powershell
# KMeans
.venv\Scripts\python kmeans_markov_monthly_rebalance.py --output-dir background_8760_outputs --solar-zero-before-hour 6 --solar-zero-after-hour 20

# v14
.venv\Scripts\python v14_conditional_wgan_gp.py --output-dir gan_background_outputs_v14_best --kmeans-output-dir background_8760_outputs

# guo
.venv\Scripts\python guo_3dconv_stattn_conditional_wgan_gp.py --output-dir gan_background_outputs_guo_kmfix_opt2_stdfix --kmeans-output-dir background_8760_outputs
```
