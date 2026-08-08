import importlib, os, sys

mods = ['numpy', 'pandas', 'scipy', 'sklearn', 'lightgbm', 'xgboost',
        'catboost', 'torch', 'numba', 'joblib', 'polars']
for m in mods:
    try:
        mod = importlib.import_module(m)
        print("%-12s %s" % (m, getattr(mod, "__version__", "?")))
    except Exception:
        print("%-12s MISSING" % m)
print("cpu_count", os.cpu_count())
print("python", sys.version.split()[0])
