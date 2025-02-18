import sys
print("Python executable:", sys.executable)
print("sys.path:")
for p in sys.path:
    print("  ", p)
try:
    import joblib
    print("joblib version:", joblib.__version__)
except Exception as e:
    print("Failed to import joblib:", e)
