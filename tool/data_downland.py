import kagglehub

# Download latest version
path = kagglehub.dataset_download("chrisfilo/urbansound8k",'urbansound8k')

print("Path to dataset files:", path)