import kagglehub

# Download latest version
path = kagglehub.dataset_download("zalandafridi/srd-dataset")

print("Path to dataset files:", path)