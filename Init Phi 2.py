from huggingface_hub import snapshot_download
model_path = snapshot_download(repo_id="amgadhasan/phi-2",repo_type="model", local_dir="./phi-2", local_dir_use_symlinks=False)
