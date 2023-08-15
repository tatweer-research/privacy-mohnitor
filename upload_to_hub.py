from huggingface_hub import HfApi, login

api = HfApi()
token = ""
username = "alzoubi36"
folder_path = "/home/Mohammad.Al-Zoubi/test-flax/privaseer-3b"
model_name = f"{username}/privat5-v1.1-3b"

login(token=token, add_to_git_credential=False)

print("Creating repo...")
api.create_repo(token=token, repo_id=model_name, repo_type="model", exist_ok=True)

print(f"Uploading folder {folder_path} as model name {model_name}...")
api.upload_folder(
    folder_path=folder_path,
    repo_id=model_name,
    repo_type="model",
)