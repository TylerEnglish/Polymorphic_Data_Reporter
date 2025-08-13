from src.config_model.model import load_config
cfg = load_config()  # reads config/config.toml by default
print("Project:", cfg.env.project_name)
print("Local raw:", cfg.storage.local.raw_root)
print("MinIO enabled:", cfg.storage.minio.enabled)
print("Cleaning outlier method:", cfg.cleaning.outliers.method)