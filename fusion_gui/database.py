"""
Database connection and initialization for fusion_gui.
"""

import os
from pathlib import Path
from typing import Generator, Optional
import appdirs
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.engine import Engine

from fusion_gui.models import Base, AppSettings


class DatabaseManager:
    """Manages database connections and initialization."""

    def __init__(self, database_path: Optional[str] = None):
        if database_path is None:
            app_dir = appdirs.user_data_dir("fusion-mlx", "fusion-mlx")
            os.makedirs(app_dir, exist_ok=True)
            database_path = os.path.join(app_dir, "fusion-gui.db")

        self.database_path = database_path
        self.database_url = f"sqlite:///{database_path}"

        self.engine = create_engine(
            self.database_url,
            echo=False,
            connect_args={
                "check_same_thread": False,
                "timeout": 30,
            }
        )

        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )

        self._initialize_database()

    def _initialize_database(self):
        Base.metadata.create_all(bind=self.engine)

        with self.engine.connect() as connection:
            connection.execute(text("PRAGMA journal_mode=WAL"))
            connection.execute(text("PRAGMA foreign_keys=ON"))
            connection.execute(text("PRAGMA temp_store=MEMORY"))
            connection.execute(text("PRAGMA synchronous=NORMAL"))
            connection.commit()

        self._insert_default_settings()
        self._reset_model_statuses()

    def _insert_default_settings(self):
        default_settings = [
            ("server_port", 8000, "Default server port"),
            ("max_concurrent_requests", 5, "Maximum concurrent inference requests"),
            ("max_concurrent_requests_per_model", 1, "Maximum concurrent requests per model"),
            ("max_concurrent_models", 3, "Maximum concurrent loaded models"),
            ("auto_unload_inactive_models", True, "Automatically unload models after inactivity"),
            ("model_inactivity_timeout_minutes", 5, "Minutes before unloading inactive models"),
            ("enable_system_tray", True, "Enable system tray integration"),
            ("log_level", "INFO", "Application logging level"),
            ("huggingface_cache_dir", "", "HuggingFace cache directory path"),
            ("enable_gpu_acceleration", True, "Enable GPU acceleration when available"),
            ("bind_to_all_interfaces", False, "Bind server to all interfaces (0.0.0.0) instead of localhost only"),
            ("max_tokens_limit", 16384, "Maximum tokens allowed per request"),
        ]

        with self.get_session() as session:
            for key, value, description in default_settings:
                existing = session.query(AppSettings).filter_by(key=key).first()
                if not existing:
                    setting = AppSettings(key=key, description=description)
                    setting.set_typed_value(value)
                    session.add(setting)
            session.commit()

    def _reset_model_statuses(self):
        try:
            with self.get_session() as session:
                from fusion_gui.models import Model
                models = session.query(Model).all()
                for model in models:
                    if model.status != "unloaded":
                        model.status = "unloaded"
                        model.last_unloaded_at = None
                session.commit()
                if models:
                    print(f"🔄 Reset {len(models)} models to unloaded status on startup")
        except Exception as e:
            print(f"Warning: Could not reset model statuses: {e}")

    def get_session(self) -> Session:
        return self.SessionLocal()

    def get_session_generator(self) -> Generator[Session, None, None]:
        session = self.SessionLocal()
        try:
            yield session
        finally:
            session.close()

    def close(self):
        self.engine.dispose()

    def get_setting(self, key: str, default=None):
        with self.get_session() as session:
            setting = session.query(AppSettings).filter_by(key=key).first()
            if setting:
                return setting.get_typed_value()
            return default

    def set_setting(self, key: str, value, description: str = None):
        with self.get_session() as session:
            setting = session.query(AppSettings).filter_by(key=key).first()
            if setting:
                setting.set_typed_value(value)
                if description:
                    setting.description = description
            else:
                setting = AppSettings(key=key, description=description)
                setting.set_typed_value(value)
                session.add(setting)
            session.commit()

    def vacuum_database(self):
        with self.engine.connect() as connection:
            connection.execute(text("VACUUM"))
            connection.commit()

    def backup_database(self, backup_path: str):
        import shutil
        shutil.copy2(self.database_path, backup_path)

    def update_model_sizes_from_disk(self):
        try:
            with self.get_session() as session:
                from fusion_gui.models import Model
                import os

                models = session.query(Model).all()
                updated_count = 0

                for model in models:
                    if model.path:
                        actual_path = self._resolve_model_path(model.path)

                        if os.path.exists(actual_path):
                            total_size_bytes = 0
                            for root, dirs, files in os.walk(actual_path):
                                for file in files:
                                    if file.endswith(('.safetensors', '.bin', '.pth', '.pt', '.gguf', '.npz')):
                                        file_path = os.path.join(root, file)
                                        if os.path.exists(file_path):
                                            total_size_bytes += os.path.getsize(file_path)

                            if total_size_bytes > 0:
                                file_size_gb = total_size_bytes / (1024**3)

                                if "whisper" in model.path.lower() or "parakeet" in model.path.lower():
                                    overhead_multiplier = 1.15
                                else:
                                    overhead_multiplier = 1.25

                                new_memory_gb = max(file_size_gb * overhead_multiplier, 0.1)
                                new_memory_gb = round(new_memory_gb, 1)

                                if abs(model.memory_required_gb - new_memory_gb) > 0.5:
                                    old_memory = model.memory_required_gb
                                    model.memory_required_gb = new_memory_gb
                                    updated_count += 1
                                    print(f"📊 Updated {model.name}: {old_memory:.1f}GB → {new_memory_gb:.1f}GB")
                        else:
                            print(f"⚠️  Model path not found: {model.name} -> {actual_path}")

                session.commit()

                if updated_count > 0:
                    print(f"✅ Updated memory requirements for {updated_count} models")
                else:
                    print("ℹ️  No models needed size updates")

        except Exception as e:
            print(f"Warning: Could not update model sizes: {e}")

    def _resolve_model_path(self, model_path: str) -> str:
        if os.path.exists(model_path):
            return model_path

        if "/" in model_path and not os.path.exists(model_path):
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "fusion-mlx")

            cache_name = "models--" + model_path.replace("/", "--")
            cache_path = os.path.join(cache_dir, cache_name)

            if os.path.exists(cache_path):
                snapshots_dir = os.path.join(cache_path, "snapshots")
                if os.path.exists(snapshots_dir):
                    snapshot_dirs = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
                    if snapshot_dirs:
                        actual_path = os.path.join(snapshots_dir, snapshot_dirs[0])
                        return actual_path

        return model_path

    def get_database_size(self) -> int:
        return os.path.getsize(self.database_path)

    def get_database_info(self) -> dict:
        with self.engine.connect() as connection:
            result = connection.execute(text("PRAGMA database_list")).fetchall()
            main_db = next((row for row in result if row[1] == "main"), None)

            if main_db:
                return {
                    "path": main_db[2],
                    "size_bytes": self.get_database_size(),
                    "journal_mode": connection.execute(text("PRAGMA journal_mode")).scalar(),
                    "foreign_keys": bool(connection.execute(text("PRAGMA foreign_keys")).scalar()),
                    "synchronous": connection.execute(text("PRAGMA synchronous")).scalar(),
                }
        return {}


# Global database manager instance
db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager()
    return db_manager


def get_db_session() -> Generator[Session, None, None]:
    db_manager = get_database_manager()
    session = db_manager.get_session()
    try:
        yield session
    finally:
        session.close()


def close_database():
    global db_manager
    if db_manager:
        db_manager.close()
        db_manager = None