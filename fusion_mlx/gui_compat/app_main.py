#!/usr/bin/env python3
"""
MLX-GUI macOS App Entry Point
Dedicated entry point for the macOS app bundle.
"""

# src/mlx_gui/app_main.py
import logging
import os
import sys
from pathlib import Path


def setup_app_environment():
    """Setup the environment for the macOS app bundle."""
    # Get the app bundle path
    if hasattr(sys, "frozen") and sys.frozen:
        # Running as app bundle
        app_bundle_path = Path(sys.executable).parent.parent
        # No need to manually add Resources to sys.path; PyInstaller handles it.
        # Removing this avoids duplicate module imports that caused nanobind errors.

        # Add FFmpeg binaries to PATH for audio transcription
        frameworks_path = app_bundle_path / "Frameworks"
        ffmpeg_paths = []

        # Look for FFmpeg binaries in the app bundle
        for ffmpeg_dir in ["ffmpeg", "ffprobe"]:
            ffmpeg_path = frameworks_path / ffmpeg_dir
            if ffmpeg_path.exists():
                ffmpeg_paths.append(str(ffmpeg_path))

        if ffmpeg_paths:
            current_path = os.environ.get("PATH", "")
            new_path = (
                ":".join(ffmpeg_paths + [current_path])
                if current_path
                else ":".join(ffmpeg_paths)
            )
            os.environ["PATH"] = new_path
            print(f"✅ Added FFmpeg to PATH: {ffmpeg_paths}")
        else:
            print(
                "⚠️  FFmpeg binaries not found in app bundle - audio transcription may not work"
            )

        # Set up logging to file in user's home directory
        log_dir = Path.home() / "Library" / "Logs" / "MLX-GUI"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "mlx-gui.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

        print("""
███╗   ███╗██╗     ██╗  ██╗      ██████╗ ██╗   ██╗██╗
████╗ ████║██║     ╚██╗██╔╝     ██╔════╝ ██║   ██║██║
██╔████╔██║██║      ╚███╔╝█████╗██║  ███╗██║   ██║██║
██║╚██╔╝██║██║      ██╔██╗╚════╝██║   ██║██║   ██║██║
██║ ╚═╝ ██║███████╗██╔╝ ██╗     ╚██████╔╝╚██████╔╝██║
╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝      ╚═════╝  ╚═════╝ ╚═╝
        """)
        from fusion_mlx.gui_compat import __version__

        print("MLX-GUI - Apple Silicon AI Model Server")
        print(f"Version {__version__}")
        print("By Matthew Rogers (@RamboRogers)")
        print("https://github.com/RamboRogers/mlx-gui")
        print()
        print("MLX-GUI App Bundle starting...")
        print(f"Logs: {log_file}")

    else:
        # Running in development
        logging.basicConfig(level=logging.INFO)
        print("""
███╗   ███╗██╗     ██╗  ██╗      ██████╗ ██╗   ██╗██╗
████╗ ████║██║     ╚██╗██╔╝     ██╔════╝ ██║   ██║██║
██╔████╔██║██║      ╚███╔╝█████╗██║  ███╗██║   ██║██║
██║╚██╔╝██║██║      ██╔██╗╚════╝██║   ██║██║   ██║██║
██║ ╚═╝ ██║███████╗██╔╝ ██╗     ╚██████╔╝╚██████╔╝██║
╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝      ╚═════╝  ╚═════╝ ╚═╝
        """)
        from fusion_mlx.gui_compat import __version__

        print("MLX-GUI - Apple Silicon AI Model Server")
        print(f"Version {__version__}")
        print("By Matthew Rogers (@RamboRogers)")
        print("https://github.com/RamboRogers/mlx-gui")
        print()
        print("MLX-GUI running in development mode")


def main():
    """Main entry point for the macOS app."""
    try:
        # Setup the environment
        setup_app_environment()

        # Test if we can import rumps
        try:
            # rumps = importlib.util.find_spec('rumps')  # noqa: F401
            print("rumps imported successfully")
        except ImportError as e:
            print(f"Failed to import rumps: {e}")
            sys.exit(1)

        # Import and run the tray app
        from fusion_mlx.gui_compat.tray import run_tray_app

        print("Starting MLX-GUI tray app...")

        # Check for audio dependencies
        audio_modules = []
        try:
            # mlx_whisper = importlib.util.find_spec('mlx_whisper')  # noqa: F401
            audio_modules.append("MLX-Whisper")
        except ImportError:
            pass

        try:
            # parakeet_mlx = importlib.util.find_spec('parakeet_mlx')  # noqa: F401
            audio_modules.append("Parakeet-MLX")
        except ImportError:
            pass

        if audio_modules:
            print(f"Audio support: ✅ {', '.join(audio_modules)} available")
        else:
            print("Audio support: ⚠️  No audio libraries found")
            print("Install with: pip install mlx-whisper parakeet-mlx")
            print(
                "MLX-GUI will work without audio support, but audio transcription won't be available."
            )

        # Start with default settings
        success = run_tray_app(port=8000, host="127.0.0.1")

        if not success:
            print("Failed to start MLX-GUI tray app")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nMLX-GUI shutting down...")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting MLX-GUI: {e}")
        logging.exception("Error starting MLX-GUI")
        # Don't exit immediately - let user see the error
        input("Press Enter to exit...")
        sys.exit(1)


if __name__ == "__main__":
    main()
