"""
macOS system tray integration for MLX-GUI.
"""

import logging
import subprocess
import threading
import time
import webbrowser
import socket
import os
from pathlib import Path
from typing import Optional

import requests
import rumps
import uvicorn

from fusion_gui.database import get_database_manager

logger = logging.getLogger(__name__)

# Global lock to prevent multiple tray instances
_tray_lock_file = None


def _acquire_tray_lock():
    """Acquire a file lock to prevent multiple tray instances."""
    global _tray_lock_file
    
    # Create lock file in user's temp directory
    import tempfile
    lock_path = os.path.join(tempfile.gettempdir(), "mlx-gui-tray.lock")
    
    try:
        # Try to create lock file exclusively
        _tray_lock_file = open(lock_path, 'x')
        _tray_lock_file.write(str(os.getpid()))
        _tray_lock_file.flush()
        return True
    except FileExistsError:
        # Lock file already exists, check if process is still running
        try:
            with open(lock_path, 'r') as f:
                pid = int(f.read().strip())
            
            # Check if process is still running (macOS/Unix)
            try:
                os.kill(pid, 0)  # Signal 0 just checks if process exists
                logger.warning(f"Another tray instance is already running (PID: {pid})")
                return False
            except OSError:
                # Process doesn't exist, remove stale lock file
                os.unlink(lock_path)
                return _acquire_tray_lock()  # Try again
                
        except (ValueError, IOError):
            # Invalid lock file, remove it
            try:
                os.unlink(lock_path)
                return _acquire_tray_lock()  # Try again
            except OSError:
                pass
        
        return False


def _release_tray_lock():
    """Release the tray lock file."""
    global _tray_lock_file
    
    if _tray_lock_file:
        try:
            _tray_lock_file.close()
            os.unlink(_tray_lock_file.name)
        except OSError:
            pass
        _tray_lock_file = None


class MLXTrayApp(rumps.App):
    """MLX-GUI system tray application for macOS."""
    
    def __init__(self, port: int = 8000, host: str = "127.0.0.1"):
        # Use simple text instead of icon - much cleaner!
        super().__init__("MLX", title="MLX")
        self.default_port = port
        self.default_host = host
        self.server_thread = None
        self.server_running = False
        
        # Read settings from database
        try:
            db_manager = get_database_manager()
            bind_to_all = db_manager.get_setting("bind_to_all_interfaces", False)
            self.port = db_manager.get_setting("server_port", self.default_port)
            self.host = "0.0.0.0" if bind_to_all else self.default_host
        except Exception as e:
            logger.warning(f"Failed to read settings from database: {e}")
            self.port = self.default_port
            self.host = self.default_host
        
        # Always use localhost for API calls, even when binding to all interfaces
        self.base_url = f"http://127.0.0.1:{self.port}"
        
        # Status tracking
        self.system_status = {}
        self.loaded_models_count = 0
        self.memory_usage = "0GB"
        
        # Create menu items
        self.status_item = rumps.MenuItem("Status: Starting...", callback=None)
        self.models_item = rumps.MenuItem("Models: Loading...", callback=None)
        self.memory_item = rumps.MenuItem("Memory: Loading...", callback=None)
        self.separator1 = rumps.MenuItem("---", callback=None)
        self.admin_item = rumps.MenuItem("Open Admin Interface", callback=self.open_admin)
        self.unload_item = rumps.MenuItem("Unload All Models", callback=self.unload_all_models)
        self.separator2 = rumps.MenuItem("---", callback=None)
        
        # Version info
        from fusion_gui import __version__
        self.version_item = rumps.MenuItem(f"ℹ️ Version {__version__}", callback=None)
        
        # GitHub link
        self.github_item = rumps.MenuItem("📂 View on GitHub", callback=self.open_github)
        self.separator3 = rumps.MenuItem("---", callback=None)
        
        # Network settings
        try:
            db_manager = get_database_manager()
            bind_to_all = db_manager.get_setting("bind_to_all_interfaces", False)
            self.bind_checkbox = rumps.MenuItem("Bind to All Interfaces", callback=self.toggle_bind_interfaces)
            self.bind_checkbox.state = bind_to_all
        except Exception as e:
            logger.warning(f"Failed to initialize bind checkbox: {e}")
            self.bind_checkbox = rumps.MenuItem("Bind to All Interfaces", callback=self.toggle_bind_interfaces)
            self.bind_checkbox.state = False
        
        self.separator4 = rumps.MenuItem("---", callback=None)
        self.quit_item = rumps.MenuItem("Quit", callback=self.quit_app)
        
        # Add items to menu
        self.menu = [
            self.status_item,
            self.models_item, 
            self.memory_item,
            self.separator1,
            self.admin_item,
            self.unload_item,
            self.separator2,
            self.version_item,
            self.github_item,
            self.separator3,
            self.bind_checkbox,
            self.separator4,
            self.quit_item
        ]
        
        # Start status update timer (every 10 seconds)
        self.status_timer = rumps.Timer(self.update_status, 10)
        self.status_timer.start()
        
        # Disable the automatic quit button to avoid duplication
        self.quit_button = None
        
    def start_server_background(self):
        """Start the MLX-GUI server in a background thread."""
        try:
            logger.info(f"Starting MLX-GUI server on {self.host}:{self.port}")
            self.server_running = True
            
            # Import server module here to avoid circular imports
            from fusion_gui.server import create_app
            fastapi_app = create_app()
            
            # Configure uvicorn directly (no CLI involvement)
            config = uvicorn.Config(
                app=fastapi_app,
                host=self.host,
                port=self.port,
                log_level="info",
                reload=False,
                workers=1,  # Single worker to prevent multiple tray instances
                access_log=False  # Reduce log noise
            )
            
            server = uvicorn.Server(config)
            
            # Start the server (this will block)
            import asyncio
            asyncio.run(server.serve())
            
        except Exception as e:
            logger.error(f"Error starting server: {e}")
            self.server_running = False
            self.title = "MLX ❌"
            self.status_item.title = "Status: Server Failed"
    
    def wait_for_server(self):
        """Wait for server to be ready."""
        max_attempts = 30
        for attempt in range(max_attempts):
            try:
                response = requests.get(f"{self.base_url}/health", timeout=2)
                if response.status_code == 200:
                    logger.info("Server is ready")
                    self.title = "MLX"
                    self.update_status()
                    return True
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(1)
        
        logger.error("Server failed to start within timeout")
        self.title = "MLX ❌"
        self.status_item.title = "Status: Server Timeout"
        return False
    
    def update_status(self, sender=None):
        """Update menu items with current system status."""
        try:
            # Get system status
            response = requests.get(f"{self.base_url}/v1/system/status", timeout=5)
            if response.status_code == 200:
                self.system_status = response.json()
                
                # Update status indicators
                status = self.system_status.get('status', 'unknown')
                self.status_item.title = f"Status: {status.title()}"
                
                # Update models count
                model_manager = self.system_status.get('model_manager', {})
                self.loaded_models_count = model_manager.get('loaded_models_count', 0)
                queue_size = model_manager.get('queue_size', 0)
                
                if queue_size > 0:
                    self.models_item.title = f"Models: {self.loaded_models_count} loaded, {queue_size} queued"
                else:
                    self.models_item.title = f"Models: {self.loaded_models_count} loaded"
                
                # Update memory usage
                total_model_memory = model_manager.get('total_model_memory_gb', 0)
                system_memory = self.system_status.get('system', {}).get('memory', {})
                total_system_memory = system_memory.get('total_gb', 0)
                
                self.memory_usage = f"{total_model_memory:.1f}GB / {total_system_memory:.1f}GB"
                memory_percent = model_manager.get('memory_usage_percent', 0)
                self.memory_item.title = f"Memory: {self.memory_usage} ({memory_percent:.1f}%)"
                
                # Update tray text status
                if status == 'running':
                    if self.loaded_models_count > 0:
                        self.title = f"MLX ({self.loaded_models_count})"
                    else:
                        self.title = "MLX"
                else:
                    self.title = "MLX ⚠️"
                    
                # Enable/disable unload button
                self.unload_item.set_callback(self.unload_all_models if self.loaded_models_count > 0 else None)
                
            else:
                raise requests.exceptions.RequestException(f"HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to get status: {e}")
            self.status_item.title = "Status: Server Unreachable"
            self.models_item.title = "Models: Unknown"
            self.memory_item.title = "Memory: Unknown"
            self.title = "MLX ❌"
            self.unload_item.set_callback(None)
    
    def open_admin(self, sender):
        """Open the admin interface in the default browser."""
        admin_url = f"{self.base_url}/admin"
        logger.info(f"Opening admin interface: {admin_url}")
        
        try:
            # Use webbrowser module for better cross-platform support
            webbrowser.open(admin_url)
        except Exception as e:
            logger.error(f"Failed to open browser: {e}")
            # Fallback to macOS open command
            try:
                subprocess.run(["open", admin_url], check=True)
            except subprocess.CalledProcessError as e2:
                logger.error(f"Failed to open with system command: {e2}")
                rumps.alert(
                    title="Error",
                    message=f"Could not open admin interface.\nPlease manually visit: {admin_url}",
                    ok="OK"
                )
    
    def unload_all_models(self, sender):
        """Unload all currently loaded models."""
        if self.loaded_models_count == 0:
            return
            
        # Confirm action
        response = rumps.alert(
            title="Unload All Models",
            message=f"Are you sure you want to unload all {self.loaded_models_count} loaded models?",
            ok="Unload All",
            cancel="Cancel"
        )
        
        if response == 1:  # OK clicked
            try:
                # Get list of loaded models
                models_response = requests.get(f"{self.base_url}/v1/manager/status", timeout=10)
                if models_response.status_code == 200:
                    manager_status = models_response.json()
                    loaded_models = manager_status.get('loaded_models', {})
                    
                    unloaded_count = 0
                    failed_count = 0
                    
                    for model_name in loaded_models.keys():
                        try:
                            unload_response = requests.post(
                                f"{self.base_url}/v1/models/{model_name}/unload",
                                timeout=30
                            )
                            if unload_response.status_code == 200:
                                unloaded_count += 1
                                logger.info(f"Unloaded model: {model_name}")
                            else:
                                failed_count += 1
                                logger.error(f"Failed to unload {model_name}: {unload_response.status_code}")
                        except requests.exceptions.RequestException as e:
                            failed_count += 1
                            logger.error(f"Error unloading {model_name}: {e}")
                    
                    # Show result
                    if failed_count == 0:
                        rumps.notification(
                            title="MLX-GUI",
                            subtitle="Models Unloaded", 
                            message=f"Successfully unloaded {unloaded_count} models",
                            sound=False
                        )
                    else:
                        rumps.alert(
                            title="Unload Complete",
                            message=f"Unloaded: {unloaded_count}\nFailed: {failed_count}",
                            ok="OK"
                        )
                    
                    # Update status immediately
                    self.update_status()
                    
                else:
                    raise requests.exceptions.RequestException(f"HTTP {models_response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to unload models: {e}")
                rumps.alert(
                    title="Error",
                    message=f"Failed to unload models: {e}",
                    ok="OK"
                )
    
    def open_github(self, sender):
        """Open the MLX-GUI GitHub repository."""
        github_url = "https://github.com/RamboRogers/mlx-gui"
        logger.info(f"Opening GitHub repository: {github_url}")
        
        try:
            webbrowser.open(github_url)
        except Exception as e:
            logger.error(f"Failed to open browser: {e}")
            try:
                subprocess.run(["open", github_url], check=True)
            except subprocess.CalledProcessError as e2:
                logger.error(f"Failed to open with system command: {e2}")
                rumps.alert(
                    title="Error", 
                    message=f"Could not open GitHub.\nPlease manually visit: {github_url}",
                    ok="OK"
                )
    
    def toggle_bind_interfaces(self, sender):
        """Toggle the bind to all interfaces setting."""
        current_state = sender.state
        new_state = not current_state
        
        # Confirm the change
        if new_state:
            response = rumps.alert(
                title="Bind to All Interfaces",
                message="This will expose the server on all network interfaces (0.0.0.0).\n"
                        "This allows access from other devices on your network but may pose security risks.\n"
                        "The server will be restarted. Continue?",
                ok="Enable & Restart",
                cancel="Cancel"
            )
        else:
            response = rumps.alert(
                title="Bind to Localhost Only",
                message="This will restrict the server to localhost only (127.0.0.1).\n"
                        "The server will be restarted. Continue?",
                ok="Disable & Restart",
                cancel="Cancel"
            )
        
        if response == 1:  # OK clicked
            try:
                # Update the setting in the database
                db_manager = get_database_manager()
                db_manager.set_setting("bind_to_all_interfaces", new_state)
                
                # Update checkbox state
                sender.state = new_state
                
                # Update internal state
                self.host = "0.0.0.0" if new_state else self.default_host
                # Re-read port setting in case it changed
                self.port = db_manager.get_setting("server_port", self.default_port)
                # Always use localhost for API calls, even when binding to all interfaces
                self.base_url = f"http://127.0.0.1:{self.port}"
                
                logger.info(f"Bind to all interfaces setting changed to: {new_state}")
                
                # Restart server
                self.restart_server()
                
            except Exception as e:
                logger.error(f"Failed to update bind setting: {e}")
                rumps.alert(
                    title="Error",
                    message=f"Failed to update setting: {e}",
                    ok="OK"
                )
                # Revert checkbox state
                sender.state = current_state

    def restart_server(self):
        """Restart the server with new settings."""
        logger.info("Restarting server...")
        
        # Stop the current server
        self.server_running = False
        
        # Wait a bit for the server to stop
        time.sleep(2)
        
        # Start a new server thread
        self.server_thread = threading.Thread(
            target=self.start_server_background,
            name="mlx_server",
            daemon=True
        )
        self.server_thread.start()
        
        # Wait for server to be ready
        ready_thread = threading.Thread(
            target=self.wait_for_server,
            name="server_ready_check",
            daemon=True
        )
        ready_thread.start()
        
        # Show notification
        rumps.notification(
            title="MLX-GUI",
            subtitle="Server Restarted",
            message=f"Server now binding to: {self.host}:{self.port}",
            sound=False
        )

    def quit_app(self, sender):
        """Quit the application and stop the server."""
        response = rumps.alert(
            title="Quit MLX-GUI",
            message="Are you sure you want to quit MLX-GUI?\nThis will stop the server and unload all models.",
            ok="Quit",
            cancel="Cancel"
        )
        
        if response == 1:  # OK clicked
            logger.info("Shutting down MLX-GUI...")
            
            # Stop status timer
            if hasattr(self, 'status_timer'):
                self.status_timer.stop()
            
            # Try graceful shutdown via API
            try:
                requests.post(f"{self.base_url}/v1/system/shutdown", timeout=5)
            except:
                pass
            
            # Release tray lock
            _release_tray_lock()
            
            # Force quit
            rumps.quit_application()
    
    def run(self):
        """Start the tray application."""
        # Start server in background thread
        self.server_thread = threading.Thread(
            target=self.start_server_background,
            name="mlx_server",
            daemon=True
        )
        self.server_thread.start()
        
        # Wait for server to be ready (in another thread to not block tray)
        ready_thread = threading.Thread(
            target=self.wait_for_server,
            name="server_ready_check", 
            daemon=True
        )
        ready_thread.start()
        
        # Run the tray app (this blocks)
        try:
            super().run()
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
        finally:
            self.server_running = False
            # Release tray lock on any exit
            _release_tray_lock()


def run_tray_app(port: int = 8000, host: str = "127.0.0.1"):
    """Run the MLX-GUI tray application."""
    # Check for existing tray instance
    if not _acquire_tray_lock():
        print("Another MLX-GUI tray instance is already running.")
        logger.warning("Another tray instance is already running, exiting.")
        return False
    
    try:
        print(f"Initializing tray app on {host}:{port}")
        app = MLXTrayApp(port=port, host=host)
        print("Tray app initialized, starting...")
        app.run()
        print("Tray app stopped")
    except ImportError as e:
        if "rumps" in str(e):
            logger.error("rumps library not installed. Install with: pip install rumps")
            print("Error: rumps library required for tray mode")
            print("Install with: pip install rumps")
            return False
        else:
            logger.error(f"Import error: {e}")
            print(f"Import error: {e}")
            raise
    except Exception as e:
        logger.error(f"Tray app error: {e}")
        print(f"Tray app error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Always release the lock when exiting
        _release_tray_lock()
    
    return True


if __name__ == "__main__":
    # For testing
    import sys
    logging.basicConfig(level=logging.INFO)
    
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    run_tray_app(port=port)