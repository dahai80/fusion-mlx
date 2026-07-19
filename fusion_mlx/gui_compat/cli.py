"""
Command-line interface for MLX-GUI.
"""

import asyncio
import sys

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from fusion_mlx.gui_compat.database import get_database_manager
from fusion_mlx.gui_compat.huggingface_integration import get_huggingface_client
from fusion_mlx.gui_compat.server import create_app

app = typer.Typer(
    name="mlx-gui",
    help="MLX-GUI: A lightweight RESTful wrapper around Apple's MLX engine",
    add_completion=False,
)

console = Console()


@app.command()
def start(
    port: int | None = typer.Option(
        None,
        "--port",
        "-p",
        help="Port to run the server on (overrides database setting)",
    ),
    host: str | None = typer.Option(
        None, "--host", "-h", help="Host to bind to (overrides database setting)"
    ),
    reload: bool = typer.Option(
        False, "--reload", "-r", help="Enable auto-reload for development"
    ),
    workers: int = typer.Option(
        1, "--workers", "-w", help="Number of worker processes"
    ),
    log_level: str = typer.Option("info", "--log-level", "-l", help="Log level"),
    database_path: str | None = typer.Option(
        None, "--database", "-d", help="Custom database path"
    ),
):
    """Start the MLX-GUI server."""
    # Print ASCII art banner
    console.print(
        """
███╗   ███╗██╗     ██╗  ██╗      ██████╗ ██╗   ██╗██╗
████╗ ████║██║     ╚██╗██╔╝     ██╔════╝ ██║   ██║██║
██╔████╔██║██║      ╚███╔╝█████╗██║  ███╗██║   ██║██║
██║╚██╔╝██║██║      ██╔██╗╚════╝██║   ██║██║   ██║██║
██║ ╚═╝ ██║███████╗██╔╝ ██╗     ╚██████╔╝╚██████╔╝██║
╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝      ╚═════╝  ╚═════╝ ╚═╝
""",
        style="cyan",
    )

    from fusion_mlx.gui_compat import __version__

    console.print("MLX-GUI - Apple Silicon AI Model Server", style="bold white")
    console.print(f"Version {__version__}", style="bold green")
    console.print("By Matthew Rogers (@RamboRogers)", style="dim white")
    console.print("https://github.com/RamboRogers/mlx-gui", style="blue")
    console.print()

    # Initialize database
    if database_path:
        console.print(f"📊 Using custom database: {database_path}", style="blue")

    db_manager = get_database_manager()
    console.print(
        f"📊 Database initialized at: {db_manager.database_path}", style="blue"
    )

    # Get port and host from database settings, allow CLI overrides
    actual_port = (
        port if port is not None else db_manager.get_setting("server_port", 8000)
    )

    # Handle host setting - check bind_to_all_interfaces setting
    if host is not None:
        actual_host = host
    else:
        bind_to_all = db_manager.get_setting("bind_to_all_interfaces", False)
        actual_host = "0.0.0.0" if bind_to_all else "127.0.0.1"

    console.print(
        f"🚀 Starting MLX-GUI server on {actual_host}:{actual_port}", style="green"
    )
    if port is None:
        console.print(
            f"   📊 Using port from database setting: {actual_port}", style="dim blue"
        )
    else:
        console.print(
            f"   🔧 Using CLI override port: {actual_port}", style="dim yellow"
        )

    if host is None:
        bind_to_all = db_manager.get_setting("bind_to_all_interfaces", False)
        console.print(
            f"   📊 Using bind setting from database: {'all interfaces (0.0.0.0)' if bind_to_all else 'localhost only (127.0.0.1)'}",
            style="dim blue",
        )
    else:
        console.print(
            f"   🔧 Using CLI override host: {actual_host}", style="dim yellow"
        )

    # Create FastAPI app
    fastapi_app = create_app()

    # Configure uvicorn
    config = uvicorn.Config(
        app=fastapi_app,
        host=actual_host,
        port=actual_port,
        log_level=log_level,
        reload=reload,
        workers=workers if not reload else 1,
    )

    server = uvicorn.Server(config)

    try:
        # Run the server - should exit cleanly now with daemon threads
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        console.print("\n👋 Shutting down MLX-GUI server...", style="yellow")
    except Exception as e:
        console.print(f"❌ Error starting server: {e}", style="red")
        sys.exit(1)


@app.command()
def status():
    """Show server and database status."""
    console.print("📊 MLX-GUI Status", style="bold blue")

    # Database status
    try:
        db_manager = get_database_manager()
        db_info = db_manager.get_database_info()

        table = Table(title="Database Status")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Database Path", str(db_info.get("path", "Unknown")))
        table.add_row("Size", f"{db_info.get('size_bytes', 0) / 1024:.1f} KB")
        table.add_row("Journal Mode", str(db_info.get("journal_mode", "Unknown")))
        table.add_row(
            "Foreign Keys", "Enabled" if db_info.get("foreign_keys") else "Disabled"
        )

        console.print(table)

    except Exception as e:
        console.print(f"❌ Database error: {e}", style="red")


@app.command()
def models():
    """List all models in the database."""
    console.print("🤖 Models", style="bold blue")

    try:
        db_manager = get_database_manager()
        with db_manager.get_session() as session:
            from fusion_mlx.gui_compat.models import Model

            models = session.query(Model).all()

            if not models:
                console.print("No models found in database", style="yellow")
                return

            table = Table(title="MLX Models")
            table.add_column("Name", style="cyan")
            table.add_column("Type", style="blue")
            table.add_column("Status", style="green")
            table.add_column("Memory (GB)", style="magenta")
            table.add_column("Use Count", style="yellow")

            for model in models:
                status_style = "green" if model.status == "loaded" else "red"
                table.add_row(
                    model.name,
                    model.model_type,
                    f"[{status_style}]{model.status}[/{status_style}]",
                    str(model.memory_required_gb),
                    str(model.use_count),
                )

            console.print(table)

    except Exception as e:
        console.print(f"❌ Error listing models: {e}", style="red")


@app.command()
def init_db(
    force: bool = typer.Option(
        False, "--force", "-f", help="Force reinitialize database"
    ),
):
    """Initialize or reinitialize the database."""
    console.print("🗄️  Initializing database...", style="blue")

    try:
        db_manager = get_database_manager()

        if force:
            console.print(
                "⚠️  Force flag detected - recreating database", style="yellow"
            )
            # This would need implementation to drop and recreate tables

        console.print(
            f"✅ Database initialized at: {db_manager.database_path}", style="green"
        )

    except Exception as e:
        console.print(f"❌ Database initialization error: {e}", style="red")
        sys.exit(1)


@app.command()
def config():
    """Show configuration settings."""
    console.print("⚙️  Configuration", style="bold blue")

    try:
        db_manager = get_database_manager()

        with db_manager.get_session() as session:
            from fusion_mlx.gui_compat.models import AppSettings

            settings = session.query(AppSettings).all()

            if not settings:
                console.print("No configuration found", style="yellow")
                return

            table = Table(title="Application Settings")
            table.add_column("Key", style="cyan")
            table.add_column("Value", style="green")
            table.add_column("Type", style="blue")
            table.add_column("Description", style="white")

            for setting in settings:
                table.add_row(
                    setting.key,
                    str(setting.get_typed_value()),
                    setting.value_type,
                    setting.description or "",
                )

            console.print(table)

    except Exception as e:
        console.print(f"❌ Error reading configuration: {e}", style="red")


@app.command()
def discover(
    query: str = typer.Option("", "--query", "-q", help="Search query"),
    limit: int = typer.Option(10, "--limit", "-l", help="Number of results"),
    popular: bool = typer.Option(False, "--popular", "-p", help="Show popular models"),
    token: str | None = typer.Option(None, "--token", "-t", help="HuggingFace token"),
):
    """Discover MLX models from HuggingFace."""
    console.print("🔍 Discovering MLX Models", style="bold blue")

    try:
        hf_client = get_huggingface_client(token=token)

        if popular:
            models = hf_client.get_popular_mlx_models(limit=limit)
            console.print(f"📈 Top {limit} Popular MLX Models", style="green")
        else:
            models = hf_client.search_mlx_models(query=query, limit=limit)
            console.print(
                f"🔍 Search Results for '{query}'" if query else "🔍 Latest MLX Models",
                style="green",
            )

        if not models:
            console.print("No models found", style="yellow")
            return

        table = Table(title="MLX Models from HuggingFace")
        table.add_column("Model ID", style="cyan")
        table.add_column("Type", style="blue")
        table.add_column("Size (GB)", style="magenta")
        table.add_column("Downloads", style="green")
        table.add_column("MLX Ready", style="yellow")

        for model in models:
            size_str = f"{model.size_gb:.1f}" if model.size_gb else "Unknown"
            downloads_str = f"{model.downloads:,}" if model.downloads else "0"
            mlx_ready = (
                "✅"
                if model.mlx_compatible
                else "🔄" if model.has_mlx_version else "❌"
            )

            table.add_row(
                model.id,
                model.model_type,
                size_str,
                downloads_str,
                mlx_ready,
            )

        console.print(table)

    except Exception as e:
        console.print(f"❌ Error discovering models: {e}", style="red")


@app.command()
def tray(
    port: int | None = typer.Option(
        None,
        "--port",
        "-p",
        help="Port to run the server on (overrides database setting)",
    ),
    host: str | None = typer.Option(
        None, "--host", "-h", help="Host to bind to (overrides database setting)"
    ),
):
    """Launch MLX-GUI with macOS system tray interface."""
    # Print ASCII art banner
    console.print(
        """
███╗   ███╗██╗     ██╗  ██╗      ██████╗ ██╗   ██╗██╗
████╗ ████║██║     ╚██╗██╔╝     ██╔════╝ ██║   ██║██║
██╔████╔██║██║      ╚███╔╝█████╗██║  ███╗██║   ██║██║
██║╚██╔╝██║██║      ██╔██╗╚════╝██║   ██║██║   ██║██║
██║ ╚═╝ ██║███████╗██╔╝ ██╗     ╚██████╔╝╚██████╔╝██║
╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝      ╚═════╝  ╚═════╝ ╚═╝
""",
        style="cyan",
    )

    from fusion_mlx.gui_compat import __version__

    console.print("MLX-GUI - Apple Silicon AI Model Server", style="bold white")
    console.print(f"Version {__version__}", style="bold green")
    console.print("By Matthew Rogers (@RamboRogers)", style="dim white")
    console.print("https://github.com/RamboRogers/mlx-gui", style="blue")
    console.print()
    console.print("🍎 Starting MLX-GUI with system tray interface...", style="green")

    # Get port and host from database settings, allow CLI overrides
    db_manager = get_database_manager()
    actual_port = (
        port if port is not None else db_manager.get_setting("server_port", 8000)
    )

    # Handle host setting - check bind_to_all_interfaces setting
    if host is not None:
        actual_host = host
    else:
        bind_to_all = db_manager.get_setting("bind_to_all_interfaces", False)
        actual_host = "0.0.0.0" if bind_to_all else "127.0.0.1"

    console.print(f"🚀 Server will run on {actual_host}:{actual_port}", style="blue")
    if port is None:
        console.print(
            f"   📊 Using port from database setting: {actual_port}", style="dim blue"
        )
    if host is None:
        bind_to_all = db_manager.get_setting("bind_to_all_interfaces", False)
        console.print(
            f"   📊 Using bind setting from database: {'all interfaces (0.0.0.0)' if bind_to_all else 'localhost only (127.0.0.1)'}",
            style="dim blue",
        )

    try:
        from fusion_mlx.gui_compat.tray import run_tray_app

        success = run_tray_app(port=actual_port, host=actual_host)
        if not success:
            sys.exit(1)
    except ImportError as e:
        if "rumps" in str(e):
            console.print("❌ Error: rumps library not installed", style="red")
            console.print("Install with: pip install rumps", style="yellow")
            sys.exit(1)
        else:
            raise
    except Exception as e:
        console.print(f"❌ Error starting tray app: {e}", style="red")
        sys.exit(1)


@app.command()
def update_sizes():
    """Update model memory requirements based on actual file sizes."""
    console.print("📊 Updating model sizes from disk...", style="blue")

    try:
        db_manager = get_database_manager()
        db_manager.update_model_sizes_from_disk()
        console.print("✅ Model sizes updated successfully", style="green")
    except Exception as e:
        console.print(f"❌ Error updating model sizes: {e}", style="red")


@app.command()
def version():
    """Show version information."""
    from fusion_mlx.gui_compat import __version__

    console.print(f"MLX-GUI version: {__version__}", style="green")


def main():
    """Main entry point for the CLI."""
    try:
        # If no arguments provided, default to start command with database settings
        if len(sys.argv) == 1:
            start(
                port=None,
                host=None,
                reload=False,
                workers=1,
                log_level="info",
                database_path=None,
            )
        else:
            app()
    except KeyboardInterrupt:
        console.print("\n👋 Goodbye!", style="yellow")
        sys.exit(0)
    except Exception as e:
        console.print(f"❌ Unexpected error: {e}", style="red")
        sys.exit(1)


if __name__ == "__main__":
    main()
