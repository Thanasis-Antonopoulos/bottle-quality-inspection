"""Command-line entry point for Bottle Quality Inspection."""

from bottle_quality_inspection import __version__


def main() -> None:
    """Run the initial application entry point."""
    print(f"Bottle Quality Inspection v{__version__}")
    print("Project environment initialized successfully.")


if __name__ == "__main__":
    main()