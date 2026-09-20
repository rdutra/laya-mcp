from laya_mcp.logging import configure_logging
from laya_mcp.server import mcp


def main() -> None:
    configure_logging()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

