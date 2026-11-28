"""One-off backfill script to provision per-user qp_tenant_{id} PostgreSQL roles."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import settings  # noqa: E402
from backend.app.core.database import (  # noqa: E402
    configure_database,
    get_readonly_role,
    is_postgres,
    sync_readonly_role_privileges,
)


def main(argv: list[str] | None = None) -> int:
    """Create qp_tenant_{id} roles and scope SELECT grants for existing users and u_* tables."""
    parser = argparse.ArgumentParser(
        description="Backfill PostgreSQL per-tenant NOLOGIN roles (qp_tenant_{id}) for existing users.",
    )
    parser.add_argument("--database-url", default=None, help="Owner/write PostgreSQL DATABASE_URL.")
    parser.add_argument(
        "--read-database-url",
        default=None,
        help="Dedicated read-only PostgreSQL READ_DATABASE_URL (e.g. querypilot_reader).",
    )
    args = parser.parse_args(argv)

    db_url = args.database_url or settings.database_url
    read_url = args.read_database_url if args.read_database_url is not None else settings.read_database_url
    if db_url.startswith("sqlite"):
        print("SQLite configured; per-user PostgreSQL roles are not applicable.")
        return 0

    configure_database(db_url, read_database_url=read_url)
    if not is_postgres():
        print("Target database is not PostgreSQL; skipping role backfill.")
        return 0

    reader_role = get_readonly_role(read_database_url=read_url, write_database_url=db_url)
    synced_user_ids = sync_readonly_role_privileges()
    print(
        f"Synced tenant roles for user_ids={synced_user_ids} "
        f"(reader_role={reader_role or 'none'})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
