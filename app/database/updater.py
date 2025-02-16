import logging
import os
import sqlite3
from app.database.versions.v1 import v1


class DatabaseUpdater:
    def __init__(self, connection, database_path):
        self.connection = connection
        self.database_path = database_path
        self.current_version = self.get_db_version()
        self.version_updates = {
            1: v1,
        }
        self.max_version = max(self.version_updates.keys())
        self.update_required = self.current_version < self.max_version
        logging.info(f"Database version: {self.current_version}")

    def get_db_version(self):
        return self.connection.execute("PRAGMA user_version").fetchone()[0]

    def backup(self):
        bkp_path = self.database_path + ".bkp"
        if os.path.exists(bkp_path):
            # TODO: auto-rollback failed upgrades
            raise Exception(
                f"Database backup already exists, this indicates that a previous upgrade failed. Please restore this backup before continuing. Backup location: {bkp_path}"
            )

        bkp = sqlite3.connect(bkp_path)
        self.connection.backup(bkp)
        bkp.close()
        logging.info("Database backup taken pre-upgrade.")
        return bkp_path

    def update(self):
        if not self.update_required:
            return None

        bkp_version = self.current_version
        bkp_path = None
        if self.current_version > 0:
            bkp_path = self.backup()

        logging.info(f"Updating database: {self.current_version} -> {self.max_version}")

        dirname = os.path.dirname(__file__)
        cursor = self.connection.cursor()
        for version in range(self.current_version + 1, self.max_version + 1):
            filename = os.path.join(dirname, f"versions/v{version}.sql")
            if not os.path.exists(filename):
                raise Exception(
                    f"Database update script for version {version} not found"
                )

            try:
                with open(filename, "r") as file:
                    sql = file.read()
                    cursor.executescript(sql)
            except Exception as e:
                raise Exception(
                    f"Failed to execute update script for version {version}: {e}"
                )

            method = self.version_updates[version]
            if method is not None:
                method(cursor)

        cursor.execute("PRAGMA user_version = %d" % self.max_version)
        self.connection.commit()
        cursor.close()
        self.current_version = self.get_db_version()

        if bkp_path:
            # Keep a copy of the backup in case something goes wrong and we need to rollback
            os.rename(bkp_path, self.database_path + f".v{bkp_version}.bkp")
        logging.info(f"Upgrade to successful.")

        return (bkp_version, self.current_version)
