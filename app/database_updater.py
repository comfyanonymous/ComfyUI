import logging
import os


class DatabaseUpdater:
    def __init__(self, connection):
        self.connection = connection
        self.current_version = self.get_db_version()
        self.version_updates = {
            1: self._update_to_v1,
        }
        self.max_version = max(self.version_updates.keys())
        self.update_required = self.current_version < self.max_version
        logging.info(f"Database version: {self.current_version}")

    def get_db_version(self):
        return self.connection.execute("PRAGMA user_version").fetchone()[0]

    def update(self):
        if not self.update_required:
            return

        logging.info(f"Updating database: {self.current_version} -> {self.max_version}")

        dirname = os.path.dirname(__file__)
        cursor = self.connection.cursor()
        for version in range(self.current_version + 1, self.max_version + 1):
            filename = os.path.join(dirname, f"db/v{version}.sql")
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

    def _update_to_v1(self, cursor):
        # TODO: migrate users and settings
        print("Updating to v1")
