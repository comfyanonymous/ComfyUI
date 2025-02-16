import logging
import os
import sqlite3
from contextlib import contextmanager
from queue import Queue, Empty, Full
import threading
from app.database.updater import DatabaseUpdater
import folder_paths
from comfy.cli_args import args


class Database:
    def __init__(self, database_path=None, pool_size=1):
        if database_path is None:
            self.exists = False
            database_path = "file::memory:?cache=shared"
        else:
            self.exists = os.path.exists(database_path)

        self.database_path = database_path
        self.pool_size = pool_size
        # Store connections in a pool, default to 1 as normal usage is going to be from a single thread at a time
        self.connection_pool: Queue = Queue(maxsize=pool_size)
        self._db_lock = threading.Lock()
        self._initialized = False
        self._closing = False
        self._after_update_callbacks = []

    def _setup(self):
        if self._initialized:
            return

        with self._db_lock:
            if not self._initialized:
                self._make_db()
                self._initialized = True

    def _create_connection(self):
        # TODO: Catch error for sqlite lib missing on linux
        logging.info(f"Creating connection to {self.database_path}")
        conn = sqlite3.connect(
            self.database_path,
            check_same_thread=False,
            uri=self.database_path.startswith("file::"),
        )
        conn.execute("PRAGMA foreign_keys = ON")
        self.exists = True
        logging.info(f"Connected!")
        return conn

    def _make_db(self):
        with self._get_connection() as con:
            updater = DatabaseUpdater(con, self.database_path)
            result = updater.update()
            if result is not None:
                old_version, new_version = result

                for callback in self._after_update_callbacks:
                    callback(old_version, new_version)

    def _transform(self, row, columns):
        return {col.name: value for value, col in zip(row, columns)}

    @contextmanager
    def _get_connection(self):
        if self._closing:
            raise Exception("Database is shutting down")

        try:
            # Try to get connection from pool
            connection = self.connection_pool.get_nowait()
        except Empty:
            # Create new connection if pool is empty
            connection = self._create_connection()

        try:
            yield connection
        finally:
            try:
                # Try to add to pool if it's empty
                self.connection_pool.put_nowait(connection)
            except Full:
                # Pool is full, close the connection
                connection.close()

    @contextmanager
    def get_connection(self):
        # Setup the database if it's not already initialized
        self._setup()
        with self._get_connection() as connection:
            yield connection

    def execute(self, sql, *args):
        with self.get_connection() as connection:
            cursor = connection.execute(sql, args)
            results = cursor.fetchall()
            return results

    def register_after_update_callback(self, callback):
        self._after_update_callbacks.append(callback)

    def close(self):
        if self._closing:
            return
        # Drain and close all connections in the pool
        self._closing = True
        while True:
            try:
                conn = self.connection_pool.get_nowait()
                conn.close()
            except Empty:
                break
        self._closing = False

    def __del__(self):
        try:
            self.close()
        except:
            pass


# Create a global instance
db_path = None
if not args.memory_database:
    db_path = folder_paths.get_user_directory() + "/comfyui.db"
db = Database(db_path)
