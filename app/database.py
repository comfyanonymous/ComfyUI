import logging
import sqlite3
from contextlib import contextmanager
from queue import Queue, Empty, Full
import threading
from app.database_updater import DatabaseUpdater
import folder_paths
from comfy.cli_args import args


class Database:
    def __init__(self, database_path=None, pool_size=1):
        if database_path is None:
            database_path = "file::memory:?cache=shared"

        self.database_path = database_path
        self.pool_size = pool_size
        # Store connections in a pool, default to 1 as normal usage is going to be from a single thread at a time
        self.connection_pool: Queue = Queue(maxsize=pool_size)
        self._db_lock = threading.Lock()
        self._initialized = False
        self._closing = False

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
        conn = sqlite3.connect(self.database_path, check_same_thread=False)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _make_db(self):
        with self._get_connection() as con:
            updater = DatabaseUpdater(con)
            updater.update()

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
