def close_connection(self):
    """Close the database connection."""
    logging.debug(f"CLASS ChatDatabase - close_connection: Preparing to close connection to database: {self.db_name}")
    
    if self.conn:
        try:
            self.conn.close()
            logging.debug(f"CLASS ChatDatabase - close_connection: Connection closed successfully for database: {self.db_name}")
        except sqlite3.OperationalError as oe:
            logging.error(f"CLASS ChatDatabase - close_connection: Operational Error while closing: {oe}")
        except sqlite3.Error as e:
            logging.error(f"CLASS ChatDatabase - close_connection: sqlite3 Error while closing: {e}")
        except Exception as e:
            logging.exception(f"CLASS ChatDatabase - close_connection: Unexpected Error while closing: {e}")
        finally:
            self.conn = None
            logging.debug(f"CLASS ChatDatabase - close_connection: Connection variable reset to None.")
    else:
        logging.warning(f"CLASS ChatDatabase - close_connection: Connection is already closed or was never initialized for database: {self.db_name}")
