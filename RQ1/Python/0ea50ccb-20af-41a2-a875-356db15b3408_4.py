if not self.conn:
    logging.critical(f"Database connection is not open when trying to execute {operation_name}")
