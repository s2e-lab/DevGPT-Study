import re

ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

def redirect_werkzeug_logs_to_loguru(record):
    log_level = logging.getLevelName(record.levelno)
    log_message = ansi_escape.sub('', record.getMessage())
    logger_opt = logger.opt(depth=6, exception=record.exc_info)
    logger_opt.log(log_level, log_message)
