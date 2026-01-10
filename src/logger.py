import logging
class LineNoFormatter(logging.Formatter):
    """Кастомный форматтер для добавления номера строки в лог."""
    def format(self, record):
        # Получаем номер строки из record, если он есть
        lineno = getattr(record, 'lineno', 'N/A')
        record.lineno_custom = lineno
        return super().format(record)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = LineNoFormatter(fmt='%(asctime)s - %(name)s - %(lineno_custom)d - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False # Предотвращаем повтор логов, если есть родительские логгеры