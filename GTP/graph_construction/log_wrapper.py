import sys
import logging
def create_logger(name, log_file, silent=False, to_disk=True, mode='w'):
    """Logger wrapper
    """
    # setup logger
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    formatter = logging.Formatter(fmt='%(asctime)s: %(levelname)s: [%(filename)s: %(lineno)d]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    if not silent:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        log.addHandler(ch)
    if to_disk:

        fh = logging.FileHandler(log_file, mode=mode)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        log.addHandler(fh)
    return log