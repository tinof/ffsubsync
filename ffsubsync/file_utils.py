import sys


class open_file:
    """
    Context manager that opens a filename and closes it on exit, but does
    nothing for file-like objects.
    """

    def __init__(self, filename, *args, **kwargs) -> None:
        self.closing = kwargs.pop("closing", False)
        if filename is None:
            stream = sys.stdout if "w" in args else sys.stdin
            self.fh = open(stream.fileno(), *args, **kwargs)  # noqa: SIM115
        elif isinstance(filename, str):
            self.fh = open(filename, *args, **kwargs)  # noqa: SIM115
            self.closing = True
        else:
            self.fh = filename

    def __enter__(self):
        return self.fh

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.closing:
            self.fh.close()

        return False
