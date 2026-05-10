try:
    from importlib.metadata import version as _version
    __version__ = _version("lmbase")
except Exception:
    __version__ = "0.0.0"
