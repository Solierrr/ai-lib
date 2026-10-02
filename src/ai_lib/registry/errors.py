class RegistryError(Exception):
    pass


class RegistryAuthError(RegistryError):
    pass


class RegistryKeysNotConfigured(RegistryError):
    pass


class RegistryKeysUnavailable(RegistryError):
    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class RegistryUnreachable(RegistryError):
    pass
