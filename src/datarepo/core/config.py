DEFAULT_AWS_PROFILE: str | None = None


def set_default_aws_profile(profile_name: str) -> None:
    global DEFAULT_AWS_PROFILE
    DEFAULT_AWS_PROFILE = profile_name
