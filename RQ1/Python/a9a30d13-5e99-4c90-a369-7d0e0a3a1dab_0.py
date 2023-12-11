# my_build_setting.bzl
load("@bazel_skylib//rules:common_settings.bzl", "string_flag")

string_flag(
    name = "my_custom_flag",
    build_setting_default = "default",
)
