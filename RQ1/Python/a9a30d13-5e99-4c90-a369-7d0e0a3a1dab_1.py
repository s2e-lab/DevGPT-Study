# BUILD.bazel
load("//:my_build_setting.bzl", "my_custom_flag")

config_setting(
    name = "custom_flag_is_foo",
    flag_values = {":my_custom_flag": "foo"},
)

cc_binary(
    name = "my_binary",
    srcs = ["my_binary.cc"],
    deps = select({
        ":custom_flag_is_foo": [":foo_dependency"],
        "//conditions:default": [":default_dependency"],
    }),
)
