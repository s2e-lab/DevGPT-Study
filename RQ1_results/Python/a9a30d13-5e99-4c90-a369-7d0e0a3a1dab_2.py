genrule(
    name = "build_docker_image",
    srcs = [...],
    outs = [...],
    cmd = select({
        "//constraints:mac_m1": "$(location :docker_build_script_mac_m1.sh) $(SRCS)",
        "//constraints:x64": "$(location :docker_build_script_x64.sh) $(SRCS)",
    }),
    tools = select({
        "//constraints:mac_m1": [":docker_build_script_mac_m1.sh"],
        "//constraints:x64": [":docker_build_script_x64.sh"],
    }),
)
