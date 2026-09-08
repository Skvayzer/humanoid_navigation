import re

from rosidl_adapter.parser import parse_message_string
from rosidl_runtime_py import get_interface_path


# Foxy message files may contain field defaults separated by whitespace (for
# example ``float64 x 0``).  Foxglove's rosbridge schema parser does not
# support that syntax, while rewriting it as ``float64 x=0`` incorrectly
# turns the field into a constant.  Remove defaults only from the schema text
# sent to Foxglove; retain the original text for ROS parsing and preserve real
# constants whose value begins with ``=``.
_FOXY_FIELD_DEFAULT = re.compile(
    r"^(\s*(?:bool|byte|char|float32|float64|u?int(?:8|16|32|64)|"
    r"string(?:<=\d+)?)(?:\[[^]]*\])?\s+"
    r"[A-Za-z_][A-Za-z0-9_]*)\s+([^#\s][^#]*?)(\s*#.*)?$"
)


def _normalize_foxy_field_defaults(definition):
    lines = []
    for line in definition.splitlines(keepends=True):
        newline = "\n" if line.endswith("\n") else ""
        content = line[:-1] if newline else line
        match = _FOXY_FIELD_DEFAULT.match(content)
        if match and not match.group(2).lstrip().startswith("="):
            content = f"{match.group(1)}{match.group(3) or ''}"
        lines.append(content + newline)
    return "".join(lines)


def stringify_field_types(root_type):
    definition = ""
    seen_types = set()
    deps = [root_type]
    is_root = True
    while deps:
        ty = deps.pop()
        parts = ty.split("/")
        if not is_root:
            definition += "\n================================================================================\n"
            definition += f"MSG: {ty}\n"
        is_root = False

        msg_name = parts[2] if len(parts) == 3 else parts[1]
        interface_name = ty if len(parts) == 3 else f"{parts[0]}/msg/{parts[1]}"
        with open(get_interface_path(interface_name), encoding="utf-8") as msg_file:
            msg_definition = msg_file.read()
        definition += _normalize_foxy_field_defaults(msg_definition)

        spec = parse_message_string(parts[0], msg_name, msg_definition)
        for field in spec.fields:
            is_builtin = field.type.pkg_name is None
            if not is_builtin:
                field_ty = f"{field.type.pkg_name}/{field.type.type}"
                if field_ty not in seen_types:
                    deps.append(field_ty)
                    seen_types.add(field_ty)

    return definition
