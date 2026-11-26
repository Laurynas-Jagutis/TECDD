"""Reading, copying and writing the XES cases of a base log."""
import copy
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
NS = "http://www.xes-standard.org/"
ET.register_namespace("", NS)


@dataclass
class Case:
    case_id: str
    attrs: list              # XES attribute elements of the trace
    events: list             # (timestamp, activity, resource, XES event element), in time order
    row: int = 0             # row in the feature matrix
    variant_id: str = ""
    block_index: int = 0
    original_case_id: str = ""

    @property
    def start(self):
        return self.events[0][0]

    @property
    def end(self):
        return self.events[-1][0]


def tag(node):
    return node.tag.split("}", 1)[-1]


def parse_timestamp(value):
    ts = datetime.fromisoformat(value)
    return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts.astimezone(timezone.utc)


def nanoseconds(ts):
    """Nanoseconds since 1970 as an exact integer."""
    return (ts - EPOCH) // timedelta(microseconds=1) * 1000


def by_start(cases):
    return sorted(cases, key=lambda case: (case.start, case.case_id))


def attribute(xes_type, key, value):
    return ET.Element(f"{{{NS}}}{xes_type}", {"key": key, "value": str(value)})


def load_base_log(path):
    """The cases of the base log (events without timestamp or activity dropped), plus the XES
    root attributes and header to copy."""
    root = ET.parse(path).getroot()
    header = [child for child in root if tag(child) in {"extension", "global", "classifier"}]
    cases = []
    for trace in (child for child in root if tag(child) == "trace"):
        attrs = [child for child in trace if tag(child) != "event"]
        events = []
        for event in (child for child in trace if tag(child) == "event"):
            values = {attr.get("key"): attr.get("value") for attr in event}
            if values.get("time:timestamp") is not None and values.get("concept:name") is not None:
                resource = values.get("org:resource") or "__missing_resource__"
                events.append((parse_timestamp(values["time:timestamp"]), values["concept:name"], resource, event))
        if events:
            events.sort(key=lambda event: event[0])
            case_id = next((a.get("value") for a in attrs if a.get("key") == "concept:name"), f"{path.stem}_{len(cases) + 1}")
            cases.append(Case(case_id, attrs, events, row=len(cases)))
    return dict(root.attrib), header, cases


def copy_case(source, case_id, block_index, shift):
    """Copy a case under a new id, moving all its dates by shift."""
    def shifted(node):
        node = copy.deepcopy(node)
        for attr in node.iter(f"{{{NS}}}date"):
            attr.set("value", (parse_timestamp(attr.get("value")) + shift).isoformat())
        return node

    attrs = [shifted(attr) for attr in source.attrs]
    for attr in attrs:
        if attr.get("key") == "concept:name":
            attr.set("value", case_id)
    events = [(ts + shift, activity, resource, shifted(event)) for ts, activity, resource, event in source.events]
    return Case(case_id, attrs, events, variant_id=source.variant_id, block_index=block_index,
                original_case_id=source.case_id)


def write_xes(path, cases, root_attrib, header, log_name):
    root = ET.Element(f"{{{NS}}}log", root_attrib)
    root.extend(copy.deepcopy(header))
    root.append(attribute("string", "concept:name", log_name))
    for case in by_start(cases):
        trace = ET.SubElement(root, f"{{{NS}}}trace")
        trace.extend(case.attrs)
        trace.append(attribute("string", "synthetic:variant_id", case.variant_id))
        trace.append(attribute("int", "synthetic:block_index", case.block_index))
        trace.append(attribute("string", "synthetic:original_case_id", case.original_case_id))
        trace.extend(event for *_, event in case.events)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)
