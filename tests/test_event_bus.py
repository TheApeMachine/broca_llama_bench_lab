from __future__ import annotations

import logging

from core.event_bus import EventBus, LogToBusHandler, get_default_bus, reset_default_bus


def test_subscribe_and_publish_round_trip():
    bus = EventBus()
    sub = bus.subscribe("frame.*")  # wildcard semantics are simple set membership; "*" is the only wildcard
    bus.publish("frame.comprehend", {"intent": "x"})
    out = bus.drain(sub)
    # "frame.*" is a literal topic, not a glob — verify exact-match semantics.
    assert out == []

    sub2 = bus.subscribe("frame.comprehend")
    bus.publish("frame.comprehend", {"intent": "x"})
    out = bus.drain(sub2)
    assert len(out) == 1
    assert out[0].topic == "frame.comprehend"
    assert out[0].payload == {"intent": "x"}


def test_wildcard_subscriber_gets_everything():
    bus = EventBus()
    sub = bus.subscribe("*")
    bus.publish("a", 1)
    bus.publish("b", 2)
    out = bus.drain(sub)
    assert [(e.topic, e.payload) for e in out] == [("a", 1), ("b", 2)]


def test_drain_clears_queue():
    bus = EventBus()
    sub = bus.subscribe("*")
    bus.publish("x", 1)
    assert len(bus.drain(sub)) == 1
    assert bus.drain(sub) == []


def test_bounded_queue_drops_oldest_when_full():
    bus = EventBus()
    sub = bus.subscribe("*", queue_size=16)
    for i in range(100):
        bus.publish("t", i)
    out = bus.drain(sub)
    # Bounded at 16; the oldest must have been dropped.
    assert len(out) == 16
    assert out[0].payload == 84
    assert out[-1].payload == 99


def test_unsubscribe_stops_delivery():
    bus = EventBus()
    sub = bus.subscribe("*")
    bus.unsubscribe(sub)
    bus.publish("x", 1)
    assert bus.drain(sub) == []


def test_log_handler_forwards_records_as_events():
    bus = EventBus()
    sub = bus.subscribe("*")
    handler = LogToBusHandler(bus, level=logging.INFO)
    log = logging.getLogger("core.event_bus.test")
    log.setLevel(logging.DEBUG)
    log.addHandler(handler)
    try:
        log.info("hello %s", "world")
        log.warning("careful")
        log.debug("filtered out")  # below handler level
    finally:
        log.removeHandler(handler)

    events = bus.drain(sub)
    assert {e.topic for e in events} == {"log.info", "log.warning"}
    msgs = {e.payload["msg"] for e in events}
    assert "hello world" in msgs
    assert "careful" in msgs


def test_default_bus_is_singleton():
    reset_default_bus()
    a = get_default_bus()
    b = get_default_bus()
    assert a is b
    reset_default_bus()
    c = get_default_bus()
    assert c is not a
