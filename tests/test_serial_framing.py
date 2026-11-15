#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checks the bytes that travel between the PC and the board, and the fingerprint
that says which model the board runs.

  - The packet the PIL bench sends (pil_benchmark.escape_payload and
    build_packet): '#' payload '@', where a payload byte equal to '#', '@' or
    ESC goes as ESC followed by the byte XOR 0x20. The packets are decoded
    here by a few lines written from the rules of the firmware (a start
    marker opens or restarts a packet, ESC un-escapes the next byte, the end
    marker closes it), and the sketch is read to make sure it still states
    those rules.
  - The frame the HIL firmware sends back (hil_benchmark.read_frame), read
    from a stand-in for the serial port that hands the bytes over in chosen
    pieces: whole, byte by byte, with an escape pair cut in two, incomplete,
    never started, never finished.
  - The 32-bit FNV-1a fingerprint, against the published test vectors of the
    algorithm.

Each bench is skipped when one of its packages is missing (pyserial, numpy,
Pillow for PIL; pyserial, numpy, OpenCV for HIL). No board, no port.

    python -m unittest discover -s tests -v
"""
import contextlib
import io
import os
import random
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("MPLBACKEND", "Agg")   # the benches draw the matrix; no window

RESERVED = (0x23, 0x40, 0x1B)                # '#', '@', ESC


def load(name):
    """Imports a module of src/, or returns None when one of its packages is
    missing (the benches end with SystemExit in that case)."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return __import__(name)
    except (ImportError, SystemExit):
        return None


pil_benchmark = load("pil_benchmark")
hil_benchmark = load("hil_benchmark")
boards = load("boards")


def firmware_decode(stream):
    """The packets a board reads from a byte stream, by the rules of readImageFromSerial()
    in the PIL sketch: bytes outside a packet are ignored, '#' opens a packet (or starts it
    again when one is open), '@' closes it, ESC makes the next byte the data byte XOR 0x20."""
    packets, current, escaped = [], None, False
    for byte in stream:
        if current is None:
            if byte == 0x23:
                current = bytearray()
        elif escaped:
            current.append(byte ^ 0x20)
            escaped = False
        elif byte == 0x40:
            packets.append(bytes(current))
            current = None
        elif byte == 0x23:
            current = bytearray()
        elif byte == 0x1B:
            escaped = True
        else:
            current.append(byte)
    return packets


@unittest.skipIf(pil_benchmark is None, "pil_benchmark needs pyserial, Pillow and numpy")
class PilPacketTest(unittest.TestCase):
    def test_markers_of_the_bench(self):
        self.assertEqual((pil_benchmark.MARKER_START, pil_benchmark.MARKER_END), (b"#", b"@"))
        self.assertEqual(pil_benchmark.ESCAPE_BYTE, 0x1B)
        self.assertEqual(set(pil_benchmark.BYTES_TO_ESCAPE), set(RESERVED))

    def test_only_the_three_reserved_bytes_are_escaped(self):
        # '#' 0x23 ^ 0x20 = 0x03, '@' 0x40 ^ 0x20 = 0x60, ESC 0x1B ^ 0x20 = 0x3B
        expected = {0x23: b"\x1b\x03", 0x40: b"\x1b\x60", 0x1B: b"\x1b\x3b"}
        for value in range(256):
            self.assertEqual(pil_benchmark.escape_payload(bytes([value])), expected.get(value, bytes([value])), value)

    def test_an_escaped_byte_is_never_a_marker_itself(self):
        for value in RESERVED:
            self.assertNotIn(value ^ 0x20, RESERVED)

    def test_packet_is_the_escaped_payload_between_the_markers(self):
        payload = bytes([0x00, 0x23, 0x7F, 0x40, 0xFF, 0x1B, 0x23])
        packet = pil_benchmark.build_packet(payload)
        self.assertEqual(packet, b"#" + b"\x00\x1b\x03\x7f\x1b\x60\xff\x1b\x3b\x1b\x03" + b"@")
        self.assertEqual(len(packet), len(payload) + 2 + 4)      # two markers, one ESC per reserved byte
        self.assertEqual(pil_benchmark.build_packet(b""), b"#@")

    def test_no_marker_inside_a_packet_whatever_the_image(self):
        # A raw '#' inside a packet tells the firmware that the image is being sent again,
        # and a raw '@' that it is over: neither may come from the image itself.
        rng = random.Random(1)
        payloads = [bytes(range(256)), bytes(RESERVED) * 50,
                    bytes(rng.randrange(256) for _ in range(5000))]
        for payload in payloads:
            body = pil_benchmark.build_packet(payload)[1:-1]
            self.assertNotIn(b"#", body)
            self.assertNotIn(b"@", body)
            self.assertEqual(len(body), len(payload) + sum(payload.count(bytes([v])) for v in RESERVED))

    def test_the_firmware_rules_give_the_image_back(self):
        rng = random.Random(2)
        payloads = [b"", bytes(range(256)), bytes(RESERVED) * 100, b"\x1b" * 7,
                    bytes(rng.randrange(256) for _ in range(160 * 120)),        # one 160x120 grayscale image
                    bytes(rng.choice(RESERVED + (0x03, 0x60, 0x3B, 0x20)) for _ in range(3000))]
        for payload in payloads:
            self.assertEqual(firmware_decode(pil_benchmark.build_packet(payload)), [payload])
        # several images one after another, with the text the board prints in between ignored
        stream = b"".join(pil_benchmark.build_packet(p) + b"\r\n" for p in payloads)
        self.assertEqual(firmware_decode(stream), payloads)

    def test_an_image_sent_again_after_a_cut_packet_arrives_whole(self):
        # What the bench does after a dropped link: the same packet from its start marker.
        payload = bytes(range(256)) * 4
        packet = pil_benchmark.build_packet(payload)
        for cut in (1, 2, 100, 500, len(packet) - 1):
            if packet[cut - 1] == 0x1B:
                cut += 1          # a cut between ESC and its byte is ended by the firmware's own timeout
            self.assertEqual(firmware_decode(packet[:cut] + packet), [payload], cut)

    def test_the_sketch_states_the_rules_used_here(self):
        text = (REPO / "deployment" / "pil_firmware" / "pil_firmware.ino").read_text(encoding="utf-8")
        reader = text[text.index("bool readImageFromSerial()"):text.index("void loadImageToInputTensor()")]
        for rule in (r"byte_in == '#'", r"byte_in == '@'", r"byte_in == 0x1B", r"Serial\.read\(\) \^ 0x20"):
            self.assertRegex(reader, rule)
        # the restart: a start marker inside a packet empties what was received
        restart = reader[reader.index("} else if (byte_in == '#')"):reader.index("} else if (byte_in == 0x1B)")]
        self.assertIn("image_bytes_received = 0;", restart)


class FakeClock:
    """Stands in for the time module inside the bench: waiting costs nothing and moves the clock."""

    def __init__(self):
        self.now = 0.0

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(0.0, seconds)


class PiecewisePort:
    """Stand-in for serial.Serial holding what the board sent. read() hands it over in the
    piece sizes given, then all that is left; an empty port blocks for its timeout."""

    def __init__(self, data, clock, pieces=()):
        self.data = bytearray(data)
        self.clock = clock
        self.pieces = list(pieces)

    @property
    def in_waiting(self):
        return len(self.data)

    def readline(self):
        end = self.data.find(b"\n")
        size = len(self.data) if end < 0 else end + 1
        return self._take(size)

    def read(self, size=1):
        if not self.data:
            self.clock.sleep(1.0)
            return b""
        if self.pieces:
            size = min(size, self.pieces.pop(0))
        return self._take(size)

    def _take(self, size):
        out = bytes(self.data[:size])
        del self.data[:size]
        return out


def frame_packet(frame_bytes):
    out = bytearray(b"#")
    for value in frame_bytes:
        if value in RESERVED:
            out += bytes([0x1B, value ^ 0x20])
        else:
            out.append(value)
    return bytes(out + b"@")


@unittest.skipIf(hil_benchmark is None, "hil_benchmark needs pyserial, OpenCV and numpy")
class HilFrameTest(unittest.TestCase):
    """read_frame(): FRAME_BEGIN:<n>, the escaped packet, FRAME_END."""

    def setUp(self):
        np = hil_benchmark.np
        self.frame = (np.arange(160 * 120, dtype=np.uint32) * 7 % 256).astype(np.uint8).reshape(120, 160)
        self.frame[0, :4] = (0x1B, 0x23, 0x40, 0x1B)     # the frame starts with the three reserved bytes
        self.frame[-1, -1] = 0x40                          # and ends with one
        self.raw = self.frame.tobytes()
        self.clock = FakeClock()

    def sent(self, raw=None, announced=None, packet=None):
        raw = self.raw if raw is None else raw
        packet = frame_packet(raw) if packet is None else packet
        return b"FRAME_BEGIN:%d\r\n" % (len(raw) if announced is None else announced) + packet + b"FRAME_END\r\n"

    def read(self, data, pieces=()):
        port = PiecewisePort(data, self.clock, pieces)
        with mock.patch.object(hil_benchmark, "time", self.clock), \
                contextlib.redirect_stdout(io.StringIO()) as log:
            frame = hil_benchmark.read_frame(port, timeout_s=5.0)
        return frame, port, log.getvalue()

    def assert_frame(self, frame):
        self.assertIsNotNone(frame)
        self.assertEqual((frame.shape, str(frame.dtype)), ((120, 160), "uint8"))
        self.assertEqual(frame.tobytes(), self.raw)

    def test_whole_frame_in_one_piece(self):
        frame, port, _ = self.read(self.sent())
        self.assert_frame(frame)
        self.assertEqual(bytes(port.data), b"")           # FRAME_END was taken with it

    def test_frame_that_arrives_byte_by_byte(self):
        data = self.sent()
        frame, port, _ = self.read(data, pieces=[1] * len(data))
        self.assert_frame(frame)
        self.assertEqual(bytes(port.data), b"")           # FRAME_END came later and was consumed too

    def test_escape_pair_cut_between_two_reads(self):
        # the packet starts with '#', ESC, 0x3B (the first pixel is ESC): the first read ends on the ESC
        packet = frame_packet(self.raw)
        self.assertEqual(packet[:3], b"#\x1b\x3b")
        frame, _, _ = self.read(self.sent(), pieces=[2])
        self.assert_frame(frame)
        # and with every read ending right after an ESC
        escapes = [i + 1 for i, value in enumerate(packet) if value == 0x1B]
        pieces = [b - a for a, b in zip([0] + escapes, escapes)]
        frame, _, _ = self.read(self.sent(), pieces=pieces)
        self.assert_frame(frame)

    def test_bytes_before_the_start_marker_are_ignored(self):
        data = b"FRAME_BEGIN:%d\r\n" % len(self.raw) + b"\r\n \x00\x1b" + frame_packet(self.raw) + b"FRAME_END\r\n"
        frame, _, _ = self.read(data)
        self.assert_frame(frame)

    def test_frame_with_fewer_bytes_than_announced_is_refused(self):
        frame, _, log = self.read(self.sent(raw=self.raw[:-10], announced=len(self.raw)))
        self.assertIsNone(frame)
        self.assertIn("Incomplete frame: 19190/19200", log)

    def test_no_frame_at_all(self):
        frame, _, log = self.read(b"TEMP_C:40\r\n1\r\n")
        self.assertIsNone(frame)
        self.assertIn("FRAME_BEGIN not received", log)
        self.assertGreaterEqual(self.clock.now, 5.0)       # it waited for the whole timeout, on the stand-in clock

    def test_frame_that_never_ends(self):
        frame, _, log = self.read(self.sent(packet=frame_packet(self.raw)[:-1]).replace(b"FRAME_END\r\n", b""))
        self.assertIsNone(frame)
        self.assertIn("Timed out while reading captured frame", log)


FNV_MODULES = [m for m in (hil_benchmark, boards) if m is not None and hasattr(m, "fnv1a32_bytes")]


@unittest.skipUnless(FNV_MODULES, "no module with the FNV-1a fingerprint could be imported")
class Fnv1aTest(unittest.TestCase):
    """The fingerprint the firmware prints as MODEL_FNV1A and the bench computes over the .tflite file."""

    # 32-bit FNV-1a test vectors published with the algorithm (offset basis 0x811C9DC5, prime 0x01000193)
    VECTORS = {b"": 0x811C9DC5, b"a": 0xE40C292C, b"b": 0xE70C2DE5, b"foo": 0xA9F37ED7, b"foobar": 0xBF9CF968}

    def test_published_vectors(self):
        for module in FNV_MODULES:
            for data, expected in self.VECTORS.items():
                self.assertEqual(module.fnv1a32_bytes(data), expected, "%s %r" % (module.__name__, data))

    def test_it_is_fnv_1a_and_not_fnv_1(self):
        # FNV-1 multiplies before the XOR and gives 0x050C5D7E for "a"
        for module in FNV_MODULES:
            self.assertNotEqual(module.fnv1a32_bytes(b"a"), 0x050C5D7E)

    def test_every_byte_and_its_position_count(self):
        data = bytes(range(256)) * 3
        for module in FNV_MODULES:
            reference = module.fnv1a32_bytes(data)
            self.assertEqual(reference, module.fnv1a32_bytes(bytes(data)))
            swapped = bytearray(data)
            swapped[10], swapped[11] = swapped[11], swapped[10]
            self.assertNotEqual(module.fnv1a32_bytes(bytes(swapped)), reference)
            self.assertNotEqual(module.fnv1a32_bytes(data + b"\x00"), reference)
            self.assertTrue(0 <= reference <= 0xFFFFFFFF)

    def test_file_read_in_pieces_gives_the_same_as_its_bytes(self):
        rng = random.Random(3)
        data = bytes(rng.randrange(256) for _ in range(10007))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.tflite")
            Path(path).write_bytes(data)
            empty = os.path.join(tmp, "empty.tflite")
            Path(empty).write_bytes(b"")
            for module in FNV_MODULES:
                if not hasattr(module, "fnv1a32_file"):
                    continue
                expected = module.fnv1a32_bytes(data)
                self.assertEqual(module.fnv1a32_file(path), expected)
                for chunk in (1, 7, 4096, 10007, 20000):
                    self.assertEqual(module.fnv1a32_file(path, chunk), expected, chunk)
                self.assertEqual(module.fnv1a32_file(empty), 0x811C9DC5)

    def test_fingerprint_printed_with_eight_hex_digits_is_found_in_the_boot_lines(self):
        # a fingerprint below 0x10000000 keeps its leading zeros in the firmware line
        if hil_benchmark is None:
            self.skipTest("hil_benchmark could not be imported")
        value = hil_benchmark.fnv1a32_bytes(b"foo") & 0x00FFFFFF
        lines = ["BOARD:PORTENTA_H7", "MODEL_BYTES:3", "MODEL_FNV1A:%08x" % value, "ARENA_USED:1024", "READY_HIL"]
        found = hil_benchmark.parse_firmware_fingerprint(lines)
        self.assertEqual(found["model_fnv1a"], "%08x" % value)
        self.assertEqual(len(found["model_fnv1a"]), 8)
        self.assertEqual(int(found["model_fnv1a"], 16), value)


if __name__ == "__main__":
    unittest.main()
