#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The rig calibrator and a board that does not answer.

The HIL firmware prints its boot lines once and then waits for commands. When
it does not answer (it is stuck, or it stopped on a camera failure), the
operator presses RESET: the USB port disappears and comes back. The calibrator
must stay on the port through that, open it again and read the boot lines,
instead of ending with a traceback or asking to be started again at the right
second. Checked here against a simulated port.

Needs pyserial, OpenCV and numpy (those of the calibrator); skipped otherwise.

    python -m unittest discover -s tests -v
"""
import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

try:
    import calibrate_hil_rig as calib
except SystemExit:          # hil_benchmark ends when pyserial, OpenCV or numpy are missing
    calib = None

BOOT = ["BOARD:PORTENTA_H7", "CAM_INIT:OK", "MODEL_FNV1A:ad34723a", "READY_HIL"]


class RestartingPort:
    """A port whose firmware is silent until the board is reset.

    The first read after `silent_reads` polls raises, as the driver does when
    the port vanishes; open() fails `failed_opens` times (the port is not back
    yet) and then the firmware boots and answers.
    """

    def __init__(self, boot=BOOT, silent_reads=3, failed_opens=2):
        self.boot = list(boot)
        self.silent_reads = silent_reads
        self.failed_opens = failed_opens
        self.state = "silent"
        self.polls = 0
        self.opens = 0
        self.lines = []
        self.written = []
        self.closed = 0

    @property
    def in_waiting(self):
        if self.state == "silent":
            self.polls += 1
            if self.polls > self.silent_reads:
                self.state = "gone"
        if self.state == "gone":
            raise calib.serial.SerialException("ClearCommError failed")
        return len(self.lines)

    def readline(self):
        return (self.lines.pop(0) + "\n").encode()

    def write(self, data):
        self.written.append(bytes(data))
        if self.state == "booted" and data == calib.hb.CMD_DUMP_ON:
            self.lines.append("FRAME_DUMP:ON")

    def flush(self):
        pass

    def close(self):
        self.closed += 1

    def open(self):
        self.opens += 1
        if self.opens <= self.failed_opens:
            raise calib.serial.SerialException("could not open port")
        self.state = "booted"
        self.lines += self.boot


@unittest.skipIf(calib is None, "the calibrator needs pyserial, OpenCV and numpy")
class CalibratorRestartTest(unittest.TestCase):
    @staticmethod
    def quick_first_request(port):
        """The request to a silent firmware gives up at once instead of after 6 s."""
        real = calib.frame_dump_on
        return mock.patch.object(
            calib, "frame_dump_on",
            side_effect=lambda ser, timeout_s: real(ser, 0.05 if port.state == "silent" else timeout_s))

    def open_board(self, port):
        with mock.patch.object(calib.serial, "Serial", return_value=port), mock.patch.object(calib.time, "sleep"), \
                contextlib.redirect_stdout(io.StringIO()) as log:
            try:
                return calib.open_board("SIM", 115200, "portenta"), log.getvalue()
            except SystemExit as exc:
                return exc, log.getvalue()

    def test_a_reset_while_waiting_is_followed_through(self):
        port = RestartingPort()
        with self.quick_first_request(port):
            (ser, announced), log = self.open_board(port)
        self.assertIs(ser, port)
        self.assertEqual(announced, "PORTENTA_H7")
        self.assertIn("Press RESET", log)
        self.assertIn("opening it again", log)
        self.assertEqual(port.opens, 3)                       # two attempts before the port was back
        self.assertEqual(port.written[-1], calib.hb.CMD_DUMP_ON)

    def test_a_camera_failure_after_the_reset_is_reported(self):
        port = RestartingPort(boot=["BOARD:PORTENTA_H7", "CAM_INIT:FAIL"])
        with self.quick_first_request(port):
            result, log = self.open_board(port)
        self.assertIsInstance(result, SystemExit)
        self.assertEqual(result.code, 1)
        self.assertIn("CAM_INIT:FAIL", log)
        self.assertNotIn("Traceback", log)

    def test_a_port_that_does_not_come_back_ends_with_a_message(self):
        port = RestartingPort(failed_opens=10 ** 6)
        with self.quick_first_request(port), mock.patch.object(calib, "reopen_port", return_value=False):
            result, log = self.open_board(port)
        self.assertIsInstance(result, SystemExit)
        self.assertEqual(result.code, 1)
        self.assertIn("did not come back", log)

    def test_a_port_lost_in_the_live_loop_is_opened_again(self):
        port = RestartingPort()
        port.state = "gone"                                   # unplugged while a frame was being read
        with mock.patch.object(calib.time, "sleep"), contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(calib.reconnect(port))
        self.assertEqual(port.opens, 3)                       # two attempts before the port was back
        self.assertEqual(port.written[-1], calib.hb.CMD_DUMP_ON)

    def test_a_port_that_stays_away_ends_the_live_loop_without_a_traceback(self):
        port = RestartingPort()
        port.state = "gone"
        with mock.patch.object(calib, "reopen_port", return_value=False):
            self.assertFalse(calib.reconnect(port))
        self.assertEqual(sorted(k for k in calib.TEXT["en"] if k.startswith("port_")),
                         ["port_back", "port_gone", "port_lost"])

    def test_a_running_firmware_is_not_asked_to_reset(self):
        port = RestartingPort()
        port.state = "booted"
        (ser, announced), log = self.open_board(port)
        self.assertIs(ser, port)
        self.assertIsNone(announced)                          # boot lines not seen: the board is not known
        self.assertNotIn("Press RESET", log)
        self.assertEqual(port.opens, 0)


if __name__ == "__main__":
    unittest.main()
