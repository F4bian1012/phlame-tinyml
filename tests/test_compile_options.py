#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checks how compile_upload_arduino.py turns its options into what arduino-cli
receives: the macros of the sketch (arena size, image buffer, and the place of
the tensor arena asked with --arena_memory internal), the build folder, and the
combinations it refuses before arduino-cli is run at all.

arduino-cli is never run here: the script's own run_command is replaced by a
stand-in that records each command and answers like the tool does. No board.

Standard library only.

    python -m unittest discover -s tests -v
"""
import contextlib
import io
import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import boards  # noqa: E402
import compile_upload_arduino as script  # noqa: E402

PIL_SKETCH = (REPO / "deployment" / "pil_firmware" / "pil_firmware.ino").read_text(encoding="utf-8")
HIL_SKETCH = (REPO / "deployment" / "hil_firmware" / "hil_firmware.ino").read_text(encoding="utf-8")
INTERNAL = "-DTENSOR_ARENA_INTERNAL_RAM=1"


class FakeArduinoCli:
    """Stands in for run_command: records the commands and answers like arduino-cli.

    The core is reported as installed and one board of the given kind as
    connected on PORT, so a run reaches the compile and, without
    --compile_only, the upload."""

    PORT = "SIM_PORT"

    def __init__(self, board_key="portenta"):
        self.board = boards.BOARDS[board_key]
        self.commands = []

    def __call__(self, command):
        self.commands.append(command)
        if command.startswith("arduino-cli core list"):
            return True, json.dumps({"platforms": [{"id": self.board["core"]}]})
        if command.startswith("arduino-cli board list"):
            return True, json.dumps({"detected_ports": [{
                "port": {"address": self.PORT},
                "matching_boards": [{"name": "Arduino " + self.board["label"], "fqbn": self.board["fqbn"]}]}]})
        if command.startswith("arduino-cli compile"):
            return True, ("Sketch uses 700144 bytes (35%) of program storage space. Maximum is 1966080 bytes.\n"
                          "Global variables use 455632 bytes (87%) of dynamic memory, leaving 67992 bytes for "
                          "local variables. Maximum is 523624 bytes.\n")
        return True, ""

    def of(self, verb):
        return [c for c in self.commands if c.startswith("arduino-cli %s" % verb)]


def run(*argv, board_key="portenta"):
    """Runs the script's main() with a stand-in for arduino-cli.

    Returns (exit code or None, what was printed, the stand-in)."""
    cli_tool = FakeArduinoCli(board_key)
    out = io.StringIO()
    code = None
    with mock.patch.object(script, "run_command", cli_tool), \
            contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        try:
            script.main(list(argv))
        except SystemExit as stop:
            code = stop.code
    return code, out.getvalue(), cli_tool


class DefinesTest(unittest.TestCase):
    def test_default_build_passes_no_macro(self):
        for board_key in boards.BOARDS:
            for target in ("pil", "hil", None):
                self.assertEqual(script.sketch_defines(board_key, target), [])

    def test_sizes_reach_the_sketch_as_before(self):
        self.assertEqual(script.sketch_defines("nicla", "pil", arena_kb=392), ["-DTENSOR_ARENA_KB=392"])
        self.assertEqual(script.sketch_defines("nicla", "hil", arena_kb=288, image_buffer_bytes=19200),
                         ["-DTENSOR_ARENA_KB=288", "-DIMAGE_BUFFER_BYTES=19200"])
        self.assertEqual(script.sketch_defines("portenta", None, image_buffer_bytes=57600),
                         ["-DIMAGE_BUFFER_BYTES=57600"])

    def test_arena_in_internal_ram_is_a_macro_of_the_pil_sketch_with_its_size(self):
        self.assertEqual(script.sketch_defines("portenta", "pil", arena_kb=368, arena_memory="internal"),
                         [INTERNAL, "-DTENSOR_ARENA_KB=368"])
        self.assertEqual(script.sketch_defines("portenta", "pil", arena_kb=192, image_buffer_bytes=19200,
                                               arena_memory="internal"),
                         [INTERNAL, "-DTENSOR_ARENA_KB=192", "-DIMAGE_BUFFER_BYTES=19200"])

    def test_the_macros_passed_are_the_ones_the_sketch_reads(self):
        for define in script.sketch_defines("portenta", "pil", arena_kb=368, image_buffer_bytes=19200,
                                            arena_memory="internal"):
            name = define[2:].split("=")[0]
            self.assertRegex(PIL_SKETCH, r"#if(ndef)? %s\b" % name)
        # the HIL sketch has the two sizes and not the place of the arena
        self.assertIn("#ifndef TENSOR_ARENA_KB", HIL_SKETCH)
        self.assertNotIn("TENSOR_ARENA_INTERNAL_RAM", HIL_SKETCH)

    def test_internal_ram_without_a_size_is_refused_and_says_what_to_pass(self):
        with self.assertRaises(ValueError) as refused:
            script.sketch_defines("portenta", "pil", arena_memory="internal")
        message = str(refused.exception)
        self.assertIn("needs --arena_kb", message)
        self.assertIn("4096", message)                       # why the default cannot be used
        self.assertIn("ARENA_USED", message)                 # where the size comes from
        self.assertIn("--arena_kb %d" % boards.BOARDS["nicla"]["default_arena_kb"], message)

    def test_internal_ram_is_refused_on_a_board_without_sdram(self):
        with self.assertRaises(ValueError) as refused:
            script.sketch_defines("nicla", "pil", arena_kb=368, arena_memory="internal")
        self.assertIn("Nicla Vision", str(refused.exception))
        self.assertIn("internal RAM already", str(refused.exception))

    def test_internal_ram_is_refused_for_another_sketch(self):
        for target in ("hil", None):
            with self.assertRaises(ValueError) as refused:
                script.sketch_defines("portenta", target, arena_kb=368, arena_memory="internal")
            self.assertIn("--target pil", str(refused.exception))


class CommandsTest(unittest.TestCase):
    FQBN = boards.BOARDS["portenta"]["fqbn"]

    def test_default_command_is_the_one_the_script_always_ran(self):
        self.assertEqual(script.compile_command(self.FQBN, "deployment/pil_firmware"),
                         'arduino-cli compile --fqbn %s --warnings none "deployment/pil_firmware"' % self.FQBN)
        self.assertEqual(script.upload_command("SIM_PORT", self.FQBN, "deployment/pil_firmware"),
                         'arduino-cli upload -p SIM_PORT --fqbn %s "deployment/pil_firmware"' % self.FQBN)

    def test_macros_travel_in_one_build_property(self):
        command = script.compile_command(self.FQBN, "sketch", [INTERNAL, "-DTENSOR_ARENA_KB=368"])
        self.assertIn(' --build-property "compiler.cpp.extra_flags=%s -DTENSOR_ARENA_KB=368" ' % INTERNAL, command)
        self.assertEqual(command.count("--build-property"), 1)

    def test_build_folder_is_used_by_the_compile_and_by_the_upload(self):
        self.assertIn(' --build-path "out/build" ', script.compile_command(self.FQBN, "sketch", build_path="out/build"))
        self.assertIn(' --input-dir "out/build" ',
                      script.upload_command("SIM_PORT", self.FQBN, "sketch", build_path="out/build"))


class ScriptTest(unittest.TestCase):
    def test_internal_ram_build_is_compiled_with_its_macros(self):
        code, log, tool = run("--target", "pil", "--arena_memory", "internal", "--arena_kb", "368", "--compile_only")
        self.assertIsNone(code, log)
        self.assertEqual(len(tool.of("compile")), 1)
        compile_command = tool.of("compile")[0]
        self.assertIn("--fqbn %s " % boards.BOARDS["portenta"]["fqbn"], compile_command)
        self.assertIn('"compiler.cpp.extra_flags=%s -DTENSOR_ARENA_KB=368"' % INTERNAL, compile_command)
        self.assertTrue(compile_command.endswith('pil_firmware"'), compile_command)
        self.assertEqual(tool.of("upload"), [])                       # --compile_only
        self.assertEqual(tool.of("board list"), [])                   # and no board is looked for
        self.assertIn("Global variables use 455632 bytes", log)       # the RAM the linker reports is shown

    def test_default_builds_get_no_macro(self):
        for board_key in boards.BOARDS:
            code, log, tool = run("--target", "pil", "--board", board_key, "--compile_only", board_key=board_key)
            self.assertIsNone(code, log)
            self.assertNotIn("--build-property", tool.of("compile")[0])
            self.assertNotIn("TENSOR_ARENA", tool.of("compile")[0])
            self.assertNotIn("--build-path", tool.of("compile")[0])

    def test_upload_follows_the_compile_and_takes_the_binary_from_the_build_folder(self):
        code, log, tool = run("--target", "pil", "--arena_memory", "internal", "--arena_kb", "368",
                              "--build_path", os.path.join("some", "build"))
        self.assertIsNone(code, log)
        folder = os.path.abspath(os.path.join("some", "build"))
        self.assertIn('--build-path "%s"' % folder, tool.of("compile")[0])
        self.assertEqual(len(tool.of("upload")), 1)
        self.assertIn("-p %s " % tool.PORT, tool.of("upload")[0])
        self.assertIn('--input-dir "%s"' % folder, tool.of("upload")[0])
        self.assertLess(tool.commands.index(tool.of("compile")[0]), tool.commands.index(tool.of("upload")[0]))
        # without --build_path the upload is left to find the build of the sketch, as before
        code, log, tool = run("--target", "pil")
        self.assertIsNone(code, log)
        self.assertNotIn("--input-dir", tool.of("upload")[0])

    def test_wrong_combinations_stop_before_arduino_cli_is_run(self):
        refused = {
            ("--target", "pil", "--arena_memory", "internal"): "needs --arena_kb",
            ("--target", "pil", "--board", "nicla", "--arena_memory", "internal", "--arena_kb", "368"): "Nicla Vision",
            ("--target", "hil", "--arena_memory", "internal", "--arena_kb", "368"): "--target pil",
            ("--sketch_path", str(REPO / "deployment" / "pil_firmware"), "--arena_memory", "internal",
             "--arena_kb", "368"): "--target pil",
            ("--target", "pil", "--arena_memory", "external", "--arena_kb", "368"): "invalid choice",
        }
        for argv, message in refused.items():
            code, log, tool = run(*argv, "--compile_only")
            self.assertEqual(code, 2, argv)
            self.assertIn(message, log, argv)
            self.assertEqual(tool.commands, [], argv)

    def test_help_names_the_option(self):
        code, log, tool = run("--help")
        self.assertEqual(code, 0)
        for text in ("--arena_memory", "{default,internal}", "--build_path"):
            self.assertIn(text, log)
        self.assertEqual(tool.commands, [])


if __name__ == "__main__":
    unittest.main()
