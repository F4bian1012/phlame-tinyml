#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cli.py - Command-line conventions shared by the scripts.

The scripts were written at different times and spell their multi-word
options in two ways (--model_path, --output-dir). With the parser defined
here every such option is accepted with '_' or '-': --output_dir and
--output-dir are the same option in every script, so a command copied from
one level of the ladder works at the next one.

Standard library only: importing it must stay cheap for --help.
"""

import argparse
import sys


class ArgumentParser(argparse.ArgumentParser):
    """argparse.ArgumentParser that accepts '_' and '-' interchangeably in option names."""

    def parse_known_args(self, args=None, namespace=None):
        args = sys.argv[1:] if args is None else list(args)
        return super().parse_known_args([self._canonical(a) for a in args], namespace)

    def _canonical(self, token):
        """Rewrites --some-option / --some_option to the spelling the script declares."""
        if not token.startswith("--") or token == "--":
            return token
        name, sep, value = token.partition("=")
        if name in self._option_string_actions:
            return token
        for alt in ("--" + name[2:].replace("-", "_"), "--" + name[2:].replace("_", "-")):
            if alt in self._option_string_actions:
                return alt + sep + value
        return token
