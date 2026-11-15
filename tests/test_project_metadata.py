#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checks pyproject.toml against the files that already state the same facts:
CITATION.cff (version, licence, author, repository) and requirements.txt
(dependencies), and that it installs no code: the scripts are run from a
checkout as `python src/<script>.py`.

Standard library only (tomllib, Python 3.11 or later).

    python -m unittest discover -s tests -v
"""
import re
import unittest
from pathlib import Path

try:
    import tomllib
except ImportError:          # Python < 3.11
    tomllib = None

REPO = Path(__file__).resolve().parent.parent


def cff_scalar(text, key):
    match = re.search(r'^\s*(?:- )?%s:\s*"?(.*?)"?\s*$' % re.escape(key), text, re.M)
    return match.group(1) if match else None


def requirement_lines():
    """The requirements of requirements.txt: every line that is not a comment."""
    lines = (REPO / "requirements.txt").read_text(encoding="utf-8").splitlines()
    return sorted(line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#"))


@unittest.skipIf(tomllib is None, "reading pyproject.toml needs tomllib (Python 3.11 or later)")
class ProjectMetadataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(REPO / "pyproject.toml", "rb") as fh:
            cls.data = tomllib.load(fh)
        cls.project = cls.data["project"]
        cls.cff = (REPO / "CITATION.cff").read_text(encoding="utf-8")

    def test_version_is_the_one_of_the_citation_file(self):
        self.assertEqual(self.project["version"], cff_scalar(self.cff, "version"))

    def test_licence_author_and_repository_are_those_of_the_citation_file(self):
        self.assertEqual(self.project["license"], cff_scalar(self.cff, "license"))
        author = "%s %s" % (cff_scalar(self.cff, "given-names"), cff_scalar(self.cff, "family-names"))
        self.assertEqual([a["name"] for a in self.project["authors"]], [author])
        self.assertEqual(self.project["urls"]["Repository"], cff_scalar(self.cff, "repository-code"))
        for name in self.project["license-files"]:
            self.assertTrue((REPO / name).is_file(), name)
        self.assertTrue((REPO / self.project["readme"]).is_file())

    def test_dependencies_are_those_of_requirements_txt(self):
        self.assertEqual(sorted(self.project["dependencies"]), requirement_lines())

    def test_no_personal_contact_data(self):
        for author in self.project["authors"]:
            self.assertEqual(list(author), ["name"])       # no e-mail address

    def test_nothing_in_src_is_installed(self):
        # The scripts are run from the checkout; several modules of src/ have names
        # (cli, images, boards, artifacts) that must not land in a shared environment.
        tool = self.data["tool"]["setuptools"]
        self.assertEqual((tool["py-modules"], tool["packages"]), ([], []))
        for table in ("scripts", "gui-scripts", "entry-points"):
            self.assertNotIn(table, self.project)

    def test_python_versions_are_stated(self):
        self.assertRegex(self.project["requires-python"], r"^>=3\.\d+$")


if __name__ == "__main__":
    unittest.main()
