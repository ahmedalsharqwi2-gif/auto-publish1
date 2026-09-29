"""Smoke tests for imports used by the publishing pipeline."""

import importlib
import unittest


class QualityCheckImportTests(unittest.TestCase):
    def test_quality_check_pipeline_imports(self):
        module = importlib.import_module("scripts.quality_check")
        self.assertTrue(callable(module.QualityCheckPipeline))


if __name__ == "__main__":
    unittest.main()
