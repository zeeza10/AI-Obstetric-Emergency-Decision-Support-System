import os
import sys
import unittest

sys.path.insert(0, os.getcwd())
suite = unittest.defaultTestLoader.discover('tests', pattern='test_core.py')
result = unittest.TextTestRunner(verbosity=1).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
