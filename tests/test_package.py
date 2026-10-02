import unittest

import ai_lib


class TestPackage(unittest.TestCase):
    def test_version_exposed(self) -> None:
        self.assertEqual(ai_lib.__version__, "0.1.1")


if __name__ == "__main__":
    unittest.main()
