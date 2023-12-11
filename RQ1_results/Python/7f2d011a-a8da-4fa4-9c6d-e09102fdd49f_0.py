import os
import shutil
import tempfile
import unittest
from script import camel_to_snake, rename_file, update_generate_statements

class ScriptTest(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_camel_to_snake(self):
        self.assertEqual(camel_to_snake("camelCase"), "camel_case")
        self.assertEqual(camel_to_snake("dcRedirectionPolicy_mock.go"), "dc_redirection_policy_mock.go")
        self.assertEqual(camel_to_snake("nDCHistoryResender_mock.go"), "ndc_historyresender_mock.go")

    def test_rename_file(self):
        # Create a dummy Go file
        file_path = os.path.join(self.test_dir, "testFile.go")
        with open(file_path, "w") as file:
            file.write("//go:generate mockgen -source TestFile.go -destination TestFile_mock.go")

        # Rename the file
        rename_file(file_path)

        # Check if the file is renamed
        self.assertFalse(os.path.exists(file_path))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "test_file.go")))

        # Check if go:generate statement is updated
        with open(os.path.join(self.test_dir, "test_file.go"), "r") as file:
            content = file.read()
            self.assertIn("-source test_file.go -destination test_file_mock.go", content)

    def test_update_generate_statements(self):
        # Create a dummy Go file
        file_path = os.path.join(self.test_dir, "testFile.go")
        with open(file_path, "w") as file:
            file.write("//go:generate mockgen -source TestFile.go -destination TestFile_mock.go")

        # Update go:generate statements
        update_generate_statements(file_path)

        # Check if go:generate statement is updated
        with open(file_path, "r") as file:
            content = file.read()
            self.assertIn("-source test_file.go -destination test_file_mock.go", content)

if __name__ == "__main__":
    unittest.main()
