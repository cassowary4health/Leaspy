This directory is used to store temporary files during tests, please keep it.
It should be empty after tests if everything went smooth.

To help your tests you should:
- if you redefine `setUpClass`, begin your class method by calling `super().setUpClass()`
- if you redefine `tearDownClass`, finish your class method by calling `super().tearDownClass()`
- if you need a tmp path, use the `rel_tmp_path` class method
- if you want to customize the subfolder tmp files are stored in redefine `TMP_SUBFOLDER` class property (default to class name)
- if you want your tmp folder not to be removed at tear down, set `TMP_REMOVE_AT_END = False` as class attribute
- if you want your tmp folder not to be reset at set up, set `TMP_RESET_AT_SETUP = False` as class attribute
