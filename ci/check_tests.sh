#!/bin/bash

for test_dir in unit_tests functional_tests; do
	# check no missing __init__.py file
	nb_init=$(find tests/$test_dir -type f -name "__init__.py" | wc -l)
	nb_folder=$(find tests/$test_dir -type d -not -name "__*" | wc -l)
	if (( $nb_init != $nb_folder )); then
		echo "Missing a '__init__.py' in a test subfolder from $test_dir."
		exit 1
	fi

	# check that all tests files begin with test_*
	files_not_beginning_by_test=$(find tests/$test_dir -name "*.py" -not -name "test_*" -not -name "__init__.py")
	nb_not_beginning_by_test=$(find tests/$test_dir -name "*.py" -not -name "test_*" -not -name "__init__.py" | wc -l)
	if (( $nb_not_beginning_by_test != 0 )); then
		echo "Test files not matching 'test_*.py' in $test_dir: $files_not_beginning_by_test"
		exit 1
	fi

	# TODO? check that all tests classes import LeaspyTestCase?
done

echo "Tests seems OK!"
exit 0
