#!/bin/bash
# Script to be run at root of repo

# Check that all Python test files comply with "test_*.py" format
folders_to_check="tests/unit_tests tests/functional_tests"

find_cmd="find $folders_to_check -name '*.py' -not -name 'test_*' -not -name '__init__.py'"
files_not_beginning_by_test=$(eval "$find_cmd")
nb_not_beginning_by_test=$(eval "$find_cmd" | wc -l)
if (( $nb_not_beginning_by_test != 0 )); then
    echo "Test files not matching 'test_*.py':"
    echo "$files_not_beginning_by_test"
    exit 1
fi

# TODO? check that all tests classes import LeaspyTestCase?

echo "Tests seems OK!"
exit 0
