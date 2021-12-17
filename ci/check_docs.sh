#!/bin/bash
# Script to be run at root of repo

# Check no missing *.rst file in docs/
rst_files=$(find docs -name "leaspy*.rst" | sed -E 's@^docs/@@g' | sed -E 's@.rst$@@g' | sort)
py_modules=$(find leaspy -name "*.py" -not -path "*/_legacy/*" | sed -E 's@(/__init__)?\.py@@g' | sed -E 's@/@.@g' | sort)

if [[ $rst_files != $py_modules ]]; then

    echo "There are some missing (-) or some extra (+) rst files in 'docs/' compared to current content of 'leaspy/'"
    echo

    # display differences
    diff -aU0 <(echo "$py_modules") <(echo "$rst_files") | tail -n +4 | grep -v "@@ "

    echo
    echo "Please, fix these issues prior to pushing (you can use \`sphinx-apidoc --separate -o docs/ leaspy/\` to help)."
    exit 1
fi

echo "Docs seems OK!"
exit 0
