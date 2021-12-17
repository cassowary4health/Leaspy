#!/bin/bash
# Script to be run at root of repo

# Check no missing __init__.py file
folders_to_check=(leaspy tests)
regex_folders_without_py="^leaspy/(datasets/(data|(model|individual)_parameters)|algo/data)$"

# regex used are extended but the flag needed is not the same between mac & linux...
find -E . -maxdepth 0 > /dev/null 2>&1 && find_pre="-E" || find_post="-regextype posix-extended"

py_init=$(find $find_pre ${folders_to_check[@]} $find_post -name "__init__.py" -not -regex ".*/_[^/]+/.*" | sed -E 's@/__init__.py@@g' | sort)
subfolders=$(find $find_pre ${folders_to_check[@]} $find_post -type d -not -regex ".*/_[^/]+(/.+|$)" -not -regex $regex_folders_without_py | sort)

if [[ $py_init != $subfolders ]]; then

    echo "There are some missing __init__.py files in $folders_to_check folders"
    echo

    # display differences
    diff -aU0 <(echo "$subfolders") <(echo "$py_init") | tail -n +4 | grep -v "@@ "

    echo
    echo "Please, fix these issues prior to pushing."
    exit 1
fi

echo "Init files seems OK!"
exit 0