import os


def _count_valid_files_in_directory(directory, white_list_formats, follow_links, split=None):
    """Count files with extension in `white_list_formats` contained in a directory.
    # Arguments
        directory: absolute path to the directory containing files to be counted
        white_list_formats: set of strings containing allowed extensions for
            the files to be counted.
    # Returns
        the count of files with extension in `white_list_formats` contained in
        the directory.
    """
    def _recursive_list(subpath):
        return sorted(os.walk(subpath, followlinks=follow_links), key=lambda tpl: tpl[0])

    samples = 0
    for _, _, files in _recursive_list(directory):
        for fname in files:
            is_valid = False
            for extension in white_list_formats:
                if fname.lower().endswith('.' + extension):
                    is_valid = True
                    break
            if is_valid:
                samples += 1
    return samples


def get_extension(filename):
    """Get extension of the filename
    There are newer methods to achieve this but this method is backwards compatible.
    """
    return os.path.splitext(filename)[1].strip('.').lower()


def _iter_valid_files(directory, white_list_formats, follow_links):
    """Iterates on files with extension in `white_list_formats` contained in `directory`.
    # Arguments
        directory: Absolute path to the directory
            containing files to be counted
        white_list_formats: Set of strings containing allowed extensions for
            the files to be counted.
        follow_links: Boolean.
    # Yields
        Tuple of (root, filename) with extension in `white_list_formats`.
    """
    def _recursive_list(subpath):
        return sorted(os.walk(subpath, followlinks=follow_links),
                      key=lambda x: x[0])

    for root, _, files in _recursive_list(directory):
        for fname in sorted(files):
            if fname.lower().endswith('.tiff'):
                warnings.warn('Using ".tiff" files with multiple bands '
                              'will cause distortion. Please verify your output.')
            if get_extension(fname) in white_list_formats:
                yield root, fname


def _list_valid_filenames_in_directory(directory, white_list_formats, split,
                                       class_indices, follow_links):
    """Lists paths of files in `subdir` with extensions in `white_list_formats`.
    # Arguments
        directory: absolute path to a directory containing the files to list.
            The directory name is used as class label
            and must be a key of `class_indices`.
        white_list_formats: set of strings containing allowed extensions for
            the files to be counted.
        split: tuple of floats (e.g. `(0.2, 0.6)`) to only take into
            account a certain fraction of files in each directory.
            E.g.: `segment=(0.6, 1.0)` would only account for last 40 percent
            of images in each directory.
        class_indices: dictionary mapping a class name to its index.
        follow_links: boolean.
    # Returns
         classes: a list of class indices
         filenames: the path of valid files in `directory`, relative from
             `directory`'s parent (e.g., if `directory` is "dataset/class1",
            the filenames will be
            `["class1/file1.jpg", "class1/file2.jpg", ...]`).
    """
    dirname = os.path.basename(directory)
    if split:
        num_files = len(list(
            _iter_valid_files(directory, white_list_formats, follow_links)))
        start, stop = int(split[0] * num_files), int(split[1] * num_files)
        valid_files = list(
            _iter_valid_files(
                directory, white_list_formats, follow_links))[start: stop]
    else:
        valid_files = _iter_valid_files(
            directory, white_list_formats, follow_links)
    classes = []
    filenames = []
    for root, fname in valid_files:
        classes.append(class_indices[dirname])
        absolute_path = os.path.join(root, fname)
        relative_path = os.path.join(
            dirname, os.path.relpath(absolute_path, directory))
        filenames.append(relative_path)

    return classes, filenames
# def _list_valid_filenames_in_directory(directory, white_list_formats,
#                                        class_indices, follow_links):
#     """List paths of files in `subdir` relative from `directory` whose extensions are in `white_list_formats`.
#     # Arguments
#         directory: absolute path to a directory containing the files to list.
#             The directory name is used as class label and must be a key of `class_indices`.
#         white_list_formats: set of strings containing allowed extensions for
#             the files to be counted.
#         class_indices: dictionary mapping a class name to its index.
#     # Returns
#         classes: a list of class indices
#         filenames: the path of valid files in `directory`, relative from
#             `directory`'s parent (e.g., if `directory` is "dataset/class1",
#             the filenames will be ["class1/file1.jpg", "class1/file2.jpg", ...]).
#     """
#     def _recursive_list(subpath):
#         return sorted(os.walk(subpath, followlinks=follow_links), key=lambda tpl: tpl[0])

#     classes = []
#     filenames = []
#     subdir = os.path.basename(directory)
#     basedir = os.path.dirname(directory)
#     for root, _, files in _recursive_list(directory):
#         for fname in files:
#             is_valid = False
#             for extension in white_list_formats:
#                 if fname.lower().endswith('.' + extension):
#                     is_valid = True
#                     break
#             if is_valid:
#                 classes.append(class_indices[subdir])
#                 # add filename relative to directory
#                 absolute_path = os.path.join(root, fname)
#                 filenames.append(os.path.relpath(absolute_path, basedir))
#     return classes, filenames