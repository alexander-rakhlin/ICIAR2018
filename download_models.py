#!/usr/bin/env python3
# """Download and populate directories with models, features and CV predictions. Partially borrowed from Keras distribution"""

import os
import zipfile
import hashlib
from six.moves.urllib.error import URLError
from six.moves.urllib.request import urlretrieve
import argparse
from tqdm import tqdm


DEFAULT_MD5 = "5d92cd351ada73ccbc7a15c879a65b40"
DEFAULT_FNAME = "ICIAR2018_data.zip"
DEFAULT_URL = "https://www.dropbox.com/s/1vwsfekuxc50cfm/ICIAR2018_data.zip?dl=1"


def _extract_archive(file_path, path="."):
    """Extracts an archive of zip format.

    # Arguments
        file_path: path to the archive file
        path: path to extract the archive file

    # Returns
        True if a match was found and an archive extraction was completed,
        False otherwise.
    """
    open_fn = zipfile.ZipFile
    is_match_fn = zipfile.is_zipfile

    if is_match_fn(file_path):
        with open_fn(file_path) as archive:
            try:
                archive.extractall(path)
            except (RuntimeError, KeyboardInterrupt):
                raise
        return True


class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def get_file(fname, origin, extract=False, md5_hash=None):
    """Downloads a file from a URL if it not already present.

    Files in zip formats can also be extracted.
    Passing a hash will verify the file after download.

    # Arguments
        fname: Name of the file. If an absolute path `/path/to/file.txt` is
            specified the file will be saved at that location.
        origin: Original URL of the file.
        md5_hash: md5 hash of the file for verification.

    # Returns
        Path to the downloaded file
    """
    fpath = fname
    download = False
    if os.path.exists(fpath):
        # File found; verify integrity if a hash was provided.
        if md5_hash is not None:
            if not validate_file(fpath, md5_hash, algorithm="md5"):
                print("A local file was found, but it seems to be "
                      "incomplete or outdated because the md5 "
                      "file hash does not match the original value of " +
                      md5_hash + " so we will re-download the data.")
                download = True
            else:
                print("A local file was found and "
                      "file hash matches the original value "
                      "so we do not re-download the data.")
    else:
        download = True

    if download:
        print("Downloading data from", origin)
        error_msg = "URL fetch failure on {}: {} -- {}"
        try:
            try:
                with TqdmUpTo(unit="B", unit_scale=True, miniters=1,
                              desc=fpath.split("/")[-1]) as t:  # all optional kwargs
                    urlretrieve(origin, fpath, reporthook=t.update_to, data=None)
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(fpath):
                os.remove(fpath)
            raise
    if extract:
        _extract_archive(fpath)

    return fpath


def _hash_file(fpath, algorithm="sha256", chunk_size=65535):
    """Calculates a file sha256 or md5 hash.

    # Example

    ```python
       >>> from keras.data_utils import _hash_file
       >>> _hash_file("/path/to/file.zip")
       "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ```

    # Arguments
        fpath: path to the file being validated
        algorithm: hash algorithm, one of "auto", "sha256", or "md5".
            The default "auto" detects the hash algorithm in use.
        chunk_size: Bytes to read at a time, important for large files.

    # Returns
        The file hash
    """
    hasher = hashlib.md5()

    with open(fpath, "rb") as fpath_file:
        for chunk in iter(lambda: fpath_file.read(chunk_size), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def validate_file(fpath, file_hash, algorithm="auto", chunk_size=65535):
    """Validates a file against a sha256 or md5 hash.

    # Arguments
        fpath: path to the file being validated
        file_hash:  The expected hash string of the file.
            The sha256 and md5 hash algorithms are both supported.
        algorithm: Hash algorithm, one of "auto", "sha256", or "md5".
            The default "auto" detects the hash algorithm in use.
        chunk_size: Bytes to read at a time, important for large files.

    # Returns
        Whether the file is valid
    """
    if ((algorithm is "sha256") or
            (algorithm is "auto" and len(file_hash) is 64)):
        hasher = "sha256"
    else:
        hasher = "md5"

    if str(_hash_file(fpath, hasher, chunk_size)) == str(file_hash):
        return True
    else:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--url",
        required=False,
        default=DEFAULT_URL,
        metavar="url",
        help="Original URL of the file.")
    arg("--fname",
        required=False,
        default=DEFAULT_FNAME,
        metavar="fname",
        help="Local file name.")
    arg("--md5",
        required=False,
        default=DEFAULT_MD5,
        metavar="md5",
        help="Feature root dir. Default: data/preprocessed/train")
    arg("-dnx",
        action="store_true",
        default=False,
        help="Do not extract archive after download.")
    args = parser.parse_args()
    url = args.url
    fname = args.fname
    md5 = args.md5
    extract = not args.dnx

    get_file(fname, url, extract=extract, md5_hash=md5)
