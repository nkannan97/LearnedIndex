import urllib.request
import hashlib
import lzma
import os.path
import struct
import argparse
import sys

import numpy as np


def download_data(url, filename, verbose=False):
    """
    Downloads an xz-compressed vector from a given URL if not already downloaded.

    Arguments:
        url {str} -- URL to download from.
        filename {str} -- Filename to check for a downloaded file, and to save to if the file does not exist.
        verbose {bool} -- Whether to print progress.
    """
    if not os.path.exists(filename):
        with urllib.request.urlopen(url) as downloaded:
            with open(filename, 'wb') as f:
                if verbose:
                    print('Downloading ' + filename)
                f.write(downloaded.read())


def decompress_array(filename, md5, dtype_str=None, verbose=False):
    """
    Decompresses an xz-compressed vector from a file, verifies its MD5 checksum, verifies its length, and parses it into a NumPy array of a given datatype.

    Arguments:
        filename {str} -- File to open.
        md5 {str} -- Expected MD5 checksum hexadecimal digest. If None, omits check.
        dtype_str {str} -- NumPy array protocol type string. If None and the dataset name is that of one of the SOSD datasets, assumes the type string of the SOSD dataset. Otherwise, assumes 'float64'.
        verbose {bool} -- Whether to print progress.

    Raises:
        RuntimeError: MD5 checksum verification failed.

    Returns:
        numpy -- Parsed data.
    """
    with lzma.open(filename) as decompressed:
        if filename in sosd_datasets:
            md5 = sosd_datasets[filename]['md5']
        if md5 is not None:
            if verbose:
                print('Verifying ' + filename)
            if md5 != hashlib.md5(decompressed.read()).hexdigest():
                raise RuntimeError('MD5 failure')
        decompressed.seek(0)
        declared_length = struct.unpack('<Q', decompressed.read(8))[0]
        if verbose:
            print('Declared vector length: ' + str(declared_length))
        if dtype_str is None:
            if filename in sosd_datasets:
                dtype_str = sosd_datasets[filename]['dtype_str']
            else:
                dtype_str = 'float64'
        array = np.frombuffer(decompressed.read(), dtype=np.dtype(dtype_str))
        actual_length = array.shape[0]
        if verbose:
            print('Actual vector length: ' + str(actual_length))
        if declared_length != actual_length:
            raise RuntimeError('Length mismatch')
        return array


# Names originally defined at https://github.com/learnedsystems/SOSD/blob/master/scripts/download.sh
sosd_datasets = {
    'wiki_ts_200M_uint64': {
        'url': 'https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/JGVF9A/ULSWQQ',
        'md5': '4f1402b1c476d67f77d2da4955432f7d',
        'dtype_str': 'u8'
    },
    'osm_cellids_200M_uint64': {
        'url': 'https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/JGVF9A/CTBUKT',
        'md5': '01666e42b2d64a55411bdc280ac9d2a3',
        'dtype_str': 'u8'
    },
    'books_200M_uint32': {
        'url': 'https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/JGVF9A/MZZUP2',
        'md5': 'c4a848fdc56130abdd167d7e6b813843',
        'dtype_str': 'u4'
    },
    'books_200M_uint64': {
        'url': 'https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/JGVF9A/PDOUMU',
        'md5': 'aa88040624be2f508f1ab6f5532ace88',
        'dtype_str': 'u8'
    },
    'fb_200M_uint32': {
        'url': 'https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/JGVF9A/GXH3ZC',
        'md5': '881eacb62c38eb8c2fdd4d59706b70a7',
        'dtype_str': 'u4'
    },
    'fb_200M_uint64': {
        'url': 'https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/JGVF9A/Y54SI9',
        'md5': '407a23758b72e3c1ee3f6384d98ce604',
        'dtype_str': 'u8'
    },
}


def generate_data(filename, dataset_size, data_distribution, std_coefficient):
    """
    Generates a synthetic dataset of floating-point numbers by drawing from a uniform distribution in [0, 1), or a mixture of Gaussian distributions with uniformly distributed means between [{-data_distribution}, {data_distribution}] and standard deviations between [0, {std_coefficient}). Compresses as xz and saves to given filename as a vector of dtype float64.

    Arguments:
        filename {str} -- Filename to which to save (overwrites existing file).
        dataset_size {int} -- Number of elements in the dataset.
        data_distribution {int}, {string} -- Either {'uniform'} to draw a uniform distribution, or the number of component Gaussian distributions to draw from a mixture of Gaussian distributions.
        std_coefficient {float} -- For a mixture of Gaussian distributions, the maximum standard deviation of a particular Gaussian distribution (controls the degree to which samples are "separated" between Gaussian distributions).
    """

    rng = np.random.default_rng()

    data = None
    if data_distribution == 'uniform':
        data = rng.random(size=dataset_size) * dataset_size
    else:
        ps = np.repeat(1 / data_distribution, data_distribution)
        means = rng.uniform(
            low=-data_distribution,
            high=data_distribution,
            size=data_distribution)
        stds = rng.uniform(size=data_distribution) * std_coefficient
        data = np.empty(dataset_size)
        for i in range(dataset_size):
            chosen_distribution = rng.choice(len(ps), p=ps)
            data[i] = rng.normal(
                means[chosen_distribution], stds[chosen_distribution])
    # Akin to https://github.com/learnedsystems/SOSD/blob/master/gen_uniform.py#L16
    with lzma.open(filename, 'wb') as f:
        f.write(struct.pack('Q', len(data)))
        f.write(data.tobytes())
        f.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name')
    parser.add_argument('--generated_dataset_size', type=int)
    parser.add_argument('--generated_data_distribution')  # int or 'uniform'
    parser.add_argument('--generated_dataset_std_coefficient', type=float)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if args.dataset_name in sosd_datasets:
        download_data(sosd_datasets[args.dataset_name]['url'],
                      args.dataset_name,
                      verbose=args.verbose)
    else:
        if args.generated_dataset_size is None or args.generated_data_distribution is None or args.generated_dataset_std_coefficient is None:
            raise ValueError(
                'Must specify all of --generated_dataset_size, --generated_data_distribution, and --generated_dataset_std_coefficient')
        data_params = {
            'filename': args.dataset_name,
            'std_coefficient': args.generated_dataset_std_coefficient,
        }
        if args.generated_dataset_size <= 0:
            raise ValueError(
                '--generated_dataset_size must be a positive integer')
        data_params['dataset_size'] = args.generated_dataset_size
        if args.generated_data_distribution == 'uniform':
            data_params['data_distribution'] = 'uniform'
        else:
            try:
                data_params['data_distribution'] = int(
                    args.generated_data_distribution)
                if data_params['data_distribution'] < 0:
                    raise ValueError
            except ValueError:
                raise ValueError(
                    '--generated_data_distribution must be a positive integer')
        generate_data(**data_params)


if __name__ == "__main__":
    main()
