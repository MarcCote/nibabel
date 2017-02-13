import os
import argparse

import nibabel as nib


def build_argparser():
    DESCRIPTION = "Merged several tractograms together."
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument('tractograms', metavar='tractogram', nargs="+", help='list of tractograms (.trk|.tck).')
    p.add_argument('--out', metavar='filename', default='merged.tck',
                   help='name of the file containing merged tractograms. Default: "%(default)s".')
    p.add_argument('-f', '--force', action="store_true", help='overwrite existing output files.')
    return p


def main():
    parser = build_argparser()
    args = parser.parse_args()

    if os.path.isfile(args.out) and not args.force:
        print("Skipping existing file: '{}'. Use -f to overwrite.".format(args.out))

    tractogram = None
    for filename in args.tractograms:
        tfile = nib.streamlines.load(filename)
        if tractogram is None:
            tractogram = tfile.tractogram
        else:
            tractogram += tfile.tractogram

    nib.streamlines.save(tractogram, args.out, header=tfile.header)

if __name__ == '__main__':
    main()
