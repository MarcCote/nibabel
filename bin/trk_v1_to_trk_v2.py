import os
import copy
import argparse
import numpy as np

import nibabel as nib
from nibabel.streamlines import Field
from nibabel.orientations import aff2axcodes


def build_argparser():
    DESCRIPTION = "Convert tractograms (TRK v1 -> TRK v2)."
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument('anatomy', help="reference anatomy (.nii|.nii.gz).")
    p.add_argument('tractograms', metavar='bundle', nargs="+", help='list of tractograms.')
    p.add_argument('-f', '--force', action="store_true", help='overwrite existing output files.')
    return p


def main():
    parser = build_argparser()
    args = parser.parse_args()

    try:
        nii = nib.load(args.anatomy)
    except:
        parser.error("Expecting anatomy image as first agument.")

    for tractogram in args.tractograms:
        if nib.streamlines.detect_format(tractogram) is not nib.streamlines.TrkFile:
            print("Skipping non TRK file: '{}'".format(tractogram))
            continue

        output_filename = tractogram[:-4] + '.v2.trk'
        if os.path.isfile(output_filename) and not args.force:
            print("Skipping existing file: '{}'. Use -f to overwrite.".format(output_filename))
            continue

        trk = nib.streamlines.load(tractogram)
        if trk.header['version'] != 1:
            print("Skipping. TRK version is: '{}'".format(trk.header['version']))
            continue

        # Undo transform done by the TRK loader.
        affine = nib.streamlines.trk.get_affine_rasmm_to_trackvis(trk.header)
        trk.tractogram.apply_affine(affine)

        # Applied the right transform.
        header = copy.deepcopy(trk.header)
        header[Field.VOXEL_TO_RASMM] = nii.affine.copy()
        affine = nib.streamlines.trk.get_affine_trackvis_to_rasmm(header)
        trk.tractogram.apply_affine(affine)

        # Continue like nothing has happened.
        trk.tractogram.affine_to_rasmm = np.eye(4)

        # Update header.
        header["version"] = 2
        header[Field.VOXEL_SIZES] = nii.header.get_zooms()[:3]
        header[Field.DIMENSIONS] = nii.shape[:3]
        header[Field.VOXEL_ORDER] = "".join(aff2axcodes(nii.affine))

        nib.streamlines.save(trk.tractogram, output_filename, header=header)

if __name__ == '__main__':
    main()

