import os.path
import os
import pdb2ms
import initial_placement
import subprocess
import postfilter
import apply_results
import merge_pdbs
import pairwise_rmsd
import clustering
import io
from contextlib import redirect_stdout

#os.environ["OMP_NUM_THREADS"] = "2"

receptor_path = "/home/nils/swifttcr/example/input/1ao7_l_u_pnon.pdb"
ligand_path = "/home/nils/swifttcr/example/input/1ao7_r_u_pnon.pdb"
output_dir = "/home/nils/swifttcr/example/output/"
restraints = "/home/nils/swifttcr/example/input/restraintsDE.json"
rotations = "/home/nils/swifttcr/example/input/filtered_cr_in_60.prm"

reference_ligand = "/home/nils/swifttcr/example/input/2bnr_r_u.pdb"
reference_receptor = "/home/nils/swifttcr/example/input/2bnr_l_u.pdb"

receptor = os.path.basename(receptor_path)
print(receptor)
receptor_pdb = os.path.basename(receptor_path)
print(receptor_pdb)

#How can i make this work better and not ugly?
receptor_ms = os.path.basename(receptor_path)
receptor_ms = receptor_ms.replace(".pdb", ".ms")
print(receptor_ms)

ligand = os.path.basename(ligand_path)
ligand_pdb = os.path.basename(ligand_path)

#Same for this part
ligand_ms = os.path.basename(ligand_path)
ligand_ms = ligand_ms.replace(".pdb", ".ms")

out_prefix = "data"

#################################################################

if __name__ == "__main__":

    initial_placement.initial_placement_main(receptor_path, ligand_path, output_dir, reference_receptor, reference_ligand)

    pdb2ms.pdb2ms_main(output_dir)

    os.chdir(output_dir)

    subprocess.run([
        "/home/nils/swifttcr/piper/piper_attr",
        "-k1",
        "--msur_k=1.0",
        "--maskr=1.0",
        "-T", "FFTW_EXHAUSTIVE",
        "-p", "/home/nils/swifttcr/piper/atoms04.prm",
        "-f", "/home/nils/swifttcr/piper/coeffs04.prm",
        "-r", rotations,
        os.path.join(output_dir, receptor_ms),
        os.path.join(output_dir, ligand_ms)
    ]
    )

    print(os.getcwd())

    postfilter.post_filter_main(output_dir, "ft.000.00", rotations, restraints, receptor, ligand, out_prefix)

    if not os.path.exists("rotated"):
        os.mkdir("rotated")
    os.chdir("rotated")

    apply_results.apply_results_main(100, None, None, out_prefix, os.path.join(output_dir, "ft.000.00"), os.path.join(output_dir, rotations), os.path.join(output_dir,ligand))

    os.chdir("..")

    if not os.path.exists("merged"):
        os.mkdir("merged")

    merge_pdbs.merge_pdbs_main(receptor,"rotated", "merged")

    print(os.getcwd())

    ## Will fix this but i got an error when trying to run it using multiprocessing and this doesn't
    pairwise_rmsd.pairwise_rmsd_main("merged", "irmsd.csv", "A", "D", "interface", 2)

    output_file = os.path.join(output_dir, 'clustering.txt')

    output_buffer = io.StringIO()

    with redirect_stdout(output_buffer):
        clustering.clustering_main(os.path.join(output_dir, 'irmsd.csv'))

    captured_output = output_buffer.getvalue()

    with open(output_file, 'w') as file:
        file.write(captured_output)

    print(f"Output written to {output_file}")