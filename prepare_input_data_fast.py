import numpy as np
import drdf
import h5py
from pathlib import Path
import sys
import ROOT
sys.path.append('../tools/display')
from read_mctruth import loadPrimariesEdepSimAll
from geom_import import load_geometry
from geometry import Geometry
sys.path.append('../tools/analysis')
from edepsim_deposits import EdepSimDeposits


def get_response_data(event) -> np.ndarray:
  #assert len(event.items())==60, "Number of cameras is not 60"
  cameras = np.empty((60,), dtype=np.float32)
  for i, (cam, img) in enumerate(event.items()):
    #print("cam :", cam)
    #print("pixel shape:", np.array(img.pixels, np.float32).shape)
    #assert np.array(img.pixels, np.float32).shape == (32,32,2), "Camera shape is not (32,32,2)" 
    times = np.array(img.pixels, np.float32)[:,:,1]
    non_zero_values = times[times != 0]

    if non_zero_values.size > 0:
        non_zero_mean = non_zero_values.mean()
    else:
        non_zero_mean = 0.0 
        print("Camera ", cam, "without signal")

    cameras[i] = non_zero_mean
  return cameras

def get_truth_data_fast(truths_dict: dict, geom: Geometry, event: int) -> np.ndarray:
  s = EdepSimDeposits(truths_dict[event], geom)
  s.voxelize()
  return s.voxels.voxels

if __name__ == "__main__":

  edepFile_new = "/storage/gpfs_data/neutrino/SAND-LAr/SAND-LAr-OPTICALSIM-PROD/GRAIN/TDR_numuCCQES/edepsim/new-skimmed-events-in-GRAIN_LAr_merged_reindex.edep-sim.root"
  edepFile_old = "/home/filippo/DUNE/data/numu-CC-QE/OLD_skimmed-events-in-GRAIN_LAr_merged_reindex.edep-sim.root"
  defs = {}
  defs['voxel_size'] = 150
  geometryPath = "/storage/gpfs_data/neutrino/SAND-LAr/SAND-LAr-GDML/MASK/GRAIN_box31_3cm_random_2row"        #path to GRAIN geometry
  geom = load_geometry(geometryPath, defs)

  response_base_path = Path('/storage/gpfs_data/neutrino/SAND-LAr/SAND-LAr-OPTICALSIM-PROD/GRAIN/TDR_numuCCQES/detresponse')
  # drdf_files = [f.name for f in response_base_path.glob('*.drdf') if f.is_file()]
  drdf_files = ["response28.drdf", "response29.drdf"]

  ROOT.gErrorIgnoreLevel = ROOT.kWarning

  events = []
  truths = []
  truths_dict = loadPrimariesEdepSimAll(edepFile_new)
  for file in drdf_files:
    reader = drdf.DRDF()
    reader.read(response_base_path / file)
    for run in reader.runs:
      for event in reader.runs[run]:
        cameras = get_response_data(reader.runs[run][event])
        truth = get_truth_data_fast(truths_dict, geom, event)
        events.append(cameras)
        truths.append(truth)
  
  events_data = np.stack(events, axis=0)
  truths_data = np.stack(truths, axis=0)
  print("events:", events_data.shape, events_data.nbytes / 1024 / 1024)
  print("truths:", truths_data.shape, truths_data.nbytes / 1024 / 1024)

  # Save to HDF5 file
  with h5py.File('/storage/gpfs_data/neutrino/SAND-LAr/SAND-LAr-EDEPSIM-PROD/new-numu-CC-QE-in-GRAIN/grain_numu_ccqe/pre-volumereco-data/test_15cm.h5', 'w') as f:
    f.create_dataset('inputs', data=events_data, compression='gzip')
    f.create_dataset('targets', data=truths_data, compression='gzip')