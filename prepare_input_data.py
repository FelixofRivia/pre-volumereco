import numpy as np
import drdf
import h5py
from pathlib import Path
import sys
sys.path.append('../tools_new/display')
from read_mctruth import loadPrimariesEdepSim
from geom_import import load_geometry
from geometry import Geometry
sys.path.append('../tools_new/analysis')
from edepsim_deposits import EdepSimDeposits


def get_response_data(event) -> np.ndarray:
  assert len(event.items())==60, "Number of cameras is not 60"
  cameras = np.empty((60,), dtype=np.float32)
  for i, (cam, img) in enumerate(event.items()):
    #print("cam :", cam)
    #print("pixel shape:", np.array(img.pixels, np.float32).shape)
    assert np.array(img.pixels, np.float32).shape == (32,32,2), "Camera shape is not (32,32,2)" 
    times = np.array(img.pixels, np.float32)[:,:,1]
    non_zero_values = times[times != 0]

    if non_zero_values.size > 0:
        non_zero_mean = non_zero_values.mean()
    else:
        non_zero_mean = 0.0 
        print("Camera ", cam, "without signal")

    cameras[i] = non_zero_mean
  return cameras

def get_truth_data(edepFile: str, geom: Geometry, event: int) -> np.ndarray:
  MCtruth = loadPrimariesEdepSim(edepFile, event)
  s = EdepSimDeposits(MCtruth, geom)
  s.voxelize()
  return s.voxels.voxels

if __name__ == "__main__":

  edepFile = "/home/filippo/DUNE/data/numu-CC-QE/skimmed-events-in-GRAIN_LAr_merged_reindex.edep-sim.root"
  defs = {}
  defs['voxel_size'] = 200
  geometryPath = "/home/filippo/DUNE/GEOMETRIES/GRAIN_official"        #path to GRAIN geometry
  geom = load_geometry(geometryPath, defs)

  response_base_path = Path('/home/filippo/DUNE/data/numu-CC-QE/detector_response')
  drdf_files = [f.name for f in response_base_path.glob('*.drdf') if f.is_file()]
  #drdf_files = ["/home/filippo/DUNE/data/numu-CC-QE/detector_response/response33.drdf"]

  print(drdf_files)

  events = []
  truths = []
  for file in drdf_files:
    reader = drdf.DRDF()
    reader.read(response_base_path / file)
    for run in reader.runs:
      for event in reader.runs[run]:
        if event<1200: print("event: ", event)
        cameras = get_response_data(reader.runs[run][event])
        truth = get_truth_data(edepFile, geom, event)
        events.append(cameras)
        truths.append(truth)
  
  events_data = np.stack(events, axis=0)
  truths_data = np.stack(truths, axis=0)
  print("events:", events_data.shape, events_data.nbytes / 1024 / 1024)
  print("truths:", truths_data.shape, truths_data.nbytes / 1024 / 1024)

  # Save to HDF5 file
  with h5py.File('/home/filippo/DUNE/data/numu-CC-QE/lightweight_dataset_20cm.h5', 'w') as f:
    f.create_dataset('inputs', data=events_data, compression='gzip')
    f.create_dataset('targets', data=truths_data, compression='gzip')