import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'local'))
from face_geometry import normalized_box,decode_heatmaps,regions_from_landmarks,reliable_regions,regions_from_five
class FaceGeometry(unittest.TestCase):
 def test_heatmaps_map_to_model_grid_without_half_cell_bias(self):
  m=np.full((1,98,16,16),-1,np.float32);m[:,:,8,4]=1
  p=decode_heatmaps(m,[100,200,420,520])
  np.testing.assert_allclose(p,np.tile([180,360],(98,1)))
 def test_regions_clip_and_reject_bad_shapes(self):
  b=normalized_box([[-10,-20],[30,40]],100,80)
  self.assertTrue(all(0<=x<=1 for x in b));self.assertLessEqual(b[0]+b[2],1.000001);self.assertLessEqual(b[1]+b[3],1.000001)
  with self.assertRaises(ValueError):regions_from_landmarks(np.zeros((5,2)),[0,0,100,100],100,100)
 def test_cartoon_inconsistent_eye_region_requires_fallback(self):
  self.assertFalse(reliable_regions({'eye_boxes':[[.3,.2,.08,.04],[.25,.2,.35,.04]]},[.2,.1,.8,.8]))
  result=regions_from_five([[30,30],[70,30],[50,50],[35,70],[65,70]],[10,10,90,90],100,100)
  self.assertEqual(len(result['landmarks']),5);self.assertLess(result['eye_boxes'][0][0],result['eye_boxes'][1][0]);self.assertGreater(result['mouth_box'][1],result['eye_boxes'][0][1])
if __name__=='__main__':unittest.main()
