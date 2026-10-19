"""Composition invariants; run with the installed local NumPy/OpenCV runtime."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'local'))
from quality import blend_mouth,photo_motion,blink_amount,blink_starts,mouth_mask,rect

class QualityTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(4)
        self.frame=rng.integers(0,256,(800,600,3),dtype=np.uint8)
        self.face=[.2,.15,.6,.5]
        self.mouth=[.35,.43,.30,.10]
    def test_original_features_and_background_are_preserved(self):
        result=blend_mouth(self.frame,np.zeros((256,256,3),np.uint8),self.face,self.mouth)
        self.assertTrue(np.array_equal(result[:330],self.frame[:330]))
        self.assertTrue(np.array_equal(result[455:],self.frame[455:]))
        self.assertTrue(np.array_equal(result[:,:185],self.frame[:,:185]))
        self.assertFalse(np.array_equal(result[365:410,230:370],self.frame[365:410,230:370]))
    def test_motion_is_opt_in_and_eye_edit_is_local(self):
        self.assertTrue(np.array_equal(photo_motion(self.frame,self.face,1.1),self.frame))
        edited=photo_motion(self.frame,self.face,1.1,[[.30,.30,.12,.05],[.57,.30,.12,.05]],True)
        self.assertTrue(np.array_equal(edited[:200],self.frame[:200]))
        self.assertTrue(np.array_equal(edited[400:],self.frame[400:]))
        self.assertFalse(np.array_equal(edited[240:280],self.frame[240:280]))
    def test_bezier_timing_and_idle(self):
        starts=blink_starts();idle=np.diff(starts)-.35
        self.assertGreaterEqual(idle.min(),3.5);self.assertLessEqual(idle.max(),6.)
        self.assertAlmostEqual(blink_amount(starts[0]+.1),.78,places=4)
        self.assertEqual(blink_amount(starts[0]+.351),0)
    def test_poisson_and_feather_preserve_all_outside_pixels(self):
        x1,y1,x2,y2=rect(self.face,600,800)
        for blend in ('feather','poisson'):
            for scope in ('lips','jaw'):
                mask=np.zeros((800,600),bool);mask[y1:y2,x1:x2]=mouth_mask((y2-y1,x2-x1),self.face,self.mouth,scope)[:,:,0]>0
                out=blend_mouth(self.frame,np.zeros((512,512,3),np.uint8),self.face,self.mouth,blend=blend,scope=scope)
                self.assertTrue(np.array_equal(out[~mask],self.frame[~mask]))
    def test_body_presets_make_small_distinct_motion(self):
        results=[]
        for motion in ('breathe','sway','nod','hand'):
            edited=photo_motion(self.frame,self.face,1.0,motion=motion,hand=[.25,.6,.23,.1])
            self.assertEqual(edited.shape,self.frame.shape)
            self.assertFalse(np.array_equal(edited,self.frame),motion)
            results.append(edited)
        self.assertFalse(np.array_equal(results[0],results[1]))
if __name__=='__main__':unittest.main()
