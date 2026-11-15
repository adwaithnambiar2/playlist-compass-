import unittest
from compass import recommend,load
class Tests(unittest.TestCase):
    def setUp(self): self.rows=load('data/demo_tracks.csv')
    def test_exclusions_and_uniqueness(self):
        out=recommend(self.rows,['track-000'],k=200)
        ids=[r['id'] for r in out]
        self.assertEqual(len(ids),119);self.assertEqual(len(set(ids)),119)
        self.assertNotIn('track-000',ids)
    def test_unit_invariance(self):
        a=recommend(self.rows,['track-000'])
        rows=[dict(r,tempo=float(r['tempo'])*60) for r in self.rows]
        self.assertEqual([r['id'] for r in a],[r['id'] for r in recommend(rows,['track-000'])])
    def test_zero_diversity_orders_relevance(self):
        scores=[r['relevance'] for r in recommend(self.rows,['track-000'],diversity=0)]
        self.assertEqual(scores,sorted(scores,reverse=True))
    def test_validation(self):
        for kw in [dict(seeds=['missing']),dict(seeds=[]),dict(seeds=['track-000'],weights=[-1]*5)]:
            with self.assertRaises(ValueError): recommend(self.rows,**kw)
if __name__=='__main__': unittest.main()
