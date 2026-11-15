import importlib.util
import pathlib
import re
import sys
import unittest
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'application'))
from inference import attention_html
spec=importlib.util.spec_from_file_location('ui_rendering',ROOT/'application/app.py')
ui=importlib.util.module_from_spec(spec)
spec.loader.exec_module(ui)

class Presentation(unittest.TestCase):
    def test_probability_card_keeps_all_six_exact_probabilities(self):
        values=[.1,.6,.08,.07,.09,.06]
        result={'probabilities':dict(zip(ui.LABELS,values)),'selected_model':'distilbert'}
        card=ui.prediction_card(result)
        rows=re.findall(r'data-emotion="([a-z]+)" data-probability="([0-9.]+)"',card)
        self.assertEqual(len(rows),6)
        self.assertEqual(rows[0][0],'sadness')
        self.assertAlmostEqual(sum(float(p) for _,p in rows),1)
        self.assertIn('data-model="distilbert"',card)
        self.assertNotIn('certainty',card.lower())

    def test_disagreement_names_the_attention_model(self):
        card=ui.attention_card({'attention_prediction':'joy','prediction':'sadness','attention_status':'Main classifier uses 128 of 162 subwords.','attention_html':'<p>tokens</p>'})
        self.assertIn('BiLSTM + Attention',card)
        self.assertIn('Second opinion',card)
        self.assertIn('<h2>Words that stood out</h2>',card)
        self.assertIn('aria-label="About word focus"',card)
        self.assertIn('not an exact contribution to the prediction',card)
        self.assertIn('first 128 of 162',card)

    def test_attention_keeps_positions_and_escapes_input(self):
        card=attention_html('happy happy <script>',np.array([.2,.2,.2,.2,.2]),128)
        self.assertEqual(card.count('>happy</span>'),2)
        self.assertNotIn('<script>',card)
        self.assertEqual(re.findall(r'data-index="(\d+)"',card),['0','1','2','3','4'])
        self.assertIn('25% or higher',ui.attention_heading())
        self.assertNotIn('Most weight:',card)
        self.assertEqual(re.findall(r'class="e-weight">([^<]+)',card),['20%']*5)

    def test_attention_discloses_truncation_and_preserves_exact_weights(self):
        card=attention_html('happy '*160,np.ones(128)/128,128)
        self.assertIn('first 128 of 160 tokens',card)
        weights=[float(x) for x in re.findall(r'data-attention="([0-9.]+)"',card)]
        self.assertEqual(len(weights),128)
        self.assertAlmostEqual(sum(weights),1,places=5)
        ordered=attention_html('happy happy sad happy',np.array([.14,.04,.56,.26]),128)
        self.assertEqual(re.findall(r'class="e-weight">([^<]+)',ordered),['14%','4.0%','56%','26%'])
        zero=attention_html('a b',np.array([0.,1.]),128)
        self.assertIn('height:0.00px',zero)

if __name__=='__main__':unittest.main()
