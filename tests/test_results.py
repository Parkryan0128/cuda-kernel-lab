"""Host-only checks for interpretation errors in the results pipeline."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("summarize", Path(__file__).resolve().parents[1] / "scripts" / "summarize.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class SummaryTests(unittest.TestCase):
    def test_percentiles_are_interpolated(self):
        self.assertEqual(module.percentile([1,4,2,3], .5), 2.5)
        self.assertAlmostEqual(module.percentile([1,2,3,4], .1), 1.3)

    def test_refuses_missing_final_correctness_and_incomplete_samples(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "metadata.json").write_text(json.dumps({"samples":2,"profile_once":False,"l2_bytes":48}))
            case = "reduce/r0/device/256"
            evidence = [{"case":case,"phase":"before","passed":True}]
            (root / "correctness.jsonl").write_text(json.dumps(evidence[0]) + "\n")
            header = "run_id,operation,variant,mode,n_elements,m,n,k,seed,config,sample,repetitions,total_ms,per_operation_us,correctness,workspace_bytes,pass_count,working_set_bytes,input_bytes,output_bytes\n"
            row = "run,reduce,r0,device,256,0,0,0,2027,B256,{sample},10,0.02,2,passed,0,1,1028,1024,4\n"
            path = root / "samples.csv"
            path.write_text(header + row.format(sample=0))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                module.summarize_file(path)
            path.write_text(header + row.format(sample=0) + row.format(sample=1))
            with self.assertRaisesRegex(ValueError, "before/after"):
                module.summarize_file(path)
            evidence.append({"case":case,"phase":"after","passed":True})
            (root / "correctness.jsonl").write_text("\n".join(json.dumps(x) for x in evidence))
            summary = module.summarize_file(path)[0]
            self.assertEqual(summary["median_us"], 2)
            self.assertAlmostEqual(summary["input_GB_per_s"], .512)

if __name__ == "__main__":
    unittest.main()
