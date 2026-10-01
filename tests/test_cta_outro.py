import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import main


class CurrentPipelineGateTests(unittest.TestCase):
    def _pipeline_with_safe_mocks(self):
        pipeline = main.AutoPublishPipeline()
        pipeline.content_generator.generate_narration = lambda _topic: "نص عربي صالح للاختبار"
        report = SimpleNamespace(is_acceptable=True, overall_score=1.0, issues=[], warnings=[])
        pipeline.quality_checker.check_text = lambda text: (text, report)
        pipeline.quality_checker.check_audio = lambda _path, _text: report
        pipeline.voice_generator.generate = lambda _text, output_path: (Path(output_path), True)
        return pipeline

    def test_rejects_audio_outside_publish_window_before_assembly(self):
        pipeline = self._pipeline_with_safe_mocks()
        with patch.object(main, "probe_duration", return_value=10.0), patch.object(
            main, "assemble_video"
        ) as assemble:
            result = pipeline.run(topic="موضوع اختباري")

        self.assertFalse(result)
        assemble.assert_not_called()

    def test_broll_gate_blocks_publication(self):
        pipeline = self._pipeline_with_safe_mocks()
        with (
            patch.object(main, "probe_duration", return_value=60.0),
            patch.object(main, "assemble_video", return_value=Path("output/video.mp4")),
            patch.object(
                main,
                "evaluate_broll",
                return_value={"passed": False, "errors": ["black frame"], "broll": {"clip_count": 0}},
            ),
            patch.object(pipeline.publisher, "publish_to_buffer") as publish,
        ):
            result = pipeline.run(topic="موضوع اختباري")

        self.assertFalse(result)
        publish.assert_not_called()


if __name__ == "__main__":
    unittest.main()
